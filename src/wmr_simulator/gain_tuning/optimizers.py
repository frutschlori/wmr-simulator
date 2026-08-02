import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.gain_parametrization import flat_params as gain_parametrization_flat_params
from wmr_simulator.gain_parametrization import num_params as gain_parametrization_num_params
from wmr_simulator.gain_parametrization import with_flat_params, zero_params
from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    sample_initial_pose_offsets,
    scheduled_closed_loop_objective_terms,
)

_NUM_GAINS = 5
_NUM_STABLE_GAINS = 4
_GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor")
_LOSS_COMPONENT_NAMES = ("tracking", "velocity_tracking", "input", "input_delta", "omega_delta", "gain_delta")


# ---------------------------------------------------------------------------
# Optimizer-value <-> controller-gain reparametrization (base gains only)
#
# The trainable vector is [gain_values(5), parametrization_flat(num_w)]. The first five entries
# are the base 5-gain vector in a bounded reparam space (log-space for the four
# "stable" gains, sqrt-space for the motor I gain); they are clipped to [0, 1].
# The trailing parametrization entries are linear (can be negative) and are not
# clipped here; each parametrization is responsible for bounding its own effect.
# ---------------------------------------------------------------------------


def _controller_gains_to_optimizer_values(gains, k_min_stab, k_max_stab, k_max_rest):
    gains = clip_controller_gains(jnp.asarray(gains, dtype=jnp.float32))
    log_ratio = jnp.log(k_max_stab / k_min_stab)
    stable = jnp.clip(gains[..., :_NUM_STABLE_GAINS], min=k_min_stab, max=k_max_stab)
    stable_values = jnp.log(stable / k_min_stab) / log_ratio
    rest = jnp.clip(gains[..., _NUM_STABLE_GAINS:] / k_max_rest, min=0.0, max=1.0)
    rest_values = jnp.sqrt(rest)
    return jnp.clip(jnp.concatenate([stable_values, rest_values], axis=-1), min=0.0, max=1.0)


def _controller_gains_from_optimizer_values(values, k_min_stab, k_max_stab, k_max_rest):
    values = jnp.clip(jnp.asarray(values, dtype=jnp.float32), min=0.0, max=1.0)
    stable_values = values[..., :_NUM_STABLE_GAINS]
    rest_values = values[..., _NUM_STABLE_GAINS:]
    stable = k_min_stab * (k_max_stab / k_min_stab) ** stable_values
    rest = k_max_rest * rest_values**2
    return jnp.concatenate([stable, rest], axis=-1)


def _clip_optimizer_values(values):
    """Clip only the gain part of the (possibly augmented) optimizer vector."""
    gain_part = jnp.clip(values[..., :_NUM_GAINS], min=0.0, max=1.0)
    w_part = values[..., _NUM_GAINS:]
    return jnp.concatenate([gain_part, w_part], axis=-1)


def _latin_hypercube_samples(key: jax.Array, num_points: int, num_dims: int) -> jax.Array:
    if num_points <= 0:
        return jnp.empty((0, num_dims), dtype=jnp.float32)
    keys = jax.random.split(key, num_dims + 1)
    jitter = jax.random.uniform(keys[0], (num_points, num_dims), dtype=jnp.float32)
    permutations = jax.vmap(lambda dim_key: jax.random.permutation(dim_key, num_points))(keys[1:])
    bins = jnp.swapaxes(permutations, 0, 1).astype(jnp.float32)
    return (bins + jitter) / float(num_points)


def _with_motor_zero_variants(values: jax.Array) -> jax.Array:
    if values.shape[0] == 0:
        return values
    zero_i = values.at[:, 4].set(0.0)
    return jnp.concatenate([values, zero_i], axis=0)


def _relative_lhs_values(unit_samples, center_values, relative_range, k_min_stab, k_max_stab, k_max_rest):
    """Map unit LHS samples into a per-gain band of +/- ``relative_range`` around
    the center gains (in gain space), returned as optimizer values.

    The gain transform is monotonic, so the gain-space band maps to a per-dim
    optimizer-space box; a disabled gain (0) keeps a zero-width band.
    """
    center_gains = _controller_gains_from_optimizer_values(
        center_values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    lo_values = _controller_gains_to_optimizer_values(
        center_gains * (1.0 - relative_range), k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    hi_values = _controller_gains_to_optimizer_values(
        center_gains * (1.0 + relative_range), k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    return lo_values + unit_samples * (hi_values - lo_values)


def _candidate_optimizer_values(
    pipeline,
    init_gain_values,
    num_w,
    num_lhs_points,
    presearch_relative_range=0.0,
    k_min_stab=1e-3,
    k_max_stab=20.0,
    k_max_rest=20.0,
    w_init=None,
):
    """Build candidate vectors of width (5 + num_w). LHS searches the gain part.

    ``init_gain_values`` may hold several rows; each becomes its own candidate.
    When ``presearch_relative_range`` > 0 the LHS samples a +/- band around the
    (first) init gains instead of the full [k_min_stab, k_max_stab] range. The
    parametrization part is the same for every candidate: ``w_init`` (identity
    when None), so the presearch scores the gains with the parametrization it
    will be optimized with.
    """
    init_gain_values = jnp.atleast_2d(init_gain_values)
    if num_lhs_points <= 0:
        gain_candidates = init_gain_values
    else:
        lhs_key = jax.random.fold_in(pipeline.robot_key, 1729)
        lhs_unit = _latin_hypercube_samples(lhs_key, num_lhs_points, _NUM_GAINS)
        if presearch_relative_range and presearch_relative_range > 0.0:
            lhs_values = _relative_lhs_values(
                lhs_unit, init_gain_values[0], presearch_relative_range, k_min_stab, k_max_stab, k_max_rest
            )
        else:
            lhs_values = lhs_unit
        base_values = jnp.concatenate([init_gain_values, lhs_values], axis=0)
        gain_candidates = _with_motor_zero_variants(base_values)
    if w_init is None:
        w_part = jnp.zeros((gain_candidates.shape[0], num_w), dtype=jnp.float32)
    else:
        w_part = jnp.broadcast_to(
            jnp.asarray(w_init, dtype=jnp.float32).reshape(1, num_w), (gain_candidates.shape[0], num_w)
        )
    return jnp.concatenate([gain_candidates, w_part], axis=-1)


def _make_terms_for_values(
    pipeline,
    schedule_template,
    schedule_enabled,
    num_w,
    replay_robot_keys,
    replay_estimator_keys,
    velocity_tracking_weight,
    input_weight,
    input_delta_weight,
    omega_delta_weight,
    gain_delta_weight,
    k_min_stab,
    k_max_stab,
    k_max_rest,
    reference_trajectories,
    initial_pose_offsets,
):
    reference_trajectories = (
        None if reference_trajectories is None else jnp.asarray(reference_trajectories, dtype=jnp.float32)
    )
    def terms_for_values(values):
        gains = _controller_gains_from_optimizer_values(
            values[..., :_NUM_GAINS], k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
        )
        if schedule_enabled:
            params = with_flat_params(values[..., _NUM_GAINS:], schedule_template)
        else:
            params = zero_params(schedule_template)

        def terms_for_reference(reference_states):
            return scheduled_closed_loop_objective_terms(
                pipeline,
                gains,
                params,
                replay_robot_keys,
                replay_estimator_keys,
                velocity_tracking_weight=velocity_tracking_weight,
                input_weight=input_weight,
                input_delta_weight=input_delta_weight,
                omega_delta_weight=omega_delta_weight,
                gain_delta_weight=gain_delta_weight,
                reference_states=reference_states,
                initial_pose_offsets=initial_pose_offsets,
            )

        if reference_trajectories is None:
            return terms_for_reference(pipeline.reference_states)
        return jnp.mean(jax.vmap(terms_for_reference)(reference_trajectories), axis=0)

    return terms_for_values


def _select_initial_optimizer_values(
    candidate_values,
    loss_for_optimizer_values,
    num_adam_optimizations,
    k_min_stab,
    k_max_stab,
    k_max_rest,
    lhs_enabled,
):
    candidate_losses = jax.vmap(loss_for_optimizer_values)(candidate_values)
    candidate_losses = np.asarray(candidate_losses, dtype=float)
    num_starts = min(int(num_adam_optimizations), int(candidate_values.shape[0]))
    best_indices = np.argsort(candidate_losses)[:num_starts]
    if lhs_enabled:
        print("Finished LHS candidate evaluation.")
        print("Best LHS candidates:")
    else:
        print("Evaluated initial gain candidates (no LHS search):")
    top_count = min(3, int(candidate_values.shape[0]))
    top_indices = best_indices[:top_count]
    top_gains = np.asarray(
        _controller_gains_from_optimizer_values(
            candidate_values[jnp.asarray(top_indices), :_NUM_GAINS],
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
        ),
        dtype=float,
    )
    for rank, (candidate_index, gains) in enumerate(zip(top_indices, top_gains), start=1):
        gain_text = ", ".join(f"{name}={value:.7g}" for name, value in zip(_GAIN_NAMES, gains))
        print(f"  {rank}. loss={candidate_losses[candidate_index]:.8f}  {gain_text}")
    return candidate_values[jnp.asarray(best_indices)], candidate_losses[best_indices], best_indices


def _run_adam_optimizer(
    initial_values,
    initial_losses,
    loss_terms_for_optimizer_values,
    validation_loss_for_optimizer_values,
    validation_loss_terms_for_optimizer_values,
    num_steps,
    learning_rate,
):
    initial_loss_terms = np.asarray(jax.vmap(loss_terms_for_optimizer_values)(initial_values), dtype=float)
    initial_validation_losses = None
    initial_validation_terms = None
    if validation_loss_for_optimizer_values is not None:
        initial_validation_losses = np.asarray(
            jax.vmap(validation_loss_for_optimizer_values)(initial_values), dtype=float
        )
        initial_validation_terms = np.asarray(
            jax.vmap(validation_loss_terms_for_optimizer_values)(initial_values), dtype=float
        )
    if num_steps <= 0:
        validation_history = None if initial_validation_losses is None else initial_validation_losses[None, :]
        validation_terms_history = None if initial_validation_terms is None else initial_validation_terms[None, :, :]
        selection_scores = (
            initial_losses if initial_validation_losses is None else initial_validation_losses
        )
        return (
            initial_values,
            np.asarray(selection_scores, dtype=float),
            initial_losses[None, :],
            validation_history,
            initial_loss_terms[None, :, :],
            validation_terms_history,
        )

    optimizer = optax.adam(learning_rate)
    current_opt_state = optimizer.init(initial_values)

    def loss_with_terms(values):
        terms = loss_terms_for_optimizer_values(values)
        return jnp.sum(terms), terms

    # Score used to keep the best iterate: the validation loss when there is a
    # validation set, else the training loss.
    def score_of(loss_values, validation_values):
        return loss_values if validation_loss_for_optimizer_values is None else validation_values

    @scan_tqdm(num_steps, desc=f"Adam optimization ({initial_values.shape[0]} starts)")
    def train_step(carry, _):
        values, opt_state, best_values, best_score = carry
        (loss_values, loss_terms), grads = jax.vmap(jax.value_and_grad(loss_with_terms, has_aux=True))(values)
        validation_values = (
            jnp.full_like(loss_values, jnp.nan)
            if validation_loss_for_optimizer_values is None
            else jax.vmap(validation_loss_for_optimizer_values)(values)
        )
        validation_terms = (
            jnp.full_like(loss_terms, jnp.nan)
            if validation_loss_terms_for_optimizer_values is None
            else jax.vmap(validation_loss_terms_for_optimizer_values)(values)
        )
        # Track the best iterate *seen*, per start. A fixed step size on this
        # objective can sit on a cliff edge (large kx blows the rollout up), so
        # the last iterate is not reliably the best one -- it can be an order of
        # magnitude worse than a point the run already passed through.
        score = score_of(loss_values, validation_values)
        improved = score < best_score
        next_best_score = jnp.where(improved, score, best_score)
        next_best_values = jnp.where(improved[:, None], values, best_values)
        updates, next_opt_state = optimizer.update(grads, opt_state, values)
        next_values = _clip_optimizer_values(optax.apply_updates(values, updates))
        return (
            (next_values, next_opt_state, next_best_values, next_best_score),
            (loss_values, validation_values, loss_terms, validation_terms),
        )

    initial_score = score_of(
        jnp.asarray(initial_losses, dtype=jnp.float32),
        jnp.full((initial_values.shape[0],), jnp.inf, dtype=jnp.float32)
        if initial_validation_losses is None
        else jnp.asarray(initial_validation_losses, dtype=jnp.float32),
    )
    (final_values, _, best_values, best_score), (
        adam_loss_history,
        adam_validation_history,
        adam_terms_history,
        adam_validation_terms_history,
    ) = jax.lax.scan(
        train_step,
        (initial_values, current_opt_state, initial_values, initial_score),
        jnp.arange(num_steps),
    )
    # The scan scores each iterate before its update, so the final iterate has
    # not been scored yet; fold it in so a run that improved to the very end
    # keeps its last step.
    final_score = score_of(
        jax.vmap(lambda v: jnp.sum(loss_terms_for_optimizer_values(v)))(final_values),
        jnp.full((final_values.shape[0],), jnp.inf, dtype=jnp.float32)
        if validation_loss_for_optimizer_values is None
        else jax.vmap(validation_loss_for_optimizer_values)(final_values),
    )
    improved = final_score < best_score
    best_values = jnp.where(improved[:, None], final_values, best_values)
    best_score = jnp.where(improved, final_score, best_score)
    loss_history = jnp.concatenate([jnp.asarray(initial_losses, dtype=jnp.float32)[None, :], adam_loss_history], axis=0)
    loss_terms_history = jnp.concatenate(
        [jnp.asarray(initial_loss_terms, dtype=jnp.float32)[None, :, :], adam_terms_history], axis=0
    )
    validation_history = None
    validation_terms_history = None
    if initial_validation_losses is not None:
        validation_history = jnp.concatenate(
            [jnp.asarray(initial_validation_losses, dtype=jnp.float32)[None, :], adam_validation_history], axis=0
        )
        validation_history = np.asarray(validation_history, dtype=float)
        validation_terms_history = jnp.concatenate(
            [jnp.asarray(initial_validation_terms, dtype=jnp.float32)[None, :, :], adam_validation_terms_history], axis=0
        )
        validation_terms_history = np.asarray(validation_terms_history, dtype=float)
    return (
        best_values,
        np.asarray(best_score, dtype=float),
        np.asarray(loss_history, dtype=float),
        validation_history,
        np.asarray(loss_terms_history, dtype=float),
        validation_terms_history,
    )


def optimize_controller_gains(
    pipeline,
    init_gains: jax.Array,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
    schedule_template,
    schedule_enabled: bool = False,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    gain_delta_weight: float = 0.0,
    k_min_stab: float = 1e-3,
    k_max_stab: float = 20.0,
    k_max_rest: float = 20.0,
    num_lhs_points: int = 0,
    num_adam_optimizations: int = 1,
    presearch_relative_range: float = 0.0,
    warm_start_schedule: bool = False,
    init_offset_radius: float = 0.0,
    init_offset_angle: float = 0.0,
    training_reference_trajectories: jax.Array | None = None,
    validation_reference_trajectories: jax.Array | None = None,
):
    """Single-stage joint optimization of base gains and the gain schedule.

    The trainable vector is ``[gain_values(5), parametrization_flat(num_w)]``. LHS
    presearch and multistart operate on the gain part, with the parametrization
    held at its start value (the warm-started template, else the identity
    mapping) so the candidates are scored under the controller that Adam then
    refines jointly.
    When ``schedule_enabled`` is False, ``num_w = 0`` and the parametrization is
    fixed at its identity mapping, reproducing the static controller.

    ``init_gains`` may be a single 5-gain vector or a batch ``(N, 5)`` of them
    (each becomes its own candidate/start).

    Returns a dict with the best start's gains/parametrization plus the raw
    per-start loss histories and selection metadata (``best_start_index``,
    ``start_candidate_indices``, ``final_gains_per_start``); use
    :func:`histories_for_start` to extract one start's history lists.
    """
    if k_min_stab <= 0.0:
        raise ValueError("k_min_stab must be positive for log-space optimization.")
    if k_max_stab <= k_min_stab:
        raise ValueError("k_max_stab must be larger than k_min_stab.")
    if k_max_rest <= 0.0:
        raise ValueError("k_max_rest must be positive.")
    if num_lhs_points < 0:
        raise ValueError("num_lhs_points must be non-negative.")
    if num_adam_optimizations <= 0:
        raise ValueError("num_adam_optimizations must be positive.")

    num_w = gain_parametrization_num_params(schedule_template) if schedule_enabled else 0

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_realizations)
    # One start-pose offset per realization, drawn once and held fixed for the
    # whole run (like the noise keys) so the objective stays a deterministic
    # function of the gains. Folded off the robot key with its own tag so the
    # offsets do not correlate with the measurement-noise draws.
    initial_pose_offsets = sample_initial_pose_offsets(
        jax.random.fold_in(pipeline.robot_key, 5813),
        num_realizations,
        init_offset_radius,
        init_offset_angle,
    )
    if init_offset_radius > 0.0 or init_offset_angle > 0.0:
        print(
            f"Randomized start poses: {num_realizations} offsets within "
            f"{init_offset_radius:.4g} m / {np.rad2deg(init_offset_angle):.3g} deg of the reference start."
        )

    def make_terms(reference_trajectories):
        return _make_terms_for_values(
            pipeline=pipeline,
            schedule_template=schedule_template,
            schedule_enabled=schedule_enabled,
            num_w=num_w,
            replay_robot_keys=replay_robot_keys,
            replay_estimator_keys=replay_estimator_keys,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            omega_delta_weight=omega_delta_weight,
            gain_delta_weight=gain_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            reference_trajectories=reference_trajectories,
            initial_pose_offsets=initial_pose_offsets,
        )

    loss_terms_for_optimizer_values = make_terms(training_reference_trajectories)
    loss_for_optimizer_values = lambda values: jnp.sum(loss_terms_for_optimizer_values(values))

    validation_loss_for_optimizer_values = None
    validation_loss_terms_for_optimizer_values = None
    if validation_reference_trajectories is not None and len(validation_reference_trajectories) > 0:
        validation_loss_terms_for_optimizer_values = make_terms(validation_reference_trajectories)
        validation_loss_for_optimizer_values = (
            lambda values: jnp.sum(validation_loss_terms_for_optimizer_values(values))
        )

    init_gain_values = _controller_gains_to_optimizer_values(
        init_gains, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    # Warm-start the parametrization from the template (e.g. the previous
    # iteration's trained schedule) instead of the identity mapping.
    w_init = gain_parametrization_flat_params(schedule_template) if (schedule_enabled and warm_start_schedule) else None
    candidate_values = _candidate_optimizer_values(
        pipeline=pipeline,
        init_gain_values=init_gain_values,
        num_w=num_w,
        num_lhs_points=num_lhs_points,
        presearch_relative_range=presearch_relative_range,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
        w_init=w_init,
    )
    if w_init is not None:
        print("Warm-starting the gain parametrization from the template (previous result).")
    schedule_state = "enabled" if schedule_enabled else "disabled"
    if num_lhs_points > 0:
        band = (
            f"+/-{100.0 * presearch_relative_range:.0f}% around the init gains"
            if presearch_relative_range and presearch_relative_range > 0.0
            else "full [k_min_stab, k_max_stab] range"
        )
        print(
            "Starting LHS candidate evaluation "
            f"({num_lhs_points} LHS points * (1, ki=0) = {candidate_values.shape[0]} candidates; "
            f"presearch {band}; gain schedule {schedule_state})."
        )
    else:
        print(f"Starting initial gain candidate evaluation (LHS disabled; gain schedule {schedule_state}).")
    initial_values, initial_losses, start_candidate_indices = _select_initial_optimizer_values(
        candidate_values=candidate_values,
        loss_for_optimizer_values=loss_for_optimizer_values,
        num_adam_optimizations=num_adam_optimizations,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
        lhs_enabled=num_lhs_points > 0,
    )
    (
        final_values,
        selection_scores,
        loss_history,
        validation_loss_history,
        loss_terms_history,
        validation_terms_history,
    ) = _run_adam_optimizer(
        initial_values=initial_values,
        initial_losses=initial_losses,
        loss_terms_for_optimizer_values=loss_terms_for_optimizer_values,
        validation_loss_for_optimizer_values=validation_loss_for_optimizer_values,
        validation_loss_terms_for_optimizer_values=validation_loss_terms_for_optimizer_values,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    # Pick the winning start on the validation trajectories when there are any.
    # Selecting on the training loss rewards the start that fit its own noise
    # draw / trajectory split best: measured over seeds 0-3, the lowest-training-
    # loss run was also the worst-generalizing one. `selection_scores` already
    # falls back to the training loss when there is no validation set, and it is
    # the score of each start's *best* iterate, not of its last one.
    best_index = int(np.argmin(selection_scores))
    best_values = final_values[best_index]
    final_gains_per_start = _controller_gains_from_optimizer_values(
        final_values[:, :_NUM_GAINS], k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    best_gains = final_gains_per_start[best_index]
    # Losses *of the returned point*. The loss histories are the raw per-step
    # traces, so their last entry belongs to the last iterate, which is not the
    # one returned; reporting it would advertise a number the exported gains do
    # not achieve.
    best_training_loss = float(loss_for_optimizer_values(best_values))
    best_validation_loss = (
        None
        if validation_loss_for_optimizer_values is None
        else float(validation_loss_for_optimizer_values(best_values))
    )
    print(
        f"Best start {best_index}: training loss {best_training_loss:.8f}"
        + ("" if best_validation_loss is None else f", validation loss {best_validation_loss:.8f}")
    )
    if schedule_enabled:
        best_schedule_params = with_flat_params(best_values[_NUM_GAINS:], schedule_template)
    else:
        best_schedule_params = None

    return {
        "gains": best_gains,
        "schedule_params": best_schedule_params,
        "final_gains_per_start": final_gains_per_start,
        "best_start_index": best_index,
        "best_training_loss": best_training_loss,
        "best_validation_loss": best_validation_loss,
        "start_candidate_indices": np.asarray(start_candidate_indices, dtype=int),
        "loss_history_per_start": loss_history,
        "validation_loss_history_per_start": validation_loss_history,
        "loss_terms_history_per_start": loss_terms_history,
        "validation_loss_terms_history_per_start": validation_terms_history,
    }


def histories_for_start(optimization_result: dict, start_index: int):
    """Extract one start's history lists from an optimization result dict.

    Returns ``(loss_history, validation_loss_history, loss_component_history,
    validation_loss_component_history)`` in the list/dict format consumed by
    reporting and plotting.
    """
    loss_history = optimization_result["loss_history_per_start"][:, start_index].tolist()
    validation_per_start = optimization_result["validation_loss_history_per_start"]
    validation_loss_history = None if validation_per_start is None else validation_per_start[:, start_index].tolist()
    terms = optimization_result["loss_terms_history_per_start"][:, start_index, :]
    loss_component_history = {name: terms[:, index].tolist() for index, name in enumerate(_LOSS_COMPONENT_NAMES)}
    validation_terms = optimization_result["validation_loss_terms_history_per_start"]
    validation_loss_component_history = None
    if validation_terms is not None:
        validation_loss_component_history = {
            name: validation_terms[:, start_index, index].tolist()
            for index, name in enumerate(_LOSS_COMPONENT_NAMES)
        }
    return loss_history, validation_loss_history, loss_component_history, validation_loss_component_history
