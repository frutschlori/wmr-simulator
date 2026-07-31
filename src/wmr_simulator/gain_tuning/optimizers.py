import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.controller import (
    ACTIVE_GAIN_INDICES,
    GAIN_NAMES,
    NUM_GAINS,
    ZERO_ALLOWED_GAIN_INDICES,
    active_gain_mask,
)
from wmr_simulator.gain_parametrization import flat_params as gain_parametrization_flat_params
from wmr_simulator.gain_parametrization import num_params as gain_parametrization_num_params
from wmr_simulator.gain_parametrization import with_flat_params, zero_params
from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    scheduled_closed_loop_objective_terms,
)

_NUM_GAINS = NUM_GAINS
_GAIN_NAMES = GAIN_NAMES
_LOSS_COMPONENT_NAMES = ("tracking", "velocity_tracking", "input", "input_delta", "omega_delta", "gain_delta")

# Per-gain search space. A gain that must stay strictly positive for the closed
# loop to be stable is searched in log space: scale-free resolution across the
# decades between k_min_stab and k_max_stab, and it can never reach 0. The
# integral gain is the one gain that is allowed to be exactly 0 (integral action
# off), which log space cannot express, so it is searched in sqrt space over
# [0, k_max_rest] instead — that also puts most of the resolution near 0, where
# the useful values are.
_LOG_SPACE_GAIN_MASK = jnp.asarray(
    [index not in ZERO_ALLOWED_GAIN_INDICES for index in range(NUM_GAINS)], dtype=bool
)


# ---------------------------------------------------------------------------
# Optimizer-value <-> controller-gain reparametrization (base gains only)
#
# The trainable vector is [gain_values(NUM_GAINS), parametrization_flat(num_w)].
# The leading entries are the base gain vector in a bounded reparam space (see
# _LOG_SPACE_GAIN_MASK); they are clipped to [0, 1]. The trailing
# parametrization entries are linear (can be negative) and are not clipped here;
# each parametrization is responsible for bounding its own effect.
# ---------------------------------------------------------------------------


def _controller_gains_to_optimizer_values(gains, k_min_stab, k_max_stab, k_max_rest):
    gains = clip_controller_gains(jnp.asarray(gains, dtype=jnp.float32))
    log_ratio = jnp.log(k_max_stab / k_min_stab)
    log_values = jnp.log(jnp.clip(gains, min=k_min_stab, max=k_max_stab) / k_min_stab) / log_ratio
    sqrt_values = jnp.sqrt(jnp.clip(gains / k_max_rest, min=0.0, max=1.0))
    values = jnp.where(_LOG_SPACE_GAIN_MASK, log_values, sqrt_values)
    return jnp.clip(values, min=0.0, max=1.0)


def _controller_gains_from_optimizer_values(values, k_min_stab, k_max_stab, k_max_rest):
    values = jnp.clip(jnp.asarray(values, dtype=jnp.float32), min=0.0, max=1.0)
    log_gains = k_min_stab * (k_max_stab / k_min_stab) ** values
    sqrt_gains = k_max_rest * values**2
    return jnp.where(_LOG_SPACE_GAIN_MASK, log_gains, sqrt_gains)


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
    """Duplicate every candidate with the integral gain(s) switched off."""
    if values.shape[0] == 0:
        return values
    zero_i = values.at[:, jnp.asarray(ZERO_ALLOWED_GAIN_INDICES)].set(0.0)
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
    trainable_gain_mask=None,
):
    """Build candidate vectors of width (NUM_GAINS + num_w). LHS searches the gain part.

    ``init_gain_values`` may hold several rows; each becomes its own candidate.
    When ``presearch_relative_range`` > 0 the LHS samples a +/- band around the
    (first) init gains instead of the full [k_min_stab, k_max_stab] range. The
    parametrization part is the same for every candidate: ``w_init`` (identity
    when None), so the presearch scores the gains with the parametrization it
    will be optimized with.

    Gains outside ``trainable_gain_mask`` (the ones the active control law never
    reads) are held at their init value rather than wasting LHS dimensions.
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
        if trainable_gain_mask is not None:
            lhs_values = jnp.where(trainable_gain_mask, lhs_values, init_gain_values[0])
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
    trainable_mask=None,
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
        return initial_values, initial_losses[None, :], validation_history, initial_loss_terms[None, :, :], validation_terms_history

    optimizer = optax.adam(learning_rate)
    current_opt_state = optimizer.init(initial_values)

    def loss_with_terms(values):
        terms = loss_terms_for_optimizer_values(values)
        return jnp.sum(terms), terms

    @scan_tqdm(num_steps, desc=f"Adam optimization ({initial_values.shape[0]} starts)")
    def train_step(carry, _):
        values, opt_state = carry
        (loss_values, loss_terms), grads = jax.vmap(jax.value_and_grad(loss_with_terms, has_aux=True))(values)
        if trainable_mask is not None:
            # Freeze the entries the active control law never reads: zero their
            # gradient so Adam's moments stay at 0 and the values never move.
            grads = jnp.where(trainable_mask, grads, 0.0)
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
        updates, next_opt_state = optimizer.update(grads, opt_state, values)
        next_values = _clip_optimizer_values(optax.apply_updates(values, updates))
        return (next_values, next_opt_state), (loss_values, validation_values, loss_terms, validation_terms)

    (final_values, _), (adam_loss_history, adam_validation_history, adam_terms_history, adam_validation_terms_history) = jax.lax.scan(
        train_step,
        (initial_values, current_opt_state),
        jnp.arange(num_steps),
    )
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
        final_values,
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
    training_reference_trajectories: jax.Array | None = None,
    validation_reference_trajectories: jax.Array | None = None,
):
    """Single-stage joint optimization of base gains and the gain schedule.

    The trainable vector is ``[gain_values(NUM_GAINS), parametrization_flat(num_w)]``.
    LHS presearch and multistart operate on the gain part, with the
    parametrization held at its start value (the warm-started template, else the
    identity mapping) so the candidates are scored under the controller that
    Adam then refines jointly.
    When ``schedule_enabled`` is False, ``num_w = 0`` and the parametrization is
    fixed at its identity mapping, reproducing the static controller.

    Which gains move depends on ``pipeline``'s controller type
    (``controller.ACTIVE_GAIN_INDICES``): the Kanayama law tunes
    [kx, ky, kth, kpmotor, kimotor], the dynamic-feedback law tunes
    [kpmotor, kimotor, kp_x, kd_x, kp_y, kd_y]. The others are held at their
    init value.

    ``init_gains`` may be a single gain vector or a batch ``(N, NUM_GAINS)`` of
    them (each becomes its own candidate/start).

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

    # Only the gains the pipeline's control law actually reads are searched; the
    # rest keep their init value (they have no effect on the rollout, so leaving
    # them free would spend LHS dimensions and write meaningless numbers into
    # the result).
    controller_type = pipeline.controller.controller_type
    trainable_gain_mask = jnp.asarray(active_gain_mask(controller_type), dtype=bool)
    trainable_mask = jnp.concatenate([trainable_gain_mask, jnp.ones(num_w, dtype=bool)])
    frozen = [name for name, active in zip(_GAIN_NAMES, active_gain_mask(controller_type)) if not active]
    print(
        f"Tuning the '{controller_type}' controller: "
        f"{', '.join(name for name, active in zip(_GAIN_NAMES, active_gain_mask(controller_type)) if active)}"
        + (f" (held fixed: {', '.join(frozen)})" if frozen else "")
    )

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_realizations)

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
        trainable_gain_mask=trainable_gain_mask,
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
    final_values, loss_history, validation_loss_history, loss_terms_history, validation_terms_history = _run_adam_optimizer(
        initial_values=initial_values,
        initial_losses=initial_losses,
        loss_terms_for_optimizer_values=loss_terms_for_optimizer_values,
        validation_loss_for_optimizer_values=validation_loss_for_optimizer_values,
        validation_loss_terms_for_optimizer_values=validation_loss_terms_for_optimizer_values,
        num_steps=num_steps,
        learning_rate=learning_rate,
        trainable_mask=trainable_mask,
    )
    final_losses = loss_history[-1]
    best_index = int(np.argmin(final_losses))
    best_values = final_values[best_index]
    final_gains_per_start = _controller_gains_from_optimizer_values(
        final_values[:, :_NUM_GAINS], k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    # Report the frozen gains exactly as configured. They never moved, but the
    # round trip through the bounded reparametrization is only float32-accurate
    # (10.0 -> 10.00001), and that error would otherwise accumulate over the
    # iterations of the active-learning loop.
    final_gains_per_start = jnp.where(
        trainable_gain_mask, final_gains_per_start, jnp.atleast_2d(jnp.asarray(init_gains, dtype=jnp.float32))[0]
    )
    best_gains = final_gains_per_start[best_index]
    if schedule_enabled:
        best_schedule_params = with_flat_params(best_values[_NUM_GAINS:], schedule_template)
    else:
        best_schedule_params = None

    return {
        "gains": best_gains,
        "schedule_params": best_schedule_params,
        "final_gains_per_start": final_gains_per_start,
        "best_start_index": best_index,
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
