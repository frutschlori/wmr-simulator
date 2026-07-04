import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_objective,
    closed_loop_objective_terms,
)

_NUM_GAINS = 6
_NUM_STABLE_GAINS = 4
_GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor", "kdmotor")
_LOSS_COMPONENT_NAMES = ("tracking", "velocity_tracking", "input", "input_delta")


def _controller_gains_to_optimizer_values(
    gains: jax.Array,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
):
    gains = clip_controller_gains(jnp.asarray(gains, dtype=jnp.float32))
    log_ratio = jnp.log(k_max_stab / k_min_stab)
    stable = jnp.clip(gains[..., :_NUM_STABLE_GAINS], min=k_min_stab, max=k_max_stab)
    stable_values = jnp.log(stable / k_min_stab) / log_ratio
    rest = jnp.clip(gains[..., _NUM_STABLE_GAINS:] / k_max_rest, min=0.0, max=1.0)
    rest_values = jnp.sqrt(rest)
    return jnp.clip(jnp.concatenate([stable_values, rest_values], axis=-1), min=0.0, max=1.0)


def _controller_gains_from_optimizer_values(
    values: jax.Array,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
):
    values = jnp.clip(jnp.asarray(values, dtype=jnp.float32), min=0.0, max=1.0)
    stable_values = values[..., :_NUM_STABLE_GAINS]
    rest_values = values[..., _NUM_STABLE_GAINS:]
    stable = k_min_stab * (k_max_stab / k_min_stab) ** stable_values
    rest = k_max_rest * rest_values**2
    return jnp.concatenate([stable, rest], axis=-1)


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
    zero_d = values.at[:, 5].set(0.0)
    zero_i_d = zero_i.at[:, 5].set(0.0)
    return jnp.concatenate([values, zero_i, zero_d, zero_i_d], axis=0)


def _candidate_optimizer_values(
    pipeline,
    init_values: jax.Array,
    num_lhs_points: int,
) -> jax.Array:
    if num_lhs_points <= 0:
        return init_values[None, :]
    lhs_key = jax.random.fold_in(pipeline.robot_key, 1729)
    lhs_values = _latin_hypercube_samples(lhs_key, num_lhs_points, _NUM_GAINS)
    base_values = jnp.concatenate([init_values[None, :], lhs_values], axis=0)
    return _with_motor_zero_variants(base_values)


def _make_loss_for_optimizer_values(
    pipeline,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    velocity_tracking_weight: float,
    input_weight: float,
    input_delta_weight: float,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
    reference_trajectories: jax.Array | None = None,
):
    reference_trajectories = (
        None if reference_trajectories is None else jnp.asarray(reference_trajectories, dtype=jnp.float32)
    )

    def loss_for_optimizer_values(values):
        gains = _controller_gains_from_optimizer_values(
            values,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
        )

        def loss_for_reference(reference_states):
            return closed_loop_objective(
                pipeline,
                gains,
                replay_robot_keys,
                replay_estimator_keys,
                velocity_tracking_weight=velocity_tracking_weight,
                input_weight=input_weight,
                input_delta_weight=input_delta_weight,
                reference_states=reference_states,
            )

        if reference_trajectories is None:
            return loss_for_reference(pipeline.reference_states)
        return jnp.mean(jax.vmap(loss_for_reference)(reference_trajectories))

    return loss_for_optimizer_values


def _make_loss_terms_for_optimizer_values(
    pipeline,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    velocity_tracking_weight: float,
    input_weight: float,
    input_delta_weight: float,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
    reference_trajectories: jax.Array | None = None,
):
    reference_trajectories = (
        None if reference_trajectories is None else jnp.asarray(reference_trajectories, dtype=jnp.float32)
    )

    def loss_terms_for_optimizer_values(values):
        gains = _controller_gains_from_optimizer_values(
            values,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
        )

        def terms_for_reference(reference_states):
            return closed_loop_objective_terms(
                pipeline,
                gains,
                replay_robot_keys,
                replay_estimator_keys,
                velocity_tracking_weight=velocity_tracking_weight,
                input_weight=input_weight,
                input_delta_weight=input_delta_weight,
                reference_states=reference_states,
            )

        if reference_trajectories is None:
            return terms_for_reference(pipeline.reference_states)
        return jnp.mean(jax.vmap(terms_for_reference)(reference_trajectories), axis=0)

    return loss_terms_for_optimizer_values


def _select_initial_optimizer_values(
    candidate_values: jax.Array,
    loss_for_optimizer_values,
    num_adam_optimizations: int,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
) -> tuple[jax.Array, np.ndarray]:
    candidate_losses = jax.vmap(loss_for_optimizer_values)(candidate_values)
    candidate_losses = np.asarray(candidate_losses, dtype=float)
    num_starts = min(int(num_adam_optimizations), int(candidate_values.shape[0]))
    best_indices = np.argsort(candidate_losses)[:num_starts]
    print("Finished LHS candidate evaluation.")
    print("Best LHS candidates:")
    top_count = min(3, int(candidate_values.shape[0]))
    top_indices = best_indices[:top_count]
    top_gains = np.asarray(
        _controller_gains_from_optimizer_values(
            candidate_values[jnp.asarray(top_indices)],
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
        ),
        dtype=float,
    )
    for rank, (candidate_index, gains) in enumerate(zip(top_indices, top_gains), start=1):
        gain_text = ", ".join(f"{name}={value:.7g}" for name, value in zip(_GAIN_NAMES, gains))
        print(f"  {rank}. loss={candidate_losses[candidate_index]:.8f}  {gain_text}")
    return candidate_values[jnp.asarray(best_indices)], candidate_losses[best_indices]


def _run_adam_optimizer(
    initial_values: jax.Array,
    initial_losses: np.ndarray,
    loss_for_optimizer_values,
    loss_terms_for_optimizer_values,
    validation_loss_for_optimizer_values,
    validation_loss_terms_for_optimizer_values,
    num_steps: int,
    learning_rate: float,
) -> tuple[jax.Array, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    initial_loss_terms = np.asarray(
        jax.vmap(loss_terms_for_optimizer_values)(initial_values),
        dtype=float,
    )
    initial_validation_losses = None
    initial_validation_terms = None
    if validation_loss_for_optimizer_values is not None:
        initial_validation_losses = np.asarray(
            jax.vmap(validation_loss_for_optimizer_values)(initial_values),
            dtype=float,
        )
        initial_validation_terms = np.asarray(
            jax.vmap(validation_loss_terms_for_optimizer_values)(initial_values),
            dtype=float,
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
        next_values = jnp.clip(optax.apply_updates(values, updates), min=0.0, max=1.0)
        return (next_values, next_opt_state), (loss_values, validation_values, loss_terms, validation_terms)

    (final_values, _), (adam_loss_history, adam_validation_history, adam_terms_history, adam_validation_terms_history) = jax.lax.scan(
        train_step,
        (initial_values, current_opt_state),
        jnp.arange(num_steps),
    )
    loss_history = jnp.concatenate([jnp.asarray(initial_losses, dtype=jnp.float32)[None, :], adam_loss_history], axis=0)
    loss_terms_history = jnp.concatenate(
        [jnp.asarray(initial_loss_terms, dtype=jnp.float32)[None, :, :], adam_terms_history],
        axis=0,
    )
    validation_history = None
    validation_terms_history = None
    if initial_validation_losses is not None:
        validation_history = jnp.concatenate(
            [jnp.asarray(initial_validation_losses, dtype=jnp.float32)[None, :], adam_validation_history],
            axis=0,
        )
        validation_history = np.asarray(validation_history, dtype=float)
        validation_terms_history = jnp.concatenate(
            [jnp.asarray(initial_validation_terms, dtype=jnp.float32)[None, :, :], adam_validation_terms_history],
            axis=0,
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
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    k_min_stab: float = 1e-3,
    k_max_stab: float = 20.0,
    k_max_rest: float = 20.0,
    num_lhs_points: int = 0,
    num_adam_optimizations: int = 1,
    training_reference_trajectories: jax.Array | None = None,
    validation_reference_trajectories: jax.Array | None = None,
):
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

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_realizations)
    loss_for_optimizer_values = _make_loss_for_optimizer_values(
        pipeline=pipeline,
        replay_robot_keys=replay_robot_keys,
        replay_estimator_keys=replay_estimator_keys,
        velocity_tracking_weight=velocity_tracking_weight,
        input_weight=input_weight,
        input_delta_weight=input_delta_weight,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
        reference_trajectories=training_reference_trajectories,
    )
    loss_terms_for_optimizer_values = _make_loss_terms_for_optimizer_values(
        pipeline=pipeline,
        replay_robot_keys=replay_robot_keys,
        replay_estimator_keys=replay_estimator_keys,
        velocity_tracking_weight=velocity_tracking_weight,
        input_weight=input_weight,
        input_delta_weight=input_delta_weight,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
        reference_trajectories=training_reference_trajectories,
    )
    validation_loss_for_optimizer_values = None
    validation_loss_terms_for_optimizer_values = None
    if validation_reference_trajectories is not None and len(validation_reference_trajectories) > 0:
        validation_loss_for_optimizer_values = _make_loss_for_optimizer_values(
            pipeline=pipeline,
            replay_robot_keys=replay_robot_keys,
            replay_estimator_keys=replay_estimator_keys,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            reference_trajectories=validation_reference_trajectories,
        )
        validation_loss_terms_for_optimizer_values = _make_loss_terms_for_optimizer_values(
            pipeline=pipeline,
            replay_robot_keys=replay_robot_keys,
            replay_estimator_keys=replay_estimator_keys,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            reference_trajectories=validation_reference_trajectories,
        )
    init_values = _controller_gains_to_optimizer_values(
        init_gains,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
    )
    candidate_values = _candidate_optimizer_values(
        pipeline=pipeline,
        init_values=init_values,
        num_lhs_points=num_lhs_points,
    )
    if num_lhs_points > 0:
        print(
            "Starting LHS candidate evaluation "
            f"({num_lhs_points} LHS points * (1, ki=0, kd=0, ki=kd=0) = {candidate_values.shape[0]} candidates."
        )
    else:
        print("Starting initial gain candidate evaluation (LHS disabled).")
    initial_values, initial_losses = _select_initial_optimizer_values(
        candidate_values=candidate_values,
        loss_for_optimizer_values=loss_for_optimizer_values,
        num_adam_optimizations=num_adam_optimizations,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
    )
    final_values, loss_history, validation_loss_history, loss_terms_history, validation_terms_history = _run_adam_optimizer(
        initial_values=initial_values,
        initial_losses=initial_losses,
        loss_for_optimizer_values=loss_for_optimizer_values,
        loss_terms_for_optimizer_values=loss_terms_for_optimizer_values,
        validation_loss_for_optimizer_values=validation_loss_for_optimizer_values,
        validation_loss_terms_for_optimizer_values=validation_loss_terms_for_optimizer_values,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    final_losses = loss_history[-1]
    best_index = int(np.argmin(final_losses))
    best_values = final_values[best_index]
    best_gains = _controller_gains_from_optimizer_values(
        best_values,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
    )
    selected_validation_history = None
    if validation_loss_history is not None:
        selected_validation_history = validation_loss_history[:, best_index].tolist()
    selected_loss_terms_history = loss_terms_history[:, best_index, :]
    selected_validation_terms_history = None
    if validation_terms_history is not None:
        selected_validation_terms_history = validation_terms_history[:, best_index, :]
    return (
        best_gains,
        loss_history[:, best_index].tolist(),
        selected_validation_history,
        {
            name: selected_loss_terms_history[:, index].tolist()
            for index, name in enumerate(_LOSS_COMPONENT_NAMES)
        },
        None
        if selected_validation_terms_history is None
        else {
            name: selected_validation_terms_history[:, index].tolist()
            for index, name in enumerate(_LOSS_COMPONENT_NAMES)
        },
    )
