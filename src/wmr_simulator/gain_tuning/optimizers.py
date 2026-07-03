import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_objective,
)

_NUM_GAINS = 6
_NUM_STABLE_GAINS = 4


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
    input_weight: float,
    input_delta_weight: float,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
):
    def loss_for_optimizer_values(values):
        gains = _controller_gains_from_optimizer_values(
            values,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
        )
        return closed_loop_objective(
            pipeline,
            gains,
            replay_robot_keys,
            replay_estimator_keys,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
        )

    return loss_for_optimizer_values


def _select_initial_optimizer_values(
    candidate_values: jax.Array,
    loss_for_optimizer_values,
    num_adam_optimizations: int,
) -> tuple[jax.Array, np.ndarray]:
    candidate_losses = jax.vmap(loss_for_optimizer_values)(candidate_values)
    candidate_losses = np.asarray(candidate_losses, dtype=float)
    num_starts = min(int(num_adam_optimizations), int(candidate_values.shape[0]))
    best_indices = np.argsort(candidate_losses)[:num_starts]
    return candidate_values[jnp.asarray(best_indices)], candidate_losses[best_indices]


def _run_adam_optimizer(
    initial_values: jax.Array,
    initial_losses: np.ndarray,
    loss_for_optimizer_values,
    num_steps: int,
    learning_rate: float,
) -> tuple[jax.Array, np.ndarray]:
    if num_steps <= 0:
        return initial_values, initial_losses[None, :]

    optimizer = optax.adam(learning_rate)
    current_opt_state = optimizer.init(initial_values)

    @scan_tqdm(num_steps, desc=f"Adam optimization ({initial_values.shape[0]} starts)")
    def train_step(carry, _):
        values, opt_state = carry
        loss_values, grads = jax.vmap(jax.value_and_grad(loss_for_optimizer_values))(values)
        updates, next_opt_state = optimizer.update(grads, opt_state, values)
        next_values = jnp.clip(optax.apply_updates(values, updates), min=0.0, max=1.0)
        return (next_values, next_opt_state), loss_values

    (final_values, _), adam_loss_history = jax.lax.scan(
        train_step,
        (initial_values, current_opt_state),
        jnp.arange(num_steps),
    )
    loss_history = jnp.concatenate([jnp.asarray(initial_losses, dtype=jnp.float32)[None, :], adam_loss_history], axis=0)
    return final_values, np.asarray(loss_history, dtype=float)


def optimize_controller_gains(
    pipeline,
    init_gains: jax.Array,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    k_min_stab: float = 1e-3,
    k_max_stab: float = 20.0,
    k_max_rest: float = 20.0,
    num_lhs_points: int = 0,
    num_adam_optimizations: int = 1,
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
        input_weight=input_weight,
        input_delta_weight=input_delta_weight,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
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
    initial_values, initial_losses = _select_initial_optimizer_values(
        candidate_values=candidate_values,
        loss_for_optimizer_values=loss_for_optimizer_values,
        num_adam_optimizations=num_adam_optimizations,
    )
    final_values, loss_history = _run_adam_optimizer(
        initial_values=initial_values,
        initial_losses=initial_losses,
        loss_for_optimizer_values=loss_for_optimizer_values,
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
    return best_gains, loss_history[:, best_index].tolist()
