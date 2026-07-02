import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_objective,
)


def _controller_gains_to_optimizer_values(gains: jax.Array):
    return clip_controller_gains(gains)


def _controller_gains_from_optimizer_values(values: jax.Array):
    return jnp.clip(values, min=0)


def optimize_controller_gains(
    pipeline,
    init_gains: jax.Array,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
):
    optimizer = optax.adam(learning_rate)
    current_values = _controller_gains_to_optimizer_values(init_gains)
    current_opt_state = optimizer.init(current_values)

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_realizations)

    if num_steps <= 0:
        return _controller_gains_from_optimizer_values(current_values), []

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        values, opt_state = carry

        def loss_for_optimizer_values(current_values):
            current_gains = _controller_gains_from_optimizer_values(current_values)
            return closed_loop_objective(
                pipeline,
                current_gains,
                replay_robot_keys,
                replay_estimator_keys,
                input_weight=input_weight,
                input_delta_weight=input_delta_weight,
            )

        loss_value, grads = jax.value_and_grad(
            loss_for_optimizer_values
        )(values)
        updates, next_opt_state = optimizer.update(grads, opt_state, values)
        next_values = jnp.clip(optax.apply_updates(values, updates), min=0)
        return (next_values, next_opt_state), loss_value

    (current_values, _), loss_history = jax.lax.scan(
        train_step,
        (current_values, current_opt_state),
        jnp.arange(num_steps),
    )

    current_gains = _controller_gains_from_optimizer_values(current_values)
    return current_gains, np.asarray(loss_history, dtype=float).tolist()
