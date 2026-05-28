import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_tracking_mse,
)


def optimize_controller_gains(
    pipeline,
    init_gains: jax.Array,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
):
    optimizer = optax.adam(learning_rate)
    current_gains = clip_controller_gains(init_gains)
    current_opt_state = optimizer.init(current_gains)

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_realizations)

    if num_steps <= 0:
        return current_gains, []

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        gains, opt_state = carry
        loss_value, grads = jax.value_and_grad(
            lambda current_gains: closed_loop_tracking_mse(
                pipeline,
                current_gains,
                replay_robot_keys,
                replay_estimator_keys,
            )
        )(gains)
        updates, next_opt_state = optimizer.update(grads, opt_state, gains)
        next_gains = optax.apply_updates(gains, updates)
        next_gains = clip_controller_gains(next_gains)
        return (next_gains, next_opt_state), loss_value

    (current_gains, _), loss_history = jax.lax.scan(
        train_step,
        (current_gains, current_opt_state),
        jnp.arange(num_steps),
    )

    return current_gains, np.asarray(loss_history, dtype=float).tolist()
