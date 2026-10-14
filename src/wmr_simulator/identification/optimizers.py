import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.identification.losses import pose_window_replay_mse
from wmr_simulator.types import (
    PhysicalParams,
    physical_params_mse,
    physical_params_to_array,
)


# Optimizer vector: log-relative values of (r, L_effective, u_max, tau_motor).
_NUM_OPTIMIZER_DIMS = 4


def _params_from_optimizer_values(values: jax.Array, init_params: PhysicalParams) -> PhysicalParams:
    # Log-relative for positive params: value = init * exp(theta) stays positive without clipping.
    positive = physical_params_to_array(init_params) * jnp.exp(values)
    return PhysicalParams(
        wheel_radius=positive[..., 0],
        base_diameter=positive[..., 1],
        max_wheel_speed=positive[..., 2],
        time_constant=positive[..., 3],
    )


def _loss_normalization_scale(initial_loss: jax.Array) -> jax.Array:
    return 1.0 / jnp.maximum(initial_loss, 1e-12)


def optimize_physical_params_adam(
    pipelines,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
):
    """Identify one parameter set from one or more target logs.

    The loss is the mean over pipelines of each one's pose and motor loss,
    normalized by its value at the initial parameters.
    """
    pipelines = list(pipelines)
    if not pipelines:
        raise ValueError("At least one identification pipeline is required.")
    optimizer = optax.adam(learning_rate)
    current_log_relative = jnp.zeros(_NUM_OPTIMIZER_DIMS, dtype=jnp.float32)
    opt_state = optimizer.init(current_log_relative)

    if num_steps <= 0:
        return init_params, [], [], []

    def params_from_optimizer_values(values):
        return _params_from_optimizer_values(values, init_params)

    def pose_loss_for_params(pipeline, params):
        return pose_window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=pipeline.target_log,
            est_params=pipeline.initial_params,
            window_length=pipeline.window_length,
            replay_segment_plan=pipeline.replay_segment_plan,
        )

    def motor_loss_for_params(pipeline, params):
        return pipeline.motor_wheel_speed_mse(params, pipeline.target_log, window_length=pipeline.window_length)

    initial_params = params_from_optimizer_values(current_log_relative)
    pose_loss_scales = [
        _loss_normalization_scale(pose_loss_for_params(pipeline, initial_params)) for pipeline in pipelines
    ]
    motor_loss_scales = [
        _loss_normalization_scale(motor_loss_for_params(pipeline, initial_params)) for pipeline in pipelines
    ]

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        log_relative, current_opt_state = carry

        def normalized_loss(next_log_relative):
            params = params_from_optimizer_values(next_log_relative)
            pose_loss_value = jnp.mean(
                jnp.stack(
                    [
                        scale * pose_loss_for_params(pipeline, params)
                        for pipeline, scale in zip(pipelines, pose_loss_scales)
                    ]
                )
            )
            motor_loss_value = jnp.mean(
                jnp.stack(
                    [
                        scale * motor_loss_for_params(pipeline, params)
                        for pipeline, scale in zip(pipelines, motor_loss_scales)
                    ]
                )
            )
            return pose_loss_value + motor_loss_value, (pose_loss_value, motor_loss_value)

        (loss_value, (pose_loss_value, motor_loss_value)), grads = jax.value_and_grad(
            normalized_loss,
            has_aux=True,
        )(log_relative)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, log_relative)
        next_log_relative = optax.apply_updates(log_relative, updates)
        next_params = params_from_optimizer_values(next_log_relative)
        parameter_mse = physical_params_mse(next_params, pipelines[0].hidden_params)
        return (next_log_relative, next_opt_state), (pose_loss_value, motor_loss_value, parameter_mse)

    (
        current_log_relative,
        _,
    ), (pose_loss_history, motor_loss_history, parameter_mse_history) = jax.lax.scan(
        train_step,
        (current_log_relative, opt_state),
        jnp.arange(num_steps),
    )
    current_params = params_from_optimizer_values(current_log_relative)

    return (
        current_params,
        np.asarray(pose_loss_history, dtype=float).tolist(),
        np.asarray(motor_loss_history, dtype=float).tolist(),
        np.asarray(parameter_mse_history, dtype=float).tolist(),
    )
