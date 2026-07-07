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


# Optimizer vector layout: 5 log-relative dims for the positive physical parameters
# (r, L_effective, u_max, tau_motor, a_slip_max).
# Notes:
#   - base_diameter is the *effective* wheelbase; tire-scrub in turns is absorbed
#     into it because a separate correction would be structurally non-identifiable
#     (Borenstein & Feng 1996, E_b).
#   - a_slip_max = init * exp(theta): a zero init keeps the burnout model disabled
#     (0 * exp(theta) = 0 with zero gradient) -- pass a positive init to identify it.
_NUM_POSITIVE_DIMS = 5
_NUM_OPTIMIZER_DIMS = 5


def _params_from_optimizer_values(values: jax.Array, init_params: PhysicalParams) -> PhysicalParams:
    # Log-relative for positive params: value = init * exp(theta) stays positive without clipping.
    scale_values = physical_params_to_array(init_params)[..., :_NUM_POSITIVE_DIMS]
    log_relative = values[..., :_NUM_POSITIVE_DIMS]
    positive = scale_values * jnp.exp(log_relative)
    return PhysicalParams(
        wheel_radius=positive[..., 0],
        base_diameter=positive[..., 1],
        max_wheel_speed=positive[..., 2],
        time_constant=positive[..., 3],
        a_slip_max=positive[..., 4],
    )


def _loss_normalization_scale(initial_loss: jax.Array) -> jax.Array:
    return 1.0 / jnp.maximum(initial_loss, 1e-12)


def optimize_physical_params_adam(
    pipeline,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
):
    optimizer = optax.adam(learning_rate)
    current_log_relative = jnp.zeros(_NUM_OPTIMIZER_DIMS, dtype=jnp.float32)
    opt_state = optimizer.init(current_log_relative)

    if num_steps <= 0:
        return init_params, [], [], []

    def params_from_optimizer_values(values):
        return _params_from_optimizer_values(values, init_params)

    def pose_loss_for_params(params):
        return pose_window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=pipeline.target_log,
            est_params=pipeline.initial_params,
            window_length=pipeline.window_length,
        )

    def motor_loss_for_params(params):
        return pipeline.motor_wheel_speed_mse(params, pipeline.target_log, window_length=pipeline.window_length)

    initial_params = params_from_optimizer_values(current_log_relative)
    pose_loss_scale = _loss_normalization_scale(pose_loss_for_params(initial_params))
    motor_loss_scale = _loss_normalization_scale(motor_loss_for_params(initial_params))

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        log_relative, current_opt_state = carry

        def normalized_loss(next_log_relative):
            params = params_from_optimizer_values(next_log_relative)
            pose_loss_value = pose_loss_scale * pose_loss_for_params(params)
            motor_loss_value = motor_loss_scale * motor_loss_for_params(params)
            return pose_loss_value + motor_loss_value, (pose_loss_value, motor_loss_value)

        (loss_value, (pose_loss_value, motor_loss_value)), grads = jax.value_and_grad(
            normalized_loss,
            has_aux=True,
        )(log_relative)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, log_relative)
        next_log_relative = optax.apply_updates(log_relative, updates)
        next_params = params_from_optimizer_values(next_log_relative)
        parameter_mse = physical_params_mse(next_params, pipeline.hidden_params)
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


def empirical_moments(samples: jax.Array) -> tuple[jax.Array, jax.Array]:
    samples = jnp.asarray(samples)
    mean = jnp.mean(samples, axis=0)
    centered = samples - mean
    denominator = jnp.maximum(samples.shape[0] - 1, 1)
    covariance = centered.T @ centered / denominator
    return mean, covariance


def _make_bootstrap_batch_optimizer(
    pipeline,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
):
    optimizer = optax.adam(learning_rate)
    initial_params = init_params
    replay_segment_plan = pipeline.replay_segment_plan

    def make_target_log(robot_key, estimator_key):
        return pipeline.run_target_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            wheel_speed_log_source=pipeline.replay_wheel_speed_source,
        )

    def params_from_optimizer_values(values):
        return _params_from_optimizer_values(values, initial_params)

    def pose_loss_one(
        log_relative,
        target_log,
    ):
        params = params_from_optimizer_values(log_relative)
        return pose_window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=target_log,
            est_params=pipeline.initial_params,
            window_length=pipeline.window_length,
            replay_segment_plan=replay_segment_plan,
        )

    def motor_loss_one(log_relative, target_log):
        params = params_from_optimizer_values(log_relative)
        return pipeline.motor_wheel_speed_mse(params, target_log, window_length=pipeline.window_length)

    def normalized_loss_one(
        log_relative,
        target_log,
        pose_loss_scale,
        motor_loss_scale,
    ):
        pose_loss_value = pose_loss_scale * pose_loss_one(
            log_relative,
            target_log,
        )
        motor_loss_value = motor_loss_scale * motor_loss_one(log_relative, target_log)
        return pose_loss_value + motor_loss_value, (pose_loss_value, motor_loss_value)

    batched_pose_loss = jax.vmap(pose_loss_one)
    batched_motor_loss = jax.vmap(motor_loss_one)
    batched_value_and_grad = jax.vmap(jax.value_and_grad(normalized_loss_one, has_aux=True))

    @jax.jit
    def optimize_batch(
        target_robot_keys: jax.Array,
        target_estimator_keys: jax.Array,
    ):
        batch_size = target_robot_keys.shape[0]
        current_log_relative = jnp.zeros((batch_size, _NUM_OPTIMIZER_DIMS), dtype=jnp.float32)
        current_opt_state = optimizer.init(current_log_relative)
        target_logs = jax.vmap(make_target_log)(target_robot_keys, target_estimator_keys)
        pose_loss_scales = _loss_normalization_scale(
            batched_pose_loss(
                current_log_relative,
                target_logs,
            )
        )
        motor_loss_scales = _loss_normalization_scale(
            batched_motor_loss(
                current_log_relative,
                target_logs,
            )
        )

        @scan_tqdm(num_steps, desc="Bootstrap optimization")
        def train_step(carry, step):
            log_relative, opt_state = carry
            (_, (pose_loss_values, motor_loss_values)), grads = batched_value_and_grad(
                log_relative,
                target_logs,
                pose_loss_scales,
                motor_loss_scales,
            )
            updates, next_opt_state = optimizer.update(grads, opt_state, log_relative)
            next_log_relative = optax.apply_updates(log_relative, updates)
            next_params = params_from_optimizer_values(next_log_relative)
            parameter_mse_values = physical_params_mse(next_params, pipeline.hidden_params)
            return (next_log_relative, next_opt_state), (pose_loss_values, motor_loss_values, parameter_mse_values)

        (
            current_log_relative,
            _,
        ), (geometry_loss_history, motor_loss_history, parameter_mse_history) = jax.lax.scan(
            train_step,
            (current_log_relative, current_opt_state),
            jnp.arange(num_steps),
        )
        estimated_params = params_from_optimizer_values(current_log_relative)

        parameter_samples = physical_params_to_array(estimated_params)
        return (
            estimated_params,
            parameter_samples,
            jnp.swapaxes(geometry_loss_history, 0, 1),
            jnp.swapaxes(motor_loss_history, 0, 1),
            jnp.swapaxes(parameter_mse_history, 0, 1),
        )

    return optimize_batch


def bootstrap_identification_adam(
    pipeline,
    initial_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    bootstrap_samples: int,
    seed: int = 0,
):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")

    master_key = jax.random.PRNGKey(seed)
    target_key, _ = jax.random.split(master_key, 2)
    target_robot_key, target_estimator_key = jax.random.split(target_key, 2)
    target_robot_keys = jax.random.split(target_robot_key, bootstrap_samples)
    target_estimator_keys = jax.random.split(target_estimator_key, bootstrap_samples)

    if num_steps <= 0:
        estimated_params = PhysicalParams(
            *(
                jnp.full((bootstrap_samples,), value, dtype=jnp.float32)
                for value in physical_params_to_array(initial_params)
            )
        )
        parameter_samples = physical_params_to_array(estimated_params)
        parameter_mean, parameter_covariance = empirical_moments(parameter_samples)
        return {
            "estimated_params": estimated_params,
            "parameter_samples": parameter_samples,
            "parameter_mean": parameter_mean,
            "parameter_covariance": parameter_covariance,
            "loss_history": jnp.empty((bootstrap_samples, 0), dtype=jnp.float32),
            "motor_loss_history": jnp.empty((bootstrap_samples, 0), dtype=jnp.float32),
            "parameter_mse_history": jnp.empty((bootstrap_samples, 0), dtype=jnp.float32),
            "target_robot_key": target_robot_keys[0],
            "target_estimator_key": target_estimator_keys[0],
        }

    optimize_batch = _make_bootstrap_batch_optimizer(
        pipeline=pipeline,
        init_params=initial_params,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    estimated_params, parameter_samples, loss_history, motor_loss_history, parameter_mse_history = optimize_batch(
        target_robot_keys,
        target_estimator_keys,
    )

    parameter_mean, parameter_covariance = empirical_moments(parameter_samples)
    return {
        "estimated_params": estimated_params,
        "parameter_samples": parameter_samples,
        "parameter_mean": parameter_mean,
        "parameter_covariance": parameter_covariance,
        "loss_history": loss_history,
        "motor_loss_history": motor_loss_history,
        "parameter_mse_history": parameter_mse_history,
        "target_robot_key": target_robot_keys[0],
        "target_estimator_key": target_estimator_keys[0],
    }
