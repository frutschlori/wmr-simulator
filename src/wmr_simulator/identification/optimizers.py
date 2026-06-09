import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.identification.losses import pose_window_replay_mse
from wmr_simulator.types import (
    PhysicalParams,
    clip_physical_params,
    physical_params_mse,
    physical_params_to_array,
)


def _params_from_geometry_and_motor_relative(
    geometry_params: jax.Array,
    motor_relative: jax.Array,
    scale_params: PhysicalParams,
) -> PhysicalParams:
    return clip_physical_params(
        PhysicalParams(
            wheel_radius=geometry_params[..., 0],
            base_diameter=geometry_params[..., 1],
            max_wheel_speed=scale_params.max_wheel_speed * motor_relative[..., 0],
            time_constant=scale_params.time_constant * motor_relative[..., 1],
        )
    )


def _geometry_params_to_array(params: PhysicalParams) -> jax.Array:
    return jnp.stack([params.wheel_radius, params.base_diameter], axis=-1)


def _clip_geometry_params(values: jax.Array) -> jax.Array:
    return jnp.clip(values, min=1e-4)


def _clip_relative_params(values: jax.Array) -> jax.Array:
    return jnp.clip(values, min=1e-6)


def _loss_normalization_scale(initial_loss: jax.Array) -> jax.Array:
    return 1.0 / jnp.maximum(initial_loss, 1e-12)


def optimize_physical_params_adam(
    pipeline,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    motor_learning_rate: float,
    num_realizations: int,
):
    geometry_optimizer = optax.adam(learning_rate)
    motor_optimizer = optax.adam(motor_learning_rate)
    scale_params = clip_physical_params(init_params)
    current_geometry_params = _geometry_params_to_array(scale_params)
    current_motor_relative = jnp.ones(2, dtype=jnp.float32)
    current_geometry_opt_state = geometry_optimizer.init(current_geometry_params)
    current_motor_opt_state = motor_optimizer.init(current_motor_relative)
    num_replay_realizations = pipeline.resolve_replay_realizations(num_realizations)

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_replay_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_replay_realizations)

    if num_steps <= 0:
        return scale_params, [], [], []

    def params_from_optimizer_values(next_geometry_params, next_motor_relative):
        return _params_from_geometry_and_motor_relative(
            next_geometry_params,
            next_motor_relative,
            scale_params,
        )

    def unnormalized_geometry_loss(next_geometry_params, motor_relative):
        params = params_from_optimizer_values(next_geometry_params, motor_relative)
        return pose_window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=pipeline.target_log,
            replay_robot_keys=replay_robot_keys,
            replay_estimator_keys=replay_estimator_keys,
            est_params=pipeline.initial_params,
            window_length=pipeline.window_length,
        )

    def unnormalized_motor_loss(geometry_params, next_motor_relative):
        params = params_from_optimizer_values(geometry_params, next_motor_relative)
        return pipeline.motor_wheel_speed_mse(params, pipeline.target_log)

    geometry_loss_scale = _loss_normalization_scale(
        unnormalized_geometry_loss(current_geometry_params, current_motor_relative)
    )
    motor_loss_scale = _loss_normalization_scale(
        unnormalized_motor_loss(current_geometry_params, current_motor_relative)
    )

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        geometry_params, motor_relative, geometry_opt_state, motor_opt_state = carry

        def geometry_loss(next_geometry_params):
            return geometry_loss_scale * unnormalized_geometry_loss(next_geometry_params, motor_relative)

        def motor_loss(next_motor_relative):
            return motor_loss_scale * unnormalized_motor_loss(geometry_params, next_motor_relative)

        pose_loss_value, geometry_grads = jax.value_and_grad(geometry_loss)(geometry_params)
        motor_loss_value, motor_grads = jax.value_and_grad(motor_loss)(motor_relative)

        geometry_updates, next_geometry_opt_state = geometry_optimizer.update(
            geometry_grads,
            geometry_opt_state,
            geometry_params,
        )
        motor_updates, next_motor_opt_state = motor_optimizer.update(
            motor_grads,
            motor_opt_state,
            motor_relative,
        )
        next_geometry_params = _clip_geometry_params(optax.apply_updates(geometry_params, geometry_updates))
        next_motor_relative = _clip_relative_params(optax.apply_updates(motor_relative, motor_updates))
        next_params = params_from_optimizer_values(next_geometry_params, next_motor_relative)
        parameter_mse = physical_params_mse(next_params, pipeline.hidden_params)
        return (
            next_geometry_params,
            next_motor_relative,
            next_geometry_opt_state,
            next_motor_opt_state,
        ), (pose_loss_value, motor_loss_value, parameter_mse)

    (
        current_geometry_params,
        current_motor_relative,
        _,
        _,
    ), (geometry_loss_history, motor_loss_history, parameter_mse_history) = jax.lax.scan(
        train_step,
        (
            current_geometry_params,
            current_motor_relative,
            current_geometry_opt_state,
            current_motor_opt_state,
        ),
        jnp.arange(num_steps),
    )
    current_params = _params_from_geometry_and_motor_relative(
        current_geometry_params,
        current_motor_relative,
        scale_params,
    )

    return (
        current_params,
        np.asarray(geometry_loss_history, dtype=float).tolist(),
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


def _bootstrap_replay_keys(replay_keys: jax.Array, num_replay_realizations: int):
    def replay_keys_for_sample(sample_key):
        robot_key, estimator_key = jax.random.split(sample_key, 2)
        return (
            jax.random.split(robot_key, num_replay_realizations),
            jax.random.split(estimator_key, num_replay_realizations),
        )

    return jax.vmap(replay_keys_for_sample)(replay_keys)


def _make_bootstrap_batch_optimizer(
    pipeline,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    motor_learning_rate: float,
):
    geometry_optimizer = optax.adam(learning_rate)
    motor_optimizer = optax.adam(motor_learning_rate)
    initial_params = clip_physical_params(init_params)

    def make_target_log(robot_key, estimator_key):
        return pipeline.run_target_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
        )

    def params_from_optimizer_values(geometry_params, motor_relative):
        return _params_from_geometry_and_motor_relative(
            geometry_params,
            motor_relative,
            initial_params,
        )

    def pose_loss_one(
        geometry_params,
        motor_relative,
        target_log,
        sample_replay_robot_keys,
        sample_replay_estimator_keys,
    ):
        params = params_from_optimizer_values(geometry_params, motor_relative)
        return pose_window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=target_log,
            replay_robot_keys=sample_replay_robot_keys,
            replay_estimator_keys=sample_replay_estimator_keys,
            est_params=pipeline.initial_params,
            window_length=pipeline.window_length,
        )

    def motor_loss_one(geometry_params, motor_relative, target_log):
        params = params_from_optimizer_values(geometry_params, motor_relative)
        return pipeline.motor_wheel_speed_mse(params, target_log)

    def normalized_pose_loss_one(
        geometry_params,
        motor_relative,
        target_log,
        sample_replay_robot_keys,
        sample_replay_estimator_keys,
        loss_scale,
    ):
        return loss_scale * pose_loss_one(
            geometry_params,
            motor_relative,
            target_log,
            sample_replay_robot_keys,
            sample_replay_estimator_keys,
        )

    def normalized_motor_loss_one(geometry_params, motor_relative, target_log, loss_scale):
        return loss_scale * motor_loss_one(geometry_params, motor_relative, target_log)

    batched_pose_loss = jax.vmap(pose_loss_one)
    batched_motor_loss = jax.vmap(motor_loss_one)
    batched_pose_value_and_grad = jax.vmap(jax.value_and_grad(normalized_pose_loss_one, argnums=0))
    batched_motor_value_and_grad = jax.vmap(jax.value_and_grad(normalized_motor_loss_one, argnums=1))

    @jax.jit
    def optimize_batch(
        target_robot_keys: jax.Array,
        target_estimator_keys: jax.Array,
        replay_robot_keys: jax.Array,
        replay_estimator_keys: jax.Array,
    ):
        batch_size = target_robot_keys.shape[0]
        initial_geometry_params = _geometry_params_to_array(initial_params)
        current_geometry_params = jnp.broadcast_to(initial_geometry_params, (batch_size, 2))
        current_motor_relative = jnp.ones((batch_size, 2), dtype=jnp.float32)
        current_geometry_opt_state = geometry_optimizer.init(current_geometry_params)
        current_motor_opt_state = motor_optimizer.init(current_motor_relative)
        target_logs = jax.vmap(make_target_log)(target_robot_keys, target_estimator_keys)
        geometry_loss_scales = _loss_normalization_scale(
            batched_pose_loss(
                current_geometry_params,
                current_motor_relative,
                target_logs,
                replay_robot_keys,
                replay_estimator_keys,
            )
        )
        motor_loss_scales = _loss_normalization_scale(
            batched_motor_loss(
                current_geometry_params,
                current_motor_relative,
                target_logs,
            )
        )

        @scan_tqdm(num_steps, desc="Bootstrap optimization")
        def train_step(carry, step):
            geometry_params, motor_relative, geometry_opt_state, motor_opt_state = carry
            pose_loss_values, geometry_grads = batched_pose_value_and_grad(
                geometry_params,
                motor_relative,
                target_logs,
                replay_robot_keys,
                replay_estimator_keys,
                geometry_loss_scales,
            )
            motor_loss_values, motor_grads = batched_motor_value_and_grad(
                geometry_params,
                motor_relative,
                target_logs,
                motor_loss_scales,
            )
            geometry_updates, next_geometry_opt_state = geometry_optimizer.update(
                geometry_grads,
                geometry_opt_state,
                geometry_params,
            )
            motor_updates, next_motor_opt_state = motor_optimizer.update(
                motor_grads,
                motor_opt_state,
                motor_relative,
            )
            next_geometry_params = _clip_geometry_params(optax.apply_updates(geometry_params, geometry_updates))
            next_motor_relative = _clip_relative_params(optax.apply_updates(motor_relative, motor_updates))
            next_params = params_from_optimizer_values(next_geometry_params, next_motor_relative)
            parameter_mse_values = physical_params_mse(next_params, pipeline.hidden_params)
            return (
                next_geometry_params,
                next_motor_relative,
                next_geometry_opt_state,
                next_motor_opt_state,
            ), (pose_loss_values, motor_loss_values, parameter_mse_values)

        (
            estimated_geometry_params,
            estimated_motor_relative,
            _,
            _,
        ), (geometry_loss_history, motor_loss_history, parameter_mse_history) = jax.lax.scan(
            train_step,
            (
                current_geometry_params,
                current_motor_relative,
                current_geometry_opt_state,
                current_motor_opt_state,
            ),
            jnp.arange(num_steps),
        )
        estimated_params = params_from_optimizer_values(estimated_geometry_params, estimated_motor_relative)

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
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    motor_learning_rate: float,
    num_realizations: int,
    bootstrap_samples: int,
    seed: int = 0,
):
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")

    num_replay_realizations = pipeline.resolve_replay_realizations(num_realizations)
    initial_params = clip_physical_params(init_params)

    if num_steps <= 0:
        estimated_params = PhysicalParams(
            wheel_radius=jnp.full((bootstrap_samples,), initial_params.wheel_radius, dtype=jnp.float32),
            base_diameter=jnp.full((bootstrap_samples,), initial_params.base_diameter, dtype=jnp.float32),
            max_wheel_speed=jnp.full((bootstrap_samples,), initial_params.max_wheel_speed, dtype=jnp.float32),
            time_constant=jnp.full((bootstrap_samples,), initial_params.time_constant, dtype=jnp.float32),
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
        }

    master_key = jax.random.PRNGKey(seed)
    target_key, replay_key = jax.random.split(master_key, 2)
    target_robot_key, target_estimator_key = jax.random.split(target_key, 2)
    target_robot_keys = jax.random.split(target_robot_key, bootstrap_samples)
    target_estimator_keys = jax.random.split(target_estimator_key, bootstrap_samples)
    replay_keys = jax.random.split(replay_key, bootstrap_samples)
    replay_robot_keys, replay_estimator_keys = _bootstrap_replay_keys(
        replay_keys,
        num_replay_realizations,
    )
    optimize_batch = _make_bootstrap_batch_optimizer(
        pipeline=pipeline,
        init_params=init_params,
        num_steps=num_steps,
        learning_rate=learning_rate,
        motor_learning_rate=motor_learning_rate,
    )
    estimated_params, parameter_samples, loss_history, motor_loss_history, parameter_mse_history = optimize_batch(
        target_robot_keys,
        target_estimator_keys,
        replay_robot_keys,
        replay_estimator_keys,
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
    }
