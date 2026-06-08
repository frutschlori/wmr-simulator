import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_tqdm import scan_tqdm

from wmr_simulator.identification.losses import window_replay_mse
from wmr_simulator.types import (
    PhysicalParams,
    clip_physical_params,
    physical_params_mse,
    physical_params_to_array,
)


def optimize_physical_params_adam(
    pipeline,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
):
    optimizer = optax.adam(learning_rate)
    current_params = clip_physical_params(init_params)
    current_opt_state = optimizer.init(current_params)
    num_replay_realizations = pipeline.resolve_replay_realizations(num_realizations)

    replay_robot_keys = jax.random.split(pipeline.robot_key, num_replay_realizations)
    replay_estimator_keys = jax.random.split(pipeline.estimator_key, num_replay_realizations)

    if num_steps <= 0:
        return current_params, [], []

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        params, opt_state = carry
        loss_value, grads = jax.value_and_grad(
            lambda p: window_replay_mse(
                pipeline=pipeline,
                params=p,
                target_log=pipeline.target_log,
                replay_robot_keys=replay_robot_keys,
                replay_estimator_keys=replay_estimator_keys,
                est_params=pipeline.initial_params,
                window_length=pipeline.window_length,
            )
        )(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = clip_physical_params(optax.apply_updates(params, updates))
        parameter_mse = physical_params_mse(next_params, pipeline.hidden_params)
        return (next_params, next_opt_state), (loss_value, parameter_mse)

    (current_params, _), (loss_history, parameter_mse_history) = jax.lax.scan(
        train_step,
        (current_params, current_opt_state),
        jnp.arange(num_steps),
    )

    return (
        current_params,
        np.asarray(loss_history, dtype=float).tolist(),
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
):
    optimizer = optax.adam(learning_rate)
    initial_params = clip_physical_params(init_params)

    def make_target_log(robot_key, estimator_key):
        return pipeline.run_target_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
        )

    def loss_one(params, target_log, sample_replay_robot_keys, sample_replay_estimator_keys):
        return window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=target_log,
            replay_robot_keys=sample_replay_robot_keys,
            replay_estimator_keys=sample_replay_estimator_keys,
            est_params=pipeline.initial_params,
            window_length=pipeline.window_length,
        )

    batched_value_and_grad = jax.vmap(jax.value_and_grad(loss_one))

    @jax.jit
    def optimize_batch(
        target_robot_keys: jax.Array,
        target_estimator_keys: jax.Array,
        replay_robot_keys: jax.Array,
        replay_estimator_keys: jax.Array,
    ):
        batch_size = target_robot_keys.shape[0]
        current_params = PhysicalParams(
            wheel_radius=jnp.full((batch_size,), initial_params.wheel_radius, dtype=jnp.float32),
            base_diameter=jnp.full((batch_size,), initial_params.base_diameter, dtype=jnp.float32),
        )
        current_opt_state = optimizer.init(current_params)
        target_logs = jax.vmap(make_target_log)(target_robot_keys, target_estimator_keys)

        @scan_tqdm(num_steps, desc="Bootstrap optimization")
        def train_step(carry, step):
            params, opt_state = carry
            loss_values, grads = batched_value_and_grad(
                params,
                target_logs,
                replay_robot_keys,
                replay_estimator_keys,
            )
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = clip_physical_params(optax.apply_updates(params, updates))
            parameter_mse_values = 0.5 * (
                (1000.0 * (next_params.wheel_radius - pipeline.hidden_params.wheel_radius)) ** 2
                + (1000.0 * (next_params.base_diameter - pipeline.hidden_params.base_diameter)) ** 2
            )
            return (next_params, next_opt_state), (loss_values, parameter_mse_values)

        (estimated_params, _), (loss_history, parameter_mse_history) = jax.lax.scan(
            train_step,
            (current_params, current_opt_state),
            jnp.arange(num_steps),
        )

        parameter_samples = physical_params_to_array(estimated_params)
        return (
            estimated_params,
            parameter_samples,
            jnp.swapaxes(loss_history, 0, 1),
            jnp.swapaxes(parameter_mse_history, 0, 1),
        )

    return optimize_batch


def bootstrap_identification_adam(
    pipeline,
    init_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
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
        )
        parameter_samples = physical_params_to_array(estimated_params)
        parameter_mean, parameter_covariance = empirical_moments(parameter_samples)
        return {
            "estimated_params": estimated_params,
            "parameter_samples": parameter_samples,
            "parameter_mean": parameter_mean,
            "parameter_covariance": parameter_covariance,
            "loss_history": jnp.empty((bootstrap_samples, 0), dtype=jnp.float32),
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
    )
    estimated_params, parameter_samples, loss_history, parameter_mse_history = optimize_batch(
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
        "parameter_mse_history": parameter_mse_history,
    }
