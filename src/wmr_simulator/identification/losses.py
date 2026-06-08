import jax
import jax.numpy as jnp

from wmr_simulator.types import PhysicalParams, SimulationLog


def pose_mse(predicted_poses, target_poses, weights=None):
    pos_error = predicted_poses[:, :2] - target_poses[:, :2]
    angle_error = predicted_poses[:, 2] - target_poses[:, 2]
    angle_loss = 2.0 - 2.0 * jnp.cos(angle_error)
    squared_error = jnp.sum(pos_error ** 2, axis=1) + angle_loss
    if weights is not None:
        weights = jnp.asarray(weights, dtype=squared_error.dtype)
        return jnp.sum(weights * squared_error) / jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.mean(squared_error)


def window_replay_mse(
    pipeline,
    params: PhysicalParams,
    target_log: SimulationLog,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    est_params: PhysicalParams | None = None,
    window_length: int | None = None,
):
    target_pose_hat = pipeline.estimator.get_est_pose(target_log.estimator_states)
    loss_weights = pipeline.target_loss_weights(target_log)

    def replay_loss(robot_key, estimator_key):
        predicted_log = pipeline.replay_rollout(
            params,
            target_log=target_log,
            est_params=est_params,
            robot_key=robot_key,
            estimator_key=estimator_key,
            window_length=window_length,
        )
        predicted_poses = pipeline._prediction_pose_series(predicted_log)
        return pose_mse(predicted_poses, target_pose_hat, loss_weights)

    losses = jax.vmap(replay_loss)(replay_robot_keys, replay_estimator_keys)
    return jnp.mean(losses)


def multi_experiment_window_replay_loss(
    pipelines,
    params: PhysicalParams,
    replay_key_sets,
    est_params: PhysicalParams | None = None,
    window_length: int | None = None,
):
    losses = [
        window_replay_mse(
            pipeline=pipeline,
            params=params,
            target_log=pipeline.target_log,
            replay_robot_keys=robot_keys,
            replay_estimator_keys=estimator_keys,
            est_params=est_params,
            window_length=window_length,
        )
        for pipeline, (robot_keys, estimator_keys) in zip(pipelines, replay_key_sets)
    ]
    return jnp.mean(jnp.asarray(losses))
