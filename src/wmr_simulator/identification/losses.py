import jax.numpy as jnp

from wmr_simulator.types import PhysicalParams, SimulationLog


def pose_mse(predicted_poses, target_poses):
    pos_error = predicted_poses[:, :2] - target_poses[:, :2]
    angle_error = predicted_poses[:, 2] - target_poses[:, 2]
    return jnp.mean(jnp.sum(pos_error**2, axis=1) + 2.0 - 2.0 * jnp.cos(angle_error))


def pose_window_replay_mse(
    pipeline,
    params: PhysicalParams,
    target_log: SimulationLog,
    est_params: PhysicalParams | None = None,
    window_length: int | None = None,
    replay_segment_plan=None,
):
    predicted_log = pipeline.replay_rollout(
        params,
        target_log=target_log,
        window_length=window_length,
        replay_segment_plan=replay_segment_plan,
    )
    prediction_states = predicted_log.pose.states
    target_states = target_log.pose.states[1:]
    return pose_mse(prediction_states, target_states)


def window_replay_mse(
    pipeline,
    params: PhysicalParams,
    target_log: SimulationLog,
    est_params: PhysicalParams | None = None,
    window_length: int | None = None,
    replay_segment_plan=None,
):
    return pose_window_replay_mse(
        pipeline=pipeline,
        params=params,
        target_log=target_log,
        est_params=est_params,
        window_length=window_length,
        replay_segment_plan=replay_segment_plan,
    ) + pipeline.motor_wheel_speed_mse(params, target_log, window_length=window_length)
