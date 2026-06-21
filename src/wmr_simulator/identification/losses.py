import jax.numpy as jnp

from wmr_simulator.types import PhysicalParams, SimulationLog


def pose_mse(predicted_poses, target_poses):
    pos_error = predicted_poses[:, :2] - target_poses[:, :2]
    angle_error = predicted_poses[:, 2] - target_poses[:, 2]
    return jnp.mean(jnp.sum(pos_error**2, axis=1) + 2.0 - 2.0 * jnp.cos(angle_error))


def weighted_pose_mse(pipeline, predicted_poses, target_log: SimulationLog):
    target_poses = target_log.pose.states[1:]
    variances = pose_loss_variances(pipeline, target_log)
    pos_error = predicted_poses[:, :2] - target_poses[:, :2]
    angle_error = predicted_poses[:, 2] - target_poses[:, 2]
    residual_terms = jnp.column_stack(
        [
            pos_error[:, 0] ** 2,
            pos_error[:, 1] ** 2,
            2.0 - 2.0 * jnp.cos(angle_error),
        ]
    )
    return jnp.mean(jnp.sum(residual_terms / variances, axis=1))


def pose_loss_variances(pipeline, target_log: SimulationLog):
    measurement_variance = jnp.diag(jnp.asarray(pipeline.estimator.R, dtype=jnp.float32))
    input_variance = input_pose_variances(pipeline, target_log)
    variance_floor = jnp.asarray([1e-6, 1e-6, 1e-4], dtype=jnp.float32)
    return jnp.maximum(measurement_variance + input_variance, variance_floor)


def input_pose_variances(pipeline, target_log: SimulationLog):
    estimator = pipeline.estimator
    pose_time = target_log.pose.time_s
    wheel_time = target_log.wheel.time_s
    dt = jnp.diff(pose_time)
    theta = target_log.pose.states[:-1, 2]
    wheel_indices = jnp.clip(
        jnp.searchsorted(wheel_time, pose_time[:-1], side="right") - 1,
        0,
        target_log.wheel.speeds.shape[0] - 1,
    )
    wheel_speeds = target_log.wheel.speeds[wheel_indices]
    dphi = wheel_speeds * dt[:, None]
    encoder_variance = estimator.enc_angle_noise ** 2
    wheel_increment_variance = jnp.column_stack(
        [
            jnp.maximum(dphi[:, 0] ** 2 - encoder_variance, 0.0) * estimator.slip_r_var + encoder_variance,
            jnp.maximum(dphi[:, 1] ** 2 - encoder_variance, 0.0) * estimator.slip_l_var + encoder_variance,
        ]
    )
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    r = estimator.r_est
    L = estimator.L_est
    input_jacobian_squared = jnp.stack(
        [
            jnp.column_stack([(0.5 * r * cos_theta) ** 2, (0.5 * r * cos_theta) ** 2]),
            jnp.column_stack([(0.5 * r * sin_theta) ** 2, (0.5 * r * sin_theta) ** 2]),
            jnp.column_stack(
                [
                    jnp.full_like(theta, (r / L) ** 2),
                    jnp.full_like(theta, (r / L) ** 2),
                ]
            ),
        ],
        axis=1,
    )
    return jnp.sum(input_jacobian_squared * wheel_increment_variance[:, None, :], axis=-1)


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
    if getattr(pipeline, "use_inverse_variance_pose_loss", True):
        return weighted_pose_mse(pipeline, prediction_states, target_log)
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
