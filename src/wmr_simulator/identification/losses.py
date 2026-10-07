import jax.numpy as jnp

from wmr_simulator.types import PhysicalParams, SimulationLog


# The position residual is scored along the measured heading only. The nominal
# model cannot move sideways, so the lateral residual holds whatever moves the
# tracked point sideways besides turning: a mocap rigid-body origin off the axle
# (lateral speed d * omega) and sideslip in turns. Fitted against it, the
# wheelbase absorbed both (2026-10-07, real robot: 98-103 mm against 87-89 from
# the heading alone and from encoder vs gyro yaw rate; in MuJoCo, with the mocap
# point on the axle, both fits give 85 mm against a true 84.9 on the identified
# phase). The longitudinal residual pins the wheel radius, the heading one the
# wheelbase.


def _longitudinal_error(predicted_poses, target_poses):
    """Position error along the measured heading."""
    pos_error = predicted_poses[:, :2] - target_poses[:, :2]
    heading = target_poses[:, 2]
    return jnp.cos(heading) * pos_error[:, 0] + jnp.sin(heading) * pos_error[:, 1]


def pose_mse(predicted_poses, target_poses):
    longitudinal = _longitudinal_error(predicted_poses, target_poses)
    angle_error = predicted_poses[:, 2] - target_poses[:, 2]
    return jnp.mean(longitudinal**2 + 2.0 - 2.0 * jnp.cos(angle_error))


def weighted_pose_mse(pipeline, predicted_poses, target_log: SimulationLog):
    target_poses = target_log.pose.states[1:]
    variances = pose_loss_variances(pipeline, target_log)
    heading = target_poses[:, 2]
    # Variance of the position error projected onto the heading (x and y are
    # independent in pose_loss_variances).
    longitudinal_variance = jnp.cos(heading) ** 2 * variances[:, 0] + jnp.sin(heading) ** 2 * variances[:, 1]
    longitudinal = _longitudinal_error(predicted_poses, target_poses)
    angle_error = predicted_poses[:, 2] - target_poses[:, 2]
    return jnp.mean(longitudinal**2 / longitudinal_variance + (2.0 - 2.0 * jnp.cos(angle_error)) / variances[:, 2])


def pose_loss_variances(pipeline, target_log: SimulationLog):
    measurement_variance = jnp.diag(jnp.asarray(pipeline.estimator.R, dtype=jnp.float32))
    input_variance = input_pose_variances(pipeline, target_log)
    variance_floor = jnp.asarray([1e-6, 1e-6, 1e-4], dtype=jnp.float32)
    return jnp.maximum(measurement_variance + input_variance, variance_floor)


def input_pose_variances(pipeline, target_log: SimulationLog):
    estimator = pipeline.estimator
    theta = target_log.pose.states[:-1, 2]
    # Encoder increment noise only, and constant per wheel. The multiplicative
    # slip term that used to scale this by dphi^2 (with the encoder variance
    # subtracted off so it was not double counted) went with the random-slippage
    # model itself, which is no longer in the plant -- weighting residuals for a
    # disturbance the simulator never generates only mis-scales the fit. Losing
    # the dphi dependence is why the wheel-speed lookup above is gone too.
    encoder_variance = estimator.enc_angle_noise ** 2
    wheel_increment_variance = jnp.full((theta.shape[0], 2), encoder_variance, dtype=jnp.float32)
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
