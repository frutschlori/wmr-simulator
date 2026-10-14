from typing import NamedTuple

import jax
import jax.numpy as jnp

from wmr_simulator.gain_parametrization import outer_gains_over_refs
from wmr_simulator.trajectory_optimization.start_offsets import sample_initial_pose_offsets


def clip_controller_gains(gains: jax.Array):
    return jnp.clip(gains, min=0)


class Realizations(NamedTuple):
    """The frozen noise keys and start offsets a rollout batch is scored under.

    Shared by the trajectory design and the gain tuner and held fixed for a
    run, so both objectives are deterministic. Each tuning trajectory derives
    its own child noise keys from these roots.
    """

    robot_keys: jax.Array       # (R, 2) plant/measurement-noise keys
    estimator_keys: jax.Array   # (R, 2) estimator-noise keys
    start_offsets: jax.Array    # (R, 3) [dx, dy, dtheta] start-pose offsets


def make_realizations(
    robot_key: jax.Array,
    estimator_key: jax.Array,
    num_realizations: int,
    offset_radius: float,
    offset_angle: float,
) -> Realizations:
    """Draw ``num_realizations`` noise-key pairs and start-pose offsets."""
    return Realizations(
        robot_keys=jax.random.split(robot_key, num_realizations),
        estimator_keys=jax.random.split(estimator_key, num_realizations),
        start_offsets=sample_initial_pose_offsets(
            jax.random.fold_in(robot_key, 5813),
            num_realizations,
            offset_radius,
            offset_angle,
        ),
    )


# ``namespace`` values for split_realization_keys_by_trajectory.
TRAINING_KEY_NAMESPACE = 0
VALIDATION_KEY_NAMESPACE = 1


def split_realization_keys_by_trajectory(
    realization_keys: jax.Array,
    num_trajectories: int,
    namespace: int = TRAINING_KEY_NAMESPACE,
) -> jax.Array:
    """``(T, R, 2)`` child noise keys per trajectory from the ``(R, 2)``
    bundle; ``namespace`` separates training and validation sets."""
    realization_keys = jnp.asarray(realization_keys, dtype=jnp.uint32)
    namespaced_keys = jax.vmap(lambda key: jax.random.fold_in(key, namespace))(
        realization_keys
    )
    keys_by_realization = jax.vmap(
        lambda key: jax.random.split(key, num_trajectories)
    )(namespaced_keys)
    return jnp.swapaxes(keys_by_realization, 0, 1)


def _resolve_initial_pose_offsets(initial_pose_offsets, num_realizations: int) -> jax.Array:
    if initial_pose_offsets is None:
        return jnp.zeros((num_realizations, 3), dtype=jnp.float32)
    return jnp.asarray(initial_pose_offsets, dtype=jnp.float32)


def weighted_realization_mean(terms: jax.Array, realization_weights) -> jax.Array:
    """Weighted mean of the per-realization loss terms ``(R, num_terms)``;
    ``realization_weights`` ``(R,)`` (None: all ones) drop diverging rollouts."""
    if realization_weights is None:
        return jnp.mean(terms, axis=0)
    weights = jnp.asarray(realization_weights, dtype=jnp.float32)
    # All-zero weights fall back to the plain mean.
    total = jnp.sum(weights)
    return jnp.where(
        total > 0.0,
        jnp.sum(terms * weights[:, None], axis=0) / jnp.maximum(total, 1e-12),
        jnp.mean(terms, axis=0),
    )


def _reference_targets(pipeline, reference_states: jax.Array):
    reference_poses = reference_states[1:, :3]
    reference_theta = reference_states[1:, 2]
    reference_velocity_xy = reference_states[1:, 3:5]
    reference_velocity = jnp.stack(
        [
            reference_velocity_xy[:, 0] * jnp.cos(reference_theta)
            + reference_velocity_xy[:, 1] * jnp.sin(reference_theta),
            reference_states[1:, 5],
        ],
        axis=1,
    )
    reference_pose_indices = jnp.arange(
        pipeline.inner_steps_per_geometry_step,
        reference_states.shape[0] * pipeline.inner_steps_per_geometry_step,
        pipeline.inner_steps_per_geometry_step,
        dtype=jnp.int32,
    )
    return reference_poses, reference_velocity, reference_pose_indices


def pose_tracking_losses(
    predicted_poses: jax.Array,
    reference_poses: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """The position (squared error) and heading (``2 - 2cos``) tracking
    losses, unweighted and separate."""

    pos_error = predicted_poses[:, :2] - reference_poses[:, :2]
    angle_error = predicted_poses[:, 2] - reference_poses[:, 2]
    position_loss = jnp.mean(jnp.sum(pos_error**2, axis=1))
    heading_loss = jnp.mean(2.0 - 2.0 * jnp.cos(angle_error))
    return position_loss, heading_loss


def _base_loss_terms(
    pipeline,
    predicted_log,
    reference_poses,
    reference_velocity,
    reference_pose_indices,
    position_tracking_weight: float,
    heading_tracking_weight: float,
    linear_velocity_tracking_weight: float,
    angular_velocity_tracking_weight: float,
    input_weight: float,
    input_delta_weight: float,
    omega_delta_weight: float,
) -> jax.Array:
    predicted_poses = predicted_log.pose.states[reference_pose_indices]
    position_tracking_loss, heading_tracking_loss = pose_tracking_losses(
        predicted_poses, reference_poses
    )
    predicted_velocity = predicted_log.wheel.vel_omega[reference_pose_indices]
    # [v, omega] normalized by their limits.
    velocity_scale = jnp.asarray([pipeline.v_max, pipeline.omega_max], dtype=jnp.float32)
    velocity_error = (predicted_velocity - reference_velocity) / velocity_scale
    linear_velocity_tracking_loss = jnp.mean(velocity_error[:, 0] ** 2)
    angular_velocity_tracking_loss = jnp.mean(velocity_error[:, 1] ** 2)
    duty_cycle = predicted_log.wheel.duty_cycle[:-1]

    input_loss = jnp.mean(jnp.sum(duty_cycle**2, axis=1))         # minimize input energy

    input_delta = jnp.diff(duty_cycle, axis=0)
    input_delta_loss = jnp.mean(jnp.sum(input_delta**2, axis=1))  # favor input smoothness

    # Step-to-step change of the yaw rate, normalized by omega_max.
    omega = predicted_log.wheel.vel_omega[:, 1] / pipeline.omega_max
    omega_delta_loss = jnp.mean(jnp.diff(omega) ** 2)
    return jnp.asarray(
        [
            position_tracking_weight * position_tracking_loss,
            heading_tracking_weight * heading_tracking_loss,
            linear_velocity_tracking_weight * linear_velocity_tracking_loss,
            angular_velocity_tracking_weight * angular_velocity_tracking_loss,
            input_weight * input_loss,
            input_delta_weight * input_delta_loss,
            omega_delta_weight * omega_delta_loss,
        ],
        dtype=jnp.float32,
    )


def closed_loop_objective(
    pipeline,
    gains: jax.Array,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    position_tracking_weight: float = 1.0,
    heading_tracking_weight: float = 1.0,
    linear_velocity_tracking_weight: float = 0.0,
    angular_velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
    initial_pose_offsets: jax.Array | None = None,
):
    return jnp.sum(
        closed_loop_objective_terms(
            pipeline,
            gains,
            replay_robot_keys,
            replay_estimator_keys,
            position_tracking_weight=position_tracking_weight,
            heading_tracking_weight=heading_tracking_weight,
            linear_velocity_tracking_weight=linear_velocity_tracking_weight,
            angular_velocity_tracking_weight=angular_velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            omega_delta_weight=omega_delta_weight,
            reference_states=reference_states,
            initial_pose_offsets=initial_pose_offsets,
        )
    )


def closed_loop_objective_terms(
    pipeline,
    gains: jax.Array,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    position_tracking_weight: float = 1.0,
    heading_tracking_weight: float = 1.0,
    linear_velocity_tracking_weight: float = 0.0,
    angular_velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
    initial_pose_offsets: jax.Array | None = None,
    realization_weights: jax.Array | None = None,
):
    reference_states = pipeline.reference_states if reference_states is None else reference_states
    reference_poses, reference_velocity, reference_pose_indices = _reference_targets(pipeline, reference_states)
    offsets = _resolve_initial_pose_offsets(initial_pose_offsets, replay_robot_keys.shape[0])
    reference_start = pipeline.initial_reference_pose(reference_states)

    def realization_loss(robot_key, estimator_key, offset):
        predicted_log = pipeline.run_closed_loop(
            pipeline.robot_params,
            controller_gains=gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
            initial_pose=reference_start + offset,
        )
        return _base_loss_terms(
            pipeline,
            predicted_log,
            reference_poses,
            reference_velocity,
            reference_pose_indices,
            position_tracking_weight,
            heading_tracking_weight,
            linear_velocity_tracking_weight,
            angular_velocity_tracking_weight,
            input_weight,
            input_delta_weight,
            omega_delta_weight,
        )

    terms = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys, offsets)
    return weighted_realization_mean(terms, realization_weights)


def scheduled_closed_loop_objective_terms(
    pipeline,
    nominal_gains: jax.Array,
    schedule_params,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    position_tracking_weight: float = 1.0,
    heading_tracking_weight: float = 1.0,
    linear_velocity_tracking_weight: float = 0.0,
    angular_velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    gain_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
    initial_pose_offsets: jax.Array | None = None,
    realization_weights: jax.Array | None = None,
):
    """Loss terms for the scheduled controller: the seven static terms plus
    a gain-schedule smoothness penalty (``gain_delta``)."""
    reference_states = pipeline.reference_states if reference_states is None else reference_states
    reference_poses, reference_velocity, reference_pose_indices = _reference_targets(pipeline, reference_states)
    offsets = _resolve_initial_pose_offsets(initial_pose_offsets, replay_robot_keys.shape[0])
    reference_start = pipeline.initial_reference_pose(reference_states)

    def realization_loss(robot_key, estimator_key, offset):
        predicted_log = pipeline.run_closed_loop(
            pipeline.robot_params,
            controller_gains=nominal_gains,
            schedule_params=schedule_params,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
            initial_pose=reference_start + offset,
        )
        return _base_loss_terms(
            pipeline,
            predicted_log,
            reference_poses,
            reference_velocity,
            reference_pose_indices,
            position_tracking_weight,
            heading_tracking_weight,
            linear_velocity_tracking_weight,
            angular_velocity_tracking_weight,
            input_weight,
            input_delta_weight,
            omega_delta_weight,
        )

    base_terms = weighted_realization_mean(
        jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys, offsets),
        realization_weights,
    )

    outer_gains = outer_gains_over_refs(nominal_gains, schedule_params, reference_states)
    gain_rate = jnp.diff(outer_gains, axis=0) / pipeline.geometry_dt
    gain_delta_loss = jnp.mean(jnp.sum(gain_rate**2, axis=1))

    return jnp.concatenate([base_terms, jnp.asarray([gain_delta_weight * gain_delta_loss], dtype=jnp.float32)])
