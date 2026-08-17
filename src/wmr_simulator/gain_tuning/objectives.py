from typing import NamedTuple

import jax
import jax.numpy as jnp

from wmr_simulator.gain_parametrization import outer_gains_over_refs
from wmr_simulator.trajectory_optimization.start_offsets import sample_initial_pose_offsets


def clip_controller_gains(gains: jax.Array):
    return jnp.clip(gains, min=0)


class Realizations(NamedTuple):
    """The frozen stochastic conditions a rollout batch is scored under.

    One root bundle per run, built once and shared by everything that rolls
    out. For a set of tuning trajectories, each trajectory gets independent
    child noise keys from these roots; the trajectory optimizer averages its
    FIM over the corresponding root realizations. Sharing the roots is what
    makes a trajectory designed to be informative about the gains informative
    about them *as the tuner sees them* -- two independent draws would score
    the two halves of the loop on different problems.

    Held fixed for the whole run (common random numbers): both objectives then
    stay deterministic functions of their decision variables.
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
    """Draw ``num_realizations`` noise-key pairs and start-pose offsets.

    The offsets are folded off the robot key with their own tag so they do not
    correlate with the measurement-noise draws.
    """
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


# ``namespace`` values for split_realization_keys_by_trajectory. Named because
# the summary figures have to reproduce the objective's keys exactly, and a
# literal 0/1 in two places is a silent way for them to drift apart.
TRAINING_KEY_NAMESPACE = 0
VALIDATION_KEY_NAMESPACE = 1


def split_realization_keys_by_trajectory(
    realization_keys: jax.Array,
    num_trajectories: int,
    namespace: int = TRAINING_KEY_NAMESPACE,
) -> jax.Array:
    """Derive frozen, independent noise keys for every trajectory.

    ``realization_keys`` is the run-level ``(R, 2)`` bundle. The returned
    ``(T, R, 2)`` array gives every trajectory a distinct child key for each
    realization while remaining deterministic across objective evaluations.
    ``namespace`` separates otherwise-independent sets such as training and
    validation trajectories.
    """
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
    """Mean of the per-realization loss terms, optionally down-weighting some.

    ``terms`` is ``(R, num_terms)``; ``realization_weights`` is ``(R,)`` or None
    (all ones). The weights exist so a diverging rollout can be dropped from the
    objective outright -- see ``optimizers.rollout_outlier_weights`` for why one
    such rollout is enough to stall the whole solve. They are constants of the
    run, never functions of the gains, so the objective stays smooth.
    """
    if realization_weights is None:
        return jnp.mean(terms, axis=0)
    weights = jnp.asarray(realization_weights, dtype=jnp.float32)
    # An all-zero weight vector would divide by zero; fall back to the plain mean
    # rather than emit NaN, since dropping every rollout is a caller bug and the
    # caller (rollout_outlier_weights) already refuses to produce it.
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


def pose_tracking_loss(
    predicted_poses: jax.Array,
    reference_poses: jax.Array,
    position_tracking_weight: float,
    heading_tracking_weight: float,
) -> jax.Array:
    """Pose-tracking loss with position and heading weighted separately.

    Same two residuals as ``SimulationPipeline.pose_mse`` (squared position
    error and the SO(2) heading error ``2 - 2cos``), but each carries its own
    weight """

    pos_error = predicted_poses[:, :2] - reference_poses[:, :2]
    angle_error = predicted_poses[:, 2] - reference_poses[:, 2]
    position_loss = jnp.mean(jnp.sum(pos_error**2, axis=1))
    heading_loss = jnp.mean(2.0 - 2.0 * jnp.cos(angle_error))
    return position_tracking_weight * position_loss + heading_tracking_weight * heading_loss


def _base_loss_terms(
    pipeline,
    predicted_log,
    reference_poses,
    reference_velocity,
    reference_pose_indices,
    position_tracking_weight: float,
    heading_tracking_weight: float,
    velocity_tracking_weight: float,
    input_weight: float,
    input_delta_weight: float,
    omega_delta_weight: float,
) -> jax.Array:
    predicted_poses = predicted_log.pose.states[reference_pose_indices]
    tracking_loss = pose_tracking_loss(
        predicted_poses,
        reference_poses,
        position_tracking_weight,
        heading_tracking_weight,
    )
    predicted_velocity = predicted_log.wheel.vel_omega[reference_pose_indices]
    # Normalize [v, omega] by their limits so the two channels are commensurate
    # (raw omega ~10 would otherwise swamp v ~1) and velocity_tracking_weight is an
    # interpretable relative weight rather than a unit-reconciliation constant.
    velocity_scale = jnp.asarray([pipeline.v_max, pipeline.omega_max], dtype=jnp.float32)
    velocity_error = (predicted_velocity - reference_velocity) / velocity_scale
    velocity_tracking_loss = jnp.mean(jnp.sum(velocity_error**2, axis=1))
    duty_cycle = predicted_log.wheel.duty_cycle[:-1]

    input_loss = jnp.mean(jnp.sum(duty_cycle**2, axis=1))         # minimize input energy

    input_delta = jnp.diff(duty_cycle, axis=0)
    input_delta_loss = jnp.mean(jnp.sum(input_delta**2, axis=1))  # favor input smoothness

    # Angular-rate chatter penalty: the step-to-step change in the robot's yaw
    # rate over the full wheel-dt series (normalized by omega_max, like the
    # velocity term). Aggressive motor gains that oscillate omega on the real
    # robot cost here even when the pose/velocity tracking still looks good in
    # sim, so the optimizer is pushed toward smoother, more transferable gains.
    omega = predicted_log.wheel.vel_omega[:, 1] / pipeline.omega_max
    omega_delta_loss = jnp.mean(jnp.diff(omega) ** 2)
    return jnp.asarray(
        [
            tracking_loss,
            velocity_tracking_weight * velocity_tracking_loss,
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
    velocity_tracking_weight: float = 0.0,
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
            velocity_tracking_weight=velocity_tracking_weight,
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
    velocity_tracking_weight: float = 0.0,
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
            velocity_tracking_weight,
            input_weight,
            input_delta_weight,
            omega_delta_weight,
        )

    terms = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys, offsets)
    return weighted_realization_mean(terms, realization_weights)


def scheduled_closed_loop_objective(
    pipeline,
    nominal_gains: jax.Array,
    schedule_params,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    position_tracking_weight: float = 1.0,
    heading_tracking_weight: float = 1.0,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    gain_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
    initial_pose_offsets: jax.Array | None = None,
    realization_weights: jax.Array | None = None,
):
    return jnp.sum(
        scheduled_closed_loop_objective_terms(
            pipeline,
            nominal_gains,
            schedule_params,
            replay_robot_keys,
            replay_estimator_keys,
            position_tracking_weight=position_tracking_weight,
            heading_tracking_weight=heading_tracking_weight,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            omega_delta_weight=omega_delta_weight,
            gain_delta_weight=gain_delta_weight,
            reference_states=reference_states,
            initial_pose_offsets=initial_pose_offsets,
            realization_weights=realization_weights,
        )
    )


def scheduled_closed_loop_objective_terms(
    pipeline,
    nominal_gains: jax.Array,
    schedule_params,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    position_tracking_weight: float = 1.0,
    heading_tracking_weight: float = 1.0,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    gain_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
    initial_pose_offsets: jax.Array | None = None,
    realization_weights: jax.Array | None = None,
):
    """Loss terms for the scheduled controller.

    Returns a 6-vector: the five static terms (tracking, velocity_tracking,
    input, input_delta, omega_delta) plus a rate-scaled gain-schedule smoothness
    penalty. The penalty is a pure function of the reference trajectory
    (independent of rollout noise). With ``gain_delta_weight = 0`` and an
    identity schedule (W = 0, b = 0) this reproduces the static objective terms
    padded with a trailing zero.
    """
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
            velocity_tracking_weight,
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
