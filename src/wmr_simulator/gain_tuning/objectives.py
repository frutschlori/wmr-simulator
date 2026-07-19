import jax
import jax.numpy as jnp

from wmr_simulator.gain_parametrization import outer_gains_over_refs


def clip_controller_gains(gains: jax.Array):
    return jnp.clip(gains, min=0)


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


def _base_loss_terms(
    pipeline,
    predicted_log,
    reference_poses,
    reference_velocity,
    reference_pose_indices,
    velocity_tracking_weight: float,
    input_weight: float,
    input_delta_weight: float,
) -> jax.Array:
    predicted_poses = predicted_log.pose.states[reference_pose_indices]
    tracking_loss = pipeline.pose_mse(predicted_poses, reference_poses)
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
    return jnp.asarray(
        [
            tracking_loss,
            velocity_tracking_weight * velocity_tracking_loss,
            input_weight * input_loss,
            input_delta_weight * input_delta_loss,
        ],
        dtype=jnp.float32,
    )


def closed_loop_objective(
    pipeline,
    gains: jax.Array,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
):
    return jnp.sum(
        closed_loop_objective_terms(
            pipeline,
            gains,
            replay_robot_keys,
            replay_estimator_keys,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            reference_states=reference_states,
        )
    )


def closed_loop_objective_terms(
    pipeline,
    gains: jax.Array,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
):
    reference_states = pipeline.reference_states if reference_states is None else reference_states
    reference_poses, reference_velocity, reference_pose_indices = _reference_targets(pipeline, reference_states)

    def realization_loss(robot_key, estimator_key):
        predicted_log = pipeline.run_closed_loop(
            pipeline.robot_params,
            controller_gains=gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
        )
        return _base_loss_terms(
            pipeline,
            predicted_log,
            reference_poses,
            reference_velocity,
            reference_pose_indices,
            velocity_tracking_weight,
            input_weight,
            input_delta_weight,
        )

    terms = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys)
    return jnp.mean(terms, axis=0)


def scheduled_closed_loop_objective(
    pipeline,
    nominal_gains: jax.Array,
    schedule_params,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    gain_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
):
    return jnp.sum(
        scheduled_closed_loop_objective_terms(
            pipeline,
            nominal_gains,
            schedule_params,
            replay_robot_keys,
            replay_estimator_keys,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            gain_delta_weight=gain_delta_weight,
            reference_states=reference_states,
        )
    )


def scheduled_closed_loop_objective_terms(
    pipeline,
    nominal_gains: jax.Array,
    schedule_params,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    gain_delta_weight: float = 0.0,
    reference_states: jax.Array | None = None,
):
    """Loss terms for the scheduled controller.

    Returns a 5-vector: the four static terms plus a rate-scaled gain-schedule
    smoothness penalty. The penalty is a pure function of the reference
    trajectory (independent of rollout noise). With ``gain_delta_weight = 0`` and
    an identity schedule (W = 0, b = 0) this reproduces the static objective terms
    padded with a trailing zero.
    """
    reference_states = pipeline.reference_states if reference_states is None else reference_states
    reference_poses, reference_velocity, reference_pose_indices = _reference_targets(pipeline, reference_states)

    def realization_loss(robot_key, estimator_key):
        predicted_log = pipeline.run_closed_loop(
            pipeline.robot_params,
            controller_gains=nominal_gains,
            schedule_params=schedule_params,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
        )
        return _base_loss_terms(
            pipeline,
            predicted_log,
            reference_poses,
            reference_velocity,
            reference_pose_indices,
            velocity_tracking_weight,
            input_weight,
            input_delta_weight,
        )

    base_terms = jnp.mean(jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys), axis=0)

    outer_gains = outer_gains_over_refs(nominal_gains, schedule_params, reference_states)
    gain_rate = jnp.diff(outer_gains, axis=0) / pipeline.geometry_dt
    gain_delta_loss = jnp.mean(jnp.sum(gain_rate**2, axis=1))

    return jnp.concatenate([base_terms, jnp.asarray([gain_delta_weight * gain_delta_loss], dtype=jnp.float32)])
