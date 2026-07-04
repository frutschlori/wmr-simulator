import jax
import jax.numpy as jnp


def clip_controller_gains(gains: jax.Array):
    return jnp.clip(gains, min=0)


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

    def realization_loss(robot_key, estimator_key):
        predicted_log = pipeline.run_closed_loop(
            pipeline.robot_params,
            controller_gains=gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
        )
        predicted_poses = predicted_log.pose.states[reference_pose_indices]
        tracking_loss = pipeline.pose_mse(predicted_poses, reference_poses)
        predicted_velocity = predicted_log.wheel.vel_omega[reference_pose_indices]
        velocity_tracking_loss = jnp.mean(jnp.sum((predicted_velocity - reference_velocity) ** 2, axis=1))
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

    terms = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys)
    return jnp.mean(terms, axis=0)
