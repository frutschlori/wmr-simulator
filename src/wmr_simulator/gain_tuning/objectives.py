import jax
import jax.numpy as jnp


def clip_controller_gains(gains: jax.Array):
    return jnp.clip(gains, min=0)


def closed_loop_objective(
    pipeline,
    gains: jax.Array,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
):
    reference_poses = pipeline.reference_states[1:, :3]
    reference_pose_indices = jnp.arange(
        pipeline.inner_steps_per_geometry_step,
        pipeline.reference_states.shape[0] * pipeline.inner_steps_per_geometry_step,
        pipeline.inner_steps_per_geometry_step,
        dtype=jnp.int32,
    )

    def realization_loss(robot_key, estimator_key):
        predicted_log = pipeline.run_closed_loop(
            pipeline.robot_params,
            controller_gains=gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
        )
        predicted_poses = predicted_log.pose.states[reference_pose_indices]
        tracking_loss = pipeline.pose_mse(predicted_poses, reference_poses)
        duty_cycle = predicted_log.wheel.duty_cycle[:-1]

        input_loss = jnp.mean(jnp.sum(duty_cycle**2, axis=1))         # minimize input energy

        input_delta = jnp.diff(duty_cycle, axis=0)
        input_delta_loss = jnp.mean(jnp.sum(input_delta**2, axis=1))  # favor input smoothness
        return tracking_loss + input_weight * input_loss + input_delta_weight * input_delta_loss

    losses = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys)
    return jnp.mean(losses)
