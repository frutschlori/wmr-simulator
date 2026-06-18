import jax
import jax.numpy as jnp


def clip_controller_gains(gains: jax.Array):
    return jnp.clip(gains, min=0)


def closed_loop_tracking_mse(
    pipeline,
    gains: jax.Array,
    replay_robot_keys: jax.Array,
    replay_estimator_keys: jax.Array,
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
        return pipeline.pose_mse(predicted_poses, reference_poses)

    losses = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys)
    return jnp.mean(losses)
