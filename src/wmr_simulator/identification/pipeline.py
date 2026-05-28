import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.estimator import EstimatorState
from wmr_simulator.robot import DiffDriveState
from wmr_simulator.identification.losses import window_replay_mse
from wmr_simulator.identification.optimizers import bootstrap_identification_adam, optimize_physical_params_adam
from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.types import (
    PhysicalParams,
    SimulationLog,
    clip_physical_params,
    physical_params_from_array,
    physical_params_mse,
)


class SystemIdentificationPipeline(SimulationPipeline):
    def __init__(
        self,
        problem_path: str,
        initial_params: PhysicalParams,
        seed: int = 0,
        reference_trajectories_dir: str | None = None,
        window_length: int | None = None,
        deterministic_replay: bool = True,
    ):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
            window_length=window_length,
        )
        self.initial_params = initial_params
        self.deterministic_replay = bool(deterministic_replay)
        self.target_log = self.run_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=self.gains,
        )

    @staticmethod
    def _clip_physical_params(params: PhysicalParams):
        return clip_physical_params(params)

    @staticmethod
    def parameter_mse(params: PhysicalParams, target_params: PhysicalParams):
        return physical_params_mse(params, target_params)

    def resolve_replay_realizations(self, num_realizations: int) -> int:
        if self.deterministic_replay:
            return 1
        num_realizations = int(num_realizations)
        if num_realizations <= 0:
            raise ValueError("num_realizations must be a positive integer.")
        return num_realizations

    def _initial_replay_states(
        self,
        init_pose,
        init_wheel_speeds,
        init_u_hat,
        init_u_true,
        init_covariance,
        robot_key,
        estimator_key,
    ):
        robot_state = DiffDriveState(
            pose=jnp.asarray(init_pose, dtype=jnp.float32),
            wheel_speeds=jnp.asarray(init_wheel_speeds, dtype=jnp.float32),
            key=robot_key,
            vel_omega=jnp.zeros(2, dtype=jnp.float32),
            wheel_cmd=jnp.zeros(2, dtype=jnp.float32),
        )
        estimator_state = EstimatorState(
            pose_hat=jnp.asarray(init_pose, dtype=jnp.float32),
            pose_meas=jnp.asarray(init_pose, dtype=jnp.float32),
            u_hat=jnp.asarray(init_u_hat, dtype=jnp.float32),
            u_true=jnp.asarray(init_u_true, dtype=jnp.float32),
            P=jnp.asarray(init_covariance, dtype=jnp.float32),
            key=estimator_key,
        )
        return robot_state, estimator_state

    def _extract_estimated_pose_series(self, sim_log: SimulationLog):
        return self.estimator.get_est_pose(sim_log.estimator_states)

    def _prediction_pose_series(self, sim_log: SimulationLog):
        if self.deterministic_replay:
            return self.robot.get_pose(sim_log.robot_states)
        return self._extract_estimated_pose_series(sim_log)

    def replay_rollout(
        self,
        robot_params,
        target_log: SimulationLog,
        est_params=None,
        robot_key=None,
        estimator_key=None,
        window_length: int | None = None,
    ):
        window_length = self.resolve_window_length(window_length)
        robot_key = self.robot_key if robot_key is None else robot_key
        estimator_key = self.estimator_key if estimator_key is None else estimator_key
        model_params = robot_params if est_params is None else est_params

        wheel_cmds = target_log.robot_states.wheel_cmd
        target_measurements = self._extract_estimated_pose_series(target_log)
        target_wheel_speeds = target_log.robot_states.wheel_speeds
        target_u_hat = target_log.estimator_states.u_hat
        target_u_true = target_log.estimator_states.u_true
        target_covariances = target_log.estimator_states.P

        num_steps = wheel_cmds.shape[0]
        num_windows = int(np.ceil(num_steps / window_length))
        padded_num_steps = num_windows * window_length
        pad_steps = padded_num_steps - num_steps

        padded_wheel_cmds = jnp.pad(wheel_cmds, ((0, pad_steps), (0, 0)))
        wheel_cmd_windows = padded_wheel_cmds.reshape(num_windows, window_length, 2)

        window_start_indices = jnp.arange(num_windows, dtype=jnp.int32) * window_length
        previous_indices = jnp.maximum(window_start_indices - 1, 0)

        init_poses = target_measurements[previous_indices]
        init_wheel_speeds = target_wheel_speeds[previous_indices]
        init_u_hat = target_u_hat[previous_indices]
        init_u_true = target_u_true[previous_indices]
        init_covariances = target_covariances[previous_indices]

        init_poses = init_poses.at[0].set(self.initial_reference_pose())
        init_wheel_speeds = init_wheel_speeds.at[0].set(jnp.zeros(2, dtype=jnp.float32))
        init_u_hat = init_u_hat.at[0].set(jnp.zeros(2, dtype=jnp.float32))
        init_u_true = init_u_true.at[0].set(jnp.zeros(2, dtype=jnp.float32))
        init_covariances = init_covariances.at[0].set(
            jnp.asarray(self.initial_estimator_covariance, dtype=jnp.float32)
        )

        window_robot_keys = jax.random.split(robot_key, num_windows)
        window_estimator_keys = jax.random.split(estimator_key, num_windows)

        def robot_replay_step(robot_state, wheel_cmd):
            next_robot_state = self.robot.step(
                robot_state,
                wheel_cmd,
                wheel_radius=robot_params.wheel_radius,
                base_diameter=robot_params.base_diameter,
            )
            return next_robot_state, next_robot_state

        def estimator_replay_step(carry, wheel_cmd):
            robot_state, est_state = carry

            ur_true, ul_true = self.robot.get_wheel_speeds(robot_state)
            pose_true = self.robot.get_pose(robot_state)
            next_est_state = self.estimator.update(
                est_state,
                ur_true,
                ul_true,
                pose_true,
                wheel_radius=model_params.wheel_radius,
                base_diameter=model_params.base_diameter,
            )
            next_robot_state = self.robot.step(
                robot_state,
                wheel_cmd,
                wheel_radius=robot_params.wheel_radius,
                base_diameter=robot_params.base_diameter,
            )
            return (next_robot_state, next_est_state), (next_robot_state, next_est_state)

        def replay_single_window(
            init_pose,
            init_window_wheel_speeds,
            init_window_u_hat,
            init_window_u_true,
            init_window_covariance,
            window_robot_key,
            window_estimator_key,
            wheel_cmd_window,
        ):
            carry0 = self._initial_replay_states(
                init_pose,
                init_window_wheel_speeds,
                init_window_u_hat,
                init_window_u_true,
                init_window_covariance,
                window_robot_key,
                window_estimator_key,
            )
            if self.deterministic_replay:
                robot_state0, _ = carry0
                _, robot_state_window = jax.lax.scan(robot_replay_step, robot_state0, wheel_cmd_window)
                estimator_state_window = EstimatorState(
                    pose_hat=robot_state_window.pose,
                    pose_meas=robot_state_window.pose,
                    u_hat=jnp.zeros_like(robot_state_window.wheel_speeds),
                    u_true=jnp.zeros_like(robot_state_window.wheel_speeds),
                    P=jnp.zeros((robot_state_window.pose.shape[0], 3, 3), dtype=robot_state_window.pose.dtype),
                    key=robot_state_window.key,
                )
                return robot_state_window, estimator_state_window

            _, outputs = jax.lax.scan(estimator_replay_step, carry0, wheel_cmd_window)
            return outputs

        robot_state_windows, estimator_state_windows = jax.vmap(replay_single_window)(
            init_poses,
            init_wheel_speeds,
            init_u_hat,
            init_u_true,
            init_covariances,
            window_robot_keys,
            window_estimator_keys,
            wheel_cmd_windows,
        )

        robot_states = DiffDriveState(
            pose=robot_state_windows.pose.reshape(-1, 3)[:num_steps],
            wheel_speeds=robot_state_windows.wheel_speeds.reshape(-1, 2)[:num_steps],
            key=robot_state_windows.key.reshape(-1, 2)[:num_steps],
            vel_omega=robot_state_windows.vel_omega.reshape(-1, 2)[:num_steps],
            wheel_cmd=robot_state_windows.wheel_cmd.reshape(-1, 2)[:num_steps],
        )
        estimator_states = EstimatorState(
            pose_hat=estimator_state_windows.pose_hat.reshape(-1, 3)[:num_steps],
            pose_meas=estimator_state_windows.pose_meas.reshape(-1, 3)[:num_steps],
            u_hat=estimator_state_windows.u_hat.reshape(-1, 2)[:num_steps],
            u_true=estimator_state_windows.u_true.reshape(-1, 2)[:num_steps],
            P=estimator_state_windows.P.reshape(-1, 3, 3)[:num_steps],
            key=estimator_state_windows.key.reshape(-1, 2)[:num_steps],
        )
        return SimulationLog(robot_states=robot_states, estimator_states=estimator_states)

    def loss(self, params: PhysicalParams, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        return window_replay_mse(
            pipeline=self,
            params=params,
            target_log=self.target_log,
            replay_robot_keys=replay_robot_keys,
            replay_estimator_keys=replay_estimator_keys,
            est_params=self.initial_params,
            window_length=self.window_length,
        )

    def optimize(self, init_params: PhysicalParams, num_steps: int, learning_rate: float, num_realizations: int):
        return optimize_physical_params_adam(
            pipeline=self,
            init_params=init_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
        )

    def optimize_bootstrap(
        self,
        init_params: PhysicalParams,
        num_steps: int,
        learning_rate: float,
        num_realizations: int,
        bootstrap_samples: int,
        seed: int = 0,
    ):
        return bootstrap_identification_adam(
            pipeline=self,
            init_params=init_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )


def run_single_experiment_identification(
    problem_path: str,
    initial_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
    window_length: int | None = None,
    bootstrap_samples: int = 1,
    bootstrap_seed: int | None = None,
    deterministic_replay: bool = True,
):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
        deterministic_replay=deterministic_replay,
    )
    init_target_log = pipeline.target_log
    init_replay_log = pipeline.replay_rollout(
        initial_params,
        target_log=pipeline.target_log,
        est_params=pipeline.initial_params,
        window_length=window_length,
    )
    bootstrap_result = None
    if bootstrap_samples <= 1:
        estimated_params, loss_history, parameter_mse_history = pipeline.optimize(
            init_params=initial_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
        )
    else:
        bootstrap_result = pipeline.optimize_bootstrap(
            init_params=initial_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
            bootstrap_samples=bootstrap_samples,
            seed=seed if bootstrap_seed is None else bootstrap_seed,
        )
        estimated_params = physical_params_from_array(bootstrap_result["parameter_mean"])
        loss_history = np.asarray(jnp.mean(bootstrap_result["loss_history"], axis=0), dtype=float).tolist()
        parameter_mse_history = np.asarray(
            jnp.mean(bootstrap_result["parameter_mse_history"], axis=0),
            dtype=float,
        ).tolist()
    pipeline.target_log = pipeline.run_closed_loop(
        estimated_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
    )
    final_replay_log = pipeline.replay_rollout(
        estimated_params,
        target_log=pipeline.target_log,
        est_params=pipeline.initial_params,
        window_length=window_length,
    )
    return {
        "pipeline": pipeline,
        "init_target_log": init_target_log,
        "init_replay_log": init_replay_log,
        "estimated_params": estimated_params,
        "loss_history": loss_history,
        "parameter_mse_history": parameter_mse_history,
        "bootstrap": bootstrap_result,
        "final_replay_log": final_replay_log,
    }
