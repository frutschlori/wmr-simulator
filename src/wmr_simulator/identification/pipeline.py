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
        max_replay_dt: float | None = None,
        target_log: SimulationLog | None = None,
        target_time_s=None,
        target_reference_states=None,
        replay_wheel_speed_source: str = "true",
    ):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
            window_length=window_length,
        )
        if target_reference_states is not None:
            self.reference_states = jnp.asarray(target_reference_states, dtype=jnp.float32)
        self.initial_params = initial_params
        self.deterministic_replay = bool(deterministic_replay)
        self.replay_wheel_speed_source = self._resolve_replay_wheel_speed_source(replay_wheel_speed_source)
        self.max_replay_dt = self.dt if max_replay_dt is None else self._resolve_max_replay_dt(max_replay_dt)
        self.uses_external_target_log = target_log is not None
        if target_log is None:
            self.target_log = self.run_target_closed_loop(
                initial_params,
                use_hidden_robot=True,
                controller_gains=self.gains,
            )
            self.target_time_s = None
        else:
            self.target_log = target_log
            if target_time_s is None:
                raise ValueError("target_time_s is required when target_log is provided.")
            self.target_time_s = jnp.asarray(target_time_s, dtype=jnp.float32)
        self.max_replay_substeps = self._max_replay_substeps(self.target_time_s)

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

    @staticmethod
    def _resolve_max_replay_dt(max_replay_dt: float) -> float:
        max_replay_dt = float(max_replay_dt)
        if max_replay_dt <= 0.0:
            raise ValueError("max_replay_dt must be positive.")
        return max_replay_dt

    @staticmethod
    def _resolve_replay_wheel_speed_source(source: str) -> str:
        source = str(source).lower()
        if source not in {"true", "noisy"}:
            raise ValueError("replay_wheel_speed_source must be 'true' or 'noisy'.")
        return source

    def run_target_closed_loop(self, *args, **kwargs) -> SimulationLog:
        return self.run_closed_loop(*args, **kwargs)

    def _target_replay_wheel_speeds(self, target_log: SimulationLog):
        if self.replay_wheel_speed_source == "true":
            return target_log.robot_states.wheel_speeds
        return target_log.estimator_states.u_hat

    def _target_interval_dts(self, target_log: SimulationLog, target_time_s=None):
        time_s = self.target_time_s if target_time_s is None else target_time_s
        if time_s is None:
            return jnp.full((target_log.robot_states.pose.shape[0] - 1,), self.dt, dtype=jnp.float32)
        time_s = jnp.asarray(time_s, dtype=jnp.float32)
        if time_s.shape[0] != target_log.robot_states.pose.shape[0]:
            raise ValueError(
                f"target_time_s length ({time_s.shape[0]}) must match target log length "
                f"({target_log.robot_states.pose.shape[0]})."
            )
        return jnp.diff(time_s)

    def _max_replay_substeps(self, target_time_s) -> int:
        if target_time_s is None:
            max_interval_dt = self.dt
        else:
            time_s = np.asarray(target_time_s, dtype=float)
            interval_dts = np.diff(time_s)
            if np.any(interval_dts <= 0.0):
                raise ValueError("target_time_s must be strictly increasing.")
            max_interval_dt = float(np.max(interval_dts))
        return max(1, int(np.ceil(max_interval_dt / self.max_replay_dt - 1e-4)))

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
        target_time_s=None,
    ):
        window_length = self.resolve_window_length(window_length)
        robot_key = self.robot_key if robot_key is None else robot_key
        estimator_key = self.estimator_key if estimator_key is None else estimator_key
        model_params = robot_params if est_params is None else est_params

        target_wheel_speeds = self._target_replay_wheel_speeds(target_log)
        replay_wheel_speeds = target_wheel_speeds[1:]
        logged_wheel_cmds = target_log.robot_states.wheel_cmd[1:]
        interval_dts = self._target_interval_dts(target_log, target_time_s)
        target_measurements = self._extract_estimated_pose_series(target_log)
        target_u_hat = target_log.estimator_states.u_hat
        target_u_true = target_log.estimator_states.u_true
        target_covariances = target_log.estimator_states.P

        num_target_steps = target_measurements.shape[0]
        num_intervals = max(num_target_steps - 1, 0)
        if num_intervals == 0:
            return target_log

        max_inner_steps = self.max_replay_substeps

        num_windows = int(np.ceil(num_intervals / window_length))
        padded_num_intervals = num_windows * window_length
        pad_steps = padded_num_intervals - num_intervals

        padded_wheel_speeds = jnp.pad(replay_wheel_speeds, ((0, pad_steps), (0, 0)))
        padded_wheel_cmds = jnp.pad(logged_wheel_cmds, ((0, pad_steps), (0, 0)))
        padded_interval_dts = jnp.pad(interval_dts, (0, pad_steps))
        wheel_speed_windows = padded_wheel_speeds.reshape(num_windows, window_length, 2)
        wheel_cmd_windows = padded_wheel_cmds.reshape(num_windows, window_length, 2)
        interval_dt_windows = padded_interval_dts.reshape(num_windows, window_length)

        window_start_indices_np = np.arange(num_windows, dtype=np.int32) * window_length
        window_start_indices = jnp.asarray(window_start_indices_np, dtype=jnp.int32)

        init_poses = target_measurements[window_start_indices]
        init_wheel_speeds = target_wheel_speeds[window_start_indices]
        init_u_hat = target_u_hat[window_start_indices]
        init_u_true = target_u_true[window_start_indices]
        init_covariances = target_covariances[window_start_indices]

        init_poses = init_poses.at[0].set(target_measurements[0])
        init_wheel_speeds = init_wheel_speeds.at[0].set(target_wheel_speeds[0])
        init_u_hat = init_u_hat.at[0].set(target_u_hat[0])
        init_u_true = init_u_true.at[0].set(target_u_true[0])
        init_covariances = init_covariances.at[0].set(
            jnp.asarray(self.initial_estimator_covariance, dtype=jnp.float32)
        )

        window_robot_keys = jax.random.split(robot_key, num_windows)
        window_estimator_keys = jax.random.split(estimator_key, num_windows)

        def robot_replay_step(robot_state, inputs):
            wheel_speeds, wheel_cmd, interval_dt = inputs
            substeps = jnp.ceil(interval_dt / self.max_replay_dt - 1e-4).astype(jnp.int32)
            safe_substeps = jnp.maximum(1, substeps)
            substep_dt = interval_dt / safe_substeps.astype(interval_dt.dtype)

            def substep(state, step_index):
                def active_step(_):
                    return self.robot.step_kinematic(
                        state,
                        wheel_speeds,
                        wheel_radius=robot_params.wheel_radius,
                        base_diameter=robot_params.base_diameter,
                        dt=substep_dt,
                        wheel_cmd=wheel_cmd,
                    )

                state = jax.lax.cond(step_index < substeps, active_step, lambda _: state, operand=None)
                return state, None

            next_robot_state, _ = jax.lax.scan(
                substep,
                robot_state,
                jnp.arange(max_inner_steps),
            )
            return next_robot_state, next_robot_state

        def estimator_replay_step(carry, inputs):
            wheel_speeds, wheel_cmd, interval_dt = inputs
            substeps = jnp.ceil(interval_dt / self.max_replay_dt - 1e-4).astype(jnp.int32)
            safe_substeps = jnp.maximum(1, substeps)
            substep_dt = interval_dt / safe_substeps.astype(interval_dt.dtype)

            def substep(subcarry, step_index):
                def active_step(_):
                    robot_state, est_state = subcarry

                    next_robot_state = self.robot.step_kinematic(
                        robot_state,
                        wheel_speeds,
                        wheel_radius=robot_params.wheel_radius,
                        base_diameter=robot_params.base_diameter,
                        dt=substep_dt,
                        wheel_cmd=wheel_cmd,
                    )
                    ur_true, ul_true = wheel_speeds
                    next_est_state = self.estimator.update(
                        est_state,
                        ur_true,
                        ul_true,
                        self.robot.get_pose(next_robot_state),
                        wheel_radius=model_params.wheel_radius,
                        base_diameter=model_params.base_diameter,
                        dt=substep_dt,
                    )
                    return next_robot_state, next_est_state

                carry = jax.lax.cond(step_index < substeps, active_step, lambda _: subcarry, operand=None)
                return carry, None

            next_robot_state, next_est_state = jax.lax.scan(
                substep,
                carry,
                jnp.arange(max_inner_steps),
            )[0]
            return (next_robot_state, next_est_state), (next_robot_state, next_est_state)

        def replay_single_window(
            init_pose,
            init_window_wheel_speeds,
            init_window_u_hat,
            init_window_u_true,
            init_window_covariance,
            window_robot_key,
            window_estimator_key,
            wheel_speed_window,
            wheel_cmd_window,
            interval_dt_window,
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
                _, robot_state_window = jax.lax.scan(
                    robot_replay_step,
                    robot_state0,
                    (wheel_speed_window, wheel_cmd_window, interval_dt_window),
                )
                estimator_state_window = EstimatorState(
                    pose_hat=robot_state_window.pose,
                    pose_meas=robot_state_window.pose,
                    u_hat=robot_state_window.wheel_speeds,
                    u_true=robot_state_window.wheel_speeds,
                    P=jnp.zeros((robot_state_window.pose.shape[0], 3, 3), dtype=robot_state_window.pose.dtype),
                    key=robot_state_window.key,
                )
                return robot_state_window, estimator_state_window

            _, outputs = jax.lax.scan(
                estimator_replay_step,
                carry0,
                (wheel_speed_window, wheel_cmd_window, interval_dt_window),
            )
            return outputs

        robot_state_windows, estimator_state_windows = jax.vmap(replay_single_window)(
            init_poses,
            init_wheel_speeds,
            init_u_hat,
            init_u_true,
            init_covariances,
            window_robot_keys,
            window_estimator_keys,
            wheel_speed_windows,
            wheel_cmd_windows,
            interval_dt_windows,
        )

        interval_robot_states = DiffDriveState(
            pose=robot_state_windows.pose.reshape(-1, 3)[:num_intervals],
            wheel_speeds=robot_state_windows.wheel_speeds.reshape(-1, 2)[:num_intervals],
            key=robot_state_windows.key.reshape(-1, 2)[:num_intervals],
            vel_omega=robot_state_windows.vel_omega.reshape(-1, 2)[:num_intervals],
            wheel_cmd=robot_state_windows.wheel_cmd.reshape(-1, 2)[:num_intervals],
        )
        interval_estimator_states = EstimatorState(
            pose_hat=estimator_state_windows.pose_hat.reshape(-1, 3)[:num_intervals],
            pose_meas=estimator_state_windows.pose_meas.reshape(-1, 3)[:num_intervals],
            u_hat=estimator_state_windows.u_hat.reshape(-1, 2)[:num_intervals],
            u_true=estimator_state_windows.u_true.reshape(-1, 2)[:num_intervals],
            P=estimator_state_windows.P.reshape(-1, 3, 3)[:num_intervals],
            key=estimator_state_windows.key.reshape(-1, 2)[:num_intervals],
        )

        initial_robot_state, initial_estimator_state = self._initial_replay_states(
            target_measurements[0],
            target_wheel_speeds[0],
            target_u_hat[0],
            target_u_true[0],
            jnp.asarray(self.initial_estimator_covariance, dtype=jnp.float32),
            robot_key,
            estimator_key,
        )
        robot_states = DiffDriveState(
            pose=jnp.concatenate([initial_robot_state.pose[None, :], interval_robot_states.pose], axis=0),
            wheel_speeds=jnp.concatenate(
                [initial_robot_state.wheel_speeds[None, :], interval_robot_states.wheel_speeds],
                axis=0,
            ),
            key=jnp.concatenate([initial_robot_state.key[None, :], interval_robot_states.key], axis=0),
            vel_omega=jnp.concatenate([initial_robot_state.vel_omega[None, :], interval_robot_states.vel_omega], axis=0),
            wheel_cmd=jnp.concatenate([target_log.robot_states.wheel_cmd[:1], interval_robot_states.wheel_cmd], axis=0),
        )
        estimator_states = EstimatorState(
            pose_hat=jnp.concatenate([initial_estimator_state.pose_hat[None, :], interval_estimator_states.pose_hat], axis=0),
            pose_meas=jnp.concatenate([initial_estimator_state.pose_meas[None, :], interval_estimator_states.pose_meas], axis=0),
            u_hat=jnp.concatenate([initial_estimator_state.u_hat[None, :], interval_estimator_states.u_hat], axis=0),
            u_true=jnp.concatenate([initial_estimator_state.u_true[None, :], interval_estimator_states.u_true], axis=0),
            P=jnp.concatenate([initial_estimator_state.P[None, :, :], interval_estimator_states.P], axis=0),
            key=jnp.concatenate([initial_estimator_state.key[None, :], interval_estimator_states.key], axis=0),
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
    max_replay_dt: float | None = None,
    target_log: SimulationLog | None = None,
    target_time_s=None,
    target_reference_states=None,
    replay_wheel_speed_source: str = "true",
):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
        deterministic_replay=deterministic_replay,
        max_replay_dt=max_replay_dt,
        target_log=target_log,
        target_time_s=target_time_s,
        target_reference_states=target_reference_states,
        replay_wheel_speed_source=replay_wheel_speed_source,
    )
    init_target_log = pipeline.target_log
    init_replay_log = pipeline.replay_rollout(
        initial_params,
        target_log=pipeline.target_log,
        est_params=pipeline.initial_params,
        window_length=window_length,
    )
    bootstrap_result = None
    if pipeline.uses_external_target_log and bootstrap_samples > 1:
        raise NotImplementedError("Bootstrap identification is not implemented for externally loaded target logs.")
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
    if not pipeline.uses_external_target_log:
        pipeline.target_log = pipeline.run_target_closed_loop(
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
