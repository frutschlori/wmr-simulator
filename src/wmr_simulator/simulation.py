import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.controller import Controller
from wmr_simulator.estimator import EstimatorState
from wmr_simulator.estimator import DiffDriveEstimator
from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.robot import DiffDrive, DiffDriveState
from wmr_simulator.types import PhysicalParams, SimulationLog


class SimulationPipeline:
    def __init__(
        self,
        problem_path: str,
        seed: int = 0,
        reference_trajectories_dir: str | None = None,
        window_length: int | None = None,
    ):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.problem = yaml.safe_load(file)

        self.problem_path = problem_path
        self.reference_trajectories_dir = reference_trajectories_dir
        self.loaded_reference_trajectory_path = None
        self.window_length = window_length
        self.dt = float(self.problem["time_step"])
        self.sim_time = float(self.problem["sim_time"])
        self.sim_steps = int(self.sim_time / self.dt)
        self.sim_time_grid = np.linspace(0.0, self.sim_steps * self.dt, self.sim_steps + 1)

        planner_cfg = self.problem["planner"]
        planner_time = float(planner_cfg["time"])
        planner_steps = int(planner_time / self.dt)
        planner_time_grid = np.linspace(0.0, planner_steps * self.dt, planner_steps + 1)

        if reference_trajectories_dir is None:
            start = self.problem["start"]
            goal = self.problem["goal"]
            waypoints = planner_cfg["waypoints"]
            reference_states, _ = compute_reference_trajectory(start, goal, waypoints, planner_time_grid)
        else:
            reference_states = self._load_latest_reference_states(reference_trajectories_dir)
        self.reference_states = jnp.asarray(self._extend_reference_states(reference_states), dtype=jnp.float32)

        self.robot_cfg = self.problem["robot"]
        self.estimator_cfg = self.problem["estimator"]
        self.controller_cfg = self.problem["controller"]

        self.robot = DiffDrive(robot_cfg=self.robot_cfg, dt=self.dt)
        self.estimator = DiffDriveEstimator(estimator_cfg=self.estimator_cfg, dt=self.dt)
        self.controller = Controller(
            robot_param=self.robot_cfg,
            gains=self.controller_cfg["gains"],
            duty_limits=[-1.0, 1.0],
            dt=self.dt,
        )
        estimator_init_state = self.estimator.get_init_state(
            key=jax.random.PRNGKey(1),
            start_pose=self.initial_reference_pose(),
        )
        self.initial_estimator_covariance = estimator_init_state.P

        self.hidden_params = PhysicalParams(
            wheel_radius=jnp.asarray(self.robot_cfg["wheel_radius"], dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot_cfg["base_diameter"], dtype=jnp.float32),
            max_wheel_speed=jnp.asarray(self.robot_cfg["max_wheel_speed"], dtype=jnp.float32),
            time_constant=jnp.asarray(self.robot_cfg["time_constant"], dtype=jnp.float32),
        )
        self.gains = jnp.array([jnp.asarray(value) for value in self.controller_cfg["gains"]])

        master_key = jax.random.PRNGKey(seed)
        target_key, replay_key = jax.random.split(master_key, 2)
        self.target_robot_key, self.target_estimator_key = jax.random.split(target_key, 2)
        self.robot_key, self.estimator_key = jax.random.split(replay_key, 2)

    def _load_latest_reference_states(self, reference_trajectories_dir: str) -> np.ndarray:
        if not os.path.isdir(reference_trajectories_dir):
            raise ValueError(f"Reference trajectory directory does not exist: {reference_trajectories_dir}")

        pickle_files = [
            os.path.join(reference_trajectories_dir, filename)
            for filename in os.listdir(reference_trajectories_dir)
            if filename.endswith(".pkl")
        ]
        if not pickle_files:
            raise ValueError(f"No pickle files found in reference trajectory directory: {reference_trajectories_dir}")

        latest_file = max(pickle_files, key=os.path.getctime)
        with open(latest_file, "rb") as file:
            reference_payload = pickle.load(file)

        reference_states = self._unpack_reference_states(reference_payload, latest_file)
        self.loaded_reference_trajectory_path = latest_file
        print(f"Loaded Reference Trajectory: {latest_file}")
        return reference_states

    @staticmethod
    def _unpack_reference_states(reference_payload, reference_path: str) -> np.ndarray:
        if isinstance(reference_payload, dict):
            if "reference_states" not in reference_payload:
                raise ValueError(
                    f"Loaded reference trajectory payload from {reference_path} must contain a "
                    "'reference_states' field."
                )
            reference_states = reference_payload["reference_states"]
        else:
            reference_states = reference_payload

        reference_states = np.asarray(reference_states, dtype=float)
        if reference_states.ndim != 2 or reference_states.shape[1] != 8:
            raise ValueError(
                f"Loaded reference states from {reference_path} must have shape (N, 8), got {reference_states.shape}"
            )
        return reference_states

    def _extend_reference_states(self, reference_states: np.ndarray) -> np.ndarray:
        if len(reference_states) >= len(self.sim_time_grid):
            return reference_states[: len(self.sim_time_grid)]

        num_extra_steps = len(self.sim_time_grid) - len(reference_states)
        last_ref_state = reference_states[-1]
        return np.vstack([reference_states, np.tile(last_ref_state, (num_extra_steps, 1))])

    def resolve_window_length(self, window_length: int | None = None) -> int:
        if window_length is None:
            if self.window_length is None:
                return len(self.reference_states)
            return int(self.window_length)

        window_length = int(window_length)
        if window_length <= 0:
            raise ValueError("window_length must be a positive integer.")
        return window_length

    def initial_reference_pose(self) -> jnp.ndarray:
        return jnp.asarray(self.reference_states[0, :3], dtype=jnp.float32)

    def _init_states(self, robot_key, estimator_key):
        initial_pose = self.initial_reference_pose()
        robot_state0 = self.robot.get_init_state(key=robot_key, init_pose=initial_pose)
        est_state0 = self.estimator.get_init_state(key=estimator_key, start_pose=initial_pose)
        ctrl_state0 = jnp.zeros(2, dtype=jnp.float32)
        return robot_state0, est_state0, ctrl_state0

    @staticmethod
    def pose_mse(predicted_poses, target_poses):
        pos_error = predicted_poses[:, :2] - target_poses[:, :2]
        angle_error = predicted_poses[:, 2] - target_poses[:, 2]
        angle_loss = 2.0 - 2.0 * jnp.cos(angle_error)
        squared_error = jnp.sum(pos_error ** 2, axis=1) + angle_loss
        return jnp.mean(squared_error)

    def run_closed_loop(
        self,
        robot_params: PhysicalParams,
        est_params: PhysicalParams | None = None,
        use_hidden_robot: bool = False,
        controller_gains=None,
        robot_key=None,
        estimator_key=None,
    ) -> SimulationLog:
        robot_key = self.target_robot_key if robot_key is None else robot_key
        estimator_key = self.target_estimator_key if estimator_key is None else estimator_key
        carry0 = self._init_states(robot_key, estimator_key)
        interval_reference_states = self.reference_states[:-1]
        model_params = robot_params if est_params is None else est_params

        def sim_step(carry, ref_k):
            robot_state, est_state, ctrl_state = carry

            pose_est = self.estimator.get_est_pose(est_state)
            wheel_est = self.estimator.get_est_wheel_speeds(est_state)
            next_ctrl_state, duty_cycle = self.controller.compute(
                ctrl_state,
                ref_k,
                pose_est,
                wheel_est,
                gains=controller_gains,
                wheel_radius=robot_params.wheel_radius,
                base_diameter=robot_params.base_diameter,
                max_wheel_speed=robot_params.max_wheel_speed,
            )
            if use_hidden_robot:
                next_robot_state = self.robot.step(robot_state, duty_cycle)
            else:
                next_robot_state = self.robot.step(
                    robot_state,
                    duty_cycle,
                    wheel_radius=robot_params.wheel_radius,
                    base_diameter=robot_params.base_diameter,
                    max_wheel_speed=robot_params.max_wheel_speed,
                    time_constant=robot_params.time_constant,
                )

            next_est_state = self.estimator.update(
                est_state,
                next_robot_state.wheel_speeds[0],
                next_robot_state.wheel_speeds[1],
                self.robot.get_pose(next_robot_state),
                wheel_radius=model_params.wheel_radius,
                base_diameter=model_params.base_diameter,
            )

            return (next_robot_state, next_est_state, next_ctrl_state), (next_robot_state, next_est_state)

        _, logs = jax.lax.scan(sim_step, carry0, interval_reference_states)
        interval_robot_states, interval_estimator_states = logs
        initial_robot_state, initial_estimator_state, _ = carry0
        robot_states = DiffDriveState(
            pose=jnp.concatenate([initial_robot_state.pose[None, :], interval_robot_states.pose], axis=0),
            wheel_speeds=jnp.concatenate(
                [initial_robot_state.wheel_speeds[None, :], interval_robot_states.wheel_speeds],
                axis=0,
            ),
            key=jnp.concatenate([initial_robot_state.key[None, :], interval_robot_states.key], axis=0),
            vel_omega=jnp.concatenate([initial_robot_state.vel_omega[None, :], interval_robot_states.vel_omega], axis=0),
            duty_cycle=jnp.concatenate([initial_robot_state.duty_cycle[None, :], interval_robot_states.duty_cycle], axis=0),
            wheel_speed_cmd=jnp.concatenate([initial_robot_state.wheel_speed_cmd[None, :], interval_robot_states.wheel_speed_cmd], axis=0),
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

    def run_open_loop_replay(
        self,
        robot_params: PhysicalParams,
        duty_cycles,
        est_params: PhysicalParams | None = None,
        robot_key=None,
        estimator_key=None,
    ) -> SimulationLog:
        robot_key = self.robot_key if robot_key is None else robot_key
        estimator_key = self.estimator_key if estimator_key is None else estimator_key
        carry0 = self._init_states(robot_key, estimator_key)
        duty_cycles = jnp.asarray(duty_cycles, dtype=jnp.float32)
        num_intervals = self.reference_states.shape[0] - 1
        if duty_cycles.shape[0] == num_intervals + 1:
            interval_duty_cycles = duty_cycles[:-1]
        elif duty_cycles.shape[0] == num_intervals:
            interval_duty_cycles = duty_cycles
        else:
            raise ValueError(
                f"duty_cycles length ({duty_cycles.shape[0]}) must match either the number of "
                f"samples ({num_intervals + 1}) or intervals ({num_intervals})."
            )
        model_params = robot_params if est_params is None else est_params

        def sim_step(carry, duty_cycle):
            robot_state, est_state, ctrl_state = carry
            next_robot_state = self.robot.step(
                robot_state,
                duty_cycle,
                wheel_radius=robot_params.wheel_radius,
                base_diameter=robot_params.base_diameter,
                max_wheel_speed=robot_params.max_wheel_speed,
                time_constant=robot_params.time_constant,
            )
            next_est_state = self.estimator.update(
                est_state,
                next_robot_state.wheel_speeds[0],
                next_robot_state.wheel_speeds[1],
                self.robot.get_pose(next_robot_state),
                wheel_radius=model_params.wheel_radius,
                base_diameter=model_params.base_diameter,
            )
            return (next_robot_state, next_est_state, ctrl_state), (next_robot_state, next_est_state)

        _, logs = jax.lax.scan(sim_step, carry0, interval_duty_cycles)
        interval_robot_states, interval_estimator_states = logs
        initial_robot_state, initial_estimator_state, _ = carry0
        robot_states = DiffDriveState(
            pose=jnp.concatenate([initial_robot_state.pose[None, :], interval_robot_states.pose], axis=0),
            wheel_speeds=jnp.concatenate(
                [initial_robot_state.wheel_speeds[None, :], interval_robot_states.wheel_speeds],
                axis=0,
            ),
            key=jnp.concatenate([initial_robot_state.key[None, :], interval_robot_states.key], axis=0),
            vel_omega=jnp.concatenate([initial_robot_state.vel_omega[None, :], interval_robot_states.vel_omega], axis=0),
            duty_cycle=jnp.concatenate([initial_robot_state.duty_cycle[None, :], interval_robot_states.duty_cycle], axis=0),
            wheel_speed_cmd=jnp.concatenate([initial_robot_state.wheel_speed_cmd[None, :], interval_robot_states.wheel_speed_cmd], axis=0),
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

    def simulate(
        self,
        robot_params,
        est_params=None,
        use_hidden_robot=False,
        controller_gains=None,
        duty_cycles=None,
        robot_key=None,
        estimator_key=None,
    ):
        if duty_cycles is None:
            return self.run_closed_loop(
                robot_params=robot_params,
                est_params=est_params,
                use_hidden_robot=use_hidden_robot,
                controller_gains=controller_gains,
                robot_key=robot_key,
                estimator_key=estimator_key,
            )

        return self.run_open_loop_replay(
            robot_params=robot_params,
            duty_cycles=duty_cycles,
            est_params=est_params,
            robot_key=robot_key,
            estimator_key=estimator_key,
        )
