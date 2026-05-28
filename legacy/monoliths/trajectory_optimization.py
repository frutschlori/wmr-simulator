import argparse
from datetime import datetime
from math import comb
import os
import pickle
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from wmr_simulator.controller_jax import Controller
from wmr_simulator.estimator_jax import DiffDriveEstimator, EstimatorState
from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.robot_jax import DiffDrive, DiffDriveState
from wmr_simulator.trajectory_vis import (
    plot_loss_history as plot_loss_history_figure,
    plot_trajectory as plot_trajectory_figure,
)


class ProblemDefinition:
    def __init__(self, problem_path: str):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.raw = yaml.safe_load(file)

        self.path = problem_path
        self.dt = float(self.raw["time_step"])
        self.sim_time = float(self.raw["sim_time"])
        self.planner_cfg = self.raw["planner"]
        self.planner_time = float(self.planner_cfg["time"])
        self.start = np.asarray(self.raw["start"], dtype=float)
        self.goal = np.asarray(self.raw["goal"], dtype=float)
        self.planner_waypoints = self.planner_cfg.get("waypoints", [])
        self.environment_cfg = self.raw.get("environment", {})
        self.environment_min = np.asarray(self.environment_cfg.get("min", [-np.inf, -np.inf]), dtype=float)
        self.environment_max = np.asarray(self.environment_cfg.get("max", [np.inf, np.inf]), dtype=float)
        self.robot_cfg = self.raw["robot"]
        self.estimator_cfg = self.raw.get("estimator", {})
        self.controller_cfg = self.raw.get("controller", {})

    def build_robot(self) -> DiffDrive:
        robot_type = self.robot_cfg.get("type")
        if robot_type != "differential_drive":
            raise ValueError(f"Unsupported robot type '{robot_type}'")
        return DiffDrive(robot_cfg=self.robot_cfg, dt=self.dt)

    def build_estimator(self) -> DiffDriveEstimator:
        return DiffDriveEstimator(estimator_cfg=self.estimator_cfg, dt=self.dt)

    def build_controller(self) -> Controller:
        return Controller(
            robot_param=self.estimator_cfg,
            gains=self.controller_cfg["gains"],
            cmd_limits=[-self.robot_cfg["max_wheel_speed"], self.robot_cfg["max_wheel_speed"]],
            dt=self.dt,
        )

    def planner_time_grid(self) -> np.ndarray:
        num_steps = int(self.planner_time / self.dt)
        return np.linspace(0.0, num_steps * self.dt, num_steps + 1)

    def sim_time_grid(self) -> np.ndarray:
        num_steps = int(self.sim_time / self.dt)
        return np.linspace(0.0, num_steps * self.dt, num_steps + 1)


class Trajectory:
    def __init__(self, time: np.ndarray, poses: np.ndarray, speeds: np.ndarray):
        self.time = np.asarray(time, dtype=float)
        self.poses = np.asarray(poses, dtype=float)
        self.speeds = np.asarray(speeds, dtype=float)

    def linear_speed(self) -> np.ndarray:
        theta = self.poses[:, 2]
        vx = self.speeds[:, 0]
        vy = self.speeds[:, 1]
        return vx * np.cos(theta) + vy * np.sin(theta)


class ReplayInitializationLog(NamedTuple):
    wheel_speeds: jax.Array
    estimator_u_hat: jax.Array
    estimator_u_true: jax.Array
    estimator_covariances: jax.Array


class ClosedLoopLog(NamedTuple):
    poses: jax.Array
    wheel_cmds: jax.Array
    measurements: jax.Array
    replay_init: ReplayInitializationLog


class OptimizationSnapshot(NamedTuple):
    step: int
    loss_value: float
    control_points: np.ndarray
    reference_states: np.ndarray
    closed_loop_log: ClosedLoopLog


class TrajectoryGenerator:
    def reference_states(self, problem: ProblemDefinition) -> jnp.ndarray:
        raise NotImplementedError("TrajectoryGenerator subclasses must implement reference_states().")

    def generate(self, problem: ProblemDefinition) -> Trajectory:
        raise NotImplementedError("TrajectoryGenerator subclasses must implement generate().")


class PlannerTrajectoryGenerator(TrajectoryGenerator):
    def reference_states(self, problem: ProblemDefinition) -> jnp.ndarray:
        time_grid = problem.planner_time_grid()
        reference_states, _ = compute_reference_trajectory(
            start=problem.start,
            goal=problem.goal,
            intermediate_waypoints=problem.planner_waypoints,
            time=time_grid,
        )
        return jnp.asarray(reference_states, dtype=jnp.float32)

    def generate(self, problem: ProblemDefinition) -> Trajectory:
        reference_states = np.asarray(self.reference_states(problem), dtype=float)
        time_grid = problem.planner_time_grid()
        return Trajectory(time_grid, reference_states[:, :3], reference_states[:, 3:6])


class BezierTrajectoryGenerator(TrajectoryGenerator):
    """
    Simple Bezier reference generator.

    The control points define a single Bezier curve in x/y over the planner time
    horizon. The resulting reference state layout matches existing planner module:
      [x, y, theta, vx, vy, omega, ax, ay]
    """

    def __init__(self, control_points=None):
        self.control_points = None if control_points is None else np.asarray(control_points, dtype=float)

    @staticmethod
    def _bezier_basis(num_control_points: int, s: jnp.ndarray) -> jnp.ndarray:
        degree = num_control_points - 1
        return jnp.stack(
            [
                comb(degree, i) * ((1.0 - s) ** (degree - i)) * (s ** i)
                for i in range(num_control_points)
            ],
            axis=1,
        )

    @classmethod
    def _evaluate_bezier(cls, control_points: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
        basis = cls._bezier_basis(control_points.shape[0], s)
        return basis @ control_points

    def resolve_control_points(self, problem: ProblemDefinition) -> jnp.ndarray:
        if self.control_points is None:
            control_points = np.asarray([problem.start[:2], problem.goal[:2]], dtype=float)
        else:
            control_points = self.control_points
        return jnp.asarray(control_points, dtype=jnp.float32)

    def reference_states_from_control_points(
        self,
        problem: ProblemDefinition,
        control_points: jnp.ndarray,
    ) -> jnp.ndarray:
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
        total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
        s = jnp.clip(time_grid / total_time, 0.0, 1.0)

        degree = control_points.shape[0] - 1
        if degree < 1:
            raise ValueError("Bezier trajectory requires at least two control points.")

        position = self._evaluate_bezier(control_points, s)

        first_diff = degree * (control_points[1:] - control_points[:-1])
        dpos_ds = self._evaluate_bezier(first_diff, s)

        if degree >= 2:
            second_diff = (degree - 1) * (first_diff[1:] - first_diff[:-1])
            d2pos_ds2 = self._evaluate_bezier(second_diff, s)
        else:
            d2pos_ds2 = jnp.zeros_like(position)

        velocity = dpos_ds / total_time
        acceleration = d2pos_ds2 / (total_time ** 2)

        tangent_heading = jnp.arctan2(velocity[:, 1], velocity[:, 0])
        theta = tangent_heading

        speed_sq = jnp.sum(velocity ** 2, axis=1)
        omega = (
            velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
        ) / (speed_sq + 1e-8)

        return jnp.column_stack(
            [
                position[:, 0],
                position[:, 1],
                theta,
                velocity[:, 0],
                velocity[:, 1],
                omega,
                acceleration[:, 0],
                acceleration[:, 1],
            ]
        )

    def reference_states(self, problem: ProblemDefinition) -> jnp.ndarray:
        control_points = self.resolve_control_points(problem)
        return self.reference_states_from_control_points(problem, control_points)

    def generate(self, problem: ProblemDefinition) -> Trajectory:
        reference_states = np.asarray(self.reference_states(problem), dtype=float)
        time_grid = problem.sim_time_grid()
        return Trajectory(time_grid, reference_states[:, :3], reference_states[:, 3:6])


def build_trajectory_generator(
    problem: ProblemDefinition,
    generator_type: str | None = None,
) -> TrajectoryGenerator:
    planner_generator_cfg = problem.planner_cfg.get("trajectory_generator", {})
    selected_generator = planner_generator_cfg.get("type", "planner") if generator_type is None else generator_type

    if selected_generator == "planner":
        return PlannerTrajectoryGenerator()
    if selected_generator == "bezier":
        return BezierTrajectoryGenerator()

    raise ValueError(f"Unsupported trajectory generator '{selected_generator}'.")


class TrajectoryOptimizationPipeline:
    def __init__(
        self,
        problem_path: str,
        trajectory_generator: TrajectoryGenerator | None = None,
        trajectory_generator_type: str | None = None,
    ):
        self.problem = ProblemDefinition(problem_path)
        self.robot = self.problem.build_robot()
        self.controller = self.problem.build_controller()
        self.controller_gains = jnp.asarray(self.problem.controller_cfg["gains"], dtype=jnp.float32)
        self.estimator = self.problem.build_estimator()
        estimator_init_state = self.estimator.get_init_state(
            key=jax.random.PRNGKey(1),
            start_pose=self.problem.estimator_cfg["start"],
        )
        self.initial_estimator_covariance = estimator_init_state.P
        master_key = jax.random.PRNGKey(0)
        target_key, replay_key = jax.random.split(master_key, 2)
        self.target_robot_key, self.target_estimator_key = jax.random.split(target_key, 2)
        self.replay_robot_key, self.replay_estimator_key = jax.random.split(replay_key, 2)
        self.bezier_generator = BezierTrajectoryGenerator()

        if trajectory_generator is None:
            self.trajectory_generator = build_trajectory_generator(
                self.problem,
                generator_type=trajectory_generator_type,
            )
        else:
            self.trajectory_generator = trajectory_generator

        self.reference_states = self.reference_sequence()
        self.trajectory = self.trajectory_generator.generate(self.problem)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))
        self.loss_history = None
        self.optimization_snapshots = None

    @staticmethod
    def _print_progress(step: int, total_steps: int, loss_value: float, bar_width: int = 30):
        if total_steps <= 0:
            return

        completed = int(bar_width * step / total_steps)
        bar = "=" * completed + "." * (bar_width - completed)
        sys.stdout.write(f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}")
        if step == total_steps:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def nominal_parameters(self) -> jnp.ndarray:
        return jnp.array([self.robot.r, self.robot.L], dtype=jnp.float32)

    def default_measurement_variances(self) -> np.ndarray:
        noise_pos = float(self.problem.estimator_cfg.get("noise_pos", 1.0))
        noise_angle = float(self.problem.estimator_cfg.get("noise_angle", 1.0))

        pos_var = max(noise_pos ** 2, 1e-6)
        angle_var = max(noise_angle ** 2, 1e-6)
        return np.array([pos_var, pos_var, angle_var], dtype=float)

    def reference_sequence(self) -> jnp.ndarray:
        return jnp.asarray(self.trajectory_generator.reference_states(self.problem), dtype=jnp.float32)

    def bezier_reference_sequence(self, control_points: jnp.ndarray) -> jnp.ndarray:
        return self.bezier_generator.reference_states_from_control_points(self.problem, control_points)

    def current_bezier_control_points(self):
        if not isinstance(self.trajectory_generator, BezierTrajectoryGenerator):
            return None
        return np.asarray(self.trajectory_generator.resolve_control_points(self.problem), dtype=float)

    def trajectory_from_reference_states(self, reference_states: jnp.ndarray) -> Trajectory:
        reference_states_np = np.asarray(reference_states, dtype=float)
        time_grid = self.problem.sim_time_grid()
        return Trajectory(time_grid, reference_states_np[:, :3], reference_states_np[:, 3:6])

    def _set_closed_loop_log(self, closed_loop_log: ClosedLoopLog):
        self.closed_loop_log = closed_loop_log
        self.closed_loop_poses = closed_loop_log.poses
        self.closed_loop_wheel_cmds = closed_loop_log.wheel_cmds
        self.closed_loop_measurements = closed_loop_log.measurements

    def resolve_window_length(self, window_length: int | None) -> int:
        if window_length is None:
            return len(self.trajectory.time)

        window_length = int(window_length)
        if window_length <= 0:
            raise ValueError("window_length must be a positive integer.")
        return window_length

    def initial_estimator_state(self, init_pose: jnp.ndarray, estimator_key: jax.Array):
        return EstimatorState(
            pose_hat=jnp.asarray(init_pose, dtype=jnp.float32),
            pose_meas=jnp.asarray(init_pose, dtype=jnp.float32),
            u_hat=jnp.zeros(2, dtype=jnp.float32),
            u_true=jnp.zeros(2, dtype=jnp.float32),
            P=jnp.asarray(self.initial_estimator_covariance, dtype=jnp.float32),
            key=estimator_key,
        )

    def initial_closed_loop_carry(self, robot_key: jax.Array, estimator_key: jax.Array):
        robot_state = self.robot.get_init_state(
            key=robot_key,
            init_pose=self.problem.start,
        )
        estimator_state = self.initial_estimator_state(
            jnp.asarray(self.problem.estimator_cfg["start"], dtype=jnp.float32),
            estimator_key,
        )
        controller_state = jnp.zeros(2, dtype=jnp.float32)
        return robot_state, estimator_state, controller_state

    def initial_replay_carry(
        self,
        init_pose: jnp.ndarray,
        robot_key: jax.Array,
        estimator_key: jax.Array,
        init_wheel_speeds: jnp.ndarray | None = None,
        init_u_hat: jnp.ndarray | None = None,
        init_u_true: jnp.ndarray | None = None,
        init_covariance: jnp.ndarray | None = None,
    ):
        if init_wheel_speeds is None:
            init_wheel_speeds = jnp.zeros(2, dtype=jnp.float32)
        if init_u_hat is None:
            init_u_hat = jnp.zeros(2, dtype=jnp.float32)
        if init_u_true is None:
            init_u_true = jnp.zeros(2, dtype=jnp.float32)
        if init_covariance is None:
            init_covariance = jnp.asarray(self.initial_estimator_covariance, dtype=jnp.float32)

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

    def closed_loop_step(self, carry, ref_k: jnp.ndarray, params: jnp.ndarray):
        robot_state, estimator_state, controller_state = carry

        ur_true, ul_true = self.robot.get_wheel_speeds(robot_state)
        pose_true = self.robot.get_pose(robot_state)

        next_estimator_state = self.estimator.update(
            estimator_state,
            ur_true,
            ul_true,
            pose_true,
            wheel_radius=params[0],
            base_diameter=params[1],
        )
        pose_est = self.estimator.get_est_pose(next_estimator_state)
        wheel_est = self.estimator.get_est_wheel_speeds(next_estimator_state)

        next_controller_state, wheel_cmd = self.controller.compute(
            controller_state,
            ref_k,
            pose_est,
            wheel_est,
            gains=self.controller_gains,
            wheel_radius=params[0],
            base_diameter=params[1],
        )
        next_robot_state = self.robot.step(
            robot_state,
            wheel_cmd,
            wheel_radius=params[0],
            base_diameter=params[1],
        )

        next_pose_true = self.robot.get_pose(next_robot_state)
        next_carry = (next_robot_state, next_estimator_state, next_controller_state)
        measurement = self.estimator.get_est_pose(next_estimator_state)
        replay_init_wheel_speeds = robot_state.wheel_speeds
        replay_init_u_hat = next_estimator_state.u_hat
        replay_init_u_true = next_estimator_state.u_true
        replay_init_covariance = next_estimator_state.P
        return next_carry, (
            next_pose_true,
            wheel_cmd,
            measurement,
            replay_init_wheel_speeds,
            replay_init_u_hat,
            replay_init_u_true,
            replay_init_covariance,
        )

    def run_closed_loop_deployment(self, reference_states: jnp.ndarray):
        params = self.nominal_parameters()
        initial_carry = self.initial_closed_loop_carry(self.target_robot_key, self.target_estimator_key)

        def scan_step(carry, ref_k):
            return self.closed_loop_step(carry, ref_k, params)

        _, outputs = jax.lax.scan(scan_step, initial_carry, reference_states)
        poses, wheel_cmds, measurements, init_wheel_speeds, init_u_hat, init_u_true, init_covariances = outputs
        replay_init = ReplayInitializationLog(
            wheel_speeds=init_wheel_speeds,
            estimator_u_hat=init_u_hat,
            estimator_u_true=init_u_true,
            estimator_covariances=init_covariances,
        )
        return ClosedLoopLog(
            poses=poses,
            wheel_cmds=wheel_cmds,
            measurements=measurements,
            replay_init=replay_init,
        )

    def open_loop_replay_step(self, carry, wheel_cmd: jnp.ndarray, params: jnp.ndarray):
        robot_state, estimator_state = carry

        next_robot_state = self.robot.step(
            robot_state,
            wheel_cmd,
            wheel_radius=params[0],
            base_diameter=params[1],
        )
        ur_true, ul_true = self.robot.get_wheel_speeds(next_robot_state)
        pose_true = self.robot.get_pose(next_robot_state)
        next_estimator_state = self.estimator.update(
            estimator_state,
            ur_true,
            ul_true,
            pose_true,
            wheel_radius=params[0],
            base_diameter=params[1],
        )
        measurement = self.estimator.get_est_pose(next_estimator_state)
        actual_pose = self.robot.get_pose(next_robot_state)
        return (next_robot_state, next_estimator_state), (actual_pose, measurement)

    def replay_rollout(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: ClosedLoopLog | None = None,
    ):
        window_length = self.resolve_window_length(window_length)
        if closed_loop_log is None:
            closed_loop_log = self.closed_loop_log

        num_steps = closed_loop_log.wheel_cmds.shape[0]
        num_windows = int(np.ceil(num_steps / window_length))
        padded_num_steps = num_windows * window_length
        pad_steps = padded_num_steps - num_steps

        padded_wheel_cmds = jnp.pad(closed_loop_log.wheel_cmds, ((0, pad_steps), (0, 0)))
        wheel_cmd_windows = padded_wheel_cmds.reshape(num_windows, window_length, 2)

        window_start_indices = jnp.arange(num_windows, dtype=jnp.int32) * window_length
        init_poses = closed_loop_log.measurements[window_start_indices]
        init_wheel_speeds = closed_loop_log.replay_init.wheel_speeds[window_start_indices]
        init_u_hat = closed_loop_log.replay_init.estimator_u_hat[window_start_indices]
        init_u_true = closed_loop_log.replay_init.estimator_u_true[window_start_indices]
        init_covariances = closed_loop_log.replay_init.estimator_covariances[window_start_indices]

        window_robot_keys = jax.random.split(self.replay_robot_key, num_windows)
        window_estimator_keys = jax.random.split(self.replay_estimator_key, num_windows)

        def rollout_single_window(
            init_pose,
            init_window_wheel_speeds,
            init_window_u_hat,
            init_window_u_true,
            init_window_covariance,
            robot_key,
            estimator_key,
            wheel_cmd_window,
        ):
            initial_carry = self.initial_replay_carry(
                init_pose,
                robot_key,
                estimator_key,
                init_wheel_speeds=init_window_wheel_speeds,
                init_u_hat=init_window_u_hat,
                init_u_true=init_window_u_true,
                init_covariance=init_window_covariance,
            )
            _, outputs = jax.lax.scan(
                lambda carry, wheel_cmd: self.open_loop_replay_step(carry, wheel_cmd, params),
                initial_carry,
                wheel_cmd_window,
            )
            return outputs

        actual_pose_windows, measurement_windows = jax.vmap(rollout_single_window)(
            init_poses,
            init_wheel_speeds,
            init_u_hat,
            init_u_true,
            init_covariances,
            window_robot_keys,
            window_estimator_keys,
            wheel_cmd_windows,
        )

        actual_poses = actual_pose_windows.reshape(-1, 3)[:num_steps]
        measurements = measurement_windows.reshape(-1, 3)[:num_steps]
        return actual_poses, measurements

    def replay_measurement_sequence(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: ClosedLoopLog | None = None,
    ) -> jnp.ndarray:
        """
        Replays commands logged from closed-loop experiment on open-loop robot and records estimates
        over window_length long sequences. Every window start pose is set to the measurement from the closed-loop
        experiment (similar to the SI problem definition), thus the state sensitivity recursion length is limited to
        the window_length (-> smaller windows will yield smaller FIM)
        """
        _, scanned_measurements = self.replay_rollout(
            params,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )
        return scanned_measurements

    def measurement_vector(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: ClosedLoopLog | None = None,
    ) -> jnp.ndarray:
        """ Flattens Nx3 measurement matrix into 3Nx1 vector """
        measurements = self.replay_measurement_sequence(
            params,
            window_length,
            closed_loop_log=closed_loop_log,
        )
        return measurements.reshape(-1)

    def compute_fim_matrix(self, measurement_variances=None,
        window_length=None, closed_loop_log=None) -> jnp.ndarray:
        """ Computes weighted Fisher matrix using relative parameter sensitivities. """

        # Get measurement variance
        if measurement_variances is None:
            measurement_variances = self.default_measurement_variances()
        else:
            measurement_variances = np.asarray(measurement_variances, dtype=float)
        params = self.nominal_parameters()
        inverse_variances = jnp.asarray(1.0 / measurement_variances, dtype=jnp.float32)

        # Get measurement vector
        measurement_vector = self.measurement_vector(
            params,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )
        # Compute measurement sensitivity
        measurement_sensitivity = jax.jacfwd(
            lambda p: self.measurement_vector(
                p,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
            )
        )(params)
        parameter_scaling = jnp.diag(params)
        # parameter_scaling = jnp.eye(2)
        weighted_measurement_sensitivity = measurement_sensitivity @ parameter_scaling

        # Stack variances to weight sensitivity
        num_measurements = measurement_vector.shape[0] // 3
        stacked_inverse_variances = jnp.tile(inverse_variances, num_measurements)
        weights = stacked_inverse_variances.reshape(-1, 1)

        # Finally compute FIM
        return weighted_measurement_sensitivity.T @ (weighted_measurement_sensitivity * weights)

    def initial_bezier_control_points(self, order: int) -> jnp.ndarray:
        if order < 1:
            raise ValueError("Bezier order must be at least 1.")
        num_control_points = order + 1
        line_samples = np.linspace(0.0, 1.0, num_control_points)[:, None]
        start = self.problem.start[:2][None, :]
        goal = self.problem.goal[:2][None, :]
        control_points = start + line_samples * (goal - start)
        return jnp.asarray(control_points, dtype=jnp.float32)

    def clamp_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        env_min = jnp.asarray(self.problem.environment_min, dtype=jnp.float32)
        env_max = jnp.asarray(self.problem.environment_max, dtype=jnp.float32)
        clamped = jnp.clip(control_points, env_min, env_max)
        return clamped.at[0].set(jnp.asarray(self.problem.start[:2], dtype=jnp.float32))

    def control_points_from_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        start_point = jnp.asarray(self.problem.start[:2], dtype=jnp.float32)[None, :]
        control_points = jnp.concatenate([start_point, decision_variables], axis=0)
        return self.clamp_control_points(control_points)

    def set_bezier_control_points(self, control_points: jnp.ndarray):
        control_points = self.clamp_control_points(control_points)
        self.trajectory_generator = BezierTrajectoryGenerator(control_points=np.asarray(control_points))
        self.reference_states = self.bezier_reference_sequence(control_points)
        self.trajectory = self.trajectory_from_reference_states(self.reference_states)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))

    def fim_loss_from_control_points(
        self,
        control_points: jnp.ndarray,
        window_length: int | None = None,
        measurement_variances=None,
    ) -> jnp.ndarray:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.bezier_reference_sequence(control_points)
        closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
        fim = self.compute_fim_matrix(
            measurement_variances=measurement_variances,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )
        regularized_fim = fim + 1e-6 * jnp.eye(fim.shape[0], dtype=fim.dtype)
        inverse_eigenvalues = jnp.linalg.eigvalsh(jnp.linalg.inv(regularized_fim))
        return jnp.max(inverse_eigenvalues)

    def optimize_bezier_trajectory(
        self,
        order: int,
        num_steps: int,
        learning_rate: float,
        window_length: int | None = None,
        measurement_variances=None,
        save_trace: bool = False,
        trace_stride: int = 5,
    ):
        initial_control_points = self.initial_bezier_control_points(order)
        initial_decision_variables = initial_control_points[1:]
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(initial_decision_variables)

        def unnormalized_loss_fn(decision_variables):
            control_points = self.control_points_from_decision_variables(decision_variables)
            return self.fim_loss_from_control_points(
                control_points,
                window_length=window_length,
                measurement_variances=measurement_variances,
            )

        initial_loss = unnormalized_loss_fn(initial_decision_variables)
        loss_scale = 1.0 / jnp.maximum(initial_loss, 1e-12)

        def loss_fn(decision_variables):
            return loss_scale * unnormalized_loss_fn(decision_variables)

        @jax.jit
        def train_step(decision_variables, opt_state):
            loss_value, grads = jax.value_and_grad(loss_fn)(decision_variables)
            updates, next_opt_state = optimizer.update(grads, opt_state, decision_variables)
            next_decision_variables = optax.apply_updates(decision_variables, updates)
            next_control_points = self.control_points_from_decision_variables(next_decision_variables)
            next_decision_variables = next_control_points[1:]
            return next_decision_variables, next_opt_state, loss_value

        decision_variables = initial_decision_variables
        loss_history = []
        snapshots = [] if save_trace else None

        if save_trace:
            initial_closed_loop_log = self.run_closed_loop_deployment(
                reference_states=self.bezier_reference_sequence(initial_control_points)
            )
            snapshots.append(
                OptimizationSnapshot(
                    step=0,
                    loss_value=float(loss_fn(initial_decision_variables)),
                    control_points=np.asarray(initial_control_points),
                    reference_states=np.asarray(self.bezier_reference_sequence(initial_control_points)),
                    closed_loop_log=initial_closed_loop_log,
                )
            )

        print(f"Initial unnormalized loss: {float(initial_loss):.8f}")
        print(f"Loss normalization scale: {float(loss_scale):.8f}")
        for step in range(num_steps):
            decision_variables, opt_state, loss_value = train_step(decision_variables, opt_state)
            loss_history.append(float(loss_value))
            self._print_progress(step + 1, num_steps, float(loss_value))
            if save_trace and ((step + 1) % trace_stride == 0 or step + 1 == num_steps):
                control_points = self.control_points_from_decision_variables(decision_variables)
                reference_states = self.bezier_reference_sequence(control_points)
                closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
                snapshots.append(
                    OptimizationSnapshot(
                        step=step + 1,
                        loss_value=float(loss_value),
                        control_points=np.asarray(control_points),
                        reference_states=np.asarray(reference_states),
                        closed_loop_log=closed_loop_log,
                    )
                )

        optimized_control_points = self.control_points_from_decision_variables(decision_variables)
        self.set_bezier_control_points(optimized_control_points)
        self.loss_history = loss_history
        self.optimization_snapshots = snapshots
        return optimized_control_points, loss_history

    def plot_trajectory(self, window_length=None, out_prefix="trajectory_plot", out_path=None):
        plot_trajectory_figure(
            self,
            window_length=window_length,
            out_prefix=out_prefix,
            out_path=out_path,
        )

    def plot_loss_history(self, out_prefix="loss_history", out_path=None):
        if self.loss_history is None:
            raise ValueError("No optimization loss history available. Run optimize_bezier_trajectory() first.")
        plot_loss_history_figure(
            self.loss_history,
            out_prefix=out_prefix,
            out_path=out_path,
        )

    def save_optimization_GIF(self, window_length=None, out_prefix="traj_opt_trace", frame_duration=0.2):
        if not self.optimization_snapshots:
            raise ValueError("No optimization snapshots available. Enable trace saving during optimization.")
        from wmr_simulator.trajectory_vis import save_optimization_trace as save_trace_figure

        save_trace_figure(
            self,
            self.optimization_snapshots,
            window_length=window_length,
            out_prefix=out_prefix,
            frame_duration=frame_duration,
        )

    def save_optimization_reference_states(
        self,
        out_dir="trajectory_opt_reference_exports",
        filename_prefix="traj_opt_reference_states",
    ):
        if not self.optimization_snapshots:
            raise ValueError("No optimization snapshots available. Enable trace saving during optimization.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(out_dir, f"{filename_prefix}_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)

        saved_paths = []
        for snapshot in self.optimization_snapshots:
            filename = f"step_{snapshot.step:05d}.pkl"
            out_path = os.path.join(export_dir, filename)
            with open(out_path, "wb") as file:
                pickle.dump(np.asarray(snapshot.reference_states), file)
            saved_paths.append(out_path)

        return export_dir, saved_paths

    def save_reference_states_pickle(self, out_dir="trajectory_exports", filename_prefix="reference_states"):
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as file:
            pickle.dump(np.asarray(self.reference_states), file)
        return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize trajectory optimization inputs.")
    parser.add_argument("problem", nargs="?", default="problems/problem_hidden.yaml")
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--trajectory-generator", choices=["planner", "bezier"], default="bezier")
    parser.add_argument("--bezier-order", type=int, default=7)
    parser.add_argument("--opt-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    # Arguments for GIF creation
    parser.add_argument("--save-opt-GIF", action="store_true", default=True)
    parser.add_argument("--export-opt-reference-states", action="store_true", default=True)
    parser.add_argument("--opt-trace-stride", type=int, default=50)
    parser.add_argument("--save-trajectory", action="store_true", default=True)
    args = parser.parse_args()

    pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        trajectory_generator_type=args.trajectory_generator,
    )

    print(f"Loaded problem: {pipeline.problem.path}")
    print(f"Robot: {type(pipeline.robot).__name__}")
    print(f"Trajectory generator: {type(pipeline.trajectory_generator).__name__}")
    print(f"Trajectory samples: {len(pipeline.trajectory.time)}")
    print(f"Start pose: {pipeline.trajectory.poses[0]}")
    print(f"Goal pose:  {pipeline.trajectory.poses[-1]}")
    print(f"Window length: {pipeline.resolve_window_length(args.window_length)}")
    print("Measurement vector shape:")
    print(pipeline.measurement_vector(pipeline.nominal_parameters(), window_length=args.window_length).shape)
    print("FIM:")
    print(pipeline.compute_fim_matrix(window_length=args.window_length))

    if args.trajectory_generator == "bezier":
        initial_control_points = pipeline.initial_bezier_control_points(args.bezier_order)
        pipeline.set_bezier_control_points(initial_control_points)

    pipeline.plot_trajectory(
        window_length=args.window_length,
        out_prefix=f"traj_initial_{args.trajectory_generator}",
    )

    if args.opt_steps and args.trajectory_generator == "bezier":
        print("Initial control points:")
        print(np.asarray(initial_control_points))
        optimized_control_points, loss_history = pipeline.optimize_bezier_trajectory(
            order=args.bezier_order,
            num_steps=args.opt_steps,
            learning_rate=args.learning_rate,
            window_length=args.window_length,
            save_trace=args.save_opt_GIF,
            trace_stride=args.opt_trace_stride,
        )
        print("Optimized control points:")
        print(np.asarray(optimized_control_points))
        print("Final optimization loss:")
        print(loss_history[-1])
        print("Optimized FIM:")
        print(pipeline.compute_fim_matrix(window_length=args.window_length))
        pipeline.plot_trajectory(
            window_length=args.window_length,
            out_prefix="traj_optimized_bezier",
        )
        pipeline.plot_loss_history(out_prefix="traj_opt_loss_history")
        if args.save_opt_GIF:
            pipeline.save_optimization_GIF(
                window_length=args.window_length,
                out_prefix="traj_opt",
            )
        if args.export_opt_reference_states:
            export_dir, saved_paths = pipeline.save_optimization_reference_states(
                filename_prefix="traj_opt_reference_states",
            )
            print("Exported optimization reference states:")
            print(export_dir)
            print(f"Saved {len(saved_paths)} snapshots")

    if args.save_trajectory:
        saved_path = pipeline.save_reference_states_pickle(
            filename_prefix=f"{args.trajectory_generator}_reference_states",
        )
        print("Saved trajectory pickle:")
        print(saved_path)
