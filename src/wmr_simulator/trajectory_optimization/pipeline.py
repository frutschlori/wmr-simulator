from datetime import datetime
import os
import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.controller import Controller
from wmr_simulator.estimator import DiffDriveEstimator, EstimatorState
from wmr_simulator.robot import DiffDrive, DiffDriveState
from wmr_simulator.trajectory_optimization.bezier import BezierTrajectoryGenerator
from wmr_simulator.trajectory_optimization.constraints import (
    clamp_control_points,
    constraint_loss_components_from_reference_states,
    constraint_loss_from_reference_states,
    constraint_weights,
    control_points_from_decision_variables,
    initial_bezier_control_points,
    motion_limits_from_robot_config,
)
from wmr_simulator.trajectory_optimization.fim import (
    compute_fim_matrix,
    default_measurement_variances,
)
from wmr_simulator.trajectory_optimization.objectives import fim_loss, trajectory_objective
from wmr_simulator.trajectory_optimization.optimizers import optimize_bezier_control_points
from wmr_simulator.trajectory_optimization.parametrization import (
    Trajectory,
    TrajectoryGenerator,
    build_trajectory_generator,
    normalize_time_scaling,
)
from wmr_simulator.visualization.trajectories import (
    plot_loss_history as plot_loss_history_figure,
    plot_trajectory as plot_trajectory_figure,
)


class ProblemDefinition:
    def __init__(self, problem_path: str):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.raw = yaml.safe_load(file)

        self.path = problem_path
        self.geometry_dt = float(self.raw["geometry_controller_dt"])
        self.wheel_dt = float(self.raw["wheel_controller_dt"])
        self.dt = self.geometry_dt
        self.inner_steps_per_geometry_step = self._inner_steps(self.geometry_dt, self.wheel_dt)
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
        return DiffDrive(robot_cfg=self.robot_cfg, dt=self.wheel_dt)

    def build_estimator(self) -> DiffDriveEstimator:
        return DiffDriveEstimator(estimator_cfg=self.estimator_cfg, dt=self.wheel_dt)

    def build_controller(self) -> Controller:
        return Controller(
            robot_param=self.robot_cfg,
            gains=self.controller_cfg["gains"],
            duty_limits=[-1.0, 1.0],
            dt=self.geometry_dt,
        )

    def planner_time_grid(self) -> np.ndarray:
        num_steps = int(self.planner_time / self.geometry_dt)
        return np.linspace(0.0, num_steps * self.geometry_dt, num_steps + 1)

    def sim_time_grid(self) -> np.ndarray:
        num_steps = int(self.sim_time / self.geometry_dt)
        return np.linspace(0.0, num_steps * self.geometry_dt, num_steps + 1)

    @staticmethod
    def _inner_steps(outer_dt: float, inner_dt: float) -> int:
        steps = int(np.round(outer_dt / inner_dt))
        if steps <= 0 or not np.isclose(steps * inner_dt, outer_dt):
            raise ValueError("geometry_controller_dt must be an integer multiple of wheel_controller_dt")
        return steps


class ReplayInitializationLog(NamedTuple):
    wheel_speeds: jax.Array
    estimator_u_hat: jax.Array
    estimator_u_true: jax.Array
    estimator_covariances: jax.Array


class ClosedLoopLog(NamedTuple):
    poses: jax.Array
    duty_cycles: jax.Array
    measurements: jax.Array
    replay_init: ReplayInitializationLog


class OptimizationSnapshot(NamedTuple):
    step: int
    loss_value: float
    control_points: np.ndarray
    reference_states: np.ndarray
    closed_loop_log: ClosedLoopLog


def reference_states_export_payload(reference_states, dt: float, **metadata):
    payload = {
        "reference_states": np.asarray(reference_states),
        "dt": float(dt),
    }
    payload.update(metadata)
    return payload


class TrajectoryOptimizationPipeline:
    def __init__(
        self,
        problem_path: str,
        trajectory_generator: TrajectoryGenerator | None = None,
        trajectory_generator_type: str | None = None,
        time_scaling: str | None = None,
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
        if trajectory_generator is None:
            self.trajectory_generator = build_trajectory_generator(
                self.problem,
                generator_type=trajectory_generator_type,
                time_scaling=time_scaling,
            )
        else:
            self.trajectory_generator = trajectory_generator

        if isinstance(self.trajectory_generator, BezierTrajectoryGenerator):
            self.time_scaling = self.trajectory_generator.time_scaling
        else:
            self.time_scaling = normalize_time_scaling(time_scaling)
        self.bezier_generator = BezierTrajectoryGenerator(time_scaling=self.time_scaling)

        self.reference_states = self.reference_sequence()
        self.trajectory = self.trajectory_generator.generate(self.problem)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))
        self.loss_history = None
        self.optimization_snapshots = None

    def nominal_parameters(self) -> jnp.ndarray:
        return jnp.array([self.robot.r, self.robot.L], dtype=jnp.float32)

    def default_measurement_variances(self) -> np.ndarray:
        return default_measurement_variances(self.problem.estimator_cfg)

    def motion_limits(self) -> dict[str, jnp.ndarray]:
        return motion_limits_from_robot_config(self.problem.robot_cfg)

    def constraint_weights(self, scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
        return constraint_weights(scale=scale, component_weights=component_weights)

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
        self.closed_loop_duty_cycles = closed_loop_log.duty_cycles
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

    @staticmethod
    def initial_pose_from_reference(reference_states: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(reference_states[0, :3], dtype=jnp.float32)

    def initial_closed_loop_carry(
        self,
        robot_key: jax.Array,
        estimator_key: jax.Array,
        reference_states: jnp.ndarray,
    ):
        initial_pose = self.initial_pose_from_reference(reference_states)
        robot_state = self.robot.get_init_state(
            key=robot_key,
            init_pose=initial_pose,
        )
        estimator_state = self.initial_estimator_state(
            initial_pose,
            estimator_key,
        )
        controller_state = jnp.zeros(2, dtype=jnp.float32)
        delayed_duty_cycle = jnp.zeros(2, dtype=jnp.float32)
        return robot_state, estimator_state, controller_state, delayed_duty_cycle

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
            duty_cycle=jnp.zeros(2, dtype=jnp.float32),
            wheel_speed_cmd=jnp.zeros(2, dtype=jnp.float32),
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
        robot_state, estimator_state, controller_state, delayed_duty_cycle = carry

        pose_est = self.estimator.get_est_pose(estimator_state)
        wheel_est = self.estimator.get_est_wheel_speeds(estimator_state)
        next_controller_state, duty_cycle = self.controller.compute(
            controller_state,
            ref_k,
            pose_est,
            wheel_est,
            gains=self.controller_gains,
            wheel_radius=params[0],
            base_diameter=params[1],
        )

        def wheel_step(inner_carry, inner_index):
            inner_robot_state, inner_estimator_state = inner_carry
            applied_duty_cycle = jnp.where(inner_index == 0, delayed_duty_cycle, duty_cycle)
            next_robot_state = self.robot.step(
                inner_robot_state,
                applied_duty_cycle,
                wheel_radius=params[0],
                base_diameter=params[1],
                dt=self.problem.wheel_dt,
            )
            next_estimator_state = self.estimator.update(
                inner_estimator_state,
                next_robot_state.wheel_speeds[0],
                next_robot_state.wheel_speeds[1],
                self.robot.get_pose(next_robot_state),
                wheel_radius=params[0],
                base_diameter=params[1],
                dt=self.problem.wheel_dt,
            )
            return (next_robot_state, next_estimator_state), None

        (next_robot_state, next_estimator_state), _ = jax.lax.scan(
            wheel_step,
            (robot_state, estimator_state),
            jnp.arange(self.problem.inner_steps_per_geometry_step),
        )

        next_pose_true = self.robot.get_pose(next_robot_state)
        next_carry = (next_robot_state, next_estimator_state, next_controller_state, duty_cycle)
        measurement = self.estimator.get_est_pose(next_estimator_state)
        replay_init_wheel_speeds = robot_state.wheel_speeds
        replay_init_u_hat = next_estimator_state.u_hat
        replay_init_u_true = next_estimator_state.u_true
        replay_init_covariance = next_estimator_state.P
        return next_carry, (
            next_pose_true,
            duty_cycle,
            measurement,
            replay_init_wheel_speeds,
            replay_init_u_hat,
            replay_init_u_true,
            replay_init_covariance,
        )

    def run_closed_loop_deployment(self, reference_states: jnp.ndarray):
        params = self.nominal_parameters()
        initial_carry = self.initial_closed_loop_carry(
            self.target_robot_key,
            self.target_estimator_key,
            reference_states,
        )

        def scan_step(carry, ref_k):
            return self.closed_loop_step(carry, ref_k, params)

        _, outputs = jax.lax.scan(scan_step, initial_carry, reference_states)
        poses, duty_cycles, measurements, init_wheel_speeds, init_u_hat, init_u_true, init_covariances = outputs
        replay_init = ReplayInitializationLog(
            wheel_speeds=init_wheel_speeds,
            estimator_u_hat=init_u_hat,
            estimator_u_true=init_u_true,
            estimator_covariances=init_covariances,
        )
        return ClosedLoopLog(
            poses=poses,
            duty_cycles=duty_cycles,
            measurements=measurements,
            replay_init=replay_init,
        )

    def open_loop_replay_step(self, carry, inputs, params: jnp.ndarray):
        robot_state, estimator_state = carry
        wheel_speeds, duty_cycle = inputs

        next_robot_state = self.robot.step_kinematic(
            robot_state,
            wheel_speeds,
            duty_cycle,
            wheel_radius=params[0],
            base_diameter=params[1],
            dt=self.problem.dt,
        )
        ur_true, ul_true = wheel_speeds
        pose_true = self.robot.get_pose(next_robot_state)
        next_estimator_state = self.estimator.update(
            estimator_state,
            ur_true,
            ul_true,
            pose_true,
            wheel_radius=params[0],
            base_diameter=params[1],
            dt=self.problem.dt,
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

        replay_wheel_speeds = closed_loop_log.replay_init.wheel_speeds[:-1]
        replay_duty_cycles = closed_loop_log.duty_cycles[:-1]

        num_intervals = replay_wheel_speeds.shape[0]
        num_windows = int(np.ceil(num_intervals / window_length))
        padded_num_intervals = num_windows * window_length
        pad_steps = padded_num_intervals - num_intervals

        padded_wheel_speeds = jnp.pad(replay_wheel_speeds, ((0, pad_steps), (0, 0)))
        padded_duty_cycles = jnp.pad(replay_duty_cycles, ((0, pad_steps), (0, 0)))
        wheel_speed_windows = padded_wheel_speeds.reshape(num_windows, window_length, 2)
        duty_cycle_windows = padded_duty_cycles.reshape(num_windows, window_length, 2)

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
            wheel_speed_window,
            duty_cycle_window,
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
                lambda carry, replay_inputs: self.open_loop_replay_step(carry, replay_inputs, params),
                initial_carry,
                (wheel_speed_window, duty_cycle_window),
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
            wheel_speed_windows,
            duty_cycle_windows,
        )

        actual_poses = actual_pose_windows.reshape(-1, 3)[:num_intervals]
        measurements = measurement_windows.reshape(-1, 3)[:num_intervals]
        actual_poses = jnp.concatenate([closed_loop_log.measurements[:1], actual_poses], axis=0)
        measurements = jnp.concatenate([closed_loop_log.measurements[:1], measurements], axis=0)
        return actual_poses, measurements

    def replay_measurement_sequence(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: ClosedLoopLog | None = None,
    ) -> jnp.ndarray:
        """
        Replays commands logged from closed-loop experiment on open-loop robot and records estimates
        over window_length long sequences. Every window is initialized from the closed-loop estimate at its start,
        but the replay log stores the integrated window endpoints. Thus the state sensitivity recursion length is
        limited to the window_length (-> smaller windows will yield smaller FIM).
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

        if measurement_variances is None:
            measurement_variances = self.default_measurement_variances()
        params = self.nominal_parameters()

        return compute_fim_matrix(
            lambda p: self.measurement_vector(
                p,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
            ),
            params=params,
            measurement_variances=measurement_variances,
        )

    def initial_bezier_control_points(self, order: int) -> jnp.ndarray:
        return initial_bezier_control_points(self.problem, order)

    def clamp_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        return clamp_control_points(self.problem, control_points)

    def control_points_from_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        return control_points_from_decision_variables(self.problem, decision_variables)

    def set_bezier_control_points(self, control_points: jnp.ndarray):
        control_points = self.clamp_control_points(control_points)
        self.trajectory_generator = BezierTrajectoryGenerator(
            control_points=np.asarray(control_points),
            time_scaling=self.time_scaling,
        )
        self.reference_states = self.bezier_reference_sequence(control_points)
        self.trajectory = self.trajectory_from_reference_states(self.reference_states)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))

    def fim_loss_from_control_points(
        self,
        control_points: jnp.ndarray,
        window_length: int | None = None,
        measurement_variances=None,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
    ) -> jnp.ndarray:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.bezier_reference_sequence(control_points)
        closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
        fim = self.compute_fim_matrix(
            measurement_variances=measurement_variances,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )
        return trajectory_objective(
            fim=fim,
            reference_states=reference_states,
            dt=self.problem.dt,
            limits=self.motion_limits(),
            weights=self.constraint_weights(
                scale=constraint_weight,
                component_weights=constraint_component_weights,
            ),
            smooth_max_beta=constraint_smooth_max_beta,
        )

    def objective_terms_from_control_points(
        self,
        control_points: jnp.ndarray,
        window_length: int | None = None,
        measurement_variances=None,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
    ) -> dict[str, jnp.ndarray]:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.bezier_reference_sequence(control_points)
        closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
        fim = self.compute_fim_matrix(
            measurement_variances=measurement_variances,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )
        fim_term = fim_loss(fim)
        constraint_term = constraint_loss_from_reference_states(
            reference_states=reference_states,
            dt=self.problem.dt,
            limits=self.motion_limits(),
            weights=self.constraint_weights(
                scale=constraint_weight,
                component_weights=constraint_component_weights,
            ),
            smooth_max_beta=constraint_smooth_max_beta,
        )
        total = fim_term + constraint_term
        return {
            "fim": fim_term,
            "constraints": constraint_term,
            "total": total,
            "constraint_share": constraint_term / jnp.maximum(total, 1e-12),
        }

    def constraint_components_from_control_points(
        self,
        control_points: jnp.ndarray,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
    ) -> dict[str, jnp.ndarray]:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.bezier_reference_sequence(control_points)
        return constraint_loss_components_from_reference_states(
            reference_states=reference_states,
            dt=self.problem.dt,
            limits=self.motion_limits(),
            weights=self.constraint_weights(
                scale=constraint_weight,
                component_weights=constraint_component_weights,
            ),
            smooth_max_beta=constraint_smooth_max_beta,
        )

    def optimize_bezier_trajectory(
        self,
        order: int,
        num_steps: int,
        learning_rate: float,
        window_length: int | None = None,
        measurement_variances=None,
        save_trace: bool = False,
        trace_stride: int = 5,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
    ):
        return optimize_bezier_control_points(
            pipeline=self,
            order=order,
            num_steps=num_steps,
            learning_rate=learning_rate,
            window_length=window_length,
            measurement_variances=measurement_variances,
            save_trace=save_trace,
            trace_stride=trace_stride,
            constraint_weight=constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
        )

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

    def save_optimization_GIF(
        self,
        window_length=None,
        out_prefix="traj_opt_trace",
        frame_duration=0.2,
        frames_dir=None,
        gif_path=None,
    ):
        if not self.optimization_snapshots:
            raise ValueError("No optimization snapshots available. Enable trace saving during optimization.")
        from wmr_simulator.visualization.trajectories import save_optimization_trace as save_trace_figure

        return save_trace_figure(
            self,
            self.optimization_snapshots,
            window_length=window_length,
            out_prefix=out_prefix,
            frame_duration=frame_duration,
            frames_dir=frames_dir,
            gif_path=gif_path,
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
                pickle.dump(
                    reference_states_export_payload(
                        snapshot.reference_states,
                        self.problem.dt,
                        step=int(snapshot.step),
                        loss_value=float(snapshot.loss_value),
                    ),
                    file,
                )
            saved_paths.append(out_path)

        return export_dir, saved_paths

    def save_reference_states_pickle(self, out_dir="trajectory_exports", filename_prefix="reference_states"):
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as file:
            pickle.dump(reference_states_export_payload(self.reference_states, self.problem.dt), file)
        return out_path
