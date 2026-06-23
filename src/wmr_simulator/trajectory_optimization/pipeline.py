from datetime import datetime
import os
import pickle
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.controller import Controller
from wmr_simulator.estimator import DiffDriveEstimator
from wmr_simulator.robot import DiffDrive
from wmr_simulator.simulation import SimulationPipeline, make_replay_segment_plan, replay_simulation_log
from wmr_simulator.trajectory_optimization.bezier import (
    clamp_control_points,
    compute_bezier_reference,
    control_points_from_decision_variables,
    decision_variables_from_control_points,
    initial_bezier_control_points,
    tangent_floor_loss,
)
from wmr_simulator.trajectory_optimization.constraints import (
    constraint_loss_components_from_reference_states,
    constraint_loss_from_reference_states,
    constraint_weights,
    motion_limits_from_robot_config,
)
from wmr_simulator.trajectory_optimization.fim import (
    compute_fim_matrix,
    default_measurement_variances,
)
from wmr_simulator.trajectory_optimization.objectives import fim_loss, trajectory_objective
from wmr_simulator.trajectory_optimization.optimizers import optimize_bezier_control_points
from wmr_simulator.trajectory_optimization.parametrization import normalize_time_scaling
from wmr_simulator.types import PhysicalParams, SimulationLog
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


class OptimizationSnapshot(NamedTuple):
    step: int
    loss_value: float
    control_points: np.ndarray
    reference_states: np.ndarray
    closed_loop_log: SimulationLog


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
        time_scaling: str | None = None,
    ):
        self.problem = ProblemDefinition(problem_path)
        self.simulation = SimulationPipeline(problem_path=problem_path, seed=0, reference_trajectories_dir=None)
        self.robot = self.simulation.robot
        self.controller = self.simulation.controller
        self.controller_gains = self.simulation.gains
        self.estimator = self.simulation.estimator
        self.time_scaling = normalize_time_scaling(time_scaling)
        self.control_points = initial_bezier_control_points(self.problem, order=2)
        self.reference_states = self.reference_states_from_control_points(self.control_points)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))
        self.loss_history = None
        self.optimization_snapshots = None

    def nominal_parameters(self) -> jnp.ndarray:
        return jnp.array([self.robot.r, self.robot.L], dtype=jnp.float32)

    def nominal_physical_params(self) -> PhysicalParams:
        return PhysicalParams(
            wheel_radius=jnp.asarray(self.robot.r, dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot.L, dtype=jnp.float32),
            max_wheel_speed=jnp.asarray(self.robot.max_wheel_speed, dtype=jnp.float32),
            time_constant=jnp.asarray(self.robot.tau, dtype=jnp.float32),
        )

    def default_measurement_variances(self) -> np.ndarray:
        return default_measurement_variances(self.problem.estimator_cfg)

    def motion_limits(self) -> dict[str, jnp.ndarray]:
        return motion_limits_from_robot_config(self.problem.robot_cfg)

    def constraint_weights(self, scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
        return constraint_weights(scale=scale, component_weights=component_weights)

    def reference_states_from_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        return compute_bezier_reference(self.problem, control_points, time_scaling=self.time_scaling)

    def current_bezier_control_points(self):
        return np.asarray(self.control_points, dtype=float)

    def _set_closed_loop_log(self, closed_loop_log: SimulationLog):
        self.closed_loop_log = closed_loop_log
        self.closed_loop_poses = closed_loop_log.pose.true_states
        self.closed_loop_duty_cycles = closed_loop_log.wheel.duty_cycle
        self.closed_loop_measurements = closed_loop_log.pose.states

    def run_closed_loop_deployment(self, reference_states: jnp.ndarray) -> SimulationLog:
        return self.simulation.run_closed_loop(
            self.nominal_physical_params(),
            controller_gains=self.controller_gains,
            wheel_speed_log_source="estimated",
            reference_states=reference_states,
        )

    @staticmethod
    def physical_params_from_vector(params: jnp.ndarray) -> PhysicalParams:
        params = jnp.asarray(params, dtype=jnp.float32)
        return PhysicalParams(
            wheel_radius=params[0],
            base_diameter=params[1],
            max_wheel_speed=jnp.asarray(1.0, dtype=jnp.float32),
            time_constant=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def replay_segment_plan(self, target_log: SimulationLog, window_length: int | None = None):
        num_pose_samples = len(target_log.pose.time_s)
        num_wheel_samples = len(target_log.wheel.time_s)
        window_length = self.simulation.resolve_replay_window_length(window_length, num_pose_samples - 1)
        pose_time = np.arange(num_pose_samples, dtype=float) * self.problem.wheel_dt
        wheel_time = np.arange(num_wheel_samples, dtype=float) * self.problem.wheel_dt - self.problem.wheel_dt
        return make_replay_segment_plan(
            pose_time,
            wheel_time,
            window_length,
        )

    def replay_log(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
    ) -> SimulationLog:
        target_log = self.closed_loop_log if closed_loop_log is None else closed_loop_log
        return replay_simulation_log(
            robot=self.robot,
            robot_key=self.simulation.robot_key,
            target_log=target_log,
            robot_params=self.physical_params_from_vector(params),
            replay_segment_plan=self.replay_segment_plan(target_log, window_length),
        )

    def replay_rollout(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
    ):
        target_log = self.closed_loop_log if closed_loop_log is None else closed_loop_log
        replay_log = self.replay_log(params, window_length=window_length, closed_loop_log=closed_loop_log)
        replay_poses = jnp.concatenate([target_log.pose.states[:1], replay_log.pose.states], axis=0)
        return replay_poses, replay_poses

    def replay_measurement_sequence(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
    ) -> jnp.ndarray:
        """
        Replays commands logged from closed-loop experiment on open-loop robot and records estimates
        over window_length long sequences. Every window is initialized from the closed-loop estimate at its start,
        but the replay log stores the integrated window endpoints. Thus the state sensitivity recursion length is
        limited to the window_length (-> smaller windows will yield smaller FIM).
        """
        return self.replay_log(
            params,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        ).pose.states

    def measurement_vector(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
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

    def decision_variables_from_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        return decision_variables_from_control_points(self.problem, control_points)

    def set_bezier_control_points(self, control_points: jnp.ndarray):
        control_points = self.clamp_control_points(control_points)
        self.control_points = control_points
        self.reference_states = self.reference_states_from_control_points(control_points)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))

    def fim_loss_from_control_points(
        self,
        control_points: jnp.ndarray,
        window_length: int | None = None,
        measurement_variances=None,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
        tangent_floor_weight: float = 1.0,
    ) -> jnp.ndarray:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.reference_states_from_control_points(control_points)
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
        ) + tangent_floor_weight * tangent_floor_loss(self.problem, control_points)

    def objective_terms_from_control_points(
        self,
        control_points: jnp.ndarray,
        window_length: int | None = None,
        measurement_variances=None,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
        tangent_floor_weight: float = 1.0,
    ) -> dict[str, jnp.ndarray]:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.reference_states_from_control_points(control_points)
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
        tangent_term = tangent_floor_weight * tangent_floor_loss(self.problem, control_points)
        total = fim_term + constraint_term + tangent_term
        return {
            "fim": fim_term,
            "constraints": constraint_term,
            "tangent_floor": tangent_term,
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
        reference_states = self.reference_states_from_control_points(control_points)
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
        tangent_floor_weight: float = 1.0,
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
            tangent_floor_weight=tangent_floor_weight,
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
