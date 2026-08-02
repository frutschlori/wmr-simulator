from datetime import datetime
import os
import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.controller import Controller
from wmr_simulator.estimator import DiffDriveEstimator
from wmr_simulator.robot import DiffDrive
from wmr_simulator.simulation import (
    SimulationPipeline,
    apply_noise_configuration,
    make_replay_segment_plan,
    replay_simulation_log,
)
from wmr_simulator.trajectory_optimization.constraints import (
    constraint_loss_components_from_reference_states,
    constraint_loss_from_reference_states,
    constraint_weights,
    motion_limits_from_robot_config,
)
from wmr_simulator.trajectory_optimization.fim import (
    compute_fim_factor,
    default_measurement_variances,
    fim_from_factor,
)
from wmr_simulator.trajectory_optimization.objectives import fim_loss, trajectory_objective
from wmr_simulator.trajectory_optimization.optimizers import optimize_control_points
from wmr_simulator.trajectory_optimization.bspline import (
    BSplinePlan,
    clamp_control_points,
    compute_bspline_reference,
    initial_line_control_points,
)
from wmr_simulator.trajectory_optimization.parametrization import normalize_time_scaling
from wmr_simulator.types import PhysicalParams, SimulationLog
from wmr_simulator.visualization.trajectories import (
    plot_loss_history as plot_loss_history_figure,
    plot_trajectory as plot_trajectory_figure,
    plot_trajectory_set as plot_trajectory_set_figure,
)


class ProblemDefinition:
    def __init__(self, problem_path: str):
        with open(problem_path, "r", encoding="utf-8") as file:
            # noise_enabled: false zeroes the measurement noise, so the FIM
            # variances and any rollout built from this config are noise-free too.
            self.raw = apply_noise_configuration(yaml.safe_load(file))

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


OBJECTIVE_MODE_IDENTIFICATION = "identification"
OBJECTIVE_MODE_GAIN_TUNING = "gain-tuning"
# Control points for the pipeline's placeholder curve, before a caller sets its
# own. The count is the parametrization's stiffness knob (see bspline.py); the
# scripts pass their own.
DEFAULT_NUM_CONTROL_POINTS = 8
# Lower bound on the relative FIM scale of a controller gain, matching the
# default k_min_stab of the gain search (gain_tuning.defaults).
GAIN_FIM_SCALING_FLOOR = 1e-3
OBJECTIVE_MODES = {OBJECTIVE_MODE_IDENTIFICATION, OBJECTIVE_MODE_GAIN_TUNING}


def normalize_objective_mode(objective_mode: str) -> str:
    objective_mode = objective_mode.strip().lower().replace("_", "-")
    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"Unsupported trajectory optimization objective mode '{objective_mode}'. "
            f"Expected one of {sorted(OBJECTIVE_MODES)}."
        )
    return objective_mode


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
        objective_mode: str = OBJECTIVE_MODE_IDENTIFICATION,
        fim_a_slip_max: bool = True,
    ):
        self.problem = ProblemDefinition(problem_path)
        self.simulation = SimulationPipeline(problem_path=problem_path, seed=0, reference_trajectories_dir=None)
        self.robot = self.simulation.robot
        self.controller = self.simulation.controller
        self.controller_gains = self.simulation.gains
        self.estimator = self.simulation.estimator
        # The encoder low-pass makes the simulated motor loop near-oscillatory,
        # which ill-conditions the FIM and roughens the descent; trajectory
        # optimization runs without it (simulation, identification, and gain
        # tuning keep the filter).
        self.estimator.wheel_lp_tau = 0.0
        self.time_scaling = normalize_time_scaling(time_scaling)
        self.objective_mode = normalize_objective_mode(objective_mode)
        # A BSplinePlan is just the basis sampled on this pipeline's time grid,
        # constant given (num_control_points, time_scaling); cache so repeated
        # eager calls (plotting, tests) don't resample it.
        self._spline_plan_cache: dict[tuple[int, str], BSplinePlan] = {}
        # a_slip_max sometimes has near-zero sensitivity, which makes the FIM
        # objective stiff; excluding it keeps the burnout model in the rollout
        # at its nominal value but drops it from the design parameters.
        self.fim_a_slip_max = bool(fim_a_slip_max) and self.robot.a_slip_max > 0.0
        self.control_points = self.initial_control_points(DEFAULT_NUM_CONTROL_POINTS)
        self.reference_states = self.reference_states_from_control_points(self.control_points)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))
        self.loss_history = None
        self.batch_loss_history = None
        self.batch_final_losses = None
        self.batch_constraint_weights = None
        self.optimization_snapshots = None

    def nominal_parameters(self) -> jnp.ndarray:
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            return jnp.asarray(self.controller_gains, dtype=jnp.float32)
        # Identification mode: deterministic replay-identifiable parameters
        # [r, L_effective] plus a_slip_max when enabled in the problem yaml
        # and not excluded via fim_a_slip_max (a disabled component has zero
        # sensitivity and would add a dead FIM column).
        values = [self.robot.r, self.robot.L]
        if self.fim_a_slip_max:
            values.append(self.robot.a_slip_max)
        return jnp.array(values, dtype=jnp.float32)

    def fim_parameter_scaling(self, params: jnp.ndarray) -> jnp.ndarray:
        # Relative scaling throughout: the FIM then measures information about a
        # *fractional* change in each parameter, so the design criterion is
        # scale-invariant and does not over-serve whichever parameter happens to
        # carry large raw sensitivity. In gain-tuning mode this also matches the
        # space the tuner searches in -- gains move in log space
        # (gain_tuning.optimizers), i.e. by relative steps.
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            # A gain may legitimately be 0 (kimotor = integral action off), and a
            # zero scale would blank that column and leave the FIM singular.
            # Floor it at the smallest gain the tuner can represent.
            return jnp.maximum(params, GAIN_FIM_SCALING_FLOOR)
        # Identification params (r, L, a_slip_max) are strictly positive.
        return params

    def nominal_physical_params(self) -> PhysicalParams:
        return PhysicalParams(
            wheel_radius=jnp.asarray(self.robot.r, dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot.L, dtype=jnp.float32),
            max_wheel_speed=jnp.asarray(self.robot.max_wheel_speed, dtype=jnp.float32),
            time_constant=jnp.asarray(self.robot.tau, dtype=jnp.float32),
            a_slip_max=jnp.asarray(self.robot.a_slip_max, dtype=jnp.float32),
        )

    def default_measurement_variances(self) -> np.ndarray:
        return default_measurement_variances(self.problem.estimator_cfg)

    def motion_limits(self) -> dict[str, jnp.ndarray]:
        return motion_limits_from_robot_config(self.problem.robot_cfg)

    def constraint_weights(self, scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
        return constraint_weights(scale=scale, component_weights=component_weights)

    def pinned_positions(self, num_control_points: int) -> dict[int, np.ndarray]:
        """Control points whose position is boundary data rather than a decision
        variable, mapped to that position.

        An identification trajectory must run from the pose the robot is
        physically placed in to the specified goal. The spline is clamped, so
        the curve begins at the first control point and ends at the last, and
        pinning those two is the whole of it. A gain-tuning trajectory has no
        meaningful start or goal -- it only has to excite the gains somewhere
        inside the environment box -- so every control point is free.
        """
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            return {}
        return {0: self.problem.start[:2], num_control_points - 1: self.problem.goal[:2]}

    def pin_start_heading(self) -> bool:
        """Whether the second control point is held on the start-heading ray.

        Only meaningful when the start itself is pinned: the heading there is
        how the robot is physically placed for a run, not a design choice.
        """
        return self.objective_mode != OBJECTIVE_MODE_GAIN_TUNING

    def _spline_plan(self, num_control_points: int, time_scaling: str) -> BSplinePlan:
        key = (num_control_points, time_scaling)
        plan = self._spline_plan_cache.get(key)
        if plan is None:
            plan = BSplinePlan(self.problem, num_control_points, time_scaling)
            self._spline_plan_cache[key] = plan
        return plan

    def reference_states_from_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        plan = self._spline_plan(control_points.shape[0], self.time_scaling)
        return compute_bspline_reference(
            self.problem, plan, control_points, time_scaling=self.time_scaling
        )

    def current_control_points(self):
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

    def physical_params_from_vector(self, params: jnp.ndarray) -> PhysicalParams:
        params = jnp.asarray(params, dtype=jnp.float32)
        # Layout mirrors nominal_parameters(): [r, L] + optional a_slip_max.
        # When excluded from the FIM, the burnout model keeps its nominal value.
        a_slip_max = (
            params[2] if self.fim_a_slip_max else jnp.asarray(self.robot.a_slip_max, dtype=jnp.float32)
        )
        return PhysicalParams(
            wheel_radius=params[0],
            base_diameter=params[1],
            max_wheel_speed=jnp.asarray(1.0, dtype=jnp.float32),
            time_constant=jnp.asarray(0.0, dtype=jnp.float32),
            a_slip_max=a_slip_max,
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
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            del window_length
            predicted_log = self.simulation.run_closed_loop(
                self.nominal_physical_params(),
                controller_gains=params,
                wheel_speed_log_source="estimated",
                reference_states=target_log.reference.states,
            )
            return predicted_log.pose.true_states, predicted_log.pose.states
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

    def closed_loop_gain_measurement_sequence(
        self,
        gains: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
        reference_states: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Runs the normal closed-loop deployment and records estimated poses as measurements.
        window_length is accepted for API symmetry with replay mode; closed-loop gain sensitivities
        intentionally use the full rollout without replay resets.

        ``reference_states`` may be passed directly to avoid an otherwise-redundant
        deployment rollout (the sensitivity rollout re-runs the closed loop anyway).

        NB: the rollout must stay noisy. The measurement/encoder noise supplies the
        high-frequency excitation that makes the motor feedback gains observable; a
        deterministic rollout drives the FIM near-singular in the ``kimotor`` direction.
        """
        del window_length
        if reference_states is None:
            target_log = self.closed_loop_log if closed_loop_log is None else closed_loop_log
            reference_states = target_log.reference.states
        predicted_log = self.simulation.run_closed_loop(
            self.nominal_physical_params(),
            controller_gains=gains,
            wheel_speed_log_source="estimated",
            reference_states=reference_states,
        )
        return predicted_log.pose.states[1:]

    def measurement_vector(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
        reference_states: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """ Flattens Nx3 measurement matrix into 3Nx1 vector """
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            measurements = self.closed_loop_gain_measurement_sequence(
                params,
                window_length,
                closed_loop_log=closed_loop_log,
                reference_states=reference_states,
            )
        else:
            measurements = self.replay_measurement_sequence(
                params,
                window_length,
                closed_loop_log=closed_loop_log,
            )
        return measurements.reshape(-1)

    def compute_fim_factor(self, measurement_variances=None,
        window_length=None, closed_loop_log=None, reference_states=None) -> jnp.ndarray:
        """Weighted relative parameter sensitivities ``J~``, with ``FIM = J~^T J~``.

        This is what the design criteria consume; assembling the FIM squares its
        condition number, which float32 cannot survive here (see ``fim.py``).
        """
        if measurement_variances is None:
            measurement_variances = self.default_measurement_variances()
        params = self.nominal_parameters()

        return compute_fim_factor(
            lambda p: self.measurement_vector(
                p,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
                reference_states=reference_states,
            ),
            params=params,
            measurement_variances=measurement_variances,
            parameter_scaling=self.fim_parameter_scaling(params),
        )

    def compute_fim_matrix(self, measurement_variances=None,
        window_length=None, closed_loop_log=None, reference_states=None) -> jnp.ndarray:
        """The assembled Fisher matrix, for reporting/inspection only."""
        return fim_from_factor(
            self.compute_fim_factor(
                measurement_variances=measurement_variances,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
                reference_states=reference_states,
            )
        )

    def initial_control_points(self, num_control_points: int) -> jnp.ndarray:
        return initial_line_control_points(self.problem, num_control_points=num_control_points)

    def clamp_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        return clamp_control_points(
            self.problem,
            control_points,
            pinned_positions=self.pinned_positions(control_points.shape[0]),
            pin_start_heading=self.pin_start_heading(),
        )

    def control_points_from_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        return self.clamp_control_points(jnp.reshape(jnp.ravel(decision_variables), (-1, 2)))

    def decision_variables_from_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(self.clamp_control_points(control_points))

    def set_control_points(self, control_points: jnp.ndarray):
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
    ) -> jnp.ndarray:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.reference_states_from_control_points(control_points)
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            # Gain-tuning sensitivities re-run the closed loop from these reference
            # states, so the separate deployment rollout is redundant: pass the
            # reference states straight through instead.
            fim_factor = self.compute_fim_factor(
                measurement_variances=measurement_variances,
                window_length=window_length,
                reference_states=reference_states,
            )
        else:
            closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
            fim_factor = self.compute_fim_factor(
                measurement_variances=measurement_variances,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
            )
        return trajectory_objective(
            fim_factor=fim_factor,
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
        reference_states = self.reference_states_from_control_points(control_points)
        closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
        fim_factor = self.compute_fim_factor(
            measurement_variances=measurement_variances,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )
        fim_term = fim_loss(fim_factor)
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

    def optimize_trajectory(
        self,
        num_control_points: int,
        num_steps: int,
        learning_rate: float,
        window_length: int | None = None,
        measurement_variances=None,
        save_trace: bool = False,
        trace_stride: int = 5,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
        verbose: bool = True,
    ):
        return optimize_control_points(
            pipeline=self,
            num_control_points=num_control_points,
            num_steps=num_steps,
            learning_rate=learning_rate,
            window_length=window_length,
            measurement_variances=measurement_variances,
            save_trace=save_trace,
            trace_stride=trace_stride,
            constraint_weight=constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            verbose=verbose,
        )

    def _random_gain_tuning_control_points(
        self,
        rng: np.random.Generator,
        num_control_points: int,
    ) -> jnp.ndarray:
        env_min = np.asarray(self.problem.environment_min, dtype=float)
        env_max = np.asarray(self.problem.environment_max, dtype=float)
        if not np.all(np.isfinite(env_min)) or not np.all(np.isfinite(env_max)):
            raise ValueError("Random gain-tuning trajectories require finite environment bounds.")
        start = rng.uniform(env_min, env_max)
        goal = rng.uniform(env_min, env_max)
        line_samples = np.linspace(0.0, 1.0, num_control_points)[:, None]
        positions = start[None, :] + line_samples * (goal[None, :] - start[None, :])
        return jnp.asarray(positions, dtype=jnp.float32)

    def initial_control_point_candidates(
        self,
        num_control_points: int,
        num_trajectories: int,
        seed: int = 0,
    ) -> list[jnp.ndarray]:
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive.")
        if num_control_points < 2:
            raise ValueError("num_control_points must be >= 2.")
        rng = np.random.default_rng(seed)
        candidates = []
        for _ in range(num_trajectories):
            if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
                control_points = self._random_gain_tuning_control_points(rng, num_control_points)
            else:
                control_points = self.initial_control_points(num_control_points)
            candidates.append(self.clamp_control_points(control_points))
        return candidates

    def initial_control_point_batch(self, num_control_points: int, num_trajectories: int) -> jnp.ndarray:
        candidates = self.initial_control_point_candidates(num_control_points=num_control_points, num_trajectories=num_trajectories)
        return jnp.stack(candidates, axis=0)

    def optimize_trajectories(
        self,
        num_control_points: int,
        num_steps: int,
        learning_rate: float,
        num_trajectories: int,
        vectorized: bool = False,
        constraint_weight_jitter: float = 0.0,
        seed: int = 0,
        window_length: int | None = None,
        measurement_variances=None,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
        verbose: bool = True,
    ):
        initial_control_point_candidates = self.initial_control_point_candidates(
            num_control_points=num_control_points,
            num_trajectories=num_trajectories,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 1)
        if constraint_weight_jitter < 0.0:
            raise ValueError("constraint_weight_jitter must be non-negative.")
        low = max(0.0, 1.0 - float(constraint_weight_jitter))
        high = 1.0 + float(constraint_weight_jitter)
        constraint_weight_factors = rng.uniform(low, high, size=num_trajectories)
        constraint_weights_per_trajectory = constraint_weight * constraint_weight_factors

        if vectorized:
            from wmr_simulator.trajectory_optimization.optimizers import optimize_control_points_batch

            initial_control_points = jnp.stack(initial_control_point_candidates, axis=0)
            group_constraint_weights = jnp.asarray(constraint_weights_per_trajectory, dtype=jnp.float32)
            optimized_control_points, loss_history = optimize_control_points_batch(
                pipeline=self,
                initial_control_points=initial_control_points,
                num_steps=num_steps,
                learning_rate=learning_rate,
                window_length=window_length,
                measurement_variances=measurement_variances,
                constraint_weight=group_constraint_weights,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=constraint_smooth_max_beta,
                verbose=verbose,
            )
            loss_history = np.asarray(loss_history, dtype=float) if loss_history else np.empty((0, num_trajectories))
            final_losses = []
            for trajectory_index in range(num_trajectories):
                optimized_control_points_one = optimized_control_points[trajectory_index]
                candidate_constraint_weight = float(constraint_weights_per_trajectory[trajectory_index])
                final_loss = self.fim_loss_from_control_points(
                    optimized_control_points_one,
                    window_length=window_length,
                    measurement_variances=measurement_variances,
                    constraint_weight=candidate_constraint_weight,
                    constraint_component_weights=constraint_component_weights,
                    constraint_smooth_max_beta=constraint_smooth_max_beta,
                )
                if not np.isfinite(float(final_loss)):
                    optimized_control_points = optimized_control_points.at[trajectory_index].set(
                        initial_control_point_candidates[trajectory_index]
                    )
                    final_loss = self.fim_loss_from_control_points(
                        optimized_control_points[trajectory_index],
                        window_length=window_length,
                        measurement_variances=measurement_variances,
                        constraint_weight=candidate_constraint_weight,
                        constraint_component_weights=constraint_component_weights,
                        constraint_smooth_max_beta=constraint_smooth_max_beta,
                    )
                # Raw (unnormalized) final loss, comparable across trajectories for
                # best-candidate selection; the per-trajectory histories stay
                # normalized and untouched.
                final_losses.append(float(final_loss))
            loss_history = loss_history.tolist()
        else:
            optimized = []
            histories = []
            final_losses = []
            for control_points, candidate_constraint_weight in zip(
                initial_control_point_candidates,
                constraint_weights_per_trajectory,
            ):
                optimized_control_points_one, loss_history_one = optimize_control_points(
                    pipeline=self,
                    num_control_points=control_points.shape[0],
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    initial_control_points=control_points,
                    window_length=window_length,
                    measurement_variances=measurement_variances,
                    save_trace=False,
                    constraint_weight=float(candidate_constraint_weight),
                    constraint_component_weights=constraint_component_weights,
                    constraint_smooth_max_beta=constraint_smooth_max_beta,
                    verbose=verbose,
                )
                final_loss = self.fim_loss_from_control_points(
                    optimized_control_points_one,
                    window_length=window_length,
                    measurement_variances=measurement_variances,
                    constraint_weight=float(candidate_constraint_weight),
                    constraint_component_weights=constraint_component_weights,
                    constraint_smooth_max_beta=constraint_smooth_max_beta,
                )
                if not np.isfinite(float(final_loss)):
                    optimized_control_points_one = control_points
                    final_loss = self.fim_loss_from_control_points(
                        optimized_control_points_one,
                        window_length=window_length,
                        measurement_variances=measurement_variances,
                        constraint_weight=float(candidate_constraint_weight),
                        constraint_component_weights=constraint_component_weights,
                        constraint_smooth_max_beta=constraint_smooth_max_beta,
                    )
                optimized.append(optimized_control_points_one)
                histories.append(loss_history_one)
                final_losses.append(float(final_loss))
            shapes = {tuple(candidate.shape) for candidate in optimized}
            optimized_control_points = jnp.stack(optimized, axis=0) if len(shapes) == 1 else optimized
            loss_history = np.asarray(histories, dtype=float).T.tolist() if histories and histories[0] else []

        self.batch_loss_history = loss_history
        final_losses = np.asarray(final_losses, dtype=float)
        self.batch_final_losses = final_losses.tolist()
        self.batch_constraint_weights = constraint_weights_per_trajectory.tolist()
        finite_losses = np.where(np.isfinite(final_losses), final_losses, np.inf)
        best_index = int(np.argmin(finite_losses))
        self.set_control_points(optimized_control_points[best_index])
        self.loss_history = np.asarray(loss_history)[:, best_index].tolist() if len(loss_history) else []
        return optimized_control_points, loss_history

    def plot_trajectory(self, window_length=None, out_prefix="trajectory_plot", out_path=None):
        plot_trajectory_figure(
            self,
            window_length=window_length,
            out_prefix=out_prefix,
            out_path=out_path,
        )

    def plot_trajectory_batch(
        self,
        control_point_batch,
        out_prefix="trajectory_set",
        out_path=None,
        title="Optimized Trajectories",
    ):
        """One figure for a whole batch of optimized trajectories. The rollouts
        run as a single vmapped, jitted batch instead of one eager (and
        separately compiled) rollout per trajectory."""
        control_point_batch = jnp.stack(
            [self.clamp_control_points(control_points) for control_points in control_point_batch]
        )
        reference_trajectories = jax.vmap(self.reference_states_from_control_points)(control_point_batch)
        closed_loop_logs = self.simulation.run_closed_loop_batch(
            self.nominal_physical_params(),
            reference_trajectories,
            controller_gains=self.controller_gains,
            wheel_speed_log_source="estimated",
        )
        return plot_trajectory_set_figure(
            reference_trajectories,
            closed_loop_logs.pose.true_states,
            control_point_batch=control_point_batch,
            out_prefix=out_prefix,
            out_path=out_path,
            title=title,
        )

    def plot_loss_history(self, out_prefix="loss_history", out_path=None):
        if self.loss_history is None:
            raise ValueError("No optimization loss history available. Run optimize_trajectory() first.")
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

    def save_reference_states_pickle(
        self,
        out_dir="trajectory_exports",
        filename_prefix="reference_states",
        reference_states=None,
    ):
        """Export a reference trajectory. Pass ``reference_states`` to export a
        trajectory other than the pipeline's current one -- exporting a batch
        via ``set_control_points`` would run (and compile) a closed-loop rollout
        per trajectory that the pickle does not use."""
        os.makedirs(out_dir, exist_ok=True)
        reference_states = self.reference_states if reference_states is None else reference_states
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as file:
            pickle.dump(reference_states_export_payload(reference_states, self.problem.dt), file)
        return out_path
