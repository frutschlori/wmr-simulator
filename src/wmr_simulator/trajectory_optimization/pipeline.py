from datetime import datetime
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.controller import KDMOTOR_INDEX, controller_gains_array
from wmr_simulator.simulation import (
    SimulationPipeline,
    apply_noise_configuration,
    make_replay_segment_plan,
    replay_simulation_log,
)
from wmr_simulator.trajectory_optimization.constraints import (
    constraint_weights,
    motion_floors_for_batch,
    motion_limits_from_robot_config,
)
from wmr_simulator.trajectory_optimization.fim import (
    compute_fim_factor,
    default_measurement_variances,
)
from wmr_simulator.trajectory_optimization.start_offsets import (
    START_OFFSET_MODE_NONE,
    START_OFFSET_MODE_RANDOM,
    START_OFFSET_MODE_STATIC,
    inverse_squash_start_offsets,
    normalize_start_offset_mode,
    resolve_start_offsets,
    sample_initial_pose_offset_batch,
    start_offset_mask,
    static_start_offsets,
)
from wmr_simulator.trajectory_optimization.objectives import (
    DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
    DEFAULT_CRITERION,
    normalize_criterion,
    trajectory_objective,
)
from wmr_simulator.trajectory_optimization.optimizers import optimize_control_points
from wmr_simulator.trajectory_optimization.bspline import (
    DEFAULT_MIN_TANGENT_FRACTION,
    BSplinePlan,
    clamp_control_points,
    compute_bspline_reference,
    initial_line_control_points,
    stretch_control_points_to_arc,
)
from wmr_simulator.trajectory_optimization.parametrization import normalize_time_scaling, phase_indices
from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.objectives import Realizations, make_realizations
from wmr_simulator.types import PhysicalParams, SimulationLog
from wmr_simulator.visualization.trajectories import (
    plot_loss_history as plot_loss_history_figure,
    plot_trajectory as plot_trajectory_figure,
)


class ProblemDefinition:
    def __init__(self, problem_path: str):
        with open(problem_path, "r", encoding="utf-8") as file:
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


OBJECTIVE_MODE_IDENTIFICATION = "identification"
OBJECTIVE_MODE_GAIN_TUNING = "gain-tuning"
# Control points of the placeholder curve before a caller sets its own.
DEFAULT_NUM_CONTROL_POINTS = 4
# Smallest distance between the random start and goal of a gain-tuning design's
# initial line [m].
DEFAULT_MIN_START_GOAL_DISTANCE = 1.0
# The gain-tuning FIM averages over the same realizations the gain tuner uses.
DEFAULT_NUM_REALIZATIONS = int(GAIN_TUNING_DEFAULTS["num_realizations"])
# Lower bound on the relative FIM scale of the log-space gains.
GAIN_FIM_SCALING_FLOOR = float(GAIN_TUNING_DEFAULTS["k_min_stab"])
KIMOTOR_INDEX = 4
# FIM scales of kimotor / kdmotor when the design point has them at 0 and no
# scale is given.
KIMOTOR_FIM_SCALE_FALLBACK = float(GAIN_TUNING_DEFAULTS["k_max_rest"])
KDMOTOR_FIM_SCALE_FALLBACK = 0.01
OBJECTIVE_MODES = {OBJECTIVE_MODE_IDENTIFICATION, OBJECTIVE_MODE_GAIN_TUNING}


def normalize_objective_mode(objective_mode: str) -> str:
    objective_mode = objective_mode.strip().lower().replace("_", "-")
    if objective_mode not in OBJECTIVE_MODES:
        raise ValueError(
            f"Unsupported trajectory optimization objective mode '{objective_mode}'. "
            f"Expected one of {sorted(OBJECTIVE_MODES)}."
        )
    return objective_mode


def reference_states_export_payload(
    reference_states, dt: float, start_offsets=None, control_points=None, **metadata
):
    """The pickle a designed trajectory is saved as.

    ``start_offsets`` (R, 3) are the start-pose offsets the trajectory was
    designed under; the gain tuner rolls out on exactly these starts.
    ``control_points`` (K, 2) are the curve's decision variables.
    """
    payload = {
        "reference_states": np.asarray(reference_states),
        "dt": float(dt),
    }
    if start_offsets is not None:
        payload["start_offsets"] = np.asarray(start_offsets, dtype=float)
    if control_points is not None:
        payload["control_points"] = np.asarray(control_points, dtype=float)
    payload.update(metadata)
    return payload


class TrajectoryOptimizationPipeline:
    def __init__(
        self,
        problem_path: str,
        time_scaling: str | None = None,
        objective_mode: str = OBJECTIVE_MODE_IDENTIFICATION,
        num_realizations: int = DEFAULT_NUM_REALIZATIONS,
        realizations: Realizations | None = None,
        criterion: str = DEFAULT_CRITERION,
        start_offset_mode: str = START_OFFSET_MODE_RANDOM,
        offset_displacement_step_factor: float = 1.0,
        offset_heading_step_factor: float = 1.0,
        min_tangent_fraction: float = DEFAULT_MIN_TANGENT_FRACTION,
        kimotor_fim_scale: float | None = None,
        kdmotor_fim_scale: float | None = None,
        motion_limits: dict | None = None,
        motion_phases: list[dict] | None = None,
        controller_gains=None,
        residual_model=None,
        min_start_goal_distance: float = DEFAULT_MIN_START_GOAL_DISTANCE,
    ):
        self.problem = ProblemDefinition(problem_path)
        self.min_start_goal_distance = float(min_start_goal_distance)
        # Motion limits of the constraint term (v_max, a_max, a_max_lateral,
        # omega_max, alpha_max), overriding the problem yaml's; the plant is unchanged.
        self.motion_limit_overrides = (
            {} if motion_limits is None else {key: float(value) for key, value in motion_limits.items()}
        )
        # Consecutive motion phases, each with a ``duration`` [s], its own
        # ``motion_limits``, optional ``min_speed`` / ``min_lateral_acceleration``
        # floors and ``identify`` (default true: scored by the identification
        # FIM). The reference rests at every phase boundary.
        self._set_motion_phases(motion_phases)
        # Floor on |dpos/ds| as a fraction of the curve's rms tangent (<= 0: off).
        self.min_tangent_fraction = float(min_tangent_fraction)
        # Optional residual dynamics the design is scored on (None: nominal plant).
        self.residual_model = residual_model
        self.simulation = SimulationPipeline(
            problem_path=problem_path,
            seed=0,
            reference_trajectories_dir=None,
            residual_model=residual_model,
        )
        self.robot = self.simulation.robot
        self.controller = self.simulation.controller
        # The design point: the static gains every rollout is driven at (None:
        # the problem yaml's gains).
        self.controller_gains = (
            self.simulation.gains
            if controller_gains is None
            else controller_gains_array(controller_gains)
        )
        if controller_gains is None and self._problem_has_a_trained_parametrization():
            print(
                "WARNING: the problem's gain parametrization is trained, so "
                f"controller.gains {np.asarray(self.controller_gains).tolist()} are its base "
                "gains, not a controller anybody runs. This designer drives the static "
                "controller; pass controller_gains= the static-tuned gains "
                "(robot_config_static_gains.yaml in an active-learning iteration)."
            )
        # Constant FIM scales of kimotor and kdmotor (see fim_parameter_scaling):
        # the given value, else the design point's value, else the fallback.
        nominal_kimotor = float(self.controller_gains[KIMOTOR_INDEX])
        if kimotor_fim_scale is None:
            self.kimotor_fim_scale = (
                nominal_kimotor if nominal_kimotor > 0.0 else KIMOTOR_FIM_SCALE_FALLBACK
            )
        else:
            self.kimotor_fim_scale = float(kimotor_fim_scale)
            if self.kimotor_fim_scale <= 0.0:
                raise ValueError("kimotor_fim_scale must be positive.")
        nominal_kdmotor = float(self.controller_gains[KDMOTOR_INDEX])
        if kdmotor_fim_scale is None:
            self.kdmotor_fim_scale = (
                nominal_kdmotor if nominal_kdmotor > 0.0 else KDMOTOR_FIM_SCALE_FALLBACK
            )
        else:
            self.kdmotor_fim_scale = float(kdmotor_fim_scale)
            if self.kdmotor_fim_scale <= 0.0:
                raise ValueError("kdmotor_fim_scale must be positive.")
        self.estimator = self.simulation.estimator
        # Time constant of the firmware's wheel-speed low-pass.
        self.wheel_lp_tau = float(self.estimator.wheel_lp_tau)
        self.time_scaling = normalize_time_scaling(time_scaling)
        self.objective_mode = normalize_objective_mode(objective_mode)
        # Optimality criterion applied to the FIM.
        self.criterion = normalize_criterion(criterion)
        # Cached B-spline bases per (num_control_points, time_scaling, phases).
        self._spline_plan_cache: dict[tuple[int, str, tuple[float, ...]], BSplinePlan] = {}
        # Noise keys and start offsets, drawn once so the objective is a
        # deterministic function of the decision variables.
        self.offset_radius = float(GAIN_TUNING_DEFAULTS["init_offset_radius"])
        self.offset_angle = float(GAIN_TUNING_DEFAULTS["init_offset_angle"])
        self.realizations = (
            make_realizations(
                self.simulation.robot_key,
                self.simulation.estimator_key,
                int(num_realizations),
                self.offset_radius,
                self.offset_angle,
            )
            if realizations is None
            else realizations
        )
        # Start-offset components that are decision variables (appended after
        # the control points).
        self.start_offset_mode = normalize_start_offset_mode(start_offset_mode)
        self.start_offset_mask = start_offset_mask(self.start_offset_mode)
        if self.start_offset_mode == START_OFFSET_MODE_STATIC:
            self.realizations = self.realizations._replace(
                start_offsets=static_start_offsets(
                    int(self.realizations.robot_keys.shape[0]), self.offset_radius, self.offset_angle
                )
            )
        elif self.start_offset_mode == START_OFFSET_MODE_NONE:
            self.realizations = self.realizations._replace(
                start_offsets=jnp.zeros_like(self.realizations.start_offsets)
            )
        self.optimize_start_offsets = bool(np.any(np.asarray(self.start_offset_mask)))
        if self.optimize_start_offsets and self.objective_mode != OBJECTIVE_MODE_GAIN_TUNING:
            raise ValueError(
                f"start_offset_mode '{self.start_offset_mode}' optimizes the rollout start poses, "
                "which only exist as design variables in gain-tuning mode; identification starts "
                "where the robot is physically placed."
            )
        self.num_offset_variables = (
            3 * int(self.realizations.start_offsets.shape[0]) if self.optimize_start_offsets else 0
        )
        self.initial_free_offsets = inverse_squash_start_offsets(
            self.realizations.start_offsets, self.offset_radius, self.offset_angle
        )
        # Learning-rate factors of the offset variables relative to the control points.
        self.offset_displacement_step_factor = float(offset_displacement_step_factor)
        self.offset_heading_step_factor = float(offset_heading_step_factor)
        if min(self.offset_displacement_step_factor, self.offset_heading_step_factor) < 0.0:
            raise ValueError("Offset step factors must be non-negative.")
        self.control_points = self.initial_control_points(DEFAULT_NUM_CONTROL_POINTS)
        self.reference_states = self.reference_states_from_control_points(self.control_points)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))
        self.loss_history = None
        self.batch_loss_history = None
        self.batch_final_losses = None
        # Per-trajectory start offsets of the last batch optimization, (T, R, 3).
        self.batch_start_offsets = None
        self.batch_constraint_weights = None

    def _problem_has_a_trained_parametrization(self) -> bool:
        """Whether the problem's gain parametrization is anything but the identity (theta != 0)."""
        if not self.simulation.gain_parametrization_enabled:
            return False
        from wmr_simulator.gain_parametrization import flat_params

        return bool(np.any(np.asarray(flat_params(self.simulation.gain_parametrization_params)) != 0.0))

    def nominal_parameters(self) -> jnp.ndarray:
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            return jnp.asarray(self.controller_gains, dtype=jnp.float32)
        # Identification mode: [r, L].
        return jnp.array([self.robot.r, self.robot.L], dtype=jnp.float32)

    def fim_parameter_scaling(self, params: jnp.ndarray) -> jnp.ndarray:
        # Relative scaling: the FIM measures information about fractional
        # parameter changes. kimotor and kdmotor may be tuned to exactly 0, so
        # they get constant scales instead.
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            return jnp.asarray(
                jnp.maximum(params, GAIN_FIM_SCALING_FLOOR)
                .at[KIMOTOR_INDEX]
                .set(self.kimotor_fim_scale)
                .at[KDMOTOR_INDEX]
                .set(self.kdmotor_fim_scale),
                dtype=jnp.float32,
            )
        return params

    def nominal_physical_params(self) -> PhysicalParams:
        return PhysicalParams(
            wheel_radius=jnp.asarray(self.robot.r, dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot.L, dtype=jnp.float32),
            max_wheel_speed=jnp.asarray(self.robot.max_wheel_speed, dtype=jnp.float32),
            time_constant=jnp.asarray(self.robot.tau, dtype=jnp.float32),
        )

    def default_measurement_variances(self) -> np.ndarray:
        return default_measurement_variances(self.problem.estimator_cfg)

    def _set_motion_phases(self, motion_phases: list[dict] | None) -> None:
        """Validate the phases (durations add up to ``sim_time``) and derive their boundaries."""
        self.motion_phases = [] if motion_phases is None else [dict(phase) for phase in motion_phases]
        if not self.motion_phases:
            self.phase_breaks = ()
            self.identified_phases = ()
            self.identified_duration = None
            return
        durations = np.asarray([float(phase["duration"]) for phase in self.motion_phases], dtype=float)
        if np.any(durations <= 0.0):
            raise ValueError(f"Motion phase durations must be positive, got {durations.tolist()}.")
        if not np.isclose(durations.sum(), self.problem.sim_time, atol=0.5 * self.problem.dt):
            raise ValueError(
                f"Motion phase durations add up to {durations.sum():.3f} s but the problem's "
                f"sim_time is {self.problem.sim_time:.3f} s."
            )
        self.phase_breaks = tuple(float(value) for value in np.cumsum(durations)[:-1] / durations.sum())
        self.identified_phases = tuple(bool(phase.get("identify", True)) for phase in self.motion_phases)
        num_identified = sum(self.identified_phases)
        if num_identified == 0 or not all(self.identified_phases[:num_identified]):
            raise ValueError(
                f"Identified motion phases must be the leading ones, got identify flags {self.identified_phases}."
            )
        self.identified_duration = float(durations[:num_identified].sum())

    def identification_measurement_mask(self, time_s) -> jnp.ndarray:
        """1 for measurement times inside a phase the identification FIM scores, else 0."""
        time_s = jnp.asarray(time_s, dtype=jnp.float32)
        if not self.motion_phases:
            return jnp.ones_like(time_s)
        index = jnp.clip(phase_indices(time_s / self.problem.sim_time, self.phase_breaks), 0, len(self.motion_phases) - 1)
        return jnp.asarray(self.identified_phases, dtype=jnp.float32)[index]

    def motion_limits(self, motion_floor=None) -> dict[str, jnp.ndarray]:
        """The limits the constraint term holds the curve under.

        ``motion_floor`` ``[v_min, a_lat_min]`` sets the lower bounds of one
        trajectory. A motion-phased pipeline returns the upper limits per time
        sample and the floors per phase, with ``phase_masks`` (P, N).
        """
        robot_cfg = self.problem.robot_cfg
        if self.motion_limit_overrides:
            robot_cfg = {**robot_cfg, **self.motion_limit_overrides}
        if self.motion_phases:
            if motion_floor is not None and np.any(np.asarray(motion_floor, dtype=float) != 0.0):
                raise ValueError("A motion-phased design takes its floors from the phases, not motion_floor.")
            return self._phased_motion_limits(robot_cfg)
        limits = motion_limits_from_robot_config(robot_cfg)
        if motion_floor is not None:
            floor = jnp.asarray(motion_floor, dtype=jnp.float32)
            limits = {**limits, "v_min": floor[0], "a_lat_min": floor[1]}
        return limits

    def _phased_motion_limits(self, robot_cfg: dict) -> dict[str, jnp.ndarray]:
        phase_limits = [
            motion_limits_from_robot_config(
                {
                    **robot_cfg,
                    **{key: float(value) for key, value in phase.get("motion_limits", {}).items()},
                    "v_min": float(phase.get("min_speed", 0.0)),
                    "a_lat_min": float(phase.get("min_lateral_acceleration", 0.0)),
                }
            )
            for phase in self.motion_phases
        ]
        fraction = self.problem.sim_time_grid() / self.problem.sim_time
        index = phase_indices(fraction, self.phase_breaks)
        limits = {
            key: jnp.stack([phase[key] for phase in phase_limits])[index]
            for key in ("v_max", "a_max", "a_lat_max", "omega_max", "alpha_max")
        }
        for key in ("v_min", "a_lat_min"):
            limits[key] = jnp.stack([phase[key] for phase in phase_limits])
        limits["phase_masks"] = jnp.stack(
            [(index == phase).astype(jnp.float32) for phase in range(len(phase_limits))]
        )
        return limits

    def constraint_weights(self, scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
        return constraint_weights(scale=scale, component_weights=component_weights)

    def pinned_positions(self) -> dict[int, np.ndarray]:
        """Fixed control points mapped to their position: the start in
        identification mode (where the robot is placed), none in gain-tuning mode."""
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            return {}
        return {0: self.problem.start[:2]}

    def pin_start_heading(self) -> bool:
        """Whether the second control point is held on the start-heading ray."""
        return self.objective_mode != OBJECTIVE_MODE_GAIN_TUNING

    def _spline_plan(self, num_control_points: int, time_scaling: str) -> BSplinePlan:
        """Cached basis for this control-point count (build it outside jit)."""
        key = (num_control_points, time_scaling, self.phase_breaks)
        plan = self._spline_plan_cache.get(key)
        if plan is None:
            plan = BSplinePlan(self.problem, num_control_points, time_scaling, phase_breaks=self.phase_breaks)
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
        # [r, L], as in nominal_parameters().
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
        """Replay the logged commands open loop in windows of ``window_length``
        samples, each starting from the closed-loop estimate, and return the
        replayed poses."""
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
        realizations: Realizations | None = None,
    ) -> jnp.ndarray:
        """Run the closed loop once per realization (start offset + noise keys)
        and stack the estimated poses, so the FIM is the average over realizations.

        ``window_length`` is unused (full rollouts). The rollout must stay noisy:
        without noise the FIM is near-singular in the motor gains.
        """
        del window_length
        if reference_states is None:
            target_log = self.closed_loop_log if closed_loop_log is None else closed_loop_log
            reference_states = target_log.reference.states

        def rollout(robot_key, estimator_key, start_offset):
            predicted_log = self.simulation.run_closed_loop(
                self.nominal_physical_params(),
                controller_gains=gains,
                robot_key=robot_key,
                estimator_key=estimator_key,
                wheel_speed_log_source="estimated",
                reference_states=reference_states,
                initial_pose=reference_states[0, :3] + start_offset,
            )
            return predicted_log.pose.states[1:]

        # Passed realizations carry optimized start offsets.
        realizations = self.realizations if realizations is None else realizations
        measurements = jax.vmap(rollout)(
            realizations.robot_keys, realizations.estimator_keys, realizations.start_offsets
        )
        return measurements.reshape(-1, 3) / jnp.sqrt(float(realizations.start_offsets.shape[0]))

    def measurement_vector(
        self,
        params: jnp.ndarray,
        window_length: int | None = None,
        closed_loop_log: SimulationLog | None = None,
        reference_states: jnp.ndarray | None = None,
        realizations: Realizations | None = None,
    ) -> jnp.ndarray:
        """ Flattens Nx3 measurement matrix into 3Nx1 vector """
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            measurements = self.closed_loop_gain_measurement_sequence(
                params,
                window_length,
                closed_loop_log=closed_loop_log,
                reference_states=reference_states,
                realizations=realizations,
            )
        else:
            target_log = self.closed_loop_log if closed_loop_log is None else closed_loop_log
            poses = self.replay_measurement_sequence(
                params,
                window_length,
                closed_loop_log=closed_loop_log,
            )
            # Position along the measured heading only, as in the identification
            # loss; the lateral channel is kept as zeros for the column layout.
            heading = target_log.pose.states[1:, 2]
            longitudinal = jnp.cos(heading) * poses[:, 0] + jnp.sin(heading) * poses[:, 1]
            measurements = jnp.column_stack(
                [longitudinal, jnp.zeros_like(longitudinal), poses[:, 2]]
            ) * self.identification_measurement_mask(target_log.pose.time_s[1:])[:, None]
        return measurements.reshape(-1)

    def compute_fim_factor(self, measurement_variances=None, window_length=None,
        closed_loop_log=None, reference_states=None, gains=None,
        realizations: Realizations | None = None) -> jnp.ndarray:
        """Weighted relative parameter sensitivities ``J~``, with ``FIM = J~^T J~``.

        ``gains`` replaces the design point in gain-tuning mode.
        """
        if measurement_variances is None:
            measurement_variances = self.default_measurement_variances()
        params = self.nominal_parameters() if gains is None else jnp.asarray(gains, dtype=jnp.float32)

        return compute_fim_factor(
            lambda p: self.measurement_vector(
                p,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
                reference_states=reference_states,
                realizations=realizations,
            ),
            params=params,
            measurement_variances=measurement_variances,
            parameter_scaling=self.fim_parameter_scaling(params),
        )

    def initial_control_points(self, num_control_points: int) -> jnp.ndarray:
        return initial_line_control_points(self.problem, num_control_points=num_control_points)

    def clamp_control_points(self, control_points: jnp.ndarray) -> jnp.ndarray:
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        return clamp_control_points(
            self.problem,
            control_points,
            pinned_positions=self.pinned_positions(),
            pin_start_heading=self.pin_start_heading(),
        )

    # The decision vector is [ravel(control_points), ravel(free_offsets)]; the
    # offset tail is empty unless start_offset_mode optimizes them.
    def control_points_from_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        decision_variables = jnp.ravel(decision_variables)
        if self.num_offset_variables:
            decision_variables = decision_variables[: -self.num_offset_variables]
        return self.clamp_control_points(jnp.reshape(decision_variables, (-1, 2)))

    def free_offsets_from_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        if not self.num_offset_variables:
            return self.initial_free_offsets
        return jnp.reshape(jnp.ravel(decision_variables)[-self.num_offset_variables:], (-1, 3))

    def start_offsets_from_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        """The start offsets of a decision vector: the free variables squashed
        into the feasible set, the frozen draw for components not optimized."""
        return resolve_start_offsets(
            self.free_offsets_from_decision_variables(decision_variables),
            self.realizations.start_offsets,
            self.start_offset_mask,
            self.offset_radius,
            self.offset_angle,
        )

    def realizations_from_decision_variables(self, decision_variables: jnp.ndarray) -> Realizations:
        if not self.num_offset_variables:
            return self.realizations
        return self.realizations._replace(
            start_offsets=self.start_offsets_from_decision_variables(decision_variables)
        )

    def decision_variables_from_control_points(
        self, control_points: jnp.ndarray, free_offsets: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        control_part = jnp.ravel(self.clamp_control_points(control_points))
        if not self.num_offset_variables:
            return control_part
        free_offsets = self.initial_free_offsets if free_offsets is None else free_offsets
        return jnp.concatenate([control_part, jnp.ravel(free_offsets)])

    def clamp_decision_variables(self, decision_variables: jnp.ndarray) -> jnp.ndarray:
        """Project a decision vector onto the feasible set (environment box and
        pins); the offset tail is squashed on read instead of clipped."""
        return self.decision_variables_from_control_points(
            self.control_points_from_decision_variables(decision_variables),
            self.free_offsets_from_decision_variables(decision_variables),
        )

    def decision_variable_step_scale(self, num_decision_variables: int) -> jnp.ndarray | None:
        """Per-coordinate multiplier on the optimizer's update (only the offset
        tail is scaled), or None."""
        if not self.num_offset_variables:
            return None
        if self.offset_displacement_step_factor == 1.0 and self.offset_heading_step_factor == 1.0:
            return None
        num_realizations = int(self.realizations.start_offsets.shape[0])
        offset_scale = jnp.tile(
            jnp.asarray(
                [
                    self.offset_displacement_step_factor,
                    self.offset_displacement_step_factor,
                    self.offset_heading_step_factor,
                ],
                dtype=jnp.float32,
            ),
            num_realizations,
        )
        num_control_variables = int(num_decision_variables) - self.num_offset_variables
        return jnp.concatenate(
            [jnp.ones((num_control_variables,), dtype=jnp.float32), offset_scale]
        )

    def loss_from_decision_variables(self, decision_variables: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if self.num_offset_variables:
            kwargs.setdefault("realizations", self.realizations_from_decision_variables(decision_variables))
        return self.fim_loss_from_control_points(
            self.control_points_from_decision_variables(decision_variables), **kwargs
        )

    def set_start_offsets(self, start_offsets: jnp.ndarray) -> None:
        """Make ``start_offsets`` the pipeline's realization offsets."""
        start_offsets = jnp.asarray(start_offsets, dtype=jnp.float32)
        self.realizations = self.realizations._replace(start_offsets=start_offsets)
        self.initial_free_offsets = inverse_squash_start_offsets(
            start_offsets, self.offset_radius, self.offset_angle
        )

    def set_control_points(self, control_points: jnp.ndarray):
        control_points = self.clamp_control_points(control_points)
        self.control_points = control_points
        self.reference_states = self.reference_states_from_control_points(control_points)
        self._set_closed_loop_log(self.run_closed_loop_deployment(reference_states=self.reference_states))

    def stabilization_loss_from_control_points(
        self,
        control_points: jnp.ndarray,
        constraint_smooth_max_beta: float = 20.0,
        constraint_violation_tolerance: float = DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
    ) -> jnp.ndarray:
        """The tangent-floor penalty, scaled like the motion limits.

        A curve whose ``|dpos/ds|`` collapses has no heading there, which the
        motion limits do not see but the FIM rewards (an untrackable heading
        flip is very sensitive to the gains). Zero for every healthy curve.
        """
        plan = self._spline_plan(control_points.shape[0], self.time_scaling)
        return plan.stabilization_loss(
            control_points,
            min_tangent_fraction=self.min_tangent_fraction,
            smooth_max_beta=constraint_smooth_max_beta,
        ) / constraint_violation_tolerance

    def tangent_diagnostics(self, control_points: jnp.ndarray) -> dict:
        """Where a curve sits relative to the guard: ``min_tangent_norm``,
        ``rms_tangent_norm``, ``guard_threshold``, ``guarded_samples``."""
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        plan = self._spline_plan(control_points.shape[0], self.time_scaling)
        return plan.tangent_diagnostics(control_points)

    def assert_curve_is_exportable(self, control_points: jnp.ndarray, label: str = "") -> dict:
        """Refuse to export a curve whose heading came from the fallback
        tangent (``|dpos/ds|`` below the guard threshold somewhere)."""
        report = self.tangent_diagnostics(control_points)
        if not report["guarded_samples"]:
            return report
        where = f" for {label}" if label else ""
        raise ValueError(
            f"Refusing to export a stalled curve{where}: "
            f"{report['guarded_samples']} of {report['num_samples']} samples have "
            f"|dpos/ds| <= {report['guard_threshold']:.4g} (the fallback-tangent "
            f"threshold), min {report['min_tangent_norm']:.4g} against an rms of "
            f"{report['rms_tangent_norm']:.4g}. Those samples carry a constant "
            f"fallback heading, not the curve's, so the exported reference is not "
            f"a drivable trajectory. Raise --min-tangent-fraction (currently "
            f"{self.min_tangent_fraction:.4g}) and redesign."
        )

    def fim_loss_from_control_points(
        self,
        control_points: jnp.ndarray,
        window_length: int | None = None,
        measurement_variances=None,
        constraint_weight: float = 1.0,
        constraint_component_weights: dict | None = None,
        constraint_smooth_max_beta: float = 20.0,
        constraint_violation_tolerance: float = DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
        criterion: str | None = None,
        gains: jnp.ndarray | None = None,
        realizations: Realizations | None = None,
        motion_floor=(0.0, 0.0),
    ) -> jnp.ndarray:
        control_points = self.clamp_control_points(control_points)
        reference_states = self.reference_states_from_control_points(control_points)
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            fim_factor = self.compute_fim_factor(
                measurement_variances=measurement_variances,
                window_length=window_length,
                reference_states=reference_states,
                gains=gains,
                realizations=realizations,
            )
        else:
            closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
            fim_factor = self.compute_fim_factor(
                measurement_variances=measurement_variances,
                window_length=window_length,
                closed_loop_log=closed_loop_log,
                gains=gains,
            )
        return trajectory_objective(
            fim_factor=fim_factor,
            reference_states=reference_states,
            dt=self.problem.dt,
            limits=self.motion_limits(motion_floor=motion_floor),
            weights=self.constraint_weights(
                scale=constraint_weight,
                component_weights=constraint_component_weights,
            ),
            smooth_max_beta=constraint_smooth_max_beta,
            constraint_violation_tolerance=constraint_violation_tolerance,
            criterion=self.criterion if criterion is None else normalize_criterion(criterion),
        ) + self.stabilization_loss_from_control_points(
            control_points,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            constraint_violation_tolerance=constraint_violation_tolerance,
        )

    def optimize_trajectory(
        self,
        num_control_points: int,
        num_steps: int,
        learning_rate: float,
        window_length: int | None = None,
        measurement_variances=None,
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
            constraint_weight=constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            verbose=verbose,
        )

    def _random_gain_tuning_control_points(
        self,
        rng: np.random.Generator,
        num_control_points: int,
        min_speed: float = 0.0,
    ) -> jnp.ndarray:
        env_min = np.asarray(self.problem.environment_min, dtype=float)
        env_max = np.asarray(self.problem.environment_max, dtype=float)
        if not np.all(np.isfinite(env_min)) or not np.all(np.isfinite(env_max)):
            raise ValueError("Random gain-tuning trajectories require finite environment bounds.")
        # Start and goal at least min_start_goal_distance apart: designs stay
        # close to their initialization, so a short line stays slow.
        if self.min_start_goal_distance >= float(np.linalg.norm(env_max - env_min)):
            raise ValueError(
                f"min_start_goal_distance {self.min_start_goal_distance} m does not fit in the "
                f"environment ({env_min} to {env_max})."
            )
        start = rng.uniform(env_min, env_max)
        goal = rng.uniform(env_min, env_max)
        while float(np.linalg.norm(goal - start)) < self.min_start_goal_distance:
            goal = rng.uniform(env_min, env_max)
        line_samples = np.linspace(0.0, 1.0, num_control_points)[:, None]
        positions = start[None, :] + line_samples * (goal[None, :] - start[None, :])
        if float(min_speed) > 0.0:
            # Lay the curve out at the length its minimum mean speed asks for.
            plan = self._spline_plan(num_control_points, self.time_scaling)
            limits = {
                name: float(value)
                for name, value in self.motion_limits().items()
                if name in ("v_max", "omega_max", "alpha_max")
            }
            positions = stretch_control_points_to_arc(
                np.asarray(plan.B0, dtype=float),
                positions,
                float(min_speed) * float(self.problem.sim_time),
                env_min,
                env_max,
                dt=float(self.problem.dt),
                limits=limits,
            )
        return jnp.asarray(positions, dtype=jnp.float32)

    def initial_decision_variable_candidates(
        self,
        num_control_points: int,
        num_trajectories: int,
        seed: int = 0,
        min_speeds=None,
    ) -> list[jnp.ndarray]:
        control_point_candidates = self.initial_control_point_candidates(
            num_control_points=num_control_points,
            num_trajectories=num_trajectories,
            seed=seed,
            min_speeds=min_speeds,
        )
        return [
            self.decision_variables_from_control_points(control_points)
            for control_points in control_point_candidates
        ]

    def initial_control_point_candidates(
        self,
        num_control_points: int,
        num_trajectories: int,
        seed: int = 0,
        min_speeds=None,
    ) -> list[jnp.ndarray]:
        """One initial control-point set per trajectory.

        ``min_speeds`` (per trajectory) stretches the initial curve to the
        length its mean-speed floor needs; 0 keeps the random line.
        """
        if num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive.")
        if num_control_points < 2:
            raise ValueError("num_control_points must be >= 2.")
        if min_speeds is None:
            min_speeds = np.zeros(num_trajectories, dtype=float)
        min_speeds = np.broadcast_to(np.asarray(min_speeds, dtype=float), (num_trajectories,))
        rng = np.random.default_rng(seed)
        candidates = []
        for index in range(num_trajectories):
            if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
                control_points = self._random_gain_tuning_control_points(
                    rng, num_control_points, min_speed=float(min_speeds[index])
                )
            else:
                control_points = self.initial_control_points(num_control_points)
            candidates.append(self.clamp_control_points(control_points))
        return candidates

    def batch_frozen_start_offsets(self, num_trajectories: int, seed: int) -> jnp.ndarray:
        """The frozen ``(T, R, 3)`` start offsets of a batch optimization:
        a separate draw per trajectory in ``random`` mode, else the shared bundle."""
        if self.start_offset_mode != START_OFFSET_MODE_RANDOM:
            return jnp.broadcast_to(
                self.realizations.start_offsets,
                (num_trajectories,) + self.realizations.start_offsets.shape,
            )
        batch_key = jax.random.fold_in(
            jax.random.fold_in(self.simulation.robot_key, 5814), seed
        )
        return sample_initial_pose_offset_batch(
            batch_key,
            num_trajectories,
            int(self.realizations.start_offsets.shape[0]),
            self.offset_radius,
            self.offset_angle,
        )

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
        min_speed: float = 0.0,
        min_speed_fraction: float = 0.5,
        min_lateral_acceleration: float = 0.0,
        min_lateral_acceleration_fraction: float = 0.25,
        verbose: bool = True,
        stretch_initialization: bool = True,
    ):
        """Design ``num_trajectories`` trajectories independently, in parallel.

        ``stretch_initialization`` lays each speed-floored trajectory's initial
        curve out at the length its floor asks for; False starts every
        trajectory on a random line.
        """
        # Build the basis before anything is traced.
        self._spline_plan(num_control_points, self.time_scaling)
        motion_floors_per_trajectory = motion_floors_for_batch(
            num_trajectories, min_speed, min_speed_fraction,
            min_lateral_acceleration, min_lateral_acceleration_fraction,
        )
        min_speeds_per_trajectory = motion_floors_per_trajectory[:, 0]
        initialization_min_speeds = min_speeds_per_trajectory if stretch_initialization else None
        initial_control_point_candidates = self.initial_control_point_candidates(
            num_control_points=num_control_points,
            num_trajectories=num_trajectories,
            seed=seed,
            min_speeds=initialization_min_speeds,
        )
        rng = np.random.default_rng(seed + 1)
        if constraint_weight_jitter < 0.0:
            raise ValueError("constraint_weight_jitter must be non-negative.")
        low = max(0.0, 1.0 - float(constraint_weight_jitter))
        high = 1.0 + float(constraint_weight_jitter)
        constraint_weight_factors = rng.uniform(low, high, size=num_trajectories)
        constraint_weights_per_trajectory = constraint_weight * constraint_weight_factors
        if verbose and float(min_speed) > 0.0:
            constrained = int(np.count_nonzero(min_speeds_per_trajectory))
            print(
                f"Minimum mean speed on {constrained}/{num_trajectories} trajectories: "
                f"{np.round(min_speeds_per_trajectory[min_speeds_per_trajectory > 0.0], 2).tolist()} m/s "
                f"(the rest are left unconstrained, so the set still covers the slow regime)"
            )
        if verbose and float(min_lateral_acceleration) > 0.0:
            lateral_floors = motion_floors_per_trajectory[:, 1]
            print(
                f"Minimum mean lateral acceleration on {int(np.count_nonzero(lateral_floors))}/{num_trajectories} "
                f"trajectories: {np.round(lateral_floors[lateral_floors > 0.0], 2).tolist()} m/s^2"
            )
        batch_frozen_start_offsets = self.batch_frozen_start_offsets(num_trajectories, seed)

        if vectorized:
            from wmr_simulator.trajectory_optimization.optimizers import optimize_control_points_batch

            initial_decision_variables = jnp.stack(
                self.initial_decision_variable_candidates(
                    num_control_points=num_control_points,
                    num_trajectories=num_trajectories,
                    seed=seed,
                    min_speeds=initialization_min_speeds,
                ),
                axis=0,
            )
            group_constraint_weights = jnp.asarray(constraint_weights_per_trajectory, dtype=jnp.float32)
            # Random mode: each trajectory has its own frozen start offsets and
            # shares the noise keys.
            batch_realizations = None
            if self.start_offset_mode == START_OFFSET_MODE_RANDOM:
                batch_realizations = Realizations(
                    robot_keys=jnp.broadcast_to(
                        self.realizations.robot_keys,
                        (num_trajectories,) + self.realizations.robot_keys.shape,
                    ),
                    estimator_keys=jnp.broadcast_to(
                        self.realizations.estimator_keys,
                        (num_trajectories,) + self.realizations.estimator_keys.shape,
                    ),
                    start_offsets=batch_frozen_start_offsets,
                )
            optimized_control_points, loss_history, best_decision_variables = optimize_control_points_batch(
                pipeline=self,
                initial_decision_variables=initial_decision_variables,
                num_steps=num_steps,
                learning_rate=learning_rate,
                window_length=window_length,
                measurement_variances=measurement_variances,
                constraint_weight=group_constraint_weights,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=constraint_smooth_max_beta,
                motion_floor=jnp.asarray(motion_floors_per_trajectory, dtype=jnp.float32),
                realizations_batch=batch_realizations,
                verbose=verbose,
            )
            self.batch_start_offsets = (
                batch_frozen_start_offsets
                if self.start_offset_mode == START_OFFSET_MODE_RANDOM
                else jax.vmap(self.start_offsets_from_decision_variables)(best_decision_variables)
            )
            loss_history = np.asarray(loss_history, dtype=float) if loss_history else np.empty((0, num_trajectories))
            final_losses = []
            for trajectory_index in range(num_trajectories):
                optimized_control_points_one = optimized_control_points[trajectory_index]
                candidate_constraint_weight = float(constraint_weights_per_trajectory[trajectory_index])
                candidate_motion_floor = tuple(float(value) for value in motion_floors_per_trajectory[trajectory_index])
                candidate_realizations = self.realizations._replace(
                    start_offsets=self.batch_start_offsets[trajectory_index]
                )
                final_loss = self.fim_loss_from_control_points(
                    optimized_control_points_one,
                    window_length=window_length,
                    measurement_variances=measurement_variances,
                    constraint_weight=candidate_constraint_weight,
                    constraint_component_weights=constraint_component_weights,
                    constraint_smooth_max_beta=constraint_smooth_max_beta,
                    realizations=candidate_realizations,
                    motion_floor=candidate_motion_floor,
                )
                if not np.isfinite(float(final_loss)):
                    # Diverged: fall back to the initialization.
                    print(f"  trajectory {trajectory_index} diverged; falling back to its "
                          f"unoptimized initialization")
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
                        realizations=candidate_realizations,
                        motion_floor=candidate_motion_floor,
                        )
                final_losses.append(float(final_loss))
            loss_history = loss_history.tolist()
        else:
            optimized = []
            candidate_start_offsets = []
            histories = []
            final_losses = []
            for control_points, candidate_constraint_weight, candidate_motion_floor in zip(
                initial_control_point_candidates,
                constraint_weights_per_trajectory,
                motion_floors_per_trajectory,
            ):
                trajectory_index = len(optimized)
                if self.start_offset_mode == START_OFFSET_MODE_RANDOM:
                    self.set_start_offsets(batch_frozen_start_offsets[trajectory_index])
                optimized_control_points_one, loss_history_one = optimize_control_points(
                    pipeline=self,
                    num_control_points=control_points.shape[0],
                    num_steps=num_steps,
                    learning_rate=learning_rate,
                    initial_control_points=control_points,
                    window_length=window_length,
                    measurement_variances=measurement_variances,
                    constraint_weight=float(candidate_constraint_weight),
                    constraint_component_weights=constraint_component_weights,
                    constraint_smooth_max_beta=constraint_smooth_max_beta,
                    motion_floor=tuple(float(value) for value in candidate_motion_floor),
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
                candidate_start_offsets.append(self.realizations.start_offsets)
                histories.append(loss_history_one)
                final_losses.append(float(final_loss))
            shapes = {tuple(candidate.shape) for candidate in optimized}
            optimized_control_points = jnp.stack(optimized, axis=0) if len(shapes) == 1 else optimized
            self.batch_start_offsets = jnp.stack(candidate_start_offsets, axis=0)
            loss_history = np.asarray(histories, dtype=float).T.tolist() if histories and histories[0] else []

        self.batch_loss_history = loss_history
        final_losses = np.asarray(final_losses, dtype=float)
        self.batch_final_losses = final_losses.tolist()
        self.batch_constraint_weights = constraint_weights_per_trajectory.tolist()
        finite_losses = np.where(np.isfinite(final_losses), final_losses, np.inf)
        best_index = int(np.argmin(finite_losses))
        self.set_start_offsets(self.batch_start_offsets[best_index])
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

    def plot_loss_history(self, out_prefix="loss_history", out_path=None):
        if self.loss_history is None:
            raise ValueError("No optimization loss history available. Run optimize_trajectory() first.")
        plot_loss_history_figure(
            self.loss_history,
            out_prefix=out_prefix,
            out_path=out_path,
        )

    def save_reference_states_pickle(
        self,
        out_dir="trajectory_exports",
        filename_prefix="reference_states",
        reference_states=None,
        start_offsets=...,
        control_points=...,
    ):
        """Export a reference trajectory (by default the pipeline's current one).

        ``start_offsets`` defaults to the realization offsets in gain-tuning
        mode and None in identification mode. ``control_points`` defaults to
        the pipeline's own unless ``reference_states`` is given. Curves are
        checked with :meth:`assert_curve_is_exportable` first.
        """
        os.makedirs(out_dir, exist_ok=True)
        if control_points is ...:
            control_points = self.control_points if reference_states is None else None
        reference_states = self.reference_states if reference_states is None else reference_states
        if start_offsets is ...:
            start_offsets = (
                self.realizations.start_offsets
                if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING
                else None
            )
        if control_points is not None:
            self.assert_curve_is_exportable(control_points, label=filename_prefix)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.pkl"
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as file:
            pickle.dump(
                reference_states_export_payload(
                    reference_states,
                    self.problem.dt,
                    start_offsets=start_offsets,
                    control_points=control_points,
                    phase_breaks=list(self.phase_breaks),
                    # Seconds of a log the identification fits (None: all).
                    identified_duration=self.identified_duration,
                ),
                file,
            )
        return out_path
