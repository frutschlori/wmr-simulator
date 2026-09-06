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
from wmr_simulator.trajectory_optimization.start_offsets import (
    START_OFFSET_MODE_RANDOM,
    START_OFFSET_MODE_STATIC,
    START_OFFSET_MODES,
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
    fim_loss,
    fim_objective_term,
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
)
from wmr_simulator.trajectory_optimization.parametrization import normalize_time_scaling
from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.objectives import Realizations, make_realizations
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
DEFAULT_NUM_CONTROL_POINTS = 4
# The gain-tuning FIM is an expectation over the same realizations the gain
# tuner scores its objective on -- noise keys and start-pose offsets alike --
# so their count and distribution come from the gain-tuning defaults rather
# than a private copy here.
DEFAULT_NUM_REALIZATIONS = int(GAIN_TUNING_DEFAULTS["num_realizations"])
# Lower bound on the relative FIM scale of a controller gain, matching the
# default k_min_stab of the gain search (gain_tuning.defaults). Only the four
# gains the tuner searches in log space use it; see fim_parameter_scaling.
GAIN_FIM_SCALING_FLOOR = float(GAIN_TUNING_DEFAULTS["k_min_stab"])
# Index of kimotor in the flat 5-gain vector [kx, ky, kth, kpmotor, kimotor].
# It is the one gain the tuner is allowed to set to exactly 0 (integral action
# off), which is why it cannot be scaled relative to its own value.
KIMOTOR_INDEX = 4
# Fallback FIM scale for kimotor, used only when the problem yaml starts it at
# 0 and there is no nominal value to scale by: the width of the range the tuner
# searches it over (gain_tuning.optimizers' sqrt space is [0, k_max_rest]).
KIMOTOR_FIM_SCALE_FALLBACK = float(GAIN_TUNING_DEFAULTS["k_max_rest"])
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
    """The pickle a designed trajectory ships as.

    ``start_offsets`` (R, 3) are the start-pose offsets the trajectory was
    designed under -- the realizations the FIM was averaged over. They travel
    with the curve because the design is only informative about the gains
    *under those conditions*: the gain tuner reads them back and rolls out on
    exactly the starts the designer scored, instead of drawing its own.

    ``control_points`` (K, 2) are the curve the sampled states came from. The
    states alone are a lossy record of a design -- they are the basis applied to
    the control points on one time grid -- so anything that wants to keep
    *optimizing* the trajectory (the joint loop's warm start) needs the decision
    variables themselves, not a reconstruction of them.

    Consumers read this payload by key and ignore what they do not know, so
    adding fields here is safe for the reference exporter, the gain tuner and
    the identification analysis alike.
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


class ReferenceStatesExport(NamedTuple):
    """A directory of designed trajectories, read back.

    ``start_offsets`` is (T, R, 3) and ``control_points`` (T, K, 2) when the
    design shipped them, None when it did not (identification-mode exports carry
    no offsets). Same all-or-none rule the gain tuner's loader applies to the
    offsets: a set where half the trajectories were scored under designed starts
    and half were not is not one experiment.
    """

    reference_states: np.ndarray            # (T, N, 8)
    start_offsets: np.ndarray | None        # (T, R, 3)
    control_points: np.ndarray | None       # (T, K, 2)
    dt: float
    paths: list[str]


def load_reference_states_exports(directory: str) -> ReferenceStatesExport:
    """Read back everything :func:`reference_states_export_payload` wrote.

    The counterpart of the writer, and the reader behind a warm start: a run
    pointed at a directory of pretuned trajectories picks up the curves, their
    control points and the start offsets they were designed under, rather than
    re-deriving any of them.
    """
    paths = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".pkl")
    )
    if not paths:
        raise ValueError(f"No trajectory pickles found in {directory}")

    reference_states, start_offsets, control_points, timesteps = [], [], [], []
    for path in paths:
        with open(path, "rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a reference-states export payload.")
        reference_states.append(np.asarray(payload["reference_states"], dtype=float))
        start_offsets.append(
            None if payload.get("start_offsets") is None
            else np.asarray(payload["start_offsets"], dtype=float)
        )
        control_points.append(
            None if payload.get("control_points") is None
            else np.asarray(payload["control_points"], dtype=float)
        )
        timesteps.append(float(payload["dt"]))

    shapes = {states.shape for states in reference_states}
    if len(shapes) != 1:
        raise ValueError(f"Trajectories in {directory} have differing shapes: {sorted(shapes)}.")
    if len(set(timesteps)) != 1:
        raise ValueError(f"Trajectories in {directory} were sampled at differing dt: {timesteps}.")

    present = [offsets is not None for offsets in start_offsets]
    if any(present) and not all(present):
        missing = [path for path, has in zip(paths, present) if not has]
        raise ValueError(
            "Some trajectories carry designed start offsets and some do not: "
            f"{missing}. Export them all with offsets, or none."
        )
    if all(present):
        offset_shapes = {offsets.shape for offsets in start_offsets}
        if len(offset_shapes) != 1 or len(next(iter(offset_shapes))) != 2:
            raise ValueError(
                f"Designed start offsets must all have the same shape (R, 3); got {sorted(offset_shapes)}."
            )

    have_control_points = [points is not None for points in control_points]
    if any(have_control_points) and not all(have_control_points):
        missing = [path for path, has in zip(paths, have_control_points) if not has]
        raise ValueError(
            f"Some trajectories carry control points and some do not: {missing}. "
            "Export them all with control points, or none."
        )
    if all(have_control_points):
        point_shapes = {points.shape for points in control_points}
        if len(point_shapes) != 1 or len(next(iter(point_shapes))) != 2:
            raise ValueError(
                f"Control points must all have the same shape (K, 2); got {sorted(point_shapes)}."
            )

    return ReferenceStatesExport(
        reference_states=np.stack(reference_states, axis=0),
        start_offsets=np.stack(start_offsets, axis=0) if all(present) else None,
        control_points=(
            np.stack(control_points, axis=0) if all(have_control_points) else None
        ),
        dt=timesteps[0],
        paths=paths,
    )


class TrajectoryOptimizationPipeline:
    def __init__(
        self,
        problem_path: str,
        time_scaling: str | None = None,
        objective_mode: str = OBJECTIVE_MODE_IDENTIFICATION,
        fim_a_slip_max: bool = True,
        num_realizations: int = DEFAULT_NUM_REALIZATIONS,
        realizations: Realizations | None = None,
        criterion: str = DEFAULT_CRITERION,
        start_offset_mode: str = START_OFFSET_MODE_RANDOM,
        offset_displacement_step_factor: float = 1.0,
        offset_heading_step_factor: float = 1.0,
        min_tangent_fraction: float = DEFAULT_MIN_TANGENT_FRACTION,
        kimotor_fim_scale: float | None = None,
        motion_limits: dict | None = None,
        controller_gains=None,
        residual_model=None,
    ):
        self.problem = ProblemDefinition(problem_path)
        # Motion limits the *constraint term* is written against, as a partial
        # robot-config block (v_max, a_max, a_max_lateral, omega_max,
        # alpha_max) overriding the problem yaml's. The plant is untouched --
        # this only moves the bar the designed curve is held under, so one
        # problem can be designed against a gentler envelope (identification,
        # which the robot has to drive open loop from a hand placement) than
        # the one gain tuning designs against.
        self.motion_limit_overrides = (
            {} if motion_limits is None else {key: float(value) for key, value in motion_limits.items()}
        )
        # See stabilization_loss_from_control_points: the floor |dpos/ds| is
        # held above, as a fraction of the curve's own rms tangent, so the
        # optimizer cannot buy information by stalling the curve into a cusp and
        # the bar rescales with the problem. <= 0 disables the term.
        self.min_tangent_fraction = float(min_tangent_fraction)
        # Optional learned residual dynamics (residual_model.load_residual_model),
        # exactly as the gain tuner takes it: None keeps the nominal plant. It is
        # part of the plant the design is scored on -- every closed-loop rollout
        # in here goes through self.simulation.run_closed_loop, which defaults to
        # the pipeline's model -- and nothing about it is exported with the curve.
        self.residual_model = residual_model
        self.simulation = SimulationPipeline(
            problem_path=problem_path,
            seed=0,
            reference_trajectories_dir=None,
            residual_model=residual_model,
        )
        self.robot = self.simulation.robot
        self.controller = self.simulation.controller
        # The design point, and the gains every rollout in here is driven at.
        #
        # This designer always drives the *static* controller -- no rollout in
        # this class passes ``schedule_params``, so the gain parametrization is
        # off everywhere -- and the design point therefore has to be the static
        # gains. A parametrized run's *base* gains are not a controller anybody
        # runs (the network absorbs whatever scale they take, and they drift far
        # from the effective gains), so designing at them while driving the
        # static controller designs for a robot that does not exist: measured on
        # real02, base kth 50 / kpmotor 21.5 against effective 9.25 / 0.0, and
        # that design point is what flattens the curves. ``controller_gains``
        # names them explicitly; None falls back to the problem yaml's, which is
        # only the same thing when no parametrization has been trained against
        # them.
        self.controller_gains = (
            self.simulation.gains
            if controller_gains is None
            else jnp.asarray(controller_gains, dtype=jnp.float32)
        )
        if self.controller_gains.shape != self.simulation.gains.shape:
            raise ValueError(
                f"controller_gains must have shape {self.simulation.gains.shape}, "
                f"got {self.controller_gains.shape}."
            )
        if controller_gains is None and self._problem_has_a_trained_parametrization():
            # Not an error -- a bare problem yaml is a legitimate thing to
            # design from -- but designing at a *trained* parametrization's base
            # gains is the flaw this argument exists for, and it is silent.
            print(
                "WARNING: the problem's gain parametrization is trained, so "
                f"controller.gains {np.asarray(self.controller_gains).tolist()} are its base "
                "gains, not a controller anybody runs. This designer drives the static "
                "controller; pass controller_gains= the static-tuned gains "
                "(robot_config_static_gains.yaml in an active-learning iteration)."
            )
        # kimotor's FIM scale: constant, and by default the problem's *nominal*
        # kimotor. See fim_parameter_scaling for why it cannot track the value.
        #
        # ``kimotor_fim_scale`` pins it instead, which is what makes a design
        # point of kimotor = 0 usable: that is where the gain tuner reliably
        # lands, and active learning writes the tuned gains into the next
        # iteration's problem yaml, so the criterion has to work there. Tying
        # the scale to the design point instead falls back to the search range
        # and leaves the column contributing nothing.
        nominal_kimotor = float(self.controller_gains[KIMOTOR_INDEX])
        if kimotor_fim_scale is None:
            self.kimotor_fim_scale = (
                nominal_kimotor if nominal_kimotor > 0.0 else KIMOTOR_FIM_SCALE_FALLBACK
            )
        else:
            self.kimotor_fim_scale = float(kimotor_fim_scale)
            if self.kimotor_fim_scale <= 0.0:
                raise ValueError("kimotor_fim_scale must be positive.")
        self.estimator = self.simulation.estimator
        # The encoder low-pass is part of the plant the designed trajectory will
        # be driven on: the firmware filters the raw wheel speeds with
        # tau = 1/(2*pi*3 Hz) = 0.053 s at its 10 ms inner step
        # (firmware/src/inner_controller.rs), which is exactly what the estimator
        # reproduces, so the simulated timing is not the problem. It is always on
        # (read from the problem config); read-only mirror for callers that want
        # the effective value without reaching into ``self.estimator``.
        self.wheel_lp_tau = float(self.estimator.wheel_lp_tau)
        self.time_scaling = normalize_time_scaling(time_scaling)
        self.objective_mode = normalize_objective_mode(objective_mode)
        # Which design criterion the objective minimizes. It says nothing about
        # the curve or the parameters -- only about how a FIM is scored.
        self.criterion = normalize_criterion(criterion)
        # A BSplinePlan is just the basis sampled on this pipeline's time grid,
        # constant given (num_control_points, time_scaling); cache so repeated
        # eager calls (plotting, tests) don't resample it.
        self._spline_plan_cache: dict[tuple[int, str], BSplinePlan] = {}
        # a_slip_max sometimes has near-zero sensitivity, which makes the FIM
        # objective stiff; excluding it keeps the burnout model in the rollout
        # at its nominal value but drops it from the design parameters.
        self.fim_a_slip_max = bool(fim_a_slip_max) and self.robot.a_slip_max > 0.0
        # Drawn once and reused at every objective evaluation: the design
        # criterion is an expectation over start poses and rollout noise, and
        # common random numbers keep it a deterministic function of the control
        # points. Pass ``realizations`` to score the design on the *same*
        # conditions the gain tuner uses (joint_tuning does).
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
        # Which components of the start offsets the optimizer may move. The
        # offsets are decision variables of the *curve's* problem -- they are
        # scored by the same FIM -- so they ride along in the decision vector,
        # appended after the control points. ``random``/``static`` leave the
        # drawn bundle alone and the tail is empty.
        self.start_offset_mode = normalize_start_offset_mode(start_offset_mode)
        self.start_offset_mask = start_offset_mask(self.start_offset_mode)
        if self.start_offset_mode == START_OFFSET_MODE_STATIC:
            self.realizations = self.realizations._replace(
                start_offsets=static_start_offsets(
                    int(self.realizations.robot_keys.shape[0]), self.offset_radius, self.offset_angle
                )
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
        # The free variables start *at* the drawn (or static) offsets, so an
        # optimizing run begins under exactly the conditions a frozen one uses.
        self.initial_free_offsets = inverse_squash_start_offsets(
            self.realizations.start_offsets, self.offset_radius, self.offset_angle
        )
        # Per-block step factors. Adam normalizes the update magnitude per
        # coordinate, so scaling the offset entries of the update is exactly an
        # effective learning rate of factor * learning_rate for them -- which is
        # the point: the control-point learning rate is tuned and known to work,
        # and the free offsets live on a different scale (unitless, pre-squash,
        # against a radius of offset_radius and an angle of offset_angle), so
        # they need their own pace rather than a rate that moves both.
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
        self.optimization_snapshots = None

    def _problem_has_a_trained_parametrization(self) -> bool:
        """Whether the problem's gain parametrization is anything but the identity.

        The stock yamls ship an *enabled* parametrization at theta = 0, which is
        exactly the static controller, so enablement alone says nothing -- only a
        non-zero trainable vector means the base gains have been tuned against a
        network.
        """
        if not self.simulation.gain_parametrization_enabled:
            return False
        from wmr_simulator.gain_parametrization import flat_params

        return bool(np.any(np.asarray(flat_params(self.simulation.gain_parametrization_params)) != 0.0))

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
            # ... but only for the four gains that must stay strictly positive
            # and are therefore searched in log space. kimotor is the exception:
            # it is allowed to be exactly 0 (integral action off), and a scale
            # proportional to its own value blanks its FIM column as the tuner
            # drives it toward 0 -- the FIM goes singular in that direction and
            # A-optimality picks up an unfixable term that then dominates the
            # trajectory objective (a floor only bounds how badly: at 1e-3 the
            # column is still dead).
            #
            # All that is needed to fix that is a *constant* scale, and the
            # constant has to be the problem's nominal kimotor rather than
            # anything larger. Relative scaling already used the nominal value
            # at the initial design point, so pinning it there leaves every
            # design that does not move the gains -- the whole standalone
            # trajectory optimizer -- bit-identical, while the joint loop's
            # moving design point no longer collapses the column. The size of
            # this constant is not cosmetic: it sets how much of trace(FIM^-1)
            # lives in the kimotor direction (35-40% at the nominal value), and
            # that share is what buys *curvature* in the design, since tight
            # turns are what excite the motor loop. Scaling by the search range
            # k_max_rest = 20 instead dropped the share to 3-5% and cost 28% of
            # the designed curves' mean |kappa| over 500 steps.
            return jnp.asarray(
                jnp.maximum(params, GAIN_FIM_SCALING_FLOOR)
                .at[KIMOTOR_INDEX]
                .set(self.kimotor_fim_scale),
                dtype=jnp.float32,
            )
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
        robot_cfg = self.problem.robot_cfg
        if self.motion_limit_overrides:
            robot_cfg = {**robot_cfg, **self.motion_limit_overrides}
        return motion_limits_from_robot_config(robot_cfg)

    def constraint_weights(self, scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
        return constraint_weights(scale=scale, component_weights=component_weights)

    def pinned_positions(self) -> dict[int, np.ndarray]:
        """Control points whose position is boundary data rather than a decision
        variable, mapped to that position.

        An identification trajectory must run from the pose the robot is
        physically placed in -- only the start is boundary data, since that is
        the one pose that has to be set by hand on the real robot before a run.
        The spline is clamped, so the curve begins at the first control point,
        and pinning it is the whole of it; the goal control point (the last
        one, since the curve also ends there) is a free decision variable,
        bounded only by the environment-box clip every control point already
        gets in ``clamp_control_points``. A gain-tuning trajectory has no
        meaningful start or goal -- it only has to excite the gains somewhere
        inside the environment box -- so every control point is free there too.
        """
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            return {}
        return {0: self.problem.start[:2]}

    def pin_start_heading(self) -> bool:
        """Whether the second control point is held on the start-heading ray.

        Only meaningful when the start itself is pinned: the heading there is
        how the robot is physically placed for a run, not a design choice.
        """
        return self.objective_mode != OBJECTIVE_MODE_GAIN_TUNING

    def _spline_plan(self, num_control_points: int, time_scaling: str) -> BSplinePlan:
        """Cached basis for this control-point count.

        Must be warmed *eagerly*: constructing a plan inside a jit trace stages
        the time grid into a tracer, and the basis is built with NumPy. Callers
        that hand a fresh control-point count to a jitted optimizer warm it
        first (see ``optimize_trajectories``).
        """
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
            # Always smooth here, including when fim_a_slip_max is off: the
            # designer's plant must not depend on which parameters the criterion
            # happens to score, or two runs of the same problem design against
            # different robots. This is the consumer the soft clip exists for.
            smooth_traction_limit=True,
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
        realizations: Realizations | None = None,
    ) -> jnp.ndarray:
        """
        Runs the closed-loop deployment once per start-pose offset and stacks the
        estimated poses. window_length is accepted for API symmetry with replay
        mode; closed-loop gain sensitivities intentionally use the full rollout
        without replay resets.

        Stacking is what makes the design criterion an *expectation* over start
        poses: the FIM of the stacked measurements is the sum of the per-rollout
        FIMs (``[J1; J2]^T [J1; J2] = J1^T J1 + J2^T J2``), so averaging happens
        in factored form and no FIM is ever assembled.

        The offsets matter because the Kanayama law multiplies kx and ky by the
        tracking errors: starting on the reference leaves almost no error, so
        those two gains are nearly invisible to the design. The realizations are
        drawn once (``self.realizations``) and reused at every evaluation --
        common random numbers, which keeps the objective deterministic, as the
        line search requires.

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

        # ``realizations`` overrides the pipeline's frozen bundle, which is what
        # makes the start offsets usable as decision variables (joint_tuning):
        # they then arrive as tracers rather than as a constructor-time draw.
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
            measurements = self.replay_measurement_sequence(
                params,
                window_length,
                closed_loop_log=closed_loop_log,
            )
        return measurements.reshape(-1)

    def compute_fim_factor(self, measurement_variances=None, window_length=None,
        closed_loop_log=None, reference_states=None, gains=None,
        realizations: Realizations | None = None) -> jnp.ndarray:
        """Weighted relative parameter sensitivities ``J~``, with ``FIM = J~^T J~``.

        This is what the design criteria consume; assembling the FIM squares its
        condition number, which float32 cannot survive here (see ``fim.py``).

        ``gains`` replaces the design point in gain-tuning mode. The pipeline's
        own ``controller_gains`` are fixed at construction, but an alternating
        gain/trajectory loop has to evaluate the information content at the
        gains the tuner has just stepped to, not at the ones the run started
        from. ``fim_parameter_scaling`` takes the same vector, so the relative
        scaling follows the moving design point and the criterion stays
        comparable across rounds.
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

    def compute_fim_matrix(self, measurement_variances=None, window_length=None,
        closed_loop_log=None, reference_states=None) -> jnp.ndarray:
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
            pinned_positions=self.pinned_positions(),
            pin_start_heading=self.pin_start_heading(),
        )

    # The decision vector is [ravel(control_points), ravel(free_offsets)]. The
    # offset tail is empty unless start_offset_mode optimizes them, so a run
    # without it sees exactly the vector it always did.
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
        """The start offsets a decision vector stands for: the free variables
        smoothly squashed into the feasible set, with the frozen draw kept for
        every component the mode does not optimize."""
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
        """Re-project a decision vector onto the feasible set (environment box
        and pins). The optimizer applies this after every step. The offset tail
        passes through untouched: it is squashed into its feasible set on read,
        never clipped, because FIM-optimized offsets sit on the boundary and a
        clip there is the constraint wall Adam bounces off."""
        return self.decision_variables_from_control_points(
            self.control_points_from_decision_variables(decision_variables),
            self.free_offsets_from_decision_variables(decision_variables),
        )

    def decision_variable_step_scale(self, num_decision_variables: int) -> jnp.ndarray | None:
        """Per-coordinate multiplier on the optimizer's update, or None when
        every coordinate steps at the plain learning rate.

        Only the offset tail is ever scaled; the control points always step at
        1.0, so a run that touches the factors leaves the curve's optimization
        exactly as it was."""
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
        """Adopt a set of start offsets as the pipeline's own: they become the
        bundle every later rollout, plot and export uses, and the point the free
        variables sit at."""
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
        """The basis's tangent-floor term, scaled like the motion limits.

        A curve whose ``|dpos/ds|`` collapses somewhere in the interior has no
        heading there, and the reference it produces is not a trajectory -- but
        nothing else in the objective can tell. The motion limits cannot,
        because the guard in ``bspline._reference_from_derivatives`` floors the
        ``dtheta_ds`` denominator: at a stall that denominator is 5x too large,
        so the reference reports 0.90x ``alpha_max`` where the geometry demands
        14x. (Measured by toggling the guard's two halves independently on the
        6cp design: unfloored, the constraint term is 1868 against 0.048 as
        shipped, while the fallback heading fed to ``arctan2`` changes it by
        exactly nothing -- the limits never read ``theta``. The limits would
        have crushed the stall on their own; the floor is what hides it.)

        Meanwhile the FIM *rewards* it: the untrackable heading flip makes the
        rollout enormously sensitive to the gains, and an unguarded criterion
        reads that sensitivity as information. Measured on the 2026-08-10 6cp designs,
        blending a stalled control polygon back to an evenly spread one over
        the same route costs 7.5x in ``trace(FIM^-1)`` -- 2.0 log units,
        monotonically. So the term has to be a real constraint, not a nudge:
        dividing by ``constraint_violation_tolerance`` (0.05 -> weight 20) puts
        a fully stalled curve at ~20 against that 2.0, and leaves every healthy
        design at exactly 0.
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
        """Refuse to export a curve whose heading came from the fallback tangent.

        Wherever ``|dpos/ds|`` falls under the guard's threshold, the exported
        reference does not carry the curve's heading at those samples -- it
        carries the constant ``[1, 0]`` fallback, and the step out of it is a
        ~166 deg discontinuity in a 50 ms sample. Everything downstream then
        consumes that as a pose command: the gain tuner rolls out against it,
        and ``pololu/reference_exporter`` puts it on the robot. Nothing further
        down can recognize it, because the sampled states are all it gets.

        The tangent-floor loss is what keeps the optimizer 2.5x clear of this,
        so a design tripping the check means the loss was disabled, the
        fraction is too low for the problem, or the optimizer found a stall the
        term did not price. Any of those is worth a failed run rather than a
        silently unusable pickle.
        """
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
            limits=self.motion_limits(),
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

    def objective_terms_from_control_points(
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
    ) -> dict[str, jnp.ndarray]:
        """The objective's two terms, separately, in the same units it sums them.

        ``fim`` is the raw A-optimality criterion, ``constraints`` the raw
        weighted penalty and ``tangent_floor`` the raw stabilization term;
        ``log_fim``, ``constraint_term`` and ``tangent_floor_term`` are what
        :meth:`fim_loss_from_control_points` actually adds up.
        """
        control_points = self.clamp_control_points(control_points)
        reference_states = self.reference_states_from_control_points(control_points)
        closed_loop_log = self.run_closed_loop_deployment(reference_states=reference_states)
        fim_factor = self.compute_fim_factor(
            measurement_variances=measurement_variances,
            window_length=window_length,
            closed_loop_log=closed_loop_log,
            gains=gains,
        )
        criterion = self.criterion if criterion is None else normalize_criterion(criterion)
        fim_term = fim_loss(fim_factor, criterion)
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
        log_fim_term = fim_objective_term(fim_factor, criterion)
        scaled_constraint_term = constraint_term / constraint_violation_tolerance
        tangent_floor_term = self.stabilization_loss_from_control_points(
            control_points,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            constraint_violation_tolerance=constraint_violation_tolerance,
        )
        return {
            "criterion": criterion,
            "fim": fim_term,
            "log_fim": log_fim_term,
            "constraints": constraint_term,
            "constraint_term": scaled_constraint_term,
            "tangent_floor": tangent_floor_term * constraint_violation_tolerance,
            "tangent_floor_term": tangent_floor_term,
            "total": log_fim_term + scaled_constraint_term + tangent_floor_term,
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

    def initial_decision_variable_candidates(
        self,
        num_control_points: int,
        num_trajectories: int,
        seed: int = 0,
    ) -> list[jnp.ndarray]:
        control_point_candidates = self.initial_control_point_candidates(
            num_control_points=num_control_points,
            num_trajectories=num_trajectories,
            seed=seed,
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

    def batch_frozen_start_offsets(self, num_trajectories: int, seed: int) -> jnp.ndarray:
        """Return the frozen ``(T, R, 3)`` starts for a batch optimization.

        ``random`` deliberately draws a different complete pose bundle per
        trajectory.  The bundle remains fixed over all of that trajectory's
        Adam steps, so the objective is still deterministic.  The other modes
        retain their existing semantics: ``static`` is the same prescribed
        spread for every trajectory, while optimizing modes receive the common
        frozen initialization which their decision variables then move.
        """
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
        verbose: bool = True,
    ):
        # Build the basis before anything is traced; see _spline_plan.
        self._spline_plan(num_control_points, self.time_scaling)
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
        batch_frozen_start_offsets = self.batch_frozen_start_offsets(num_trajectories, seed)

        if vectorized:
            from wmr_simulator.trajectory_optimization.optimizers import optimize_control_points_batch

            initial_decision_variables = jnp.stack(
                self.initial_decision_variable_candidates(
                    num_control_points=num_control_points,
                    num_trajectories=num_trajectories,
                    seed=seed,
                ),
                axis=0,
            )
            group_constraint_weights = jnp.asarray(constraint_weights_per_trajectory, dtype=jnp.float32)
            # In random mode each trajectory is evaluated under its own
            # frozen start-pose realization bundle.  This is independent across
            # trajectories, yet fixed for every step of each objective.
            batch_realizations = None
            if self.start_offset_mode == START_OFFSET_MODE_RANDOM:
                batch_realizations = Realizations(
                    # The noise roots stay common across candidates; only the
                    # designed start poses are newly sampled per trajectory.
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
                realizations_batch=batch_realizations,
                verbose=verbose,
            )
            # Random mode gives every trajectory a distinct frozen draw;
            # optimizing modes return their separately optimized starts.
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
                # Scored under this trajectory's own offsets, which is what it
                # was optimized against.
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
                )
                if not np.isfinite(float(final_loss)):
                    # The trajectory diverged and never recorded a finite point.
                    # Falling back to its unoptimized initialization keeps the
                    # batch usable, but silently shipping a straight line as an
                    # "optimized" trajectory would be worse than saying so.
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
                        )
                # Final objective value, comparable across trajectories for
                # best-candidate selection (they differ only in their constraint
                # weight jitter).
                final_losses.append(float(final_loss))
            loss_history = loss_history.tolist()
        else:
            optimized = []
            candidate_start_offsets = []
            histories = []
            final_losses = []
            for control_points, candidate_constraint_weight in zip(
                initial_control_point_candidates,
                constraint_weights_per_trajectory,
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

    def plot_trajectory_batch(
        self,
        control_point_batch,
        out_prefix="trajectory_set",
        out_path=None,
        title="Optimized Trajectories",
        start_offset_batch=None,
    ):
        """One figure for a whole batch of optimized trajectories. The rollouts
        run as a single vmapped, jitted batch instead of one eager (and
        separately compiled) rollout per trajectory.

        In gain-tuning mode they start from the same start-pose offsets the FIM
        was averaged over, so the plot shows the conditions the design was
        actually optimized for rather than an idealized start on the reference."""
        control_point_batch = jnp.stack(
            [self.clamp_control_points(control_points) for control_points in control_point_batch]
        )
        reference_trajectories = jax.vmap(self.reference_states_from_control_points)(control_point_batch)
        if self.objective_mode == OBJECTIVE_MODE_GAIN_TUNING:
            def rollout(reference_states, robot_key, estimator_key, start_offset):
                return self.simulation.run_closed_loop(
                    self.nominal_physical_params(),
                    controller_gains=self.controller_gains,
                    robot_key=robot_key,
                    estimator_key=estimator_key,
                    wheel_speed_log_source="estimated",
                    reference_states=reference_states,
                    initial_pose=reference_states[0, :3] + start_offset,
                ).pose.true_states

            realizations = self.realizations
            # One offset set per trajectory when the design optimized them
            # (each curve then has its own starts), else the shared bundle.
            if start_offset_batch is None:
                start_offset_batch = self.batch_start_offsets
            if start_offset_batch is not None and (
                jnp.asarray(start_offset_batch).shape[0] == reference_trajectories.shape[0]
            ):
                start_offset_batch = jnp.asarray(start_offset_batch, dtype=jnp.float32)
                closed_loop_poses = jax.jit(
                    jax.vmap(
                        jax.vmap(rollout, in_axes=(None, 0, 0, 0)),
                        in_axes=(0, None, None, 0),
                    )
                )(
                    reference_trajectories,
                    realizations.robot_keys,
                    realizations.estimator_keys,
                    start_offset_batch,
                )
            else:
                closed_loop_poses = jax.jit(
                    jax.vmap(
                        jax.vmap(rollout, in_axes=(None, 0, 0, 0)),
                        in_axes=(0, None, None, None),
                    )
                )(
                    reference_trajectories,
                    realizations.robot_keys,
                    realizations.estimator_keys,
                    realizations.start_offsets,
                )
        else:
            closed_loop_poses = self.simulation.run_closed_loop_batch(
                self.nominal_physical_params(),
                reference_trajectories,
                controller_gains=self.controller_gains,
                wheel_speed_log_source="estimated",
            ).pose.true_states
        return plot_trajectory_set_figure(
            reference_trajectories,
            closed_loop_poses,
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
        start_offsets=...,
        control_points=...,
    ):
        """Export a reference trajectory. Pass ``reference_states`` to export a
        trajectory other than the pipeline's current one -- exporting a batch
        via ``set_control_points`` would run (and compile) a closed-loop rollout
        per trajectory that the pickle does not use.

        ``start_offsets`` defaults to the design's own: in gain-tuning mode the
        realization offsets the FIM was averaged over, so the gain tuner reads
        back the conditions the trajectory was designed under. Identification
        mode exports none -- there the start is where the robot is physically
        placed, not a design variable. Pass an explicit array (or None) to
        override.

        ``control_points`` defaults to the pipeline's own -- but only when
        ``reference_states`` was not overridden, since a batch export passes
        someone else's states and the pipeline's current curve would then be the
        wrong one. A batch caller passes the matching control points explicitly;
        writing none is better than writing a curve that is not this
        trajectory's.

        Exports carrying control points are checked against the fallback-tangent
        threshold first; see :meth:`assert_curve_is_exportable`."""
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
                ),
                file,
            )
        return out_path
