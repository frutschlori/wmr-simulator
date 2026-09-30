"""Fixed (not designed) reference sets: the fixed-trajectory side of the evaluation.

The active-learning loop designs its identification trajectory and its tuning
set every iteration. The fixed alternatives replace one or both with references
that are chosen once, before any data exists, and reused unchanged by every
iteration and every experiment. They reach an experiment as
``baseline_identification_trajectory`` (one pickle) and
``baseline_tuning_trajectories_dir`` (a directory of pickles);
scripts/generate_fixed_trajectory_sets.py writes them.

Tuning-set families:

- ``matched``: analytic shapes (circles, arcs, slaloms, lines) with the designed
  set's count, duration and motion limits, graded over the speed and turning
  range the designed sets reach. What an engineer would pick by hand. None of
  them is a benchmark shape: the benchmark circles have radius 1.0 at 5-6 s and
  its lemniscates run 7 s.
- ``benchmark``: the benchmark set itself, converted to pickles. Training on the
  test set, so an optimistic bound for a fixed set, not a fair competitor. Its
  references run 5-7.3 s, so the tuning horizon has to be raised to hold them.
- ``random_bspline``: the designer's own curve family (clamped B-spline, s-curve
  time scaling, same control-point count, limits and speed floors) with the
  information criterion taken out. Random-walk control points are projected onto
  the limits by a short constraint-only Adam run, and whatever still violates a
  limit is redrawn. Against the designed set it isolates the FIM.
- ``random_twist``: smooth random forward-speed / yaw-rate profiles (the twist
  is a fixed linear map of the two wheel speeds), scaled into the limits and
  integrated to a path. Classic system-identification excitation, in a
  different curve family from the designer's.

The fixed identification reference mirrors the designed one's phases: a slow
left arc, a stop and a right arc inside the ``identify`` phase's limits, a rest,
then a fast arc inside the fast phase's limits. ``identified_duration`` is written
into the pickle, so the identify stage cuts the logs exactly as it does for a
designed trajectory.

Fixed tuning pickles carry no start offsets, so the gain tuner draws its own
from ``init_offset_radius``/``init_offset_angle`` (a designed set ships the
offsets it was optimized under).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from wmr_simulator.trajectory_optimization.baselines import (
    chain_references,
    circle_reference,
    place_reference,
    sine_reference,
    summarize_motion,
)

# summarize_motion key bounded by each robot-config motion limit.
LIMIT_SUMMARY_KEYS = {
    "v_max": "v_peak",
    "a_max": "a_peak",
    "a_max_lateral": "a_lat_peak",
    "omega_max": "omega_peak",
    "alpha_max": "alpha_peak",
}

# The matched tuning set: (name, generator, parameters). Every entry is drawn
# over the set's duration and then placed in the environment, and all of them
# stay inside the problem's limits (v 2.5, a 5, a_lat 7, omega 10, alpha 20) at
# 4 s. Measured at 4 s: mean speed 0.49-1.03 m/s, mean |a_lat| up to 1.74 m/s^2,
# alpha up to 16 rad/s^2 -- the range iteration 1's designed set covers (test31:
# mean speed 0.55-1.04, mean |a_lat| 0.37-1.74, alpha 13-19). Mirrored pairs keep
# the set symmetric in turning direction.
MATCHED_SET_SHAPES = (
    ("circle_r035_ccw", "circle", {"radius": 0.35}),
    ("circle_r035_cw", "circle", {"radius": 0.35, "clockwise": True}),
    ("circle_r050_ccw", "circle", {"radius": 0.5}),
    ("circle_r050_cw", "circle", {"radius": 0.5, "clockwise": True}),
    ("arc_r090_ccw", "circle", {"radius": 0.9, "sweep": 4.4}),
    ("arc_r090_cw", "circle", {"radius": 0.9, "sweep": 4.4, "clockwise": True}),
    ("arc_r120_ccw", "circle", {"radius": 1.2, "sweep": 3.0}),
    ("s_bend_gentle", "sine", {"length": 3.0, "amplitude": 0.3, "periods": 1.0}),
    ("s_bend_long", "sine", {"length": 3.8, "amplitude": -0.4, "periods": 1.0}),
    ("s_bend_wide", "sine", {"length": 3.0, "amplitude": 0.6, "periods": 1.0}),
    ("slalom_3", "sine", {"length": 3.4, "amplitude": -0.2, "periods": 1.5}),
    ("slalom_tight", "sine", {"length": 2.4, "amplitude": 0.25, "periods": 1.5}),
    ("slalom_4", "sine", {"length": 3.6, "amplitude": 0.15, "periods": 2.0}),
    ("line_long", "sine", {"length": 4.0, "amplitude": 0.0}),
    ("line_short", "sine", {"length": 2.0, "amplitude": 0.0}),
)

# Fixed identification reference, per phase of
# identification_trajectory.phases. The identified phase is two arcs, left then
# right with a stop between (each half the phase), inside its limits (v 1.2,
# a 2, a_lat 2, omega 6, alpha 10; worst ratio 0.98, total acceleration): mean
# speed 0.49 m/s, 3.0 rad of heading travel, mean |omega| 0.98 rad/s. The
# designed one reaches 0.54 m/s, 3.59 rad and 1.18 rad/s (test31); a single
# slalom over the whole phase cannot turn that much under a_lat 2 (best 2.0 rad),
# which is what makes the wheelbase observable. The fast arc sits inside the
# fast phase's limits (v 2.5, a 4.5, a_lat 5, omega 10, alpha 15) at 1.04 m/s
# and mean |a_lat| 1.73 m/s^2, against the designed 1.06 / 1.82: the floors
# 1.3 m/s and 2.0 m/s^2 cannot both be met under a_max 4.5 by an s-curve arc,
# and the designer misses them too.
IDENTIFICATION_SLOW_ARC = {"radius": 0.5, "sweep": 1.5}
IDENTIFICATION_FAST_ARC = {"radius": 0.9, "sweep": 4.67}


# ---------------------------------------------------------------------------
# limits, placement, export
# ---------------------------------------------------------------------------


def motion_limits(robot_cfg: dict, overrides: dict | None = None) -> dict[str, float]:
    """The upper motion limits of a problem ``robot`` block, by config name."""
    merged = {**robot_cfg, **(overrides or {})}
    limits = {name: float(merged[name]) for name in ("v_max", "a_max", "omega_max", "alpha_max")}
    limits["a_max_lateral"] = float(merged.get("a_max_lateral", merged["a_max"]))
    return limits


def limit_ratios(reference_states: np.ndarray, dt: float, limits: dict[str, float]) -> dict[str, float]:
    """Each limit's peak as a fraction of it; a feasible reference has all <= 1."""
    summary = summarize_motion(reference_states, dt)
    return {name: summary[LIMIT_SUMMARY_KEYS[name]] / float(value) for name, value in limits.items()}


def is_within_limits(reference_states: np.ndarray, dt: float, limits: dict[str, float]) -> bool:
    return max(limit_ratios(reference_states, dt, limits).values()) <= 1.0


def fit_into_environment(
    reference_states: np.ndarray,
    environment_min,
    environment_max,
    margin: float = 0.1,
    rng: np.random.Generator | None = None,
    num_rotations: int = 72,
) -> np.ndarray | None:
    """The reference moved rigidly so its path lies inside the environment box.

    Without ``rng`` the placement is deterministic: of ``num_rotations`` evenly
    spaced rotations, the one leaving the most room, centred in the box. With
    ``rng`` it is random: a random rotation that fits and a random position in
    the room it leaves. None if no rotation fits.
    """
    env_min = np.asarray(environment_min, dtype=float) + margin
    env_max = np.asarray(environment_max, dtype=float) - margin
    room = env_max - env_min
    rotations = np.linspace(0.0, 2.0 * np.pi, num_rotations, endpoint=False)
    if rng is not None:
        rotations = rng.permutation(rotations + rng.uniform(0.0, 2.0 * np.pi / num_rotations))
    best = None
    for rotation in rotations:
        rotated = place_reference(reference_states, rotation=float(rotation))
        low, high = rotated[:, :2].min(axis=0), rotated[:, :2].max(axis=0)
        slack = room - (high - low)
        if np.any(slack < 0.0):
            continue
        if rng is not None:
            target_low = env_min + rng.uniform(0.0, 1.0, size=2) * slack
            return place_reference(rotated, translation=tuple(target_low - low))
        if best is None or slack.min() > best[0]:
            best = (float(slack.min()), rotated, low, slack)
    if best is None:
        return None
    _, rotated, low, slack = best
    return place_reference(rotated, translation=tuple(env_min + 0.5 * slack - low))


def export_reference_set(
    references: dict[str, np.ndarray],
    out_dir: str | Path,
    dt: float,
    control_points: dict[str, np.ndarray] | None = None,
    **metadata,
) -> list[Path]:
    """One ``<name>.pkl`` per reference in the designed-trajectory pickle format.

    Refuses to write into a directory that already holds pickles: a fixed set
    is an input shared by every experiment that names it, and regenerating it
    in place would silently change what those experiments were run on.
    """
    from wmr_simulator.trajectory_optimization.pipeline import reference_states_export_payload

    out_dir = Path(out_dir)
    if out_dir.is_dir() and any(out_dir.glob("*.pkl")):
        raise FileExistsError(f"{out_dir} already holds a reference set; delete it to regenerate.")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, states in references.items():
        payload = reference_states_export_payload(
            np.asarray(states, dtype=float),
            dt,
            control_points=None if control_points is None else control_points[name],
            **metadata,
        )
        path = out_dir / f"{name}.pkl"
        with path.open("wb") as file:
            pickle.dump(payload, file)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# matched analytic set, benchmark set, identification reference
# ---------------------------------------------------------------------------


def _analytic_reference(generator: str, total_time: float, dt: float, params: dict) -> np.ndarray:
    if generator == "circle":
        return circle_reference(total_time=total_time, dt=dt, **params)
    if generator == "sine":
        return sine_reference(total_time=total_time, dt=dt, **params)
    raise ValueError(f"Unknown analytic generator '{generator}'.")


def matched_reference_set(
    total_time: float,
    dt: float,
    limits: dict[str, float],
    environment_min,
    environment_max,
) -> dict[str, np.ndarray]:
    """:data:`MATCHED_SET_SHAPES`, each checked against ``limits`` and placed in the box."""
    references = {}
    for name, generator, params in MATCHED_SET_SHAPES:
        states = _analytic_reference(generator, total_time, dt, params)
        ratios = limit_ratios(states, dt, limits)
        if max(ratios.values()) > 1.0:
            raise ValueError(f"Matched shape {name} exceeds the motion limits at {total_time} s: {ratios}.")
        placed = fit_into_environment(states, environment_min, environment_max)
        if placed is None:
            raise ValueError(f"Matched shape {name} does not fit into the environment.")
        references[name] = placed
    return references


def benchmark_reference_set(directory: str | Path, dt: float) -> dict[str, np.ndarray]:
    """Every ``.JSN``/``.pkl`` of a benchmark directory as reference states.

    Goes through pololu.reference_importer, the same reader the benchmark stage
    drives them with, so the tuner sees what the benchmark scores.
    """
    from wmr_simulator.pololu.reference_importer import load_reference, reference_states_from_pololu_reference

    paths = sorted(
        path for path in Path(directory).iterdir() if path.suffix.lower() in (".jsn", ".pkl")
    )
    if not paths:
        raise ValueError(f"No .JSN or .pkl references in {directory}.")
    references = {}
    for path in paths:
        reference = load_reference(path)
        if not np.isclose(reference.dt, dt):
            raise ValueError(f"{path} is sampled at {reference.dt} s, the problem at {dt} s.")
        references[path.stem] = np.asarray(reference_states_from_pololu_reference(reference), dtype=float)
    return references


def fixed_identification_reference(
    phases: list[dict],
    dt: float,
    robot_cfg: dict,
    environment_min,
    environment_max,
) -> tuple[np.ndarray, float]:
    """The fixed identification reference and its identified duration [s].

    ``phases`` is ``identification_trajectory.phases``: exactly one identify
    phase followed by one fast phase, as the designed trajectory has. Each part
    is checked against its own phase's limits.
    """
    if len(phases) != 2 or not phases[0].get("identify", True) or phases[1].get("identify", True):
        raise ValueError("The fixed identification reference expects one identify phase, then one fast phase.")
    slow_phase, fast_phase = phases
    half = 0.5 * float(slow_phase["duration"])
    slow = chain_references(
        circle_reference(total_time=half, dt=dt, **IDENTIFICATION_SLOW_ARC),
        circle_reference(total_time=half, dt=dt, clockwise=True, **IDENTIFICATION_SLOW_ARC),
    )
    fast = circle_reference(total_time=float(fast_phase["duration"]), dt=dt, **IDENTIFICATION_FAST_ARC)
    for label, states, phase in (("slow", slow, slow_phase), ("fast", fast, fast_phase)):
        limits = motion_limits(robot_cfg, phase.get("motion_limits"))
        ratios = limit_ratios(states, dt, limits)
        if max(ratios.values()) > 1.0:
            raise ValueError(f"The {label} identification phase exceeds its limits: {ratios}.")
    placed = fit_into_environment(chain_references(slow, fast), environment_min, environment_max)
    if placed is None:
        raise ValueError("The fixed identification reference does not fit into the environment.")
    return placed, float(slow_phase["duration"])


# ---------------------------------------------------------------------------
# random twist profiles
# ---------------------------------------------------------------------------


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def _twist_peaks(v, omega, dt_fine):
    a_t = np.gradient(v, dt_fine)
    alpha = np.gradient(omega, dt_fine)
    lateral = np.abs(v * omega)
    return {
        "v_max": np.max(np.abs(v)),
        "a_max": np.max(np.hypot(a_t, lateral)),
        "a_max_lateral": np.max(lateral),
        "omega_max": np.max(np.abs(omega)),
        "alpha_max": np.max(np.abs(alpha)),
    }


def _scale_twist_into_limits(v, omega, dt_fine, limits, iterations: int = 50):
    """Shrink the speed and yaw-rate profiles until every limit holds.

    Speed alone sets v and the tangential acceleration, the yaw rate alone sets
    omega and alpha, and the two together set the lateral (and so the total)
    acceleration, so each is scaled by its own worst ratio first and both by
    the square root of the shared one after.
    """
    for _ in range(iterations):
        peaks = _twist_peaks(v, omega, dt_fine)
        ratios = {name: peaks[name] / limits[name] for name in limits}
        speed_ratio = max(ratios["v_max"], 1.0)
        yaw_ratio = max(ratios["omega_max"], ratios["alpha_max"], 1.0)
        if speed_ratio > 1.0 or yaw_ratio > 1.0:
            v, omega = v / speed_ratio, omega / yaw_ratio
            continue
        shared = max(ratios["a_max"], ratios["a_max_lateral"])
        if shared <= 1.0:
            return v, omega
        # Slightly past the square root so the loop cannot creep up on 1.
        factor = 1.001 * np.sqrt(shared)
        v, omega = v / factor, omega / factor
    raise RuntimeError("Twist scaling did not converge.")


def random_twist_reference(
    rng: np.random.Generator,
    total_time: float,
    dt: float,
    limits: dict[str, float],
    num_knots: int = 6,
    peak_speed_range: tuple[float, float] = (0.8, 2.5),
    peak_yaw_rate_range: tuple[float, float] = (0.5, 5.0),
    min_speed_fraction: float = 0.2,
    ramp_time: float = 0.6,
    oversample: int = 20,
) -> np.ndarray:
    """One random reference from smooth random forward-speed / yaw-rate profiles.

    Both profiles are natural cubic splines through ``num_knots`` random values
    (speed in ``[min_speed_fraction, 1] x`` a random peak, so it never stops in
    the interior and there is no turning on the spot; yaw rate in ``+/-`` a
    random peak), faded in and out over ``ramp_time`` so the reference starts
    and ends at rest, then scaled into ``limits`` and integrated on a grid
    ``oversample`` times finer than ``dt``.
    """
    from scipy.interpolate import CubicSpline

    num_steps = int(round(total_time / dt))
    fine = np.linspace(0.0, num_steps * dt, num_steps * oversample + 1)
    dt_fine = dt / oversample
    knots = np.linspace(0.0, fine[-1], num_knots)
    peak_speed = rng.uniform(*peak_speed_range)
    peak_yaw_rate = rng.uniform(*peak_yaw_rate_range)
    speed_profile = CubicSpline(knots, rng.uniform(min_speed_fraction, 1.0, num_knots) * peak_speed, bc_type="natural")
    yaw_profile = CubicSpline(knots, rng.uniform(-1.0, 1.0, num_knots) * peak_yaw_rate, bc_type="natural")
    envelope = _smoothstep(fine / ramp_time) * _smoothstep((fine[-1] - fine) / ramp_time)
    v = envelope * np.maximum(speed_profile(fine), min_speed_fraction * peak_speed)
    omega = envelope * yaw_profile(fine)
    v, omega = _scale_twist_into_limits(v, omega, dt_fine, limits)

    a_t = np.gradient(v, dt_fine)
    theta = rng.uniform(-np.pi, np.pi) + np.concatenate([[0.0], np.cumsum(0.5 * (omega[1:] + omega[:-1]) * dt_fine)])
    vx, vy = v * np.cos(theta), v * np.sin(theta)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (vx[1:] + vx[:-1]) * dt_fine)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (vy[1:] + vy[:-1]) * dt_fine)])
    ax = a_t * np.cos(theta) - v * omega * np.sin(theta)
    ay = a_t * np.sin(theta) + v * omega * np.cos(theta)
    return np.column_stack([x, y, theta, vx, vy, omega, ax, ay])[::oversample]


def random_twist_reference_set(
    num_references: int,
    total_time: float,
    dt: float,
    limits: dict[str, float],
    environment_min,
    environment_max,
    seed: int = 0,
    max_attempts_per_reference: int = 200,
    **profile_options,
) -> dict[str, np.ndarray]:
    """``num_references`` feasible random-twist references placed in the box.

    A draw that no rotation fits into the box is redrawn; a sampled reference
    that exceeds a limit after the ``dt`` subsampling (the scaling works on the
    fine grid) is redrawn too.
    """
    rng = np.random.default_rng(seed)
    references = {}
    for index in range(num_references):
        for _ in range(max_attempts_per_reference):
            states = random_twist_reference(rng, total_time, dt, limits, **profile_options)
            if not is_within_limits(states, dt, limits):
                continue
            placed = fit_into_environment(states, environment_min, environment_max, rng=rng)
            if placed is not None:
                references[f"random_twist_{index:03d}"] = placed
                break
        else:
            raise RuntimeError(f"No feasible random-twist reference in {max_attempts_per_reference} draws.")
    return references


# ---------------------------------------------------------------------------
# random B-splines (the designer's family without the FIM)
# ---------------------------------------------------------------------------


def random_walk_control_points(
    rng: np.random.Generator,
    position_basis,
    curve_length: float,
    environment_min,
    environment_max,
    max_turn: float = 0.5 * np.pi,
    margin: float = 0.1,
    max_attempts: int = 1000,
) -> np.ndarray:
    """A random control polygon whose B-spline is ``curve_length`` long, inside the box.

    The polygon walks in a random direction, turning by up to ``max_turn`` at
    every control point, and is then scaled about its centroid until the curve
    ``position_basis`` draws from it has the requested length (a curve that
    folds back is much shorter than its polygon, so the polygon length alone
    does not set the speed). A polygon that then does not fit the box is redrawn.
    """
    from wmr_simulator.trajectory_optimization.bspline import sampled_arc_length

    env_min = np.asarray(environment_min, dtype=float) + margin
    env_max = np.asarray(environment_max, dtype=float) - margin
    num_control_points = int(np.asarray(position_basis).shape[1])
    for _ in range(max_attempts):
        heading = rng.uniform(-np.pi, np.pi)
        points = [np.zeros(2)]
        for _ in range(num_control_points - 1):
            points.append(points[-1] + np.array([np.cos(heading), np.sin(heading)]))
            heading += rng.uniform(-max_turn, max_turn)
        points = np.asarray(points)
        points = (points - points.mean(axis=0)) * (curve_length / sampled_arc_length(position_basis, points))
        low, high = points.min(axis=0), points.max(axis=0)
        slack = (env_max - env_min) - (high - low)
        if np.all(slack >= 0.0):
            return points + (env_min + rng.uniform(0.0, 1.0, size=2) * slack - low)
    raise RuntimeError(f"No random control polygon for a {curve_length:.2f} m curve fits the box.")


def random_bspline_reference_set(
    problem,
    num_references: int,
    num_control_points: int,
    time_scaling: str = "s-curve",
    seed: int = 0,
    min_speed: float = 0.0,
    min_speed_fraction: float = 0.5,
    min_lateral_acceleration: float = 0.0,
    min_lateral_acceleration_fraction: float = 0.25,
    mean_speed_range: tuple[float, float] = (0.4, 1.1),
    max_turn: float = 0.5 * np.pi,
    num_steps: int = 200,
    learning_rate: float = 1e-2,
    constraint_smooth_max_beta: float = 10.0,
    limit_margin: float = 0.97,
    floor_fraction: float = 0.9,
    check_every: int = 10,
    max_rounds: int = 30,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Random designer-family references: ``(references, control_points, report)``.

    ``problem`` is a pipeline.ProblemDefinition whose ``sim_time`` is the set's
    duration. Each slot gets the designer's speed floors for its index
    (constraints.motion_floors_for_batch), a random-walk control polygon whose curve
    is as long as a random mean speed in ``mean_speed_range`` (or its floor, if
    higher) needs, and then up to ``num_steps`` of Adam on the constraint and
    tangent-floor terms alone -- the design objective minus the FIM, and no
    rollout. The limits in that loss are shrunk by ``limit_margin`` so the
    projection ends inside them.

    A slot stops moving as soon as it is inside the true limits, has no stalled
    tangent and reaches ``floor_fraction`` of its floors (checked every
    ``check_every`` steps), and a draw that already does is not optimized at
    all. Without that the projection shrinks every curve: the smooth max is
    slightly positive even inside the limits, and Adam's normalized steps walk
    down that gradient at full size for the whole budget (measured: mean speeds
    down to 0.07 m/s from a 0.4 m/s draw). A slot still over a true limit or
    stalled after the budget is redrawn, up to ``max_rounds`` times; one that
    is only short of its floors is kept, as the designer keeps its designs.
    """
    import jax
    import jax.numpy as jnp
    import optax

    from wmr_simulator.trajectory_optimization.bspline import (
        DEFAULT_MIN_TANGENT_FRACTION,
        BSplinePlan,
        clamp_control_points,
        compute_bspline_reference,
    )
    from wmr_simulator.trajectory_optimization.constraints import (
        constraint_loss_from_reference_states,
        constraint_weights,
        motion_floors_for_batch,
        motion_limits_from_robot_config,
    )
    from wmr_simulator.trajectory_optimization.objectives import DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE

    dt = float(problem.dt)
    total_time = float(problem.sim_time)
    plan = BSplinePlan(problem, num_control_points, time_scaling)
    true_limits = motion_limits(problem.robot_cfg)
    loss_limits = {
        name: (value * limit_margin if name in ("v_max", "a_max", "a_lat_max", "omega_max", "alpha_max") else value)
        for name, value in motion_limits_from_robot_config(problem.robot_cfg).items()
    }
    weights = constraint_weights()
    tolerance = DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE
    floors = motion_floors_for_batch(
        num_references, min_speed, min_speed_fraction, min_lateral_acceleration, min_lateral_acceleration_fraction
    )

    def loss(control_points, floor):
        control_points = clamp_control_points(problem, control_points)
        states = compute_bspline_reference(problem, plan, control_points, time_scaling=time_scaling)
        limits = {**loss_limits, "v_min": floor[0], "a_lat_min": floor[1]}
        constraint = constraint_loss_from_reference_states(
            states, dt, limits, weights, smooth_max_beta=constraint_smooth_max_beta
        )
        tangent = plan.stabilization_loss(
            control_points, min_tangent_fraction=DEFAULT_MIN_TANGENT_FRACTION, smooth_max_beta=constraint_smooth_max_beta
        )
        return (constraint + tangent) / tolerance

    optimizer = optax.adam(learning_rate)
    value_and_grad = jax.vmap(jax.value_and_grad(loss))

    @jax.jit
    def step(control_points, opt_state, floor, active):
        _, grads = value_and_grad(control_points, floor)
        updates, opt_state = optimizer.update(grads, opt_state, control_points)
        updated = jax.vmap(lambda points: clamp_control_points(problem, points))(
            optax.apply_updates(control_points, updates)
        )
        return jnp.where(active[:, None, None], updated, control_points), opt_state

    def assess(points, floor):
        """(hard-feasible, also meets floor_fraction of its floors, states)."""
        states = np.asarray(compute_bspline_reference(problem, plan, points, time_scaling=time_scaling), dtype=float)
        feasible = plan.tangent_diagnostics(points)["guarded_samples"] == 0 and is_within_limits(
            states, dt, true_limits
        )
        summary = summarize_motion(states, dt)
        floors_met = summary["v_mean"] >= floor_fraction * floor[0] and summary["a_lat_mean"] >= floor_fraction * floor[1]
        return feasible, feasible and floors_met, states

    rng = np.random.default_rng(seed)
    references: dict[str, np.ndarray] = {}
    control_points_out: dict[str, np.ndarray] = {}
    pending = list(range(num_references))
    draws = 0
    for _ in range(max_rounds):
        if not pending:
            break
        initial = []
        for index in pending:
            mean_speed = max(rng.uniform(*mean_speed_range), float(floors[index, 0]))
            initial.append(
                random_walk_control_points(
                    rng,
                    plan.B0,
                    mean_speed * total_time,
                    problem.environment_min,
                    problem.environment_max,
                    max_turn=max_turn,
                )
            )
        draws += len(pending)
        control_points = jnp.asarray(np.stack(initial), dtype=jnp.float32)
        control_points = jax.vmap(lambda points: clamp_control_points(problem, points))(control_points)
        batch_floors = jnp.asarray(floors[pending], dtype=jnp.float32)
        opt_state = optimizer.init(control_points)
        active = np.ones(len(pending), dtype=bool)
        for step_index in range(num_steps + 1):
            if step_index % check_every == 0:
                for slot, index in enumerate(pending):
                    if active[slot] and assess(control_points[slot], floors[index])[1]:
                        active[slot] = False
                if not active.any():
                    break
            if step_index < num_steps:
                control_points, opt_state = step(control_points, opt_state, batch_floors, jnp.asarray(active))

        still_pending = []
        for slot, index in enumerate(pending):
            feasible, _, states = assess(control_points[slot], floors[index])
            if not feasible:
                still_pending.append(index)
                continue
            name = f"random_bspline_{index:03d}"
            references[name] = states
            control_points_out[name] = np.asarray(control_points[slot], dtype=float)
        pending = still_pending
    if pending:
        raise RuntimeError(f"{len(pending)} random B-spline slots stayed infeasible after {max_rounds} rounds.")

    names = [f"random_bspline_{index:03d}" for index in range(num_references)]
    references = {name: references[name] for name in names}
    control_points_out = {name: control_points_out[name] for name in names}
    report = {
        "draws": draws,
        "acceptance_rate": num_references / draws,
        "floors": floors,
        "total_time": total_time,
    }
    return references, control_points_out, report


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def load_reference_set(directory: str | Path) -> dict[str, np.ndarray]:
    """``{stem: reference states}`` of every pickle in ``directory``; unlike
    pipeline.load_reference_states_exports the lengths may differ."""
    references = {}
    for path in sorted(Path(directory).glob("*.pkl")):
        with path.open("rb") as file:
            payload = pickle.load(file)
        states = payload["reference_states"] if isinstance(payload, dict) else payload
        references[path.stem] = np.asarray(states, dtype=float)
    if not references:
        raise ValueError(f"No reference pickles in {directory}.")
    return references


def reference_set_summary(references: dict[str, np.ndarray], dt: float, limits: dict[str, float] | None = None) -> dict:
    """Per-reference motion statistics plus the set's ranges, as plain floats.

    With ``limits`` each reference also gets its worst limit ratio.
    """
    per_reference = {}
    for name, states in references.items():
        summary = summarize_motion(states, dt)
        if limits is not None:
            summary["worst_limit_ratio"] = max(limit_ratios(states, dt, limits).values())
        per_reference[name] = {key: float(value) for key, value in summary.items()}
    keys = next(iter(per_reference.values())).keys()
    ranges = {
        key: [min(entry[key] for entry in per_reference.values()), max(entry[key] for entry in per_reference.values())]
        for key in keys
    }
    return {"num_references": len(references), "ranges": ranges, "references": per_reference}
