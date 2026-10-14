"""Clamped uniform cubic B-spline parametrization for the FIM trajectory
optimizer: control points are the decision variables, and the basis matrices
mapping them to curve value / derivatives on a fixed time grid are constant,
so evaluation is a matmul per derivative order per axis.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import BSpline

from wmr_simulator.trajectory_optimization.constraints import smooth_positive_max
from wmr_simulator.trajectory_optimization.parametrization import (
    normalize_phase_breaks,
    normalize_time_scaling,
    time_scaling_derivatives,
)


HIGHEST_PRECISION = jax.lax.Precision.HIGHEST

# Cubic: C^2, so position, velocity and acceleration are continuous.
DEFAULT_SPLINE_DEGREE = 3

# Floor on |dpos/ds|^2, as a fraction of the curve's own mean, below which the
# tangent is treated as absent (see _reference_from_derivatives).
TANGENT_FLOOR_FRACTION = 1e-4

# The same guard as a fraction of the rms of ``|dpos/ds|``.
GUARD_TANGENT_FRACTION = float(np.sqrt(TANGENT_FLOOR_FRACTION))

# Floor on |dpos/ds| that ``tangent_floor_loss`` holds the curve above, as a
# fraction of its rms; <= 0 disables the term.
DEFAULT_MIN_TANGENT_FRACTION = 0.03


def _reference_from_derivatives(
    position: jnp.ndarray,
    dpos_ds: jnp.ndarray,
    d2pos_ds2: jnp.ndarray,
    s_dot: jnp.ndarray,
    s_ddot: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble the [T, 8] reference-state matrix from curve derivatives."""
    velocity = dpos_ds * s_dot[:, None]
    acceleration = dpos_ds * s_ddot[:, None] + d2pos_ds2 * (s_dot[:, None] ** 2)

    # Heading and yaw rate are singular where the tangent vanishes. Floor the
    # denominator at a fraction of the mean tangent and give arctan2 a fixed
    # unit tangent below that floor (on the input, so no NaN gradient leaks).
    tangent_norm_sq = jnp.sum(dpos_ds**2, axis=1)
    tangent_floor = TANGENT_FLOOR_FRACTION * jnp.mean(tangent_norm_sq) + 1e-12
    fallback_tangent = jnp.array([1.0, 0.0], dtype=dpos_ds.dtype)
    safe_tangent = jnp.where(
        (tangent_norm_sq > tangent_floor)[:, None], dpos_ds, fallback_tangent
    )
    theta = jnp.arctan2(safe_tangent[:, 1], safe_tangent[:, 0])

    dtheta_ds = (
        dpos_ds[:, 0] * d2pos_ds2[:, 1] - dpos_ds[:, 1] * d2pos_ds2[:, 0]
    ) / (tangent_norm_sq + tangent_floor)
    omega = dtheta_ds * s_dot

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


def clamped_uniform_knots(num_control_points: int, degree: int = DEFAULT_SPLINE_DEGREE) -> np.ndarray:
    """Knot vector on [0, 1] with ``degree + 1`` repeats at each end (clamped:
    the curve starts at the first control point and ends at the last)."""
    if degree < 1:
        raise ValueError("degree must be >= 1.")
    if num_control_points < degree + 1:
        raise ValueError(
            f"A degree-{degree} B-spline needs at least {degree + 1} control points, "
            f"got {num_control_points}."
        )
    num_interior = num_control_points - degree - 1
    return np.concatenate(
        [
            np.zeros(degree + 1),
            np.linspace(0.0, 1.0, num_interior + 2)[1:-1],
            np.ones(degree + 1),
        ]
    )


def bspline_basis_matrices(
    s: np.ndarray, num_control_points: int, degree: int = DEFAULT_SPLINE_DEGREE
):
    """Constant ``[len(s), num_control_points]`` matrices mapping control points
    to curve value / d/ds / d2/ds2 at each sample (float64)."""
    knots = clamped_uniform_knots(num_control_points, degree)
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    matrices = []
    for order in (0, 1, 2):
        spline = BSpline(knots, np.eye(num_control_points), degree)
        if order:
            spline = spline.derivative(order)
        matrices.append(np.asarray(spline(s), dtype=np.float64))
    return tuple(matrices)


def initial_line_control_points(problem, num_control_points: int) -> jnp.ndarray:
    """``num_control_points`` points evenly spaced along the start-goal line."""
    line_samples = np.linspace(0.0, 1.0, num_control_points)[:, None]
    start = problem.start[:2][None, :]
    goal = problem.goal[:2][None, :]
    return jnp.asarray(start + line_samples * (goal - start), dtype=jnp.float32)


def start_heading_unit(problem) -> jnp.ndarray:
    start_theta = jnp.asarray(problem.start[2], dtype=jnp.float32)
    return jnp.stack([jnp.cos(start_theta), jnp.sin(start_theta)])


def max_start_heading_distance(problem) -> jnp.ndarray:
    start = jnp.asarray(problem.start[:2], dtype=jnp.float32)
    heading = start_heading_unit(problem)
    env_min = jnp.asarray(problem.environment_min, dtype=jnp.float32)
    env_max = jnp.asarray(problem.environment_max, dtype=jnp.float32)
    safe_heading = jnp.where(jnp.abs(heading) > 1e-6, heading, 1.0)
    positive_limits = jnp.where(heading > 1e-6, (env_max - start) / safe_heading, jnp.inf)
    negative_limits = jnp.where(heading < -1e-6, (env_min - start) / safe_heading, jnp.inf)
    return jnp.maximum(jnp.min(jnp.minimum(positive_limits, negative_limits)), 0.0)


def _start_heading_control_point(problem, control_point: jnp.ndarray) -> jnp.ndarray:
    """Projection of the second control point onto the start-heading ray
    (clipped to the environment box), which pins the start heading."""
    start = jnp.asarray(problem.start[:2], dtype=jnp.float32)
    heading = start_heading_unit(problem)
    raw_distance = jnp.dot(jnp.asarray(control_point, dtype=jnp.float32) - start, heading)
    distance = jnp.clip(raw_distance, 0.0, max_start_heading_distance(problem))
    return start + distance * heading


def clamp_control_points(
    problem,
    control_points: jnp.ndarray,
    pinned_positions=None,
    pin_start_heading: bool = False,
) -> jnp.ndarray:
    """Clip control points to the environment box (which keeps the curve
    inside it) and write the pinned ones back.

    ``pinned_positions`` maps a control-point index to its fixed ``[x, y]``;
    pinned points receive no gradient.
    """
    control_points = jnp.asarray(control_points, dtype=jnp.float32)[:, :2]
    env_min = jnp.asarray(problem.environment_min, dtype=jnp.float32)
    env_max = jnp.asarray(problem.environment_max, dtype=jnp.float32)
    clamped = jnp.clip(control_points, env_min, env_max)
    if pin_start_heading:
        clamped = clamped.at[1].set(_start_heading_control_point(problem, clamped[1]))
    for index, position in (pinned_positions or {}).items():
        clamped = clamped.at[index].set(jnp.asarray(position[:2], dtype=jnp.float32))
    return clamped


def rms_tangent_norm(reference_basis: jnp.ndarray, control_points: jnp.ndarray) -> jnp.ndarray:
    """``sqrt(mean(|dpos/ds|^2))``, the curve's own scale."""
    tangent = jnp.matmul(reference_basis, control_points, precision=HIGHEST_PRECISION)
    return jnp.sqrt(jnp.mean(jnp.sum(tangent**2, axis=1)) + 1e-12)


def tangent_floor_loss(
    tangent_basis: jnp.ndarray,
    reference_basis: jnp.ndarray,
    control_points: jnp.ndarray,
    min_tangent_fraction: float = DEFAULT_MIN_TANGENT_FRACTION,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    """Squared smooth-max fractional shortfall of ``|dpos/ds|`` below
    ``min_tangent_fraction * rms(|dpos/ds|)`` (same form as the motion limits).

    The floor is ``stop_gradient``-wrapped. ``tangent_basis`` is sampled
    uniformly in ``s``; ``reference_basis`` is the time-grid basis the rms is
    taken on.
    """
    if min_tangent_fraction <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    control_points = jnp.asarray(control_points, dtype=jnp.float32)
    tangent = jnp.matmul(tangent_basis, control_points, precision=HIGHEST_PRECISION)
    tangent_norm = jnp.sqrt(jnp.sum(tangent**2, axis=1) + 1e-8)
    floor = jax.lax.stop_gradient(
        min_tangent_fraction * rms_tangent_norm(reference_basis, control_points)
    )
    # Keep 0/0 out of the gradient for a degenerate curve.
    shortfall = (floor - tangent_norm) / jnp.maximum(floor, 1e-6)
    return smooth_positive_max(shortfall, beta=smooth_max_beta) ** 2


class BSplinePlan:
    """Precomputed evaluation matrices of a clamped uniform B-spline on the
    problem's time grid and time scaling (one matmul per derivative order)."""

    def __init__(self, problem, num_control_points: int, time_scaling: str,
                 degree: int = DEFAULT_SPLINE_DEGREE, phase_breaks=()):
        self.num_control_points = num_control_points
        self.degree = degree
        self.time_scaling = normalize_time_scaling(time_scaling)
        # Motion phases (parametrization.time_scaling_derivatives); () is one s-curve.
        self.phase_breaks = normalize_phase_breaks(phase_breaks)

        time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
        total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
        s, _, _ = time_scaling_derivatives(
            time_grid, total_time, time_scaling=self.time_scaling, phase_breaks=self.phase_breaks
        )

        B0, B1, B2 = bspline_basis_matrices(np.asarray(s, dtype=np.float64),
                                            num_control_points, degree)
        self.B0 = jnp.asarray(B0, dtype=jnp.float32)
        self.B1 = jnp.asarray(B1, dtype=jnp.float32)
        self.B2 = jnp.asarray(B2, dtype=jnp.float32)

        # d/ds basis on a uniform s grid, for the tangent-floor loss.
        _, tangent_B1, _ = bspline_basis_matrices(
            np.linspace(0.0, 1.0, int(self.B0.shape[0])), num_control_points, degree
        )
        self.tangent_B1 = jnp.asarray(tangent_B1, dtype=jnp.float32)

    def stabilization_loss(
        self,
        control_points: jnp.ndarray,
        min_tangent_fraction: float = DEFAULT_MIN_TANGENT_FRACTION,
        smooth_max_beta: float = 20.0,
    ) -> jnp.ndarray:
        """The tangent-floor penalty; see :func:`tangent_floor_loss`."""
        return tangent_floor_loss(
            self.tangent_B1,
            self.B1,
            control_points,
            min_tangent_fraction=min_tangent_fraction,
            smooth_max_beta=smooth_max_beta,
        )

    def tangent_diagnostics(self, control_points: jnp.ndarray) -> dict:
        """Tangent statistics on the time grid; ``guarded_samples`` counts the
        samples whose heading is the fallback tangent."""
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        tangent = jnp.matmul(self.B1, control_points, precision=HIGHEST_PRECISION)
        tangent_norm = jnp.sqrt(jnp.sum(tangent**2, axis=1) + 1e-12)
        rms = rms_tangent_norm(self.B1, control_points)
        threshold = GUARD_TANGENT_FRACTION * rms
        return {
            "min_tangent_norm": float(jnp.min(tangent_norm)),
            "rms_tangent_norm": float(rms),
            "guard_threshold": float(threshold),
            "guarded_samples": int(jnp.sum(tangent_norm <= threshold)),
            "num_samples": int(tangent_norm.shape[0]),
        }

    def evaluate(self, control_points: jnp.ndarray):
        # precision=HIGHEST: the default reduced-precision matmul is too coarse.
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        matmul = lambda basis: jnp.matmul(basis, control_points, precision=HIGHEST_PRECISION)
        return matmul(self.B0), matmul(self.B1), matmul(self.B2)


def compute_bspline_reference(
    problem,
    plan: BSplinePlan,
    control_points: jnp.ndarray,
    time_scaling: str | None = "s_curve",
) -> jnp.ndarray:
    position, dpos_ds, d2pos_ds2 = plan.evaluate(control_points)

    time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
    total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
    _, s_dot, s_ddot = time_scaling_derivatives(
        time_grid, total_time, time_scaling=normalize_time_scaling(time_scaling),
        phase_breaks=plan.phase_breaks,
    )
    return _reference_from_derivatives(position, dpos_ds, d2pos_ds2, s_dot, s_ddot)


def sampled_arc_length(position_basis, control_points) -> float:
    """Length of the curve ``position_basis`` samples from ``control_points``."""
    positions = np.asarray(position_basis, dtype=float) @ np.asarray(control_points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def sampled_motion_peaks(position_basis, control_points, dt: float) -> dict[str, float]:
    """Peak speed / yaw rate / yaw acceleration of the sampled curve (finite differences)."""
    positions = np.asarray(position_basis, dtype=float) @ np.asarray(control_points, dtype=float)
    velocity = np.gradient(positions, dt, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    omega = np.gradient(heading, dt)
    return {
        "v_peak": float(np.max(speed)),
        "omega_peak": float(np.max(np.abs(omega))),
        "alpha_peak": float(np.max(np.abs(np.gradient(omega, dt)))),
    }


def stretch_control_points_to_arc(
    position_basis,
    control_points,
    target_arc: float,
    environment_min,
    environment_max,
    dt: float | None = None,
    limits: dict | None = None,
    num_iterations: int = 40,
):
    """Bow the interior control points out until the curve is ``target_arc`` long.

    The displacement is a half sine along the control index (zero at both
    ends); its amplitude is bisected on the sampled curve, capped by ``limits``
    (``v_max``/``omega_max``/``alpha_max``, with ``dt``) and clipped to the box.
    """
    control_points = np.asarray(control_points, dtype=float)
    environment_min = np.asarray(environment_min, dtype=float)
    environment_max = np.asarray(environment_max, dtype=float)
    num_points = control_points.shape[0]
    if target_arc <= 0.0 or num_points < 3:
        return control_points

    span = control_points[-1] - control_points[0]
    span_norm = float(np.linalg.norm(span))
    if span_norm < 1e-6:
        normal = np.array([0.0, 1.0])
    else:
        direction = span / span_norm
        normal = np.array([-direction[1], direction[0]])

    # One half period of a sine: zero at the endpoints, peak in the middle.
    shape = np.sin(np.pi * np.linspace(0.0, 1.0, num_points))

    def stretched(amplitude: float):
        return np.clip(
            control_points + amplitude * shape[:, None] * normal[None, :],
            environment_min,
            environment_max,
        )

    def bisect(is_below, high: float) -> float:
        low = 0.0
        for _ in range(int(num_iterations)):
            mid = 0.5 * (low + high)
            if is_below(mid):
                low = mid
            else:
                high = mid
        return high

    reach = float(np.linalg.norm(environment_max - environment_min))
    if sampled_arc_length(position_basis, control_points) >= target_arc:
        return control_points

    if sampled_arc_length(position_basis, stretched(reach)) < target_arc:
        amplitude = reach
    else:
        amplitude = bisect(
            lambda a: sampled_arc_length(position_basis, stretched(a)) < target_arc, reach
        )

    if limits and dt:
        def feasible(amplitude_value: float) -> bool:
            peaks = sampled_motion_peaks(position_basis, stretched(amplitude_value), float(dt))
            return (
                peaks["v_peak"] <= float(limits.get("v_max", np.inf))
                and peaks["omega_peak"] <= float(limits.get("omega_max", np.inf))
                and peaks["alpha_peak"] <= float(limits.get("alpha_max", np.inf))
            )

        if not feasible(amplitude):
            amplitude = min(amplitude, bisect(feasible, amplitude))

    return stretched(amplitude)
