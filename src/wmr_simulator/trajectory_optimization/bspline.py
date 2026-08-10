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
    normalize_time_scaling,
    time_scaling_derivatives,
)


HIGHEST_PRECISION = jax.lax.Precision.HIGHEST

# Cubic: C^2, which is exactly what the reference state needs (position,
# velocity, acceleration all continuous).
DEFAULT_SPLINE_DEGREE = 3

# Floor on |dpos/ds|^2, as a fraction of the curve's own mean, below which the
# tangent is treated as numerically absent (see _reference_from_derivatives).
# It has to be small enough not to bias the yaw rate of ordinary curves -- 1e-2
# already cost 2.5% of omega on a plain circle -- and large enough to swallow
# the singular neighbourhood. At 1e-4 a sample carrying the mean tangent is
# damped by 0.01%, while the floor still dominates by two orders of magnitude at
# the stalled configurations that produced the 1e4 gradient spikes.
TANGENT_FLOOR_FRACTION = 1e-4

# The guard above, restated as the floor on ``|dpos/ds|`` itself: it fires at
# ``sqrt(TANGENT_FLOOR_FRACTION) * rms(|dpos/ds|)``, i.e. at 1% of the curve's
# own rms tangent. Named so the loss below can be defined against it.
GUARD_TANGENT_FRACTION = float(np.sqrt(TANGENT_FLOOR_FRACTION))

# Floor on |dpos/ds| that ``tangent_floor_loss`` holds the curve above, as a
# fraction of the curve's own rms tangent; <= 0 disables the term. Below the
# floor the curve has effectively stalled: heading and yaw rate are read off the
# tangent, so a vanishing one is not a slow robot, it is an undefined pose.
#
# Relative, and to the same quantity the guard scales by, for two reasons. It
# makes the margin over the guard a constant of the code rather than of the
# problem -- 0.025 against GUARD_TANGENT_FRACTION's 0.01 is 2.5x at every scale,
# so a larger environment or a longer path never quietly lets the optimizer
# reach the fallback tangent. And it is the more faithful statement of the
# pathology: what breaks the reference is one stretch of curve being far slower
# than the rest, not the curve being slow in absolute metres. (A uniformly tiny
# curve is scale-invariant to this term and is left to the FIM, which hates it.)
#
# 0.025 reproduces the absolute 0.15 that was tuned by eye on
# ``problems/pololu_gains.yaml``, whose designs sit at rms |dpos/ds| ~ 6.3
# (spread 3.2-9.3 over ten trajectories, so the effective floor is now
# 0.08-0.23 per design rather than one number for all of them).
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

    # Heading and yaw rate come from the tangent, and both are singular where
    # the tangent vanishes: arctan2(0, 0) has a NaN derivative, and
    # dtheta/ds = cross / |dpos/ds|^2 differentiates like 1/|dpos/ds|^3. Taking
    # the derivative w.r.t. s rather than t already removed the s_dot -> 0 half
    # of this (the s-curve's endpoints), but the |dpos/ds| -> 0 half is still
    # reachable: every control point is free in gain-tuning mode, and the
    # environment clip can collapse neighbouring ones onto the same box edge.
    # Measured on a stalled curve, |dpos/ds| = 4e-3 turned a typical gradient of
    # 1e-2 into 2.3e2, and an exactly coincident pair produced NaN.
    #
    # So: floor the denominator at a fraction of the curve's own mean tangent
    # (absolute epsilons cannot work -- |dpos/ds| carries the curve's length in
    # metres per unit s), and hand arctan2 a fixed unit tangent wherever the
    # real one is below that floor. The substitution has to happen on the
    # *input*: a jnp.where on the output still evaluates arctan2 at the singular
    # point and leaks its NaN back through the cotangent.
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
    """Knot vector on [0, 1] with ``degree + 1`` repeats at each end.

    The repeats are what make the spline *clamped*: the curve then starts
    exactly at the first control point and ends exactly at the last, which is
    how a start/goal pin is expressed (fix those two control points) and why
    the start tangent is simply along ``P1 - P0``.
    """
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
    to curve value / d/ds / d2/ds2 at each sample (float64).

    Column ``i`` is the ``i``-th basis function sampled on ``s``, so the whole
    curve is ``B0 @ control_points`` -- one matmul, no per-segment bookkeeping.
    """
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
    """``num_control_points`` points evenly spaced along the start-goal line.

    Because the basis is a partition of unity and the spline is clamped, control
    points on a straight line reproduce that straight line exactly, so this is a
    zero-curvature initialization.
    """
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
    """Projection of the second control point onto the start-heading ray.

    A clamped spline's tangent at ``s = 0`` is along ``P1 - P0``, so pinning the
    start heading is exactly "keep P1 on the ray from the start pose". The
    projection keeps the along-ray distance as the free (and differentiable)
    coordinate and discards the perpendicular one; it is clipped to the
    environment box along that ray so it cannot push the curve straight out.
    """
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
    """Clip control points to the environment box, then write the pinned ones
    back over the result.

    ``pinned_positions`` maps a control-point index to its fixed ``[x, y]``.
    Writing the values in here (rather than baking them into the basis) keeps
    the basis independent of the problem's start/goal, and makes the pin
    gradient-safe: ``.at[i].set(constant)`` under ``jax.grad`` zeroes the
    incoming gradient, so a pinned control point can never be optimized even
    though it travels in the same array as the free ones.

    A B-spline lies in the convex hull of its control points, so control
    points inside the box put the whole curve inside the box.
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
    """``sqrt(mean(|dpos/ds|^2))`` -- the curve's own scale, and exactly the
    quantity ``_reference_from_derivatives`` floors against, so a floor stated
    as a fraction of it keeps a fixed ratio to the guard at any problem size."""
    tangent = jnp.matmul(reference_basis, control_points, precision=HIGHEST_PRECISION)
    return jnp.sqrt(jnp.mean(jnp.sum(tangent**2, axis=1)) + 1e-12)


def tangent_floor_loss(
    tangent_basis: jnp.ndarray,
    reference_basis: jnp.ndarray,
    control_points: jnp.ndarray,
    min_tangent_fraction: float = DEFAULT_MIN_TANGENT_FRACTION,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    """Squared smooth-max *fractional* shortfall of ``|dpos/ds|`` below
    ``min_tangent_fraction * rms(|dpos/ds|)`` -- the same
    ``smooth_positive_max(g)**2`` form as every motion limit in
    ``constraints.py``, because this is one: the curve has to keep moving.

    Writing it that way (rather than as a mean over samples) is what makes it
    commensurable with the rest of the objective, and it is what gives the term
    any authority. A stall occupies ~20 of 101 samples, so a mean dilutes a
    93% shortfall to a loss of 0.11 -- against a FIM term that pays 2.0 log
    units for the stall (measured, 6cp design 0: ``trace(FIM^-1)`` 7.4e-5
    stalled against 5.6e-4 for the same route spread out). The smooth max keeps
    the 0.93, and the pipeline then divides by the same
    ``constraint_violation_tolerance`` as the limits, so a fractional tangent
    shortfall costs exactly what an equal fractional over-limit costs.

    The shortfall is measured against the floor itself, so the term saturates at
    ~1 for a fully stalled curve and one fraction keeps its meaning across
    problems.

    The floor is ``stop_gradient``-wrapped: it sets the bar, and must not become
    something the optimizer can push on. Without that, lowering the whole
    curve's rms tangent -- a shorter, duller path -- would satisfy the term as
    readily as fixing the stall, which is the opposite of the intent.

    ``tangent_basis`` must be sampled uniformly in ``s``, *not* on the time
    grid: under the s-curve scaling the time samples cluster at ``s = 0`` and
    ``s = 1``, which is exactly where a stall does not happen.
    ``reference_basis`` is the time-grid one, because the guard's own mean is
    taken there.

    This is complementary to ``TANGENT_FLOOR_FRACTION`` in
    :func:`_reference_from_derivatives`: that guard keeps the gradient finite at
    a stalled curve, while this loss keeps the optimizer 2.5x away from ever
    reaching it.
    """
    if min_tangent_fraction <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    control_points = jnp.asarray(control_points, dtype=jnp.float32)
    tangent = jnp.matmul(tangent_basis, control_points, precision=HIGHEST_PRECISION)
    tangent_norm = jnp.sqrt(jnp.sum(tangent**2, axis=1) + 1e-8)
    floor = jax.lax.stop_gradient(
        min_tangent_fraction * rms_tangent_norm(reference_basis, control_points)
    )
    # A degenerate curve has no scale to be relative to, so the floor collapses
    # with it; the clamp only keeps 0/0 out of the gradient. Such a curve is not
    # a *stall* -- nothing about it is slower than the rest of it -- and it is
    # the FIM's problem, not this term's.
    shortfall = (floor - tangent_norm) / jnp.maximum(floor, 1e-6)
    return smooth_positive_max(shortfall, beta=smooth_max_beta) ** 2


class BSplinePlan:
    """Precomputed, control-point-independent evaluation matrices for a clamped
    uniform B-spline on a given time grid and time scaling.

    Construction samples the basis once in float64; ``evaluate`` is then one
    matmul per derivative order. Nothing about the problem's start/goal enters,
    so one plan serves any of them.

    ``problem`` is read only for its time grid; everything else about the curve
    comes from the arguments and, at evaluation time, the control points.
    """

    def __init__(self, problem, num_control_points: int, time_scaling: str,
                 degree: int = DEFAULT_SPLINE_DEGREE):
        self.num_control_points = num_control_points
        self.degree = degree
        self.time_scaling = normalize_time_scaling(time_scaling)

        time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
        total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
        s, _, _ = time_scaling_derivatives(time_grid, total_time, time_scaling=self.time_scaling)

        B0, B1, B2 = bspline_basis_matrices(np.asarray(s, dtype=np.float64),
                                            num_control_points, degree)
        self.B0 = jnp.asarray(B0, dtype=jnp.float32)
        self.B1 = jnp.asarray(B1, dtype=jnp.float32)
        self.B2 = jnp.asarray(B2, dtype=jnp.float32)

        # Second d/ds basis, on a *uniform* s grid, for the tangent-floor loss
        # only: the time grid above is warped by the time scaling, and the
        # s-curve puts most of its samples at the two ends, leaving the middle
        # of the curve -- where the control points bunch and stall it -- barely
        # observed. Same sample count, so the term costs one more matmul.
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
        """The basis's own well-posedness term; see :func:`tangent_floor_loss`.

        The fraction is an argument rather than plan state because the plan is
        cached per ``(num_control_points, time_scaling)`` and it is a property
        of the run, not of the basis.
        """
        return tangent_floor_loss(
            self.tangent_B1,
            self.B1,
            control_points,
            min_tangent_fraction=min_tangent_fraction,
            smooth_max_beta=smooth_max_beta,
        )

    def tangent_diagnostics(self, control_points: jnp.ndarray) -> dict:
        """Where this curve sits relative to the guard, in plain numbers.

        Reported on the *time* grid -- the samples that actually become the
        exported reference states, and the grid the guard's own mean is taken
        on -- so ``guarded_samples`` is the count of exported poses whose
        heading is the fallback tangent rather than the curve's.
        """
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
        # precision=HIGHEST: XLA's default lowers these matmuls to a reduced
        # -mantissa path, which costs ~2^-13 relative here. That is far too
        # coarse -- it left 3.9e-3 rad/s of phantom yaw rate on a dead-straight
        # control polygon, pushed the curve outside its own convex hull, and
        # fed noise straight into a FIM that is sensitive to it.
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
        time_grid, total_time, time_scaling=normalize_time_scaling(time_scaling)
    )
    return _reference_from_derivatives(position, dpos_ds, d2pos_ds2, s_dot, s_ddot)
