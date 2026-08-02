"""Piecewise-quintic minimum-acceleration spline: the trajectory
parametrization behind the FIM trajectory-optimization pipeline.

The central design point: the coefficient map from decision variables ``z``
(waypoints, plus a start-tangent magnitude in identification mode) to the
per-segment quintic coefficients ``C`` is AFFINE and CONSTANT, ``C = A z + b``,
because the objective (minimum integrated squared acceleration) is quadratic
in ``C`` and every continuity/interpolation/boundary constraint is linear in
``(C, z)``. The KKT system for that constrained QP is solved exactly ONCE per
``(num_segments, objective_mode, time_scaling, heading_knots)`` combination, in float64
NumPy, at construction. The constant evaluation matrices ``E0 = B0 @ A`` etc.
(``B0``/``B1``/``B2`` being the position/velocity/acceleration sampling
matrices on the fixed problem time grid) are precomputed too, so the hot path
-- everything under ``jax.grad``/``jax.jit`` -- is three constant matmuls per
axis. There is no iterative solver, no implicit differentiation, and no
linear solve inside the differentiated objective.

Decision variables are waypoints ``[x, y]``, or ``[x, y, theta]`` when
headings are pinned (``heading_knots``). A pinned heading enters ``z`` as the
tangent *components* ``m*cos(theta)``, ``m*sin(theta)``, computed in JAX, so the
trigonometry never touches the constraint matrix and the map stays affine even
when the headings themselves are optimized.
"""

import numpy as np
import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.parametrization import (
    _reference_from_derivatives,
    normalize_time_scaling,
    time_scaling_derivatives,
)

def initial_line_waypoints(problem, num_segments: int) -> jnp.ndarray:
    """``num_segments + 1`` points evenly spaced along the start-goal line."""
    line_samples = np.linspace(0.0, 1.0, num_segments + 1)[:, None]
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


def clamp_waypoints(problem, waypoints: jnp.ndarray, pin_start_and_goal: bool) -> jnp.ndarray:
    """Clip waypoint positions to the environment box, and (identification mode)
    pin the first and last to the problem's start and goal.

    Only the [x, y] columns are touched; a heading column passes through
    untouched, since an angle has no environment bound.

    Note this bounds the *waypoints*, not the curve: unlike a Bezier's convex
    hull, a spline can bulge outside the box between two in-box knots. That is
    acceptable here (the box is a lab-area soft bound, not a safety constraint).
    """
    waypoints = jnp.asarray(waypoints, dtype=jnp.float32)
    env_min = jnp.asarray(problem.environment_min, dtype=jnp.float32)
    env_max = jnp.asarray(problem.environment_max, dtype=jnp.float32)
    positions = jnp.clip(waypoints[:, :2], env_min, env_max)
    if pin_start_and_goal:
        positions = positions.at[0].set(jnp.asarray(problem.start[:2], dtype=jnp.float32))
        positions = positions.at[-1].set(jnp.asarray(problem.goal[:2], dtype=jnp.float32))
    return jnp.concatenate([positions, waypoints[:, 2:]], axis=1)


# Mirrors pipeline.OBJECTIVE_MODE_GAIN_TUNING. Duplicated as a literal (rather
# than imported) to avoid a pipeline.py <-> quintic_spline.py import cycle --
# pipeline.py imports this module.
_OBJECTIVE_MODE_GAIN_TUNING = "gain-tuning"

# eps * (integrated jerk term), relative to trace(H)/n, added so the KKT
# matrix is safely invertible: the pure curvature Gram is singular on affine
# segments (p'' = 0 pointwise).
_JERK_REGULARIZATION_RELATIVE = 1e-6
_KKT_RESIDUAL_TOLERANCE = 1e-6

# Quintic monomial basis [1, u, u^2, u^3, u^4, u^5] on the local coordinate
# u = (s - s_j) / h in [0, 1]; rows below are the basis (or its 1st/2nd
# derivative) evaluated at u = 0 or u = 1.
_C0_START = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_C0_END = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
_C1_START = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_C1_END = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
_C2_START = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
_C2_END = np.array([0.0, 0.0, 2.0, 6.0, 12.0, 20.0])


def quintic_num_segments(num_control_points: int, objective_mode: str) -> int:
    """Recovers the segment count S from a ``control_points`` array.

    The layout is one row per knot, [w_0, ..., w_S], in both objective modes;
    identification mode simply pins w_0 to the problem start and w_S to the
    goal instead of leaving them free. The start tangent no longer needs a
    marker row -- it is a heading constraint like any other now.
    """
    return num_control_points - 1


def initial_quintic_waypoints(problem, num_segments: int, objective_mode: str) -> jnp.ndarray:
    """Uniform line of ``num_segments + 1`` knots from start to goal, with a
    heading column seeded from the straight-line direction so that turning
    heading constraints on does not start from a degenerate tangent."""
    positions = initial_line_waypoints(problem, num_segments)
    spans = jnp.concatenate(
        [positions[1:2] - positions[0:1], positions[1:] - positions[:-1]], axis=0
    )
    headings = jnp.arctan2(spans[:, 1], spans[:, 0])
    return jnp.concatenate([positions, headings[:, None]], axis=1)


def _chord_magnitudes(positions: jnp.ndarray) -> jnp.ndarray:
    """Tangent magnitude per knot, from the local chord length.

    A pinned heading fixes the tangent *direction*; the spline still needs a
    magnitude for it. Taking half the distance between a knot's neighbours (the
    usual Catmull-Rom choice, one-sided at the ends) scales the tangent with the
    local waypoint spacing, so a heading constraint does not distort the curve
    just because the waypoints are far apart. It is a smooth function of the
    waypoint positions, so it stays differentiable -- and it lives here rather
    than in the constraint matrix, which is what keeps that matrix constant.
    """
    padded = jnp.concatenate([positions[:1], positions, positions[-1:]], axis=0)
    spans = padded[2:] - padded[:-2]
    return 0.5 * jnp.sqrt(jnp.sum(spans**2, axis=1) + 1e-8)


def _z_from_control_points(
    problem,
    control_points: jnp.ndarray,
    objective_mode: str,
    heading_knots: tuple[int, ...] = (),
):
    """Splits a ``control_points`` array into the per-axis decision vectors the
    affine map (folded into E0/E1/E2) was built for.

    ``control_points`` is ``[N, 2]`` (positions only) or ``[N, 3]``
    (``[x, y, theta]``); the heading column is read only for the knots listed in
    ``heading_knots``. The map itself is identical for both axes -- only this
    extraction differs, via the tangent projection (x: cos, y: sin).
    """
    control_points = jnp.asarray(control_points, dtype=jnp.float32)
    positions = control_points[:, :2]
    heading_knots = tuple(sorted(set(heading_knots)))

    tangent_x, tangent_y = [], []
    if heading_knots:
        magnitudes = _chord_magnitudes(positions)
        for knot in heading_knots:
            if knot == 0:
                # The start heading is always the problem's, never a decision
                # variable: it is how the robot is physically placed.
                direction = start_heading_unit(problem)
                magnitude = jnp.clip(magnitudes[0], 0.0, max_start_heading_distance(problem))
            else:
                theta = control_points[knot, 2]
                direction = jnp.stack([jnp.cos(theta), jnp.sin(theta)])
                magnitude = magnitudes[knot]
            tangent_x.append((magnitude * direction[0])[None])
            tangent_y.append((magnitude * direction[1])[None])

    if objective_mode == _OBJECTIVE_MODE_GAIN_TUNING:
        free_x, free_y = positions[:, 0], positions[:, 1]
    else:
        free_x, free_y = positions[1:-1, 0], positions[1:-1, 1]
    z_x = jnp.concatenate(tangent_x + [free_x]) if tangent_x else free_x
    z_y = jnp.concatenate(tangent_y + [free_y]) if tangent_y else free_y
    return z_x, z_y


def _gram_matrix(derivative_order: int, h: float) -> np.ndarray:
    """Gram matrix, over local u in [0, 1], of the `derivative_order`-th
    derivative of the monomial basis [1, u, ..., u^5], scaled for the
    s = s_j + h*u substitution (see module docstring)."""
    idx = np.arange(6, dtype=np.float64)
    if derivative_order == 2:
        coeff = idx * (idx - 1.0)
    elif derivative_order == 3:
        coeff = idx * (idx - 1.0) * (idx - 2.0)
    else:
        raise ValueError(derivative_order)
    exponents = idx - derivative_order
    denom = exponents[:, None] + exponents[None, :] + 1.0
    mask = (coeff[:, None] != 0.0) & (coeff[None, :] != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gram = np.where(mask, coeff[:, None] * coeff[None, :] / denom, 0.0)
    h_power = 2 * derivative_order - 1
    return gram / (h ** h_power)


def _build_linear_system(num_segments: int, objective_mode: str, heading_knots: tuple[int, ...] = ()):
    """Assembles the constraint system G C = D_z z + start_sel*start + goal_sel*goal
    (see module docstring): interpolation at every segment boundary, C1/C2
    continuity at interior knots, natural-spline curvature=0 at both ends,
    and a pinned tangent at every knot in ``heading_knots``.

    A pinned tangent is ``p'(s_j) = h * z[tangent_slot_j]``, where the z entry
    holds the tangent *component* for this axis -- ``m_j*cos(theta_j)`` for x,
    ``m_j*sin(theta_j)`` for y. The trigonometry therefore lives in
    ``_z_from_control_points`` (differentiable JAX), never in this matrix, so
    the coefficient map stays affine and identical for both axes even when the
    headings themselves are decision variables.

    Knot 0 always carries a tangent constraint in identification mode (the
    start heading is fixed by the problem); ``heading_knots`` may add more.
    """
    num_segments_S = num_segments
    h = 1.0 / num_segments_S
    gain_tuning = objective_mode == _OBJECTIVE_MODE_GAIN_TUNING
    heading_knots = tuple(sorted(set(heading_knots)))
    for knot in heading_knots:
        if not 0 <= knot <= num_segments_S:
            raise ValueError(f"heading knot {knot} outside [0, {num_segments_S}]")
    # z layout: [tangent components (one per heading knot), free waypoints].
    # Identification mode pins knot 0 / knot S positions, so only the interior
    # ones are free; gain-tuning mode has every waypoint free.
    num_tangents = len(heading_knots)
    num_free_waypoints = (num_segments_S + 1) if gain_tuning else (num_segments_S - 1)
    dim_z = num_tangents + num_free_waypoints
    tangent_column = {knot: index for index, knot in enumerate(heading_knots)}

    rows_G, rows_Dz, start_sel, goal_sel = [], [], [], []

    def waypoint_slot(waypoint_index: int):
        if gain_tuning:
            return ("z", num_tangents + waypoint_index)
        if waypoint_index == 0:
            return ("start",)
        if waypoint_index == num_segments_S:
            return ("goal",)
        # Free interior waypoints follow the tangent block: w_i -> z[num_tangents + i - 1].
        return ("z", num_tangents + waypoint_index - 1)

    def add_row(coeff_by_segment, slot=None, z_extra=None):
        g_row = np.zeros(6 * num_segments_S)
        for segment_index, coeffs in coeff_by_segment:
            g_row[6 * segment_index: 6 * segment_index + 6] += coeffs
        dz_row = np.zeros(dim_z)
        s_sel = g_sel = 0.0
        if slot is not None:
            kind = slot[0]
            if kind == "z":
                dz_row[slot[1]] += 1.0
            elif kind == "start":
                s_sel = 1.0
            elif kind == "goal":
                g_sel = 1.0
        if z_extra is not None:
            col, value = z_extra
            dz_row[col] += value
        rows_G.append(g_row)
        rows_Dz.append(dz_row)
        start_sel.append(s_sel)
        goal_sel.append(g_sel)

    # Interpolation: each segment hits its two bounding waypoints. Position
    # continuity across segments follows for free (both rows referencing a
    # shared interior waypoint route to the same z column / constant).
    for j in range(num_segments_S):
        add_row([(j, _C0_START)], waypoint_slot(j))
        add_row([(j, _C0_END)], waypoint_slot(j + 1))

    # C1 / C2 continuity at the S - 1 interior knots (homogeneous rows).
    for j in range(num_segments_S - 1):
        add_row([(j, _C1_END), (j + 1, -_C1_START)])
        add_row([(j, _C2_END), (j + 1, -_C2_START)])

    # Natural-spline boundary: zero curvature at both ends.
    add_row([(0, _C2_START)])
    add_row([(num_segments_S - 1, _C2_END)])

    # Pinned tangents: p'(s_j) = h * z[tangent_column[j]]. C1 continuity already
    # ties the two sides of an interior knot together, so constraining one side
    # constrains both (and is not redundant with it).
    for knot in heading_knots:
        if knot == num_segments_S:
            add_row([(num_segments_S - 1, _C1_END)], z_extra=(tangent_column[knot], h))
        else:
            add_row([(knot, _C1_START)], z_extra=(tangent_column[knot], h))

    G = np.stack(rows_G, axis=0)
    D_z = np.stack(rows_Dz, axis=0)
    if np.linalg.matrix_rank(G) < G.shape[0]:
        raise ValueError(
            f"Quintic spline constraints are rank deficient ({G.shape[0]} rows, "
            f"rank {np.linalg.matrix_rank(G)}) for S={num_segments_S}, "
            f"heading knots {heading_knots}. Too many pinned headings for the "
            "segment count; use more segments or fewer heading constraints."
        )
    if G.shape[0] > 6 * num_segments_S:
        raise ValueError(
            f"Quintic spline is over-constrained: {G.shape[0]} constraints on "
            f"{6 * num_segments_S} coefficients (S={num_segments_S})."
        )
    return G, D_z, np.asarray(start_sel), np.asarray(goal_sel), dim_z


def _solve_kkt(G: np.ndarray, D_z: np.ndarray, start_sel: np.ndarray, goal_sel: np.ndarray, num_segments: int, dim_z: int):
    """Solves the KKT system for the min-acceleration QP ONCE (float64), and
    returns the constant pieces of C = A z + P_start*start + P_goal*goal."""
    h = 1.0 / num_segments
    H2 = _gram_matrix(2, h)
    H3 = _gram_matrix(3, h)
    H2_full = np.kron(np.eye(num_segments), H2)
    H3_full = np.kron(np.eye(num_segments), H3)
    n = 6 * num_segments
    eps = _JERK_REGULARIZATION_RELATIVE * np.trace(H2_full) / n
    H = H2_full + eps * H3_full
    # Normalize H to unit mean diagonal. The Gram matrices carry h^-3 / h^-5
    # factors, so with many segments the H block dwarfs the O(1) constraint
    # block and the KKT matrix becomes badly *scaled* (condition number ~1e20 at
    # S=30) even though the system itself is fine. Scaling H by a positive
    # constant leaves the argmin untouched, so this is free.
    H = H / (np.trace(H) / n)

    num_rows = G.shape[0]
    M = np.zeros((n + num_rows, n + num_rows))
    M[:n, :n] = 2.0 * H
    M[:n, n:] = G.T
    M[n:, :n] = G
    condition_number = float(np.linalg.cond(M))

    rhs = np.zeros((n + num_rows, num_rows))
    rhs[n:, :] = np.eye(num_rows)
    solution = np.linalg.solve(M, rhs)
    P = solution[:n, :]  # C = P @ (D_z z + start_sel*start + goal_sel*goal)

    A = P @ D_z
    P_start = P @ start_sel
    P_goal = P @ goal_sel

    # Construction-time correctness check: the affine map must satisfy the
    # constraint system exactly (up to solve precision) for an arbitrary z.
    rng = np.random.default_rng(0)
    z_test = rng.normal(size=dim_z)
    start_test, goal_test = 1.3, -0.7
    C_test = A @ z_test + P_start * start_test + P_goal * goal_test
    rhs_test = D_z @ z_test + start_sel * start_test + goal_sel * goal_test
    residual = float(np.max(np.abs(G @ C_test - rhs_test)))
    if residual > _KKT_RESIDUAL_TOLERANCE:
        raise RuntimeError(
            f"Quintic spline KKT solve residual too large ({residual:.3e} > "
            f"{_KKT_RESIDUAL_TOLERANCE:.1e}); the affine map C = Az + b does not "
            "satisfy its own constraints."
        )

    return A, P_start, P_goal, condition_number


def _basis_matrices(s: np.ndarray, num_segments: int):
    """Constant [T, 6S] matrices mapping per-segment quintic coefficients C
    to position / d/ds / d2/ds2 at each sample s (float64, local coordinate
    u = (s - s_j) / h)."""
    h = 1.0 / num_segments
    s_clipped = np.clip(s, 0.0, 1.0)
    segment_index = np.clip(np.floor(s_clipped * num_segments).astype(int), 0, num_segments - 1)
    u = (s_clipped - segment_index * h) / h

    num_samples = s.shape[0]
    powers = np.stack([u ** k for k in range(6)], axis=1)
    dpowers = np.stack(
        [np.zeros_like(u)] + [k * u ** (k - 1) for k in range(1, 6)], axis=1
    ) / h
    ddpowers = np.stack(
        [np.zeros_like(u), np.zeros_like(u)] + [k * (k - 1) * u ** (k - 2) for k in range(2, 6)],
        axis=1,
    ) / (h ** 2)

    B0 = np.zeros((num_samples, 6 * num_segments))
    B1 = np.zeros((num_samples, 6 * num_segments))
    B2 = np.zeros((num_samples, 6 * num_segments))
    for t in range(num_samples):
        j = segment_index[t]
        B0[t, 6 * j: 6 * j + 6] = powers[t]
        B1[t, 6 * j: 6 * j + 6] = dpowers[t]
        B2[t, 6 * j: 6 * j + 6] = ddpowers[t]
    return B0, B1, B2


class QuinticSplinePlan:
    """Precomputed, decision-variable-independent evaluation maps for a
    piecewise-quintic minimum-acceleration spline with ``num_segments``
    segments, for a fixed problem, objective mode, and time scaling.

    Construction solves the KKT system once (float64 NumPy) and precomputes
    ``E0/E1/E2`` (each ``[T, dim_z]``, identical for both axes) plus additive
    per-axis bias terms ``b*_x``/``b*_y`` (nonzero only when identification
    mode pins the start/goal). ``evaluate`` is then three matmuls per axis.
    """

    def __init__(self, problem, num_segments: int, objective_mode: str, time_scaling: str,
                 heading_knots: tuple[int, ...] = ()):
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1.")
        self.num_segments = num_segments
        self.objective_mode = objective_mode
        self.time_scaling = normalize_time_scaling(time_scaling)
        self.heading_knots = tuple(sorted(set(heading_knots)))

        G, D_z, start_sel, goal_sel, dim_z = _build_linear_system(
            num_segments, objective_mode, self.heading_knots
        )
        A, P_start, P_goal, condition_number = _solve_kkt(G, D_z, start_sel, goal_sel, num_segments, dim_z)
        self.dim_z = dim_z
        self.kkt_condition_number = condition_number
        # The affine coefficient map itself (C = coefficient_map @ z + bias).
        # Not needed to evaluate the curve -- E0/E1/E2 below already fold it in
        # -- but kept so the affine/interpolation/continuity properties can be
        # checked directly rather than inferred from sampled positions.
        self.coefficient_map = A

        start_x, start_y = float(problem.start[0]), float(problem.start[1])
        goal_x, goal_y = float(problem.goal[0]), float(problem.goal[1])
        b_x = P_start * start_x + P_goal * goal_x
        b_y = P_start * start_y + P_goal * goal_y

        time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
        total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
        s, _, _ = time_scaling_derivatives(time_grid, total_time, time_scaling=self.time_scaling)
        s_np = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)

        B0, B1, B2 = _basis_matrices(s_np, num_segments)

        self.E0 = jnp.asarray(B0 @ A, dtype=jnp.float32)
        self.E1 = jnp.asarray(B1 @ A, dtype=jnp.float32)
        self.E2 = jnp.asarray(B2 @ A, dtype=jnp.float32)
        self.b0_x = jnp.asarray(B0 @ b_x, dtype=jnp.float32)
        self.b1_x = jnp.asarray(B1 @ b_x, dtype=jnp.float32)
        self.b2_x = jnp.asarray(B2 @ b_x, dtype=jnp.float32)
        self.b0_y = jnp.asarray(B0 @ b_y, dtype=jnp.float32)
        self.b1_y = jnp.asarray(B1 @ b_y, dtype=jnp.float32)
        self.b2_y = jnp.asarray(B2 @ b_y, dtype=jnp.float32)

    def evaluate(self, z_x: jnp.ndarray, z_y: jnp.ndarray):
        position = jnp.column_stack([self.E0 @ z_x + self.b0_x, self.E0 @ z_y + self.b0_y])
        dpos_ds = jnp.column_stack([self.E1 @ z_x + self.b1_x, self.E1 @ z_y + self.b1_y])
        d2pos_ds2 = jnp.column_stack([self.E2 @ z_x + self.b2_x, self.E2 @ z_y + self.b2_y])
        return position, dpos_ds, d2pos_ds2


def compute_quintic_reference(
    problem,
    plan: QuinticSplinePlan,
    control_points: jnp.ndarray,
    objective_mode: str,
    time_scaling: str | None = "s_curve",
) -> jnp.ndarray:
    z_x, z_y = _z_from_control_points(problem, control_points, objective_mode, plan.heading_knots)
    position, dpos_ds, d2pos_ds2 = plan.evaluate(z_x, z_y)

    time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
    total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
    _, s_dot, s_ddot = time_scaling_derivatives(
        time_grid, total_time, time_scaling=normalize_time_scaling(time_scaling)
    )
    return _reference_from_derivatives(position, dpos_ds, d2pos_ds2, s_dot, s_ddot)
