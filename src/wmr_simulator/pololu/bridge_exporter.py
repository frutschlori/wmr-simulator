"""Append a wait + bridge-back segment to a Pololu reference JSN.

The bridged reference replays the original trajectory, waits at its goal, and
then drives a planned bridge path back to the start pose, so an experiment can
be repeated without repositioning the robot by hand. The bridged JSN is a
regular Pololu reference and is exported next to the unbridged one by default
(``<name>_bridge.JSN``).

The CLI wrapper lives in scripts/pololu_append_bridge_reference.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.pololu.reference_exporter import ReferenceTrajectory, format_pololu_reference
from wmr_simulator.pololu.reference_importer import (
    load_pololu_reference,
    reference_states_from_pololu_reference,
)


# Peak of the quintic s-curve rate over its mean: 15/8.
S_CURVE_PEAK_RATE = 15.0 / 8.0


def time_grid(duration: float, dt: float) -> np.ndarray:
    steps = int(np.round(duration / dt))
    if steps <= 0:
        raise ValueError("duration must contain at least one dt interval.")
    return np.linspace(0.0, steps * dt, steps + 1)


def wait_reference_states(pose: np.ndarray, wait_time: float, dt: float) -> np.ndarray:
    intervals = int(np.round(wait_time / dt))
    if intervals <= 0:
        return np.empty((0, 8), dtype=float)
    states = np.zeros((intervals, 8), dtype=float)
    states[:, :3] = pose[:3]
    return states


def bridge_reference_states(start_pose: np.ndarray, goal_pose: np.ndarray, bridge_time: float, dt: float) -> np.ndarray:
    reference_states, _ = compute_reference_trajectory(
        start=start_pose[:3],
        goal=goal_pose[:3],
        intermediate_waypoints=[],
        time=time_grid(bridge_time, dt),
    )
    return reference_states


class ArcReturn(NamedTuple):
    """How ``arc_line_arc_states`` drives a return: arcs of ``radius`` (halved up
    to twice when no route at full radius stays in the box) at up to
    ``lateral_acceleration``, a straight line at up to ``peak_speed``, each
    piece rest to rest and at least ``min_piece_duration`` long, inside
    ``box_min``/``box_max`` by ``margin`` when a box is given. ``goal_radius``
    is the arc onto the goal pose, when it needs a different radius than the
    first one (a return between two corners of the box); None: ``radius``.
    ``smooth`` drives the smoothest B-spline near the route instead, in one
    s-curve and no tighter than ``min_radius`` (``smooth_route_states``)."""

    radius: float = 0.25
    lateral_acceleration: float = 3.0
    peak_speed: float = 1.5
    min_piece_duration: float = 0.75
    box_min: tuple[float, float] | None = None
    box_max: tuple[float, float] | None = None
    margin: float = 0.15
    goal_radius: float | None = None
    smooth: bool = False
    min_radius: float | None = None


def _tangent_routes(start_pose: np.ndarray, goal_pose: np.ndarray, radius: float, goal_radius: float | None = None):
    """(first arc angle, line length, second arc angle) of the four forward
    circle - tangent - circle routes (LSL, LSR, RSL, RSR) from ``start_pose`` to
    ``goal_pose``, on circles of ``radius`` and ``goal_radius`` (default: the
    same); a route whose circles are too close for its tangent is left out. Arc
    angles are signed (positive: left) and in (-2 pi, 2 pi)."""
    goal_radius = radius if goal_radius is None else goal_radius
    two_pi = 2.0 * math.pi
    theta_start, theta_goal = float(start_pose[2]), float(goal_pose[2])
    for sigma_start in (1.0, -1.0):
        for sigma_goal in (1.0, -1.0):
            center_start = np.asarray(start_pose[:2], dtype=float) + sigma_start * radius * np.array(
                [-math.sin(theta_start), math.cos(theta_start)]
            )
            center_goal = np.asarray(goal_pose[:2], dtype=float) + sigma_goal * goal_radius * np.array(
                [-math.sin(theta_goal), math.cos(theta_goal)]
            )
            offset = center_goal - center_start
            distance = float(np.linalg.norm(offset))
            if distance < 1e-9:
                if sigma_start != sigma_goal or abs(goal_radius - radius) > 1e-12:
                    continue
                direction, line = theta_start, 0.0
            else:
                # The tangent's left normal n satisfies n . offset = sigma_goal
                # goal_radius - sigma_start radius.
                ratio = (sigma_goal * goal_radius - sigma_start * radius) / distance
                if abs(ratio) > 1.0:
                    continue
                bearing = math.atan2(offset[1], offset[0])
                direction = bearing - math.asin(ratio)
                line = distance * math.cos(bearing - direction)

            def sweep(sigma, angle):
                swept = (sigma * angle) % two_pi
                return 0.0 if swept > two_pi - 1e-9 else sigma * swept

            yield sweep(sigma_start, direction - theta_start), line, sweep(sigma_goal, theta_goal - direction)


def _box_violation(xy: np.ndarray, route: ArcReturn) -> float:
    if route.box_min is None or route.box_max is None:
        return 0.0
    low = np.asarray(route.box_min, dtype=float) + route.margin
    high = np.asarray(route.box_max, dtype=float) - route.margin
    return float(max(np.max(low - xy), np.max(xy - high), 0.0))


def _route_poses(start_pose: np.ndarray, pieces: list[tuple[float, float]], s: np.ndarray) -> np.ndarray:
    """``[x, y, theta]`` at arc lengths ``s`` along ``pieces`` ``(length,
    curvature)`` from ``start_pose`` (exact on arcs and lines)."""
    lengths = np.array([length for length, _ in pieces])
    ends = np.cumsum(lengths)
    x0, y0, theta0 = (float(value) for value in start_pose[:3])
    piece_start = []
    for length, curvature in pieces:
        piece_start.append((x0, y0, theta0))
        if curvature == 0.0:
            x0, y0 = x0 + length * math.cos(theta0), y0 + length * math.sin(theta0)
        else:
            theta1 = theta0 + curvature * length
            x0 += (math.sin(theta1) - math.sin(theta0)) / curvature
            y0 -= (math.cos(theta1) - math.cos(theta0)) / curvature
            theta0 = theta1
    index = np.minimum(np.searchsorted(ends, s, side="left"), len(pieces) - 1)
    poses = np.empty((len(s), 3))
    for k, (local, piece) in enumerate(zip(s - (ends - lengths)[index], index)):
        px, py, ptheta = piece_start[piece]
        curvature = pieces[piece][1]
        if curvature == 0.0:
            poses[k] = px + local * math.cos(ptheta), py + local * math.sin(ptheta), ptheta
        else:
            theta = ptheta + curvature * local
            poses[k] = (px + (math.sin(theta) - math.sin(ptheta)) / curvature,
                        py - (math.cos(theta) - math.cos(ptheta)) / curvature, theta)
    return poses


def _s_curve_states(s: np.ndarray, poses: np.ndarray, curvature: np.ndarray, dt: float, route: ArcReturn) -> np.ndarray:
    """Samples after the start of the path given densely by arc length ``s``
    (from 0), ``poses`` and ``curvature``, driven with one rest-to-rest quintic
    s-curve over its whole length. The duration is the shortest (in whole steps)
    that keeps the speed under ``route.peak_speed`` and the lateral acceleration
    ``v^2 |curvature|`` under ``route.lateral_acceleration``, and is at least
    ``route.min_piece_duration``."""
    total_length = float(s[-1])
    tau_grid = np.linspace(0.0, 1.0, 20001)
    position_grid = 10 * tau_grid**3 - 15 * tau_grid**4 + 6 * tau_grid**5

    def rate(tau):
        return 30 * tau**2 - 60 * tau**3 + 30 * tau**4

    tau_at_s = np.interp(s / total_length, position_grid, tau_grid)
    duration = max(
        route.min_piece_duration,
        S_CURVE_PEAK_RATE * total_length / route.peak_speed,
        float(np.max(total_length * rate(tau_at_s) * np.sqrt(np.abs(curvature) / route.lateral_acceleration))),
    )
    steps = max(int(np.ceil(duration / dt - 1e-9)), 1)
    total = steps * dt
    tau = np.arange(1, steps + 1) / steps
    s_k = total_length * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    speed = total_length * rate(tau) / total
    accel = total_length * (60 * tau - 180 * tau**2 + 120 * tau**3) / total**2
    x, y, theta = (np.interp(s_k, s, poses[:, i]) for i in range(3))
    omega = np.interp(s_k, s, curvature) * speed

    from wmr_simulator.trajectory_optimization.reference_extension import _path_states

    return _path_states(x, y, theta, speed, omega, accel, 8)


def _clamped_spline_parametrization(num_control_points: int, start_pose: np.ndarray, goal_pose: np.ndarray):
    """Knots and the affine map ``z -> control points`` (flattened ``x, y``) of a
    clamped quintic B-spline on [0, 1] that starts at ``start_pose`` and ends at
    ``goal_pose`` with its tangent along each pose's heading: ``z = [a, b,
    P_2, ..., P_{n-3}]``, where ``a``/``b`` are how far the second and the
    second-to-last control point sit along the start and goal headings (both
    must stay positive for the curve to leave and arrive driving forward)."""
    degree, n = 5, num_control_points
    knots = np.concatenate([np.zeros(degree), np.linspace(0.0, 1.0, n - degree + 1), np.ones(degree)])
    p0, p_end = np.asarray(start_pose[:2], dtype=float), np.asarray(goal_pose[:2], dtype=float)
    constant = np.zeros((n, 2))
    constant[0], constant[1], constant[-2], constant[-1] = p0, p0, p_end, p_end
    columns = []
    for heading, row, sign in ((start_pose[2], 1, 1.0), (goal_pose[2], n - 2, -1.0)):
        column = np.zeros((n, 2))
        column[row] = sign * np.array([math.cos(heading), math.sin(heading)])
        columns.append(column.ravel())
    for k in range(n - 4):
        for axis in range(2):
            column = np.zeros((n, 2))
            column[2 + k, axis] = 1.0
            columns.append(column.ravel())
    return knots, constant.ravel(), np.stack(columns, axis=1)


def _spline_design(knots: np.ndarray, u: np.ndarray, order: int) -> np.ndarray:
    """``(len(u) * 2, 2 n)`` matrix taking flattened control points to the
    flattened ``order``-th derivative of the curve at ``u``."""
    from scipy.interpolate import BSpline

    n = len(knots) - 6
    basis = np.stack([BSpline(knots, np.eye(n)[i], 5).derivative(order)(u) if order else
                      BSpline(knots, np.eye(n)[i], 5)(u) for i in range(n)], axis=1)
    return np.kron(basis, np.eye(2))


def _smooth_spline(guide: np.ndarray, start_pose, goal_pose, num_control_points: int, min_radius: float,
                   route: ArcReturn):
    """The smoothest clamped quintic B-spline from ``start_pose`` to
    ``goal_pose`` (end poses and headings exact, forward at both ends): the
    least-squares fit to ``guide`` refined by SLSQP to minimize the change of
    curvature along the path (plus a little bending energy and length) with
    the curvature bounded by ``1 / min_radius`` and the curve inside the box.
    Returns the scipy ``BSpline``."""
    from scipy.interpolate import BSpline
    from scipy.optimize import minimize

    knots, constant, mapping = _clamped_spline_parametrization(num_control_points, start_pose, goal_pose)
    fit = _spline_design(knots, np.linspace(0.0, 1.0, len(guide)), 0) @ mapping
    target = guide.ravel() - _spline_design(knots, np.linspace(0.0, 1.0, len(guide)), 0) @ constant
    z0, *_ = np.linalg.lstsq(fit, target, rcond=None)

    u = np.linspace(0.0, 1.0, 300)
    du = u[1] - u[0]
    designs = [_spline_design(knots, u, order) for order in range(4)]
    length_scale = float(np.sum(np.hypot(*np.diff(guide, axis=0).T)))

    def geometry(z):
        points = constant + mapping @ z
        xy, d1, d2, d3 = ((design @ points).reshape(-1, 2) for design in designs)
        speed = np.hypot(d1[:, 0], d1[:, 1])
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        curvature = cross / speed**3
        d_cross = d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]
        d_speed = (d1 * d2).sum(axis=1) / speed
        d_curvature_ds = (d_cross / speed**3 - 3.0 * cross * d_speed / speed**4) / speed
        return xy, speed, curvature, d_curvature_ds

    def objective(z):
        _, speed, curvature, d_curvature_ds = geometry(z)
        ds = speed * du
        return (1e-2 * length_scale**3 * np.sum(d_curvature_ds**2 * ds) + length_scale * np.sum(curvature**2 * ds)
                + np.sum(ds) / length_scale)

    constraints = [
        {"type": "ineq", "fun": lambda z: 1.0 - (min_radius * geometry(z)[2]) ** 2},
        {"type": "ineq", "fun": lambda z: geometry(z)[1] - 0.05 * length_scale},
        {"type": "ineq", "fun": lambda z: z[:2] - 0.02},
    ]
    if route.box_min is not None and route.box_max is not None:
        low = np.asarray(route.box_min, dtype=float) + route.margin
        high = np.asarray(route.box_max, dtype=float) - route.margin
        constraints.append({"type": "ineq", "fun": lambda z: np.concatenate(
            [(geometry(z)[0] - low).ravel(), (high - geometry(z)[0]).ravel()])})
    result = minimize(objective, z0, constraints=constraints, method="SLSQP",
                      options={"maxiter": 500, "ftol": 1e-9})
    return BSpline(knots, (constant + mapping @ result.x).reshape(-1, 2), 5)


def smooth_route_states(start_pose: np.ndarray, goal_pose: np.ndarray, dt: float, route: ArcReturn) -> np.ndarray:
    """A return with continuous curvature and a turning-radius floor: the
    shortest arc - line - arc route inside the box (``_tangent_routes``, radii
    as in ``arc_line_arc_states``) is the initial guess for the smoothest
    clamped quintic B-spline with end poses and headings exact, no radius under
    ``route.min_radius`` (default: the smaller arc radius) and inside the box
    (``_smooth_spline``), driven forward in one s-curve (``_s_curve_states``).
    The optimization is non-convex, so several control-point counts are tried
    and the fastest return that meets the radius floor and the box on a fine
    check grid is kept (else the one closest to that). Unlike the arc - line -
    arc return there is no stop and no yaw-rate step where an arc meets the
    line. Starts at ``start_pose`` (included)."""
    start = np.zeros(8)
    start[:3] = start_pose[:3]
    goal_pose = np.asarray(goal_pose, dtype=float)
    goal_radius = route.radius if route.goal_radius is None else route.goal_radius
    min_radius = min(route.radius, goal_radius) if route.min_radius is None else route.min_radius

    guide_pieces = None
    for scale in (1.0, 0.5, 0.25):
        radius, last_radius = scale * route.radius, scale * goal_radius
        for first, line, second in _tangent_routes(start, goal_pose, radius, last_radius):
            pieces = [(length, curvature) for length, curvature in (
                (radius * abs(first), math.copysign(1.0 / radius, first)),
                (line, 0.0),
                (last_radius * abs(second), math.copysign(1.0 / last_radius, second)),
            ) if length > 1e-9]
            length = sum(length for length, _ in pieces)
            violation = _box_violation(_route_poses(start, pieces, np.linspace(0.0, length, 400))[:, :2], route)
            score = (round(violation, 3), length)
            if guide_pieces is None or score < guide_pieces[0]:
                guide_pieces = (score, pieces)
        if guide_pieces[0][0] == 0.0:
            break
    pieces = guide_pieces[1]
    length = sum(length for length, _ in pieces)
    guide = _route_poses(start, pieces, np.linspace(0.0, length, 400))[:, :2]

    best = None
    u = np.linspace(0.0, 1.0, 4001)
    for num_control_points in range(10, 17):
        spline = _smooth_spline(guide, start, goal_pose, num_control_points, min_radius, route)
        xy, first, second = spline(u), spline.derivative(1)(u), spline.derivative(2)(u)
        norm = np.hypot(first[:, 0], first[:, 1])
        if norm.min() < 1e-6 * norm.max():
            continue  # a cusp: not drivable forward
        curvature = (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / norm**3
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        theta = np.unwrap(np.arctan2(first[:, 1], first[:, 0]))
        theta += float(start[2]) - theta[0]
        states = _s_curve_states(s, np.column_stack([xy, theta]), curvature, dt, route)
        radius_shortfall = max(0.0, min_radius - 1.0 / float(np.max(np.abs(curvature))))
        score = (round(_box_violation(xy, route), 3), round(radius_shortfall, 3), len(states))
        if best is None or score < best[0]:
            best = (score, states)
    return np.vstack([start[None, :], best[1]])


def arc_line_arc_states(start_pose: np.ndarray, goal_pose: np.ndarray, dt: float, route: ArcReturn) -> np.ndarray:
    """A return a differential drive tracks with heading feedback all the way:
    a forward arc onto the tangent line, the line, a forward arc onto the goal
    pose -- the fastest of the four tangent routes that stays in the box, each
    piece rest to rest. The turn-line-turn return it replaces turned on the spot,
    where the Kanayama law has no heading feedback (``v_ref * kth * sin
    theta_e``), and the robot arrived wherever its wheelbase belief put it: 119 of
    159 iteration-2 repeats in Phase 2 v6 started more than 0.5 rad off heading.
    When no route fits the box at any radius, the one leaving it least is
    driven. Starts at ``start_pose`` (included). With ``route.smooth`` the
    return is ``smooth_route_states`` instead."""
    from wmr_simulator.trajectory_optimization.reference_extension import arc_turn, straight_line

    if route.smooth:
        return smooth_route_states(start_pose, goal_pose, dt, route)
    start = np.zeros(8)
    start[:3] = start_pose[:3]
    best = None
    goal_radius = route.radius if route.goal_radius is None else route.goal_radius
    for scale in (1.0, 0.5, 0.25):
        radius, last_radius = scale * route.radius, scale * goal_radius
        arc = dict(lateral_acceleration=route.lateral_acceleration, min_duration=route.min_piece_duration)
        for first, line, second in _tangent_routes(start, np.asarray(goal_pose, dtype=float), radius, last_radius):
            states = start[None, :]
            states = np.vstack([states, arc_turn(states[-1], first, radius, dt, 8, **arc)])
            states = np.vstack([states, straight_line(
                states[-1], line, dt, 8, peak_speed=route.peak_speed, min_duration=route.min_piece_duration
            )])
            states = np.vstack([states, arc_turn(states[-1], second, last_radius, dt, 8, **arc)])
            score = (round(_box_violation(states[:, :2], route), 3), len(states))
            if best is None or score < best[0]:
                best = (score, states)
        if best[0][0] == 0.0:
            break
    return best[1]


def bridged_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_bridge.JSN")


def append_bridge_reference(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    wait_time: float,
    bridge_time: float | None = None,
    result_index: int = 0,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
    plot_path: str | Path | None = None,
    arc_return: ArcReturn | None = None,
    pieces: list[dict] | None = None,
) -> Path:
    """Write ``<name>_bridge.JSN`` (default: next to the input) and return its path.

    With ``arc_return`` the return is arc - straight line - arc
    (``arc_line_arc_states``) instead of the free-form bridge, and
    ``bridge_time`` is unused. ``pieces`` (an extended identification
    reference's, ``reference_extension.extended_payload``) only changes the
    plot: each piece is drawn on its own and its end is marked."""
    if output_path is None:
        output_path = bridged_output_path(input_path)
    reference = load_pololu_reference(input_path, result_index=result_index)
    reference_states = reference_states_from_pololu_reference(
        reference,
        final_action="zero",
        acceleration="finite-difference",
        dtype=float,
    )
    reference_states = reference_states.copy()
    reference_states[-1, 3:] = 0.0

    wait_states = wait_reference_states(reference_states[-1, :3], wait_time, reference.dt)
    if arc_return is None:
        if bridge_time is None:
            raise ValueError("The free-form bridge needs a bridge_time.")
        bridge_states = bridge_reference_states(
            start_pose=reference_states[-1, :3],
            goal_pose=reference_states[0, :3],
            bridge_time=bridge_time,
            dt=reference.dt,
        )
    else:
        bridge_states = arc_line_arc_states(reference_states[-1, :3], reference_states[0, :3], reference.dt, arc_return)
    full_reference_states = np.vstack([reference_states, wait_states, bridge_states[1:]])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plot_path is not None:
        plot_bridged_reference(
            reference_states=reference_states,
            wait_states=wait_states,
            bridge_states=bridge_states,
            dt=reference.dt,
            output_path=plot_path,
            pieces=pieces,
        )
    formatted = format_pololu_reference(
        ReferenceTrajectory(
            path=Path(input_path),
            reference_states=full_reference_states,
            dt=reference.dt,
        ),
        cost=cost,
        time_stamp=time_stamp,
        decimals=decimals,
    )
    with output_path.open("w", encoding="utf-8") as file:
        # Compact: the firmware reads the JSN into a 48 KiB buffer, and indented
        # JSON spends about half of it on whitespace (serde_json_core reads both).
        json.dump(formatted, file, separators=(",", ":"))
        file.write("\n")
    from wmr_simulator.pololu.reference_exporter import check_firmware_limits

    for warning in check_firmware_limits(
        num_states=full_reference_states.shape[0],
        file_bytes=output_path.stat().st_size,
        label=str(output_path),
    ):
        print(f"WARNING: {warning}")
    return output_path


def append_bridge_references_in_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    wait_time: float,
    bridge_time: float,
    result_index: int = 0,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
) -> list[tuple[Path, Path]]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-bridged references (they live next to the unbridged ones).
    input_paths = sorted(path for path in input_dir.glob("*.JSN") if not path.stem.endswith("_bridge"))
    if not input_paths:
        raise ValueError(f"No unbridged JSN files found in: {input_dir}")

    outputs = []
    for input_path in input_paths:
        output_path = output_dir / f"{input_path.stem}_bridge.JSN"
        plot_path = output_path.with_suffix(".pdf")
        append_bridge_reference(
            input_path,
            output_path,
            wait_time=wait_time,
            bridge_time=bridge_time,
            result_index=result_index,
            cost=cost,
            time_stamp=time_stamp,
            decimals=decimals,
            plot_path=plot_path,
        )
        outputs.append((output_path, plot_path))
    return outputs


def _piece_label(piece: dict) -> str:
    if piece["kind"] == "identified":
        return "Identified phase"
    if piece["kind"] == "turn":
        return "Arc turn"
    # Designed tuning pickles carry a _YYYYMMDD_HHMMSS stamp; the index names them.
    name = re.sub(r"_\d{8}_\d{6}$", "", piece["name"])
    return f"Appended: {name}"


def _mark_piece_end(ax, pose: np.ndarray, half_length: float) -> None:
    """A short stroke across the path at ``pose`` [x, y, theta]: where a piece ends."""
    normal = np.array([-math.sin(pose[2]), math.cos(pose[2])])
    ends = pose[:2] + np.outer([-half_length, half_length], normal)
    ax.plot(ends[:, 0], ends[:, 1], color="black", linewidth=1.5, solid_capstyle="butt", zorder=4)


def plot_bridged_reference(
    reference_states: np.ndarray,
    wait_states: np.ndarray,
    bridge_states: np.ndarray,
    dt: float,
    output_path: str | Path,
    pieces: list[dict] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_reference_states = np.vstack([reference_states, wait_states, bridge_states[1:]])
    time = np.arange(full_reference_states.shape[0], dtype=float) * dt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), gridspec_kw={"height_ratios": [1.3, 1.0]})
    ax_xy, ax_state = axes

    if pieces:
        # The identified phase, the wait and the bridge keep the colours of the
        # plain plot (C0-C2); appended trajectories take the next ones in order,
        # arc turns are neutral.
        extent = np.ptp(full_reference_states[:, :2], axis=0).max()
        start, appended = 0, 0
        for piece in pieces:
            end = int(piece["end"])
            if piece["kind"] == "turn":
                style = {"color": "0.55", "linestyle": "--"}
            elif piece["kind"] == "identified":
                style = {"color": "C0"}
            else:
                style = {"color": f"C{3 + appended}"}
                appended += 1
            ax_xy.plot(reference_states[start:end + 1, 0], reference_states[start:end + 1, 1], linewidth=1.4,
                       label=_piece_label(piece), **style)
            _mark_piece_end(ax_xy, reference_states[end, :3], 0.025 * extent)
            ax_state.axvline(end * dt, color="0.6", linestyle=":", linewidth=1.0)
            start = end
    else:
        ax_xy.plot(reference_states[:, 0], reference_states[:, 1], linewidth=1.4, label="Reference")
    if len(wait_states):
        ax_xy.scatter(wait_states[:, 0], wait_states[:, 1], s=12, color="C1", label="Wait")
    ax_xy.plot(bridge_states[:, 0], bridge_states[:, 1], linewidth=1.4, color="C2", label="Bridge")
    ax_xy.scatter(reference_states[0, 0], reference_states[0, 1], marker="o", s=35, color="black", label="Start")
    ax_xy.scatter(reference_states[-1, 0], reference_states[-1, 1], marker="x", s=45, color="black", label="Original goal")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Bridged Pololu Reference")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    # One entry per label: every arc turn shares one.
    legend = dict(zip(*reversed(ax_xy.get_legend_handles_labels())))
    # Beside the axes: the equal-aspect path panel leaves room there, and inside
    # it the legend covers the path.
    ax_xy.legend(legend.values(), legend.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax_state.plot(time, full_reference_states[:, 0], label="x")
    ax_state.plot(time, full_reference_states[:, 1], label="y")
    ax_state.plot(time, full_reference_states[:, 2], label="theta")
    ax_state.set_xlabel("time [s]")
    ax_state.set_ylabel("state")
    ax_state.grid(True)
    ax_state.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a wait and bridge-back path to a Pololu reference JSN.")
    parser.add_argument("--input", type=str, default="Pololu Data/References/dt comparison/")
    parser.add_argument("--output", type=str, default="Pololu Data/References/dt comparison/bridged/")
    parser.add_argument("--plot-output", type=str, default=None)
    parser.add_argument("--wait-time", type=float, default=1.0)
    parser.add_argument("--bridge-time", type=float, default=5.0)
    parser.add_argument("--result-index", type=int, default=0)
    parser.add_argument("--cost", type=float, default=100.0)
    parser.add_argument("--time-stamp", type=float, default=0.0)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if input_path.is_dir():
        if args.plot_output is not None:
            parser.error("--plot-output can only be used when --input points to a single file.")
        output_dir = Path(args.output) if args.output is not None else input_path
        saved_paths = append_bridge_references_in_directory(
            input_path,
            output_dir,
            wait_time=args.wait_time,
            bridge_time=args.bridge_time,
            result_index=args.result_index,
            cost=args.cost,
            time_stamp=args.time_stamp,
            decimals=args.decimals,
        )
        for output_path, plot_path in saved_paths:
            print(output_path)
            print(plot_path)
        return 0

    if not input_path.is_file():
        parser.error(f"--input must point to a JSN file or directory: {input_path}")
    output_path = Path(args.output) if args.output is not None else bridged_output_path(input_path)
    plot_path = Path(args.plot_output) if args.plot_output is not None else output_path.with_suffix(".pdf")
    saved_path = append_bridge_reference(
        input_path,
        output_path,
        wait_time=args.wait_time,
        bridge_time=args.bridge_time,
        result_index=args.result_index,
        cost=args.cost,
        time_stamp=args.time_stamp,
        decimals=args.decimals,
        plot_path=plot_path,
    )
    print(saved_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
