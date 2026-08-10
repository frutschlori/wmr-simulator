"""Planner: fixed waypoint list from a problem yaml's `planner:` block."""

import numpy as np

from wmr_simulator.planner import compute_reference_trajectory


def test_fixed_waypoint_reference_hits_its_waypoints_and_headings():
    """The `planner:` entry point: waypoints given by the yaml, not optimized.

    Every waypoint must lie on the curve, and a waypoint that carries a third
    entry must be crossed with that heading -- these are the constraints the
    reference is supposed to encode, and the yaml relies on them (`[x, y]` =
    heading free, `[x, y, theta]` = heading pinned).
    """
    start = [0.0, 0.0, 0.0]
    goal = [2.0, 1.0, 1.57]
    waypoints = [[0.5, 0.0, -1.57], [1.0, 0.5]]
    time = np.linspace(0.0, 5.0, 5001)

    reference_states, _ = compute_reference_trajectory(start, goal, waypoints, time)
    assert reference_states.shape == (time.size, 8)
    assert np.all(np.isfinite(reference_states))

    for knot in [start] + waypoints + [goal]:
        distances = np.linalg.norm(reference_states[:, :2] - np.asarray(knot[:2]), axis=1)
        nearest = int(np.argmin(distances))
        assert distances[nearest] < 1e-3, f"waypoint {knot[:2]} is not on the curve"
        if len(knot) > 2:
            delta = abs((reference_states[nearest, 2] - knot[2] + np.pi) % (2 * np.pi) - np.pi)
            assert delta < 1e-2, f"heading at {knot[:2]} is {reference_states[nearest, 2]}, want {knot[2]}"


def test_fixed_waypoint_reference_starts_and_ends_at_rest():
    """The s-curve time scaling has sigma_dot(0) = sigma_dot(T) = 0, so the
    reference velocity must vanish at both ends however the path is shaped."""
    time = np.linspace(0.0, 4.0, 401)
    reference_states, _ = compute_reference_trajectory(
        [0.0, 0.0, 0.0], [2.0, 1.0, 1.57], [[0.5, 0.0, -1.57]], time
    )
    for index in (0, -1):
        assert float(np.linalg.norm(reference_states[index, 3:6])) < 1e-6
