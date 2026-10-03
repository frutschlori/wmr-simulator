import json
import pickle

import numpy as np
import pytest

from wmr_simulator.pololu.bridge_exporter import append_bridge_reference, bridged_output_path
from wmr_simulator.pololu.reference_exporter import (
    export_reference_trajectory,
    load_reference_trajectory,
)


def _export_jsn(tmp_path, dt=0.05, duration=3.0):
    """Synthetic reference pickle -> unbridged JSN, mirroring the export chain."""
    time = np.arange(0.0, duration + dt, dt)
    x = 0.3 * time
    y = 0.1 * np.sin(time)
    theta = np.arctan2(np.gradient(y, dt), np.gradient(x, dt))
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    omega = np.gradient(np.unwrap(theta), dt)
    states = np.column_stack([x, y, theta, vx, vy, omega, np.zeros_like(x), np.zeros_like(x)])
    pickle_path = tmp_path / "reference.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump({"reference_states": states, "dt": dt}, file)
    return export_reference_trajectory(load_reference_trajectory(pickle_path), tmp_path)


def test_bridged_jsn_returns_to_start(tmp_path):
    jsn_path = _export_jsn(tmp_path)
    bridged_path = append_bridge_reference(jsn_path, wait_time=1.0, bridge_time=4.0)

    # Same directory, derived name.
    assert bridged_path == bridged_output_path(jsn_path)
    assert bridged_path.parent == jsn_path.parent

    original = json.loads(jsn_path.read_text())["result"][0]
    bridged = json.loads(bridged_path.read_text())["result"][0]
    dt = original["dt"]
    assert bridged["dt"] == dt

    original_states = np.asarray(original["states"], dtype=float)
    bridged_states = np.asarray(bridged["states"], dtype=float)

    # Original trajectory + wait (1 s) + bridge (4 s) samples.
    assert len(bridged_states) > len(original_states) + round(1.0 / dt)
    # Replays the original path first, then loops back to its start pose.
    assert bridged_states[: len(original_states), :2] == pytest.approx(original_states[:, :2], abs=1e-5)
    assert bridged_states[-1, :2] == pytest.approx(original_states[0, :2], abs=1e-3)


def test_zero_wait_time_adds_no_wait_samples(tmp_path):
    jsn_path = _export_jsn(tmp_path)
    with_wait = append_bridge_reference(
        jsn_path, tmp_path / "with_wait.JSN", wait_time=2.0, bridge_time=4.0
    )
    without_wait = append_bridge_reference(
        jsn_path, tmp_path / "without_wait.JSN", wait_time=0.0, bridge_time=4.0
    )
    dt = json.loads(jsn_path.read_text())["result"][0]["dt"]
    num_with = json.loads(with_wait.read_text())["result"][0]["num_states"]
    num_without = json.loads(without_wait.read_text())["result"][0]["num_states"]
    assert num_with - num_without == round(2.0 / dt)


def _wrap(angle):
    import numpy as np

    return (angle + np.pi) % (2 * np.pi) - np.pi


def test_arc_return_reaches_the_goal_pose_moving_forward():
    import numpy as np

    from wmr_simulator.pololu.bridge_exporter import ArcReturn, arc_line_arc_states

    route = ArcReturn(radius=0.25, lateral_acceleration=3.0, peak_speed=1.5, min_piece_duration=0.75)
    goal = np.array([-0.5, -1.0, 0.0])
    # Behind, beside, ahead, reversed, and the goal pose itself rotated.
    for start in ([1.0, 1.5, 2.5], [-0.4, -0.3, -1.0], [-1.2, -1.0, 0.0], [0.5, -1.0, 0.0], [-0.5, -1.0, 2.0]):
        states = arc_line_arc_states(np.array(start), goal, 0.05, route)
        np.testing.assert_allclose(states[0, :3], start)
        np.testing.assert_allclose(states[-1, :2], goal[:2], atol=1e-9)
        assert abs(_wrap(states[-1, 2] - goal[2])) < 1e-9
        speed = np.hypot(states[:, 3], states[:, 4])
        assert speed.max() <= 1.5 + 1e-9
        assert np.abs(states[:, 5]).max() <= np.sqrt(3.0 / 0.25) + 1e-9  # v_peak / radius
        # The heading only changes while the robot moves: no turn on the spot.
        assert np.all(speed[np.abs(states[:, 5]) > 1e-9] > 0.0)
        # Forward driving (velocity along the heading), and no jumps.
        moving = speed > 1e-9
        np.testing.assert_allclose(np.cos(np.arctan2(states[moving, 4], states[moving, 3]) - states[moving, 2]), 1.0,
                                   atol=1e-9)
        assert np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).max() <= 1.5 * 0.05 + 1e-9


def test_arc_return_stays_in_the_box_with_a_smaller_radius_when_needed():
    import numpy as np

    from wmr_simulator.pololu.bridge_exporter import ArcReturn, arc_line_arc_states

    # Facing the wall 0.2 m away, the goal behind: a 0.25 m arc would hit it.
    start, goal = np.array([1.0, 0.0, 0.0]), np.array([0.6, 0.0, np.pi])
    free = arc_line_arc_states(start, goal, 0.05, ArcReturn())
    assert free[:, 0].max() > 1.2
    boxed = arc_line_arc_states(start, goal, 0.05, ArcReturn(box_min=(-1.3, -2.3), box_max=(1.3, 2.3), margin=0.1))
    assert boxed[:, 0].max() <= 1.2 + 1e-9
    np.testing.assert_allclose(boxed[-1, :2], goal[:2], atol=1e-9)
