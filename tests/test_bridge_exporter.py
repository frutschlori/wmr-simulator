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
