"""The deployment video's inputs and observer hook (rendering itself needs OpenGL)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from wmr_simulator.mujoco_sim.render import (
    STATIC_VARIANT,
    record_pose_track,
    resolve_controller,
)
from wmr_simulator.pololu.reference_importer import load_pololu_reference, load_reference


def _reference_states(num: int = 21, dt: float = 0.05) -> np.ndarray:
    """An arc as the simulator's 8-column reference states."""
    time = np.arange(num) * dt
    speed, yaw_rate = 0.5, 0.4
    yaw = yaw_rate * time
    radius = speed / yaw_rate
    return np.column_stack(
        [
            radius * np.sin(yaw),
            radius * (1.0 - np.cos(yaw)),
            yaw,
            speed * np.cos(yaw),
            speed * np.sin(yaw),
            np.full(num, yaw_rate),
            np.zeros(num),
            np.zeros(num),
        ]
    ).astype(np.float32)


def test_pickle_and_its_exported_jsn_load_to_the_same_reference(tmp_path):
    """The pickle route is the export route, so the robot drives the same curve."""
    from wmr_simulator.pololu.reference_exporter import export_reference_trajectory, load_reference_trajectory

    pickle_path = tmp_path / "trajectory.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump({"reference_states": _reference_states(), "dt": 0.05}, file)
    jsn_path = export_reference_trajectory(load_reference_trajectory(pickle_path), tmp_path / "export")

    from_pickle = load_reference(pickle_path)
    from_jsn = load_pololu_reference(jsn_path)

    np.testing.assert_allclose(from_pickle.states, from_jsn.states)
    np.testing.assert_allclose(from_pickle.actions, from_jsn.actions)
    assert from_pickle.dt == from_jsn.dt


def _iteration(tmp_path, *, static: bool, gain_mlp: bool = True):
    root = tmp_path / "exp" / "iteration_03"
    (root / "tuning_trajectories").mkdir(parents=True)
    (root / "ROBOTCFG.CFG").write_text("WHEEL_RADIUS 0.016\n")
    if gain_mlp:
        (root / "GAINMLP.JSN").write_text("{}")
    if static:
        (root / "ROBOTCFG_static.CFG").write_text("WHEEL_RADIUS 0.016\n")
    trajectory = root / "tuning_trajectories" / "tuning_trajectory_00.pkl"
    trajectory.write_bytes(b"")
    return root, trajectory


def test_static_variant_is_staged_away_from_the_deployed_network(tmp_path):
    """FirmwareConfig.from_file picks up a GAINMLP.JSN beside the config, so the
    static baseline cannot be driven from the iteration root."""
    root, _ = _iteration(tmp_path, static=True)
    staging = tmp_path / "staging"

    controller = resolve_controller(root, STATIC_VARIANT, staging)
    assert controller.config_path.parent == staging
    assert controller.config_path.name == "ROBOTCFG.CFG"
    assert not (staging / "GAINMLP.JSN").exists()
    assert controller.config_path.read_text() == (root / "ROBOTCFG_static.CFG").read_text()


@pytest.fixture(scope="module")
def robot_config(tmp_path_factory):
    from types import SimpleNamespace

    from wmr_simulator.pololu.robot_config import export_robot_config

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0)
    return export_robot_config(
        tmp_path_factory.mktemp("config") / "ROBOTCFG.CFG",
        physical_params=physical,
        controller_gains=[2.0, 6.7, 7.4, 3.2, 0.0],
    )


@pytest.fixture(scope="module")
def short_trajectory(tmp_path_factory):
    """1 s of the same arc, as a reference pickle -- the other input format."""
    path = tmp_path_factory.mktemp("trajectory") / "trajectory.pkl"
    with path.open("wb") as file:
        pickle.dump({"reference_states": _reference_states(num=21), "dt": 0.05}, file)
    return path


def test_record_pose_track_returns_one_true_pose_per_frame(robot_config, short_trajectory, tmp_path):
    result, poses = record_pose_track(robot_config, short_trajectory, seed=0, fps=30)

    assert poses.ndim == 2 and poses.shape[1] == 3
    # 1 s of reference at 30 fps, give or take the trailing partial frame.
    assert 25 <= len(poses) <= 35
    # The frames land on a fixed step cadence, so the last one is up to a frame
    # of motion short of where the run actually ended.
    np.testing.assert_allclose(poses[0][:2], result.start_pose[:2], atol=0.02)
    np.testing.assert_allclose(poses[-1], result.final_pose, atol=0.05)
    assert not (tmp_path / "TR00").exists()  # the log went to a scratch directory


