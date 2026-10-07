"""Mocap rigid-body calibration: the correction, the calibration run, the
estimator against a MuJoCo plant with a hidden misalignment, and the export."""

from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wmr_simulator.pololu.mocap_calibration import (
    CONFIG_KEYS,
    MocapCalibration,
    calibration_reference_states,
    estimate_mocap_calibration,
    export_calibration_reference,
)

YAW_OFFSET = math.radians(1.0)
# Robot frame, as the hidden plant takes it; the height is left in on purpose:
# the planar calibration has to recover the rest regardless of it.
OFFSET_ROBOT = np.array([0.011, -0.004, 0.030])


def rz2(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def test_correction_recovers_the_axle_pose():
    rng = np.random.default_rng(0)
    n = 200
    axle = rng.uniform(-1.0, 1.0, size=(n, 2))
    robot_yaw = rng.uniform(-math.pi, math.pi, size=n)
    calibration = MocapCalibration(YAW_OFFSET, (0.011, -0.004))
    mocap_yaw = robot_yaw + calibration.yaw_offset
    offset_world = np.stack([rz2(yaw) @ np.asarray(calibration.offset) for yaw in mocap_yaw])
    frames = np.column_stack([axle + offset_world, np.zeros((n, 3)), mocap_yaw])
    corrected = calibration.correct(frames)
    np.testing.assert_allclose(corrected[:, :2], axle, atol=1e-12)
    np.testing.assert_allclose(np.cos(corrected[:, 5] - robot_yaw), 1.0, atol=1e-12)


def test_composition_matches_applying_both():
    first = MocapCalibration(YAW_OFFSET, (0.011, -0.004))
    residual = MocapCalibration(math.radians(-0.3), (0.002, 0.001))
    poses = np.array([[1.0, 2.0, 0.05, 0.01, 0.02, 0.7], [-0.4, 0.3, 0.05, 0.0, 0.0, -2.9]])
    np.testing.assert_allclose(
        residual.correct(first.correct(poses)), first.compose(residual).correct(poses), atol=1e-12
    )


def test_config_values_round_trip():
    calibration = MocapCalibration(YAW_OFFSET, (0.011, -0.004))
    assert set(calibration.config_values()) == set(CONFIG_KEYS)
    assert MocapCalibration.from_config_values(calibration.config_values()) == calibration
    assert MocapCalibration.from_mapping(calibration.to_mapping()) == calibration
    assert MocapCalibration.from_config_values({}) == MocapCalibration()


def test_calibration_run_fits_the_firmware_and_the_arena(tmp_path):
    states = calibration_reference_states()
    speed = np.hypot(states[:, 3], states[:, 4])
    assert len(states) <= 350
    assert speed.max() < 1.0 and np.abs(states[:, 5]).max() <= 5.0 + 1e-9
    np.testing.assert_allclose(states[-1, :3], states[0, :3], atol=1e-9)  # repeatable in place
    assert np.all(np.abs(states[:, 0]) <= 1.3) and np.all(np.abs(states[:, 1]) <= 2.3)
    # Turns on the spot (the heading offset needs straights, the offset turns).
    assert np.any((speed < 1e-9) & (np.abs(states[:, 5]) > 3.0))
    _, jsn = export_calibration_reference(tmp_path)
    payload = json.loads(jsn.read_text())["result"][0]
    assert payload["num_states"] == len(states)


@pytest.fixture(scope="module")
def robot_config(tmp_path_factory):
    from wmr_simulator.pololu.robot_config import export_robot_config

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0845, max_wheel_speed=230.0)
    return export_robot_config(
        tmp_path_factory.mktemp("config") / "ROBOTCFG.CFG",
        physical_params=physical,
        controller_gains=[2.8, 4.7, 7.0, 3.0, 0.0, 0.0],
    )


def _decoded(log):
    from wmr_simulator.pololu.decode_binary import decode_file

    csv = log.with_name(f"{log.name}.csv")
    assert decode_file(str(log), str(csv))
    return csv


def test_estimator_recovers_a_hidden_misalignment_in_mujoco(tmp_path, robot_config):
    from wmr_simulator.mujoco_sim.deploy import run_deployment
    from wmr_simulator.mujoco_sim.plant import load_hidden_plant_config
    from wmr_simulator.pololu.robot_config import export_robot_config, load_robot_config_file

    plant = load_hidden_plant_config()
    plant = dataclasses.replace(
        plant, mocap=dataclasses.replace(plant.mocap, offset_xyz=tuple(OFFSET_ROBOT), yaw_offset=YAW_OFFSET)
    )
    _, trajectory = export_calibration_reference(tmp_path / "trajectory")
    log = run_deployment(robot_config, trajectory, tmp_path / "raw", seed=0, plant_config=plant).log_path
    calibration, diagnostics = estimate_mocap_calibration([_decoded(log)], wheel_radius=0.016)
    truth = rz2(-YAW_OFFSET) @ OFFSET_ROBOT[:2]  # the plant's offset, in the mocap body frame
    assert math.degrees(calibration.yaw_offset) == pytest.approx(1.0, abs=0.03)
    np.testing.assert_allclose(calibration.offset, truth, atol=1e-3)
    assert diagnostics["speed_scale"] == pytest.approx(1.0, abs=0.02)

    # With the calibration in the config the firmware port logs axle poses, and
    # estimating on them again leaves next to nothing.
    calibrated = export_robot_config(
        tmp_path / "calibrated" / "ROBOTCFG.CFG",
        physical_params=SimpleNamespace(wheel_radius=0.016, base_diameter=0.0845, max_wheel_speed=230.0),
        controller_gains=[2.8, 4.7, 7.0, 3.0, 0.0, 0.0],
        overrides=calibration.config_values(),
    )
    assert load_robot_config_file(calibrated)["mocap_yaw_offset"] == pytest.approx(calibration.yaw_offset)
    assert "mocap_yaw_offset" not in load_robot_config_file(robot_config)
    log = run_deployment(calibrated, trajectory, tmp_path / "corrected", seed=0, plant_config=plant).log_path
    residual, _ = estimate_mocap_calibration([_decoded(log)], wheel_radius=0.016)
    assert abs(math.degrees(residual.yaw_offset)) < 0.03
    assert np.max(np.abs(residual.offset)) < 1e-3


def test_firmware_port_reads_the_calibration_from_a_robot_config_yaml(tmp_path):
    from wmr_simulator.mujoco_sim.firmware import FirmwareConfig

    calibration = MocapCalibration(YAW_OFFSET, (0.011, -0.004))
    config = {
        "robot": {"wheel_radius": 0.016, "base_diameter": 0.0845, "max_wheel_speed": 230.0},
        "controller": {"gains": [2.8, 4.7, 7.0, 3.0, 0.0, 0.0]},
    }
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "robot_config.yaml").write_text(yaml.safe_dump(config))
    assert FirmwareConfig.from_file(tmp_path / "plain" / "robot_config.yaml").mocap_calibration == MocapCalibration()
    (tmp_path / "calibrated").mkdir()
    (tmp_path / "calibrated" / "robot_config.yaml").write_text(
        yaml.safe_dump({**config, "mocap_calibration": calibration.to_mapping()})
    )
    loaded = FirmwareConfig.from_file(tmp_path / "calibrated" / "robot_config.yaml").mocap_calibration
    assert loaded.yaw_offset == pytest.approx(calibration.yaw_offset)
    np.testing.assert_allclose(loaded.offset, calibration.offset)
