"""The deployment driver: does it produce a log the pipeline accepts as real?"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from wmr_simulator.mujoco_sim.deploy import (
    DEFAULT_START_OFFSET_ANGLE,
    DEFAULT_START_OFFSET_RADIUS,
    next_log_name,
    run_deployment,
    sample_start_offset,
)

TUNED_STATIC_GAINS = [2.0, 6.7, 7.4, 3.2, 0.0]


@pytest.fixture(scope="module")
def robot_config(tmp_path_factory):
    from wmr_simulator.pololu.robot_config import export_robot_config

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0)
    return export_robot_config(
        tmp_path_factory.mktemp("config") / "ROBOTCFG.CFG",
        physical_params=physical,
        controller_gains=TUNED_STATIC_GAINS,
    )


@pytest.fixture(scope="module")
def short_trajectory(tmp_path_factory):
    """2 s of a gentle arc, written as a Pololu reference JSN."""
    dt = 0.05
    num = 41
    time = np.arange(num) * dt
    speed, yaw_rate = 0.5, 0.4
    yaw = yaw_rate * time
    radius = speed / yaw_rate
    states = np.column_stack([radius * np.sin(yaw), radius * (1.0 - np.cos(yaw)), yaw])
    actions = np.column_stack([np.full(num - 1, speed), np.full(num - 1, yaw_rate)])
    path = tmp_path_factory.mktemp("trajectory") / "TRJ0001.JSN"
    path.write_text(
        json.dumps(
            {
                "result": [
                    {
                        "dt": dt,
                        "num_states": num,
                        "num_actions": num - 1,
                        "states": states.tolist(),
                        "actions": actions.tolist(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_deployment_writes_a_log_the_loader_accepts(tmp_path, robot_config, short_trajectory):
    from wmr_simulator.pololu.decode_binary import decode_file
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log

    result = run_deployment(robot_config, short_trajectory, tmp_path, seed=0)
    assert result.log_path.name == "TR00"
    assert result.duration == pytest.approx(41 * 0.05)

    csv_path = result.log_path.with_name("TR00.csv")
    assert decode_file(str(result.log_path), str(csv_path))
    log = load_pololu_traj_control_log(csv_path)
    # 20 Hz outer, 100 Hz inner and mocap, over ~2 s.
    assert len(log.reference.time_s) > 35
    assert len(log.pose.time_s) > 180
    assert len(log.wheel.time_s) > 180
    assert len(log.pose.command_time_s) == len(log.reference.time_s)


def test_the_log_looks_like_the_robots_own(tmp_path, robot_config, short_trajectory):
    """Same seven tags at the same rates, and no column the robot leaves empty."""
    import csv
    import struct

    from wmr_simulator.pololu.decode_binary import FLOAT_COUNTS, decode_file

    result = run_deployment(robot_config, short_trajectory, tmp_path, seed=0)
    raw = result.log_path.read_bytes()
    position = 4
    counts: dict[int, int] = {}
    while position + 5 <= len(raw):
        _, tag = struct.unpack("<IB", raw[position : position + 5])
        counts[tag] = counts.get(tag, 0) + 1
        position += 5 + 4 * FLOAT_COUNTS[tag]
    assert sorted(counts) == [2, 3, 4, 5, 6, 7, 9]
    assert counts[2] == counts[3] == counts[4]  # one outer tick, three records
    assert counts[5] == counts[6]  # the inner loop writes both together
    assert counts[7] == counts[9] == pytest.approx(counts[5], abs=2)

    csv_path = result.log_path.with_name("TR00.csv")
    decode_file(str(result.log_path), str(csv_path))
    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        assert row["x"] == row["y"] == row["yaw"] == ""
        assert row["v_actual"] == row["w_actual"] == ""


def test_runs_are_reproducible_and_take_the_next_free_slot(tmp_path, robot_config, short_trajectory):
    first = run_deployment(robot_config, short_trajectory, tmp_path, seed=0)
    second = run_deployment(robot_config, short_trajectory, tmp_path, seed=1)
    assert [first.log_path.name, second.log_path.name] == ["TR00", "TR01"]
    assert first.start_offset != second.start_offset

    repeat = run_deployment(robot_config, short_trajectory, tmp_path, seed=0)
    assert repeat.log_path.name == "TR02"
    assert repeat.start_offset == first.start_offset
    assert repeat.log_path.read_bytes() == first.log_path.read_bytes()


def test_an_existing_log_is_never_overwritten(tmp_path, robot_config, short_trajectory):
    run_deployment(robot_config, short_trajectory, tmp_path, seed=0)
    with pytest.raises(FileExistsError):
        run_deployment(robot_config, short_trajectory, tmp_path, seed=0, log_name="TR00")


def test_next_log_name_treats_a_decoded_csv_as_taken(tmp_path):
    assert next_log_name(tmp_path) == "TR00"
    (tmp_path / "TR00").touch()
    assert next_log_name(tmp_path) == "TR01"
    # The pipeline writes TR01.csv next to the binary; reusing TR01 would mix runs.
    (tmp_path / "TR01.csv").touch()
    assert next_log_name(tmp_path) == "TR02"


def test_start_offsets_match_the_hand_placement_distribution():
    rng = np.random.default_rng(0)
    offsets = np.array([sample_start_offset(rng, DEFAULT_START_OFFSET_RADIUS, DEFAULT_START_OFFSET_ANGLE)
                        for _ in range(2000)])
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    assert radii.max() <= DEFAULT_START_OFFSET_RADIUS
    assert np.abs(offsets[:, 2]).max() <= DEFAULT_START_OFFSET_ANGLE
    # Uniform by area, not clustered at the centre: the median sits at r/sqrt(2).
    assert np.median(radii) == pytest.approx(DEFAULT_START_OFFSET_RADIUS / math.sqrt(2), rel=0.05)
    assert sample_start_offset(rng, 0.0, 0.0) == pytest.approx([0.0, 0.0, 0.0])


def test_the_robot_actually_drives_the_trajectory(tmp_path, robot_config, short_trajectory):
    """Ground truth: it tracks, from a hand-placement offset, without saturating."""
    result = run_deployment(robot_config, short_trajectory, tmp_path, seed=0)
    assert np.hypot(*result.start_offset[:2]) > 0.01  # it really was placed off-reference
    assert result.tracking_rmse < 0.10
    assert result.final_pose_error < 0.10
    assert result.duty_saturated_fraction == 0.0
