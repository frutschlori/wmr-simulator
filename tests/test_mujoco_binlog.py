"""The deployment's SD-card log writer, round-tripped through the real decoder."""

from __future__ import annotations

import csv
import math
import struct

import numpy as np
import pytest

from wmr_simulator.mujoco_sim.binlog import MAGIC_HEADER, BinaryLogWriter, LogTag

# f32 storage plus the decoder's "%.6f" rendering.
TOLERANCE = dict(rel=1e-6, abs=1e-6)


def decode(path):
    """Run the pipeline's own decoder and return the csv rows as dicts."""
    from wmr_simulator.pololu.decode_binary import decode_file

    csv_path = path.with_name(path.name + ".csv")
    assert decode_file(str(path), str(csv_path))
    with open(csv_path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_round_trip_through_the_decoder(tmp_path):
    log_path = tmp_path / "TR00"
    with BinaryLogWriter(log_path) as writer:
        writer.setpoint(1000, 0.25, -1.5, 0.75, 0.8, -0.3)
        writer.wheel_cmd(1000, 61.25, 48.5)
        writer.tracking_error(1000, 0.012, -0.004, 0.03)
        writer.motor(1010, 0.42, -0.37)
        writer.encoder(1010, 55.5, -12.25)
        writer.mocap(1015, [0.5, -0.25, 0.016, 0.01, -0.02, 1.25])
        writer.imu(1020, [0.001, -0.002, 1.022], [1.4, -0.7, 63.35])
    assert writer.num_records == 7

    rows = {int(row["ts"]): row for row in decode(log_path)}
    assert sorted(rows) == [1000, 1010, 1015, 1020]

    first = rows[1000]
    assert float(first["x_des"]) == pytest.approx(0.25, **TOLERANCE)
    assert float(first["y_des"]) == pytest.approx(-1.5, **TOLERANCE)
    assert float(first["yaw_des"]) == pytest.approx(0.75, **TOLERANCE)
    assert float(first["v_ff"]) == pytest.approx(0.8, **TOLERANCE)
    assert float(first["w_ff"]) == pytest.approx(-0.3, **TOLERANCE)
    assert float(first["omega_l_cmd"]) == pytest.approx(61.25, **TOLERANCE)
    assert float(first["omega_r_cmd"]) == pytest.approx(48.5, **TOLERANCE)
    assert float(first["yaw_err"]) == pytest.approx(0.03, **TOLERANCE)

    assert float(rows[1010]["duty_l"]) == pytest.approx(0.42, **TOLERANCE)
    assert float(rows[1010]["omega_r_meas"]) == pytest.approx(-12.25, **TOLERANCE)
    assert float(rows[1015]["x_raw"]) == pytest.approx(0.5, **TOLERANCE)
    assert float(rows[1015]["yaw_raw"]) == pytest.approx(1.25, **TOLERANCE)
    assert float(rows[1020]["acc_z"]) == pytest.approx(1.022, **TOLERANCE)
    assert float(rows[1020]["gyro_z"]) == pytest.approx(63.35, **TOLERANCE)


def test_columns_the_real_robot_never_fills_stay_empty(tmp_path):
    """EkfState and Odom are never queued on the robot, so those columns are blank."""
    log_path = tmp_path / "TR01"
    with BinaryLogWriter(log_path) as writer:
        writer.setpoint(500, 0.0, 0.0, 0.0, 0.5, 0.0)
        writer.mocap(500, [0.0, 0.0, 0.016, 0.0, 0.0, 0.0])
    for row in decode(log_path):
        assert row["x"] == row["y"] == row["yaw"] == ""
        assert row["v_actual"] == row["w_actual"] == ""


def test_unwritten_tags_are_refused(tmp_path):
    with BinaryLogWriter(tmp_path / "TR02") as writer:
        with pytest.raises(ValueError, match="never written"):
            writer.write(0, LogTag.EKF_STATE, [0.0] * 6)
        with pytest.raises(ValueError, match="never written"):
            writer.write(0, LogTag.ODOM, [0.0, 0.0])


def test_payload_shape_and_finiteness_are_checked(tmp_path):
    with BinaryLogWriter(tmp_path / "TR03") as writer:
        with pytest.raises(ValueError, match="takes 2 floats"):
            writer.write(0, LogTag.MOTOR, [0.1])
        # A NaN would decode to an empty csv cell instead of failing.
        with pytest.raises(ValueError, match="finite"):
            writer.motor(0, 0.1, math.nan)
        with pytest.raises(ValueError, match="u32"):
            writer.motor(-1, 0.1, 0.2)


def test_file_layout_matches_the_firmware(tmp_path):
    log_path = tmp_path / "TR04"
    with BinaryLogWriter(log_path) as writer:
        writer.motor(1234, 0.5, -0.5)
    raw = log_path.read_bytes()
    assert raw[:4] == MAGIC_HEADER
    assert len(raw) == 4 + 5 + 2 * 4
    t_ms, tag = struct.unpack("<IB", raw[4:9])
    assert (t_ms, tag) == (1234, int(LogTag.MOTOR))
    assert struct.unpack("<2f", raw[9:]) == (0.5, -0.5)


def test_a_synthetic_run_loads_through_the_pipeline_log_loader(tmp_path):
    """End to end: a written log must satisfy the loader every downstream stage uses."""
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log

    log_path = tmp_path / "TR05"
    boot_ms = 15_000  # real logs start at the firmware's uptime, not at 0
    with BinaryLogWriter(log_path) as writer:
        for inner_tick in range(400):  # 4 s at 100 Hz
            t_ms = boot_ms + 10 * inner_tick
            t = 0.01 * inner_tick
            # A quarter-circle drive, with the robot idle for the first second
            # so the loader's trajectory-start clipping has something to clip.
            moving = t >= 1.0
            speed = 0.6 if moving else 0.0
            yaw = 0.4 * (t - 1.0) if moving else 0.0
            radius = 1.5
            x = radius * math.sin(yaw)
            y = radius * (1.0 - math.cos(yaw))
            if inner_tick % 5 == 0:  # 20 Hz outer loop
                writer.setpoint(t_ms, x, y, yaw, speed, 0.4 * moving)
                writer.wheel_cmd(t_ms, 36.0 * moving, 39.0 * moving)
                writer.tracking_error(t_ms, 0.01, -0.005, 0.002)
            writer.motor(t_ms, 0.31 * moving, 0.34 * moving)
            writer.encoder(t_ms, 36.2 * moving, 38.7 * moving)
            writer.mocap(t_ms + 2, [x, y, 0.016, 0.0, 0.0, yaw])
            writer.imu(t_ms + 4, [0.0, 0.0, 1.022], [0.0, 0.0, math.degrees(0.4 * moving)])

    csv_path = log_path.with_name(log_path.name + ".csv")
    decode(log_path)
    log = load_pololu_traj_control_log(csv_path)

    assert float(log.pose.time_s[0]) == pytest.approx(0.0, abs=0.05)
    assert len(log.wheel.time_s) > 250
    assert np.asarray(log.pose.states)[-1, 0] == pytest.approx(1.5 * math.sin(1.2), abs=0.05)
    # The encoder speeds survive the loader's smoothing and realignment.
    assert float(np.median(np.asarray(log.wheel.speeds)[:, 0])) == pytest.approx(38.7, abs=1.0)
