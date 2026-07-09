import numpy as np
import pytest

from wmr_simulator.pololu.log_loader import (
    POLOLU_TRAJ_CONTROL_COLUMNS,
    load_imu_gyro_z,
    load_pololu_traj_control_log,
)


def _write_synthetic_log(path, *, duration=4.0, dt=0.01, radius=0.5, omega=1.0):
    """Minimal traj-control csv: constant-twist arc with all event streams.

    Every pose row is duplicated once (same pose, +3 ms) partway through to
    mimic the double-logged mocap frames seen in the real logs.
    """
    rng = np.random.default_rng(0)
    rows = []
    time_s = np.arange(0.0, duration, dt)
    for i, t in enumerate(time_s):
        ts = 1000.0 * t
        yaw = omega * t
        x = radius * np.sin(yaw)
        y = radius * (1.0 - np.cos(yaw))
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        row = {name: "" for name in POLOLU_TRAJ_CONTROL_COLUMNS}
        row["ts"] = f"{ts:.0f}"
        row.update(x_raw=f"{x:.6f}", y_raw=f"{y:.6f}", yaw_raw=f"{yaw:.6f}")
        row.update(x_des=f"{x:.6f}", y_des=f"{y:.6f}", yaw_des=f"{yaw:.6f}", v_ff="0.5", w_ff="1.0")
        row.update(omega_l_meas="10.0", omega_r_meas="12.0", omega_l_cmd="10.0", omega_r_cmd="12.0")
        row.update(v_actual="0.5", w_actual="1.0", duty_l="0.3", duty_r="0.35")
        row["gyro_z"] = f"{np.degrees(omega):.4f}"  # the csv logs the gyro in deg/s
        rows.append(row)
        if 100 <= i < 105:  # duplicated mocap frames, +3 ms
            dup = {name: "" for name in POLOLU_TRAJ_CONTROL_COLUMNS}
            dup["ts"] = f"{ts + 3.0:.0f}"
            dup.update(x_raw=row["x_raw"], y_raw=row["y_raw"], yaw_raw=row["yaw_raw"])
            rows.append(dup)

    lines = [",".join(POLOLU_TRAJ_CONTROL_COLUMNS)]
    lines += [",".join(row[name] for name in POLOLU_TRAJ_CONTROL_COLUMNS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def synthetic_log(tmp_path):
    log_path = tmp_path / "TR99.csv"
    _write_synthetic_log(log_path)
    return log_path


def test_loader_fills_savgol_twists(synthetic_log):
    log = load_pololu_traj_control_log(synthetic_log)
    twists = np.asarray(log.pose.twists, dtype=float)
    assert twists.shape == (len(log.pose.time_s), 3)
    # Constant-twist arc: v_x = radius * omega = 0.5, v_y = 0, omega = 1.
    interior = slice(20, -20)
    np.testing.assert_allclose(twists[interior, 0], 0.5, atol=0.02)
    np.testing.assert_allclose(twists[interior, 1], 0.0, atol=0.02)
    np.testing.assert_allclose(twists[interior, 2], 1.0, atol=0.05)


def test_loader_keeps_raw_poses_and_smooths_states(synthetic_log):
    log = load_pololu_traj_control_log(synthetic_log)
    raw = np.asarray(log.pose.true_states, dtype=float)
    smooth = np.asarray(log.pose.states, dtype=float)
    assert raw.shape == smooth.shape
    # Raw stream still contains the duplicated frames; the smoothed one and the
    # twists must not produce the zero-velocity artefact on those intervals.
    dup = np.all(np.diff(raw, axis=0) == 0.0, axis=1)
    assert dup.sum() == 5
    twists = np.asarray(log.pose.twists, dtype=float)
    assert np.all(np.abs(twists[:-1][dup, 0] - 0.5) < 0.05)
    # Smoothed poses stay close to the (noise-free) raw ones.
    assert np.max(np.abs(smooth[:, :2] - raw[:, :2])) < 5e-3


def test_load_imu_gyro_z_converts_to_rad_s(synthetic_log):
    gyro_time, gyro_z = load_imu_gyro_z(synthetic_log)
    assert len(gyro_time) == len(gyro_z) > 0
    np.testing.assert_allclose(gyro_z, 1.0, atol=1e-4)  # written as deg/s of omega = 1 rad/s


def test_residual_dataset_uses_savgol_twists(synthetic_log):
    import jax.numpy as jnp

    from wmr_simulator.residual_model.residual import build_residual_dataset
    from wmr_simulator.types import PhysicalParams

    params = PhysicalParams(
        wheel_radius=jnp.asarray(0.016),
        base_diameter=jnp.asarray(0.09),
    )
    log = load_pololu_traj_control_log(synthetic_log)
    dataset = build_residual_dataset(log, params)
    measured = dataset["measured_twist"]
    # Twist targets come from the filter: constant twist, no differencing spikes.
    np.testing.assert_allclose(measured[10:-10, 0], 0.5, atol=0.03)
    np.testing.assert_allclose(measured[10:-10, 2], 1.0, atol=0.06)
