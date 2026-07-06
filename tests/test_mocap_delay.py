import numpy as np
import pytest

from wmr_simulator.identification.mocap_delay import (
    estimate_delay_by_crosscorrelation,
    estimate_mocap_delay_from_log_file,
    mocap_yaw_rate,
)
from wmr_simulator.pololu.log_loader import POLOLU_TRAJ_CONTROL_COLUMNS


def _yaw_rate_signal(time_s: np.ndarray) -> np.ndarray:
    # Broadband yaw-rate profile with enough excitation for a sharp peak.
    return (
        1.5 * np.sin(2.0 * np.pi * 0.7 * time_s)
        + 0.8 * np.sin(2.0 * np.pi * 2.3 * time_s + 0.4)
        + 0.3 * np.sin(2.0 * np.pi * 5.1 * time_s + 1.1)
    )


def _yaw_signal(time_s: np.ndarray) -> np.ndarray:
    # Analytic integral of _yaw_rate_signal (yaw observed by mocap).
    return (
        -1.5 / (2.0 * np.pi * 0.7) * np.cos(2.0 * np.pi * 0.7 * time_s)
        - 0.8 / (2.0 * np.pi * 2.3) * np.cos(2.0 * np.pi * 2.3 * time_s + 0.4)
        - 0.3 / (2.0 * np.pi * 5.1) * np.cos(2.0 * np.pi * 5.1 * time_s + 1.1)
    )


@pytest.mark.parametrize("true_delay", [0.0, 0.012, 0.043, -0.02])
def test_recovers_synthetic_delay(true_delay):
    rng = np.random.default_rng(0)
    imu_time = np.arange(0.0, 20.0, 0.005)  # 200 Hz gyro
    gyro = _yaw_rate_signal(imu_time) + 0.05 * rng.standard_normal(len(imu_time))
    # Mocap samples the same physical signal but is logged `true_delay` late:
    # the value observed at physical time t appears at timestamp t + delay.
    mocap_time = np.arange(0.0, 20.0, 0.01) + 0.001 * rng.standard_normal(2000)  # jittered ~100 Hz
    mocap_time = np.sort(mocap_time)
    omega_mocap = _yaw_rate_signal(mocap_time - true_delay) + 0.05 * rng.standard_normal(len(mocap_time))

    result = estimate_delay_by_crosscorrelation(
        mocap_time, omega_mocap, imu_time, gyro, max_delay_s=0.1
    )
    assert result["delay_s"] == pytest.approx(true_delay, abs=0.002)
    assert result["correlation"] > 0.9


def test_inverted_gyro_axis_still_locks_with_negative_correlation():
    imu_time = np.arange(0.0, 15.0, 0.005)
    gyro = -_yaw_rate_signal(imu_time)  # inverted z axis
    mocap_time = np.arange(0.0, 15.0, 0.01)
    omega_mocap = _yaw_rate_signal(mocap_time - 0.03)

    result = estimate_delay_by_crosscorrelation(
        mocap_time, omega_mocap, imu_time, gyro, max_delay_s=0.1
    )
    assert result["delay_s"] == pytest.approx(0.03, abs=0.002)
    assert result["correlation"] < -0.9


def test_mocap_yaw_rate_unwraps_and_uses_midpoints():
    time_s = np.arange(0.0, 1.01, 0.1)
    omega_true = 4.0  # rad/s, wraps past +-pi within the window
    yaw = (omega_true * time_s + np.pi) % (2.0 * np.pi) - np.pi
    midpoints, rate = mocap_yaw_rate(time_s, yaw)
    assert np.allclose(rate, omega_true, atol=1e-6)
    assert np.allclose(midpoints, 0.5 * (time_s[:-1] + time_s[1:]))


def _write_synthetic_log(path, true_delay: float, with_imu: bool = True) -> None:
    """Pololu-format csv: 100 Hz mocap rows (delayed timestamps) + 200 Hz gyro rows."""
    columns = list(POLOLU_TRAJ_CONTROL_COLUMNS)
    rows = []

    def row(ts_ms: float, **values) -> list[str]:
        record = {name: "" for name in columns}
        record["ts"] = f"{ts_ms:.3f}"
        for name, value in values.items():
            record[name] = f"{value:.6f}"
        return [record[name] for name in columns]

    for t in np.arange(0.0, 12.0, 0.005):  # gyro at 200 Hz
        if with_imu:
            rows.append(row(1000.0 * t, gyro_z=_yaw_rate_signal(t)))
    for t in np.arange(0.0, 12.0, 0.01):  # mocap at 100 Hz, logged late
        # The pose sampled at physical time t appears with timestamp t + delay.
        rows.append(row(1000.0 * (t + true_delay), x_raw=0.0, y_raw=0.0, yaw_raw=_yaw_signal(t)))

    lines = [",".join(columns)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_estimation_from_synthetic_csv(tmp_path):
    true_delay = 0.025
    log_path = tmp_path / "synthetic.csv"
    _write_synthetic_log(log_path, true_delay)
    result = estimate_mocap_delay_from_log_file(log_path, max_delay_s=0.1)
    assert result["delay_s"] == pytest.approx(true_delay, abs=0.004)


def test_missing_imu_data_raises(tmp_path):
    log_path = tmp_path / "no_imu.csv"
    _write_synthetic_log(log_path, 0.02, with_imu=False)
    with pytest.raises(ValueError, match="no IMU gyro samples"):
        estimate_mocap_delay_from_log_file(log_path)


def test_insufficient_overlap_raises():
    time_short = np.arange(0.0, 0.3, 0.01)
    signal = np.sin(time_short)
    with pytest.raises(ValueError, match="overlap"):
        estimate_delay_by_crosscorrelation(time_short, signal, time_short, signal, max_delay_s=0.2)
