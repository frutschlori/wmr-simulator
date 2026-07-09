from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np

from wmr_simulator.pololu.measurement_smoothing import (
    smooth_and_align_encoder_speeds,
    smooth_pose_stream,
)
from wmr_simulator.types import PoseLog, ReferenceLog, SimulationLog, WheelLog


POLOLU_TRAJ_CONTROL_COLUMNS = (
    "ts",
    "x",
    "y",
    "yaw",
    "x_des",
    "y_des",
    "yaw_des",
    "v_ff",
    "w_ff",
    "v_actual",
    "w_actual",
    "omega_l_cmd",
    "omega_r_cmd",
    "omega_l_meas",
    "omega_r_meas",
    "duty_l",
    "duty_r",
    "x_err",
    "y_err",
    "yaw_err",
    "x_raw",
    "y_raw",
    "yaw_raw",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


def load_pololu_traj_control_log(
    path: str | Path,
    *,
    clip_after_first_trajectory: bool = False,
    mocap_delay_s: float = 0.0,
) -> SimulationLog:
    """Load one Pololu traj-control csv into a SimulationLog.

    ``mocap_delay_s`` > 0 compensates the mocap transport latency (network +
    radio/UART; see identification.mocap_delay): the pose logged at time t was
    assumed at t - mocap_delay_s, so all mocap timestamps are shifted back by
    that amount before use.

    The mocap poses are always smoothed with a Savitzky-Golay filter
    (measurement_smoothing.smooth_pose_stream): repeated frames, samples too
    close together, and residual-outlier poses are dropped, ``pose.states``
    holds the filtered poses and ``pose.twists`` holds the filter-derivative
    body twists at the pose times, so downstream consumers never
    finite-difference raw mocap. The raw poses are kept in ``pose.true_states``
    for diagnostics/plots; the survivors of rejection in ``pose.clean_states``.

    The encoder wheel speeds get a light Savitzky-Golay smoothing and are
    advanced by the firmware low-pass group delay to realign them with the
    mocap motion (measurement_smoothing.smooth_and_align_encoder_speeds).

    Smoothing and outlier-rejection defaults for both mocap and encoders are
    configured in the measurement_smoothing submodule (DEFAULT_* constants).
    """
    columns, data = _read_time_normalized(Path(path), clip_after_first_trajectory)

    reference_time, reference_values = _sparse_stream(
        columns,
        data,
        ("x_des", "y_des", "yaw_des", "v_ff", "w_ff"),
    )
    pose_time, pose_states = _sparse_stream(columns, data, ("x_raw", "y_raw", "yaw_raw"))
    pose_time = pose_time - np.float32(mocap_delay_s)
    raw_pose_states = pose_states
    # Duplicate frames are dropped inside the filter but the smoothed grid is
    # interpolated back to all raw timestamps, so every stream keeps its
    # original length/time base.
    smoothed = smooth_pose_stream(pose_time, pose_states)
    pose_states = smoothed.pose(pose_time)
    pose_twists = smoothed.body_twist(pose_time)
    wheel_time, wheel_speeds = _sparse_stream(
        columns,
        data,
        ("omega_r_meas", "omega_l_meas"),
        trigger_names=("omega_r_meas", "omega_l_meas"),
    )
    # Light extra smoothing, then advance by the firmware LP group delay so the
    # encoder speeds line up in time with the mocap-derived motion
    # (measurement_smoothing.smooth_and_align_encoder_speeds, DEFAULT_ENCODER_* knobs).
    wheel_speeds = smooth_and_align_encoder_speeds(wheel_time, wheel_speeds)
    wheel_vel_omega = _latest_values_at_times(columns, data, ("v_actual", "w_actual"), wheel_time)
    duty_cycle = _latest_values_at_times(columns, data, ("duty_r", "duty_l"), wheel_time)
    command_time, wheel_cmd = _sparse_stream(
        columns,
        data,
        ("omega_r_cmd", "omega_l_cmd"),
        trigger_names=("omega_r_cmd", "omega_l_cmd"),
    )

    if len(reference_time) == 0 or len(pose_time) == 0 or len(wheel_time) == 0 or len(command_time) == 0:
        raise ValueError(f"Log {path} does not contain all required event streams.")

    zeros = np.zeros(len(reference_time), dtype=np.float32)
    reference_states = np.column_stack(
        [
            reference_values[:, 0],
            reference_values[:, 1],
            reference_values[:, 2],
            reference_values[:, 3],
            zeros,
            reference_values[:, 4],
            zeros,
            zeros,
        ]
    )

    return SimulationLog(
        reference=ReferenceLog(
            time_s=jnp.asarray(reference_time, dtype=jnp.float32),
            states=jnp.asarray(reference_states, dtype=jnp.float32),
        ),
        wheel=WheelLog(
            time_s=jnp.asarray(wheel_time, dtype=jnp.float32),
            speeds=jnp.asarray(wheel_speeds, dtype=jnp.float32),
            vel_omega=jnp.asarray(wheel_vel_omega, dtype=jnp.float32),
            duty_cycle=jnp.asarray(duty_cycle, dtype=jnp.float32),
        ),
        pose=PoseLog(
            time_s=jnp.asarray(pose_time, dtype=jnp.float32),
            states=jnp.asarray(pose_states, dtype=jnp.float32),
            true_states=jnp.asarray(raw_pose_states, dtype=jnp.float32),
            command_time_s=jnp.asarray(command_time, dtype=jnp.float32),
            wheel_cmd=jnp.asarray(wheel_cmd, dtype=jnp.float32),
            twists=jnp.asarray(pose_twists, dtype=jnp.float32),
            clean_time_s=jnp.asarray(smoothed.clean_time, dtype=jnp.float32),
            clean_states=jnp.asarray(smoothed.clean_pose, dtype=jnp.float32),
        ),
    )


def load_imu_gyro_z(
    path: str | Path,
    *,
    clip_after_first_trajectory: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """IMU yaw rate stream from one traj-control csv, converted to rad/s.

    The csv logs the gyro in deg/s; timestamps share the time base of
    load_pololu_traj_control_log (seconds, zeroed at the first logged row).
    Returns empty arrays for logs without IMU samples (older logs have the
    gyro columns in the header but never fill them).
    """
    columns, data = _read_time_normalized(Path(path), clip_after_first_trajectory)
    gyro_time, gyro = _sparse_stream(columns, data, ("gyro_z",), trigger_names=("gyro_z",))
    return gyro_time, np.deg2rad(gyro[:, 0]) if len(gyro) else gyro.reshape(0)


def _read_time_normalized(path: Path, clip_after_first_trajectory: bool) -> tuple[list[str], np.ndarray]:
    """Read a traj-control csv with the ts column sorted and in seconds from the first row."""
    columns, data = _read_csv(path)
    if tuple(columns) != POLOLU_TRAJ_CONTROL_COLUMNS:
        raise ValueError(f"Unexpected columns in {path}: {tuple(columns)}")

    ts_index = columns.index("ts")
    data = data[np.isfinite(data[:, ts_index])]
    data = data[np.argsort(data[:, ts_index])]
    time_s = data[:, ts_index] / 1000.0
    time_s = time_s - time_s[0]
    data = data.copy()
    data[:, ts_index] = time_s

    if clip_after_first_trajectory:
        data = _clip_after_first_reference_stop(columns, data)
    return columns, data


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if not any(ch.isalpha() for ch in first_line):
        raise ValueError(f"Expected a header row in {path}.")

    loaded = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    columns = list(loaded.dtype.names or ())
    data = np.column_stack([loaded[name] for name in columns])
    if data.ndim == 1:
        data = data[None, :]
    return columns, data


def _rows_with(columns: list[str], data: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    indices = [columns.index(name) for name in names]
    return data[np.all(np.isfinite(data[:, indices]), axis=1)]


def _sparse_stream(
    columns: list[str],
    data: np.ndarray,
    names: tuple[str, ...],
    *,
    trigger_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    value_indices = [columns.index(name) for name in names]
    trigger_indices = value_indices if trigger_names is None else [columns.index(name) for name in trigger_names]
    ts_index = columns.index("ts")
    latest = np.full(len(value_indices), np.nan, dtype=np.float32)
    times: list[float] = []
    values: list[np.ndarray] = []

    for row in data:
        row_values = row[value_indices]
        updates = np.isfinite(row_values)
        if np.any(updates):
            latest[updates] = row_values[updates]
        if not np.any(np.isfinite(row[trigger_indices])) or not np.all(np.isfinite(latest)):
            continue
        times.append(float(row[ts_index]))
        values.append(latest.copy())

    if not values:
        return np.empty(0, dtype=np.float32), np.empty((0, len(value_indices)), dtype=np.float32)
    return np.asarray(times, dtype=np.float32), np.vstack(values).astype(np.float32, copy=False)


def _latest_values_at_times(
    columns: list[str],
    data: np.ndarray,
    names: tuple[str, ...],
    target_time_s: np.ndarray,
    *,
    default: float = 0.0,
) -> np.ndarray:
    value_indices = [columns.index(name) for name in names]
    ts_index = columns.index("ts")
    latest = np.full(len(value_indices), default, dtype=np.float32)
    output = np.empty((len(target_time_s), len(value_indices)), dtype=np.float32)
    source_index = 0

    for target_index, target_time in enumerate(target_time_s):
        while source_index < len(data) and data[source_index, ts_index] <= target_time:
            row_values = data[source_index, value_indices]
            updates = np.isfinite(row_values)
            if np.any(updates):
                latest[updates] = row_values[updates]
            source_index += 1
        output[target_index] = latest
    return output


def _col(columns: list[str], data: np.ndarray, name: str) -> np.ndarray:
    return data[:, columns.index(name)]


def _clip_after_first_reference_stop(
    columns: list[str],
    data: np.ndarray,
    *,
    min_zero_rows: int = 3,
    zero_tolerance: float = 1e-6,
) -> np.ndarray:
    reference_rows = _rows_with(columns, data, ("x_des", "y_des", "yaw_des", "v_ff", "w_ff"))
    if len(reference_rows) < min_zero_rows:
        return data

    ts_index = columns.index("ts")
    v_ff = _col(columns, reference_rows, "v_ff")
    w_ff = _col(columns, reference_rows, "w_ff")
    zero_reference = (np.abs(v_ff) <= zero_tolerance) & (np.abs(w_ff) <= zero_tolerance)
    motion_seen = False
    for index in range(len(reference_rows) - min_zero_rows + 1):
        if not zero_reference[index]:
            motion_seen = True
            continue
        if motion_seen and np.all(zero_reference[index : index + min_zero_rows]):
            return data[data[:, ts_index] < reference_rows[index, ts_index]]
    return data


def list_pololu_log_paths(log_dir: str | Path) -> list[Path]:
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise ValueError(f"Log path must be a directory: {log_dir}")
    paths = sorted(path for path in log_dir.iterdir() if path.is_file() and _looks_like_pololu_log(path))
    if not paths:
        raise ValueError(f"No log files found in: {log_dir}")
    return paths


def _looks_like_pololu_log(path: Path) -> bool:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeDecodeError, IndexError):
        return False
    columns = tuple(name.strip() for name in first_line.split(","))
    return columns == POLOLU_TRAJ_CONTROL_COLUMNS


def print_log_summary(log_path: Path, log: SimulationLog):
    print(f"Loaded {log_path}")
    print(f"Reference samples: {len(log.reference.time_s)}")
    print(f"Pose samples: {len(log.pose.time_s)}")
    print(f"Wheel samples: {len(log.wheel.time_s)}")
    print(f"Command samples: {len(log.pose.command_time_s)}")


if __name__ == "__main__":
    from wmr_simulator.visualization.pololu import plot_logged_summary

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="Pololu Data/Experiments/2026_07_07/12/binaries/decoded/TR07.csv")
    parser.add_argument("--output", type=str, default=None, help="Output filename prefix")
    parser.add_argument("--out-dir", type=str, default="visualize")
    parser.add_argument("--clip-after-first-trajectory", default=True, action="store_true")
    # Mocap/encoder smoothing defaults are configured in the measurement_smoothing submodule.
    args = parser.parse_args()

    log_path = Path(args.log)
    if log_path.is_dir():
        output_dir = Path(args.out_dir) / log_path.name
        for path in list_pololu_log_paths(log_path):
            log = load_pololu_traj_control_log(
                path,
                clip_after_first_trajectory=args.clip_after_first_trajectory,
            )
            print_log_summary(path, log)
            imu_time, imu_gyro_z = load_imu_gyro_z(
                path, clip_after_first_trajectory=args.clip_after_first_trajectory
            )
            out_prefix = f"{args.output}_{path.stem}" if args.output is not None else f"pololu_{path.stem}"
            plot_logged_summary(
                log,
                out_prefix=out_prefix,
                out_dir=output_dir,
                imu_time_s=imu_time,
                imu_gyro_z=imu_gyro_z,
            )
        raise SystemExit(0)

    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
    )
    print_log_summary(log_path, log)
    imu_time, imu_gyro_z = load_imu_gyro_z(
        log_path, clip_after_first_trajectory=args.clip_after_first_trajectory
    )
    out_prefix = args.output if args.output is not None else f"pololu_{log_path.stem}"
    plot_logged_summary(
        log,
        out_prefix=out_prefix,
        out_dir=args.out_dir,
        imu_time_s=imu_time,
        imu_gyro_z=imu_gyro_z,
    )
