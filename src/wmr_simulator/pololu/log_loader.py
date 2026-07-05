from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np

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


def zero_phase_moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Centered moving average with no phase delay.

    Uses a symmetric window around each sample ('same' convolution) and
    normalizes by the actual number of contributing samples, so the edges are
    unbiased and the filter introduces no time shift.
    """
    if window_samples <= 1:
        return values
    kernel = np.ones(int(window_samples), dtype=float)
    counts = np.convolve(np.ones(len(values), dtype=float), kernel, mode="same")
    if values.ndim == 1:
        return (np.convolve(values.astype(float), kernel, mode="same") / counts).astype(values.dtype)
    filtered = [
        np.convolve(values[:, i].astype(float), kernel, mode="same") / counts
        for i in range(values.shape[1])
    ]
    return np.column_stack(filtered).astype(values.dtype)


def _filter_mocap_poses(pose_time_s: np.ndarray, pose_states: np.ndarray, window_s: float) -> np.ndarray:
    """Zero-phase moving average on mocap poses; yaw is filtered unwrapped."""
    if window_s <= 0.0 or len(pose_time_s) < 3:
        return pose_states
    median_dt = float(np.median(np.diff(pose_time_s)))
    if median_dt <= 0.0:
        return pose_states
    window_samples = max(int(round(window_s / median_dt)), 1)
    window_samples += 1 - window_samples % 2  # force odd for a symmetric window
    if window_samples <= 1:
        return pose_states
    xy = zero_phase_moving_average(pose_states[:, :2], window_samples)
    yaw_unwrapped = np.unwrap(pose_states[:, 2].astype(float))
    yaw = zero_phase_moving_average(yaw_unwrapped, window_samples)
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
    return np.column_stack([xy, yaw]).astype(pose_states.dtype)


def load_pololu_traj_control_log(
    path: str | Path,
    *,
    clip_after_first_trajectory: bool = False,
    mocap_filter_window_s: float = 0.0,
) -> SimulationLog:
    columns, data = _read_csv(Path(path))
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

    reference_time, reference_values = _sparse_stream(
        columns,
        data,
        ("x_des", "y_des", "yaw_des", "v_ff", "w_ff"),
    )
    pose_time, pose_states = _sparse_stream(columns, data, ("x_raw", "y_raw", "yaw_raw"))
    pose_states = _filter_mocap_poses(pose_time, pose_states, mocap_filter_window_s)
    wheel_time, wheel_speeds = _sparse_stream(
        columns,
        data,
        ("omega_r_meas", "omega_l_meas"),
        trigger_names=("omega_r_meas", "omega_l_meas"),
    )
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
            true_states=jnp.asarray(pose_states, dtype=jnp.float32),
            command_time_s=jnp.asarray(command_time, dtype=jnp.float32),
            wheel_cmd=jnp.asarray(wheel_cmd, dtype=jnp.float32),
        ),
    )


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
    parser.add_argument("--log", type=str, default="Pololu Data/Experiments/2026_07_01/TR03.csv")
    parser.add_argument("--output", type=str, default=None, help="Output filename prefix")
    parser.add_argument("--out-dir", type=str, default="visualize")
    parser.add_argument("--clip-after-first-trajectory", default=True, action="store_true")
    # Zero-phase moving-average window (seconds) on the mocap positions; 0 disables.
    parser.add_argument("--mocap-filter-window", type=float, default=0.0)
    args = parser.parse_args()

    log_path = Path(args.log)
    if log_path.is_dir():
        output_dir = Path(args.out_dir) / log_path.name
        for path in list_pololu_log_paths(log_path):
            log = load_pololu_traj_control_log(
                path,
                clip_after_first_trajectory=args.clip_after_first_trajectory,
                mocap_filter_window_s=args.mocap_filter_window,
            )
            print_log_summary(path, log)
            out_prefix = f"{args.output}_{path.stem}" if args.output is not None else f"pololu_{path.stem}"
            plot_logged_summary(log, out_prefix=out_prefix, out_dir=output_dir)
        raise SystemExit(0)

    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
        mocap_filter_window_s=args.mocap_filter_window,
    )
    print_log_summary(log_path, log)
    out_prefix = args.output if args.output is not None else f"pololu_{log_path.stem}"
    plot_logged_summary(log, out_prefix=out_prefix, out_dir=args.out_dir)
