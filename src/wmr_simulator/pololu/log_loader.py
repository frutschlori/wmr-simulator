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


def load_pololu_traj_control_log(path: str | Path, *, clip_after_first_trajectory: bool = False) -> SimulationLog:
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

    reference_rows = _rows_with(columns, data, ("x_des", "y_des", "yaw_des", "v_ff", "w_ff"))
    pose_rows = _rows_with(columns, data, ("x_raw", "y_raw", "yaw_raw"))
    wheel_rows = _rows_with(
        columns,
        data,
        ("v_actual", "w_actual", "omega_r_meas", "omega_l_meas", "duty_r", "duty_l"),
    )
    command_rows = _rows_with(columns, data, ("omega_r_cmd", "omega_l_cmd"))

    if len(reference_rows) == 0 or len(pose_rows) == 0 or len(wheel_rows) == 0 or len(command_rows) == 0:
        raise ValueError(f"Log {path} does not contain all required event streams.")

    zeros = np.zeros(len(reference_rows), dtype=np.float32)
    reference_states = np.column_stack(
        [
            _col(columns, reference_rows, "x_des"),
            _col(columns, reference_rows, "y_des"),
            _col(columns, reference_rows, "yaw_des"),
            _col(columns, reference_rows, "v_ff"),
            zeros,
            _col(columns, reference_rows, "w_ff"),
            zeros,
            zeros,
        ]
    )

    pose_states = np.column_stack(
        [
            _col(columns, pose_rows, "x_raw"),
            _col(columns, pose_rows, "y_raw"),
            _col(columns, pose_rows, "yaw_raw"),
        ]
    )

    return SimulationLog(
        reference=ReferenceLog(
            time_s=jnp.asarray(_col(columns, reference_rows, "ts"), dtype=jnp.float32),
            states=jnp.asarray(reference_states, dtype=jnp.float32),
        ),
        wheel=WheelLog(
            time_s=jnp.asarray(_col(columns, wheel_rows, "ts"), dtype=jnp.float32),
            speeds=jnp.asarray(
                np.column_stack(
                    [
                        _col(columns, wheel_rows, "omega_r_meas"),
                        _col(columns, wheel_rows, "omega_l_meas"),
                    ]
                ),
                dtype=jnp.float32,
            ),
            vel_omega=jnp.asarray(
                np.column_stack(
                    [
                        _col(columns, wheel_rows, "v_actual"),
                        _col(columns, wheel_rows, "w_actual"),
                    ]
                ),
                dtype=jnp.float32,
            ),
            duty_cycle=jnp.asarray(
                np.column_stack(
                    [
                        _col(columns, wheel_rows, "duty_r"),
                        _col(columns, wheel_rows, "duty_l"),
                    ]
                ),
                dtype=jnp.float32,
            ),
        ),
        pose=PoseLog(
            time_s=jnp.asarray(_col(columns, pose_rows, "ts"), dtype=jnp.float32),
            states=jnp.asarray(pose_states, dtype=jnp.float32),
            true_states=jnp.asarray(pose_states, dtype=jnp.float32),
            command_time_s=jnp.asarray(_col(columns, command_rows, "ts"), dtype=jnp.float32),
            wheel_cmd=jnp.asarray(
                np.column_stack(
                    [
                        _col(columns, command_rows, "omega_r_cmd"),
                        _col(columns, command_rows, "omega_l_cmd"),
                    ]
                ),
                dtype=jnp.float32,
            ),
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
    parser.add_argument("--log", type=str, default="Pololu Data/Experiments/2026_06_22/Logs/optimized/50ms_turbo_2/")
    parser.add_argument("--output", type=str, default=None, help="Output filename prefix")
    parser.add_argument("--out-dir", type=str, default="Pololu Data/Experiments/2026_06_22/Plots/optimized/")
    parser.add_argument("--clip-after-first-trajectory", default=True, action="store_true")
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
            out_prefix = f"{args.output}_{path.stem}" if args.output is not None else f"pololu_{path.stem}"
            plot_logged_summary(log, out_prefix=out_prefix, out_dir=output_dir)
        raise SystemExit(0)

    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
    )
    print_log_summary(log_path, log)
    out_prefix = args.output if args.output is not None else f"pololu_{log_path.stem}"
    plot_logged_summary(log, out_prefix=out_prefix, out_dir=args.out_dir)
