from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_COLUMNS = (
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


def looks_like_log(path: Path) -> bool:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeDecodeError, IndexError):
        return False
    return tuple(name.strip() for name in first_line.split(",")) == EXPECTED_COLUMNS


def log_files(directory: Path) -> list[Path]:
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
    if not paths:
        raise ValueError(f"No CSV files found in {directory}")
    return paths


def load_columns(path: str | Path, max_time: float | None = None) -> tuple[list[str], np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    columns = list(data.dtype.names or ())
    values = np.column_stack([data[name] for name in columns])
    if values.ndim == 1:
        values = values[None, :]
    ts_index = columns.index("ts")
    values = values[np.isfinite(values[:, ts_index])]
    values = values[np.argsort(values[:, ts_index])]
    if max_time is not None:
        if max_time < 0:
            raise ValueError("--max-time must be non-negative")
        start_ts = values[0, ts_index]
        values = values[(values[:, ts_index] - start_ts) / 1000.0 <= max_time]
        if len(values) == 0:
            raise ValueError(f"No rows available in {path}")
    values = values.copy()
    values[:, ts_index] = (values[:, ts_index] - values[0, ts_index]) / 1000.0
    return columns, values


def rows_with(columns: list[str], data: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    indices = [columns.index(name) for name in names]
    return data[np.all(np.isfinite(data[:, indices]), axis=1)]


def col(columns: list[str], data: np.ndarray, name: str) -> np.ndarray:
    return data[:, columns.index(name)]


def mocap_vel_omega(
    time_s: np.ndarray,
    pose: np.ndarray,
    diff_window: int = 5,
    min_dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) < 2:
        return time_s, np.zeros((len(time_s), 2))

    order = np.argsort(time_s)
    time_s = time_s[order]
    pose = pose[order]
    increasing = np.concatenate([[True], np.diff(time_s) > 0.0])
    time_s = time_s[increasing]
    pose = pose[increasing]
    if len(time_s) < 2:
        return time_s, np.zeros((len(time_s), 2))

    theta = np.unwrap(pose[:, 2])
    diff_window = max(1, int(diff_window))

    if diff_window <= 1 or len(time_s) < 3:
        dt = np.diff(time_s)
        valid = dt >= min_dt
        dxy = np.diff(pose[:, :2], axis=0)[valid]
        heading = theta[1:][valid]
        if not np.any(valid):
            return np.empty(0), np.empty((0, 2))
        v = (dxy[:, 0] * np.cos(heading) + dxy[:, 1] * np.sin(heading)) / dt[valid]
        omega = np.diff(theta)[valid] / dt[valid]
        return time_s[1:][valid], np.column_stack([v, omega])

    half_window = max(1, diff_window // 2)
    if len(time_s) <= 2 * half_window:
        half_window = 1

    dt = time_s[2 * half_window :] - time_s[: -2 * half_window]
    valid = dt >= min_dt
    if not np.any(valid):
        return np.empty(0), np.empty((0, 2))

    dxy = pose[2 * half_window :, :2] - pose[: -2 * half_window, :2]
    dtheta = theta[2 * half_window :] - theta[: -2 * half_window]
    heading = theta[half_window:-half_window]
    time_out = time_s[half_window:-half_window]

    dt = dt[valid]
    dxy = dxy[valid]
    dtheta = dtheta[valid]
    heading = heading[valid]
    time_out = time_out[valid]

    v = (dxy[:, 0] * np.cos(heading) + dxy[:, 1] * np.sin(heading)) / dt
    omega = dtheta / dt
    return time_out, np.column_stack([v, omega])


def print_mocap_timing_stats(log_path: str | Path, time_s: np.ndarray) -> None:
    if len(time_s) < 2:
        print(f"{Path(log_path).name}: mocap samples={len(time_s)}")
        return
    dt_ms = np.diff(np.sort(time_s)) * 1000.0
    dt_ms = dt_ms[dt_ms > 0.0]
    if len(dt_ms) == 0:
        print(f"{Path(log_path).name}: mocap samples={len(time_s)}, no positive dt")
        return
    p = np.percentile(dt_ms, [1, 5, 50, 95, 99])
    print(
        f"{Path(log_path).name}: mocap samples={len(time_s)}, "
        f"dt_ms p1/p5/median/p95/p99={p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}/{p[3]:.1f}/{p[4]:.1f}"
    )


def stair_series(time: np.ndarray, values: np.ndarray, end_time: float | None) -> tuple[np.ndarray, np.ndarray]:
    if end_time is None or len(time) == 0 or end_time <= time[-1]:
        return time, values
    return np.concatenate([time, [end_time]]), np.concatenate([values, values[-1:]])


def sparse_stream(columns: list[str], data: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    value_indices = [columns.index(name) for name in names]
    ts_index = columns.index("ts")
    latest = np.full(len(value_indices), np.nan, dtype=float)
    times = []
    values = []

    for row in data:
        row_values = row[value_indices]
        updates = np.isfinite(row_values)
        if not np.any(updates):
            continue
        latest[updates] = row_values[updates]
        if np.all(np.isfinite(latest)):
            times.append(row[ts_index])
            values.append(latest.copy())

    if not values:
        return np.empty(0), np.empty((0, len(value_indices)))
    return np.asarray(times), np.vstack(values)


def clip_after_first_trajectory(columns: list[str], data: np.ndarray, min_zero_rows: int = 3) -> np.ndarray:
    reference_rows = rows_with(columns, data, ("x_des", "y_des", "yaw_des", "v_ff", "w_ff"))
    if len(reference_rows) < min_zero_rows:
        return data
    ts_index = columns.index("ts")
    zero = (np.abs(col(columns, reference_rows, "v_ff")) <= 1e-6) & (
        np.abs(col(columns, reference_rows, "w_ff")) <= 1e-6
    )
    motion_seen = False
    for index in range(len(reference_rows) - min_zero_rows + 1):
        if not zero[index]:
            motion_seen = True
            continue
        if motion_seen and np.all(zero[index : index + min_zero_rows]):
            return data[data[:, ts_index] < reference_rows[index, ts_index]]
    return data


def plot_log(
    log_path: str | Path,
    output_path: str | Path,
    clip: bool = False,
    max_time: float | None = None,
    mocap_diff_window: int = 5,
    mocap_min_dt: float = 0.005,
    print_mocap_stats: bool = True,
):
    columns, data = load_columns(log_path, max_time=max_time)
    if clip:
        data = clip_after_first_trajectory(columns, data)

    reference_rows = rows_with(columns, data, ("x_des", "y_des", "yaw_des", "v_ff", "w_ff"))
    pose_rows = rows_with(columns, data, ("x_raw", "y_raw", "yaw_raw"))
    wheel_rows = rows_with(columns, data, ("omega_r_meas", "omega_l_meas"))
    command_rows = rows_with(columns, data, ("omega_r_cmd", "omega_l_cmd"))
    duty_time, duty_cycle = sparse_stream(columns, data, ("duty_r", "duty_l"))
    imu_time, gyro_z = sparse_stream(columns, data, ("gyro_z",))

    ref_time = col(columns, reference_rows, "ts")
    reference = np.column_stack(
        [
            col(columns, reference_rows, "x_des"),
            col(columns, reference_rows, "y_des"),
            col(columns, reference_rows, "yaw_des"),
            col(columns, reference_rows, "v_ff"),
            col(columns, reference_rows, "w_ff"),
        ]
    )
    pose_time = col(columns, pose_rows, "ts")
    measured_pose = np.column_stack(
        [col(columns, pose_rows, "x_raw"), col(columns, pose_rows, "y_raw"), col(columns, pose_rows, "yaw_raw")]
    )
    if print_mocap_stats:
        print_mocap_timing_stats(log_path, pose_time)
    wheel_time = col(columns, wheel_rows, "ts")
    wheel_speeds = np.column_stack(
        [col(columns, wheel_rows, "omega_r_meas"), col(columns, wheel_rows, "omega_l_meas")]
    )
    command_time = col(columns, command_rows, "ts")
    wheel_cmd = np.column_stack(
        [col(columns, command_rows, "omega_r_cmd"), col(columns, command_rows, "omega_l_cmd")]
    )
    mocap_time, mocap_vel = mocap_vel_omega(
        pose_time,
        measured_pose,
        diff_window=mocap_diff_window,
        min_dt=mocap_min_dt,
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Pololu Log Summary ({Path(log_path).stem})", fontsize=16)

    ax = axes[0, 0]
    ax.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    ax.plot(measured_pose[:, 0], measured_pose[:, 1], color="tab:blue", linewidth=0.95, label="Measured")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    ax_v = axes[0, 1]
    ax_w = ax_v.twinx()
    line_ref_v = ax_v.step(ref_time, np.abs(reference[:, 3]), where="post", color="lightskyblue", linestyle="--", linewidth=0.9, label="ref v")[0]
    line_mocap_v = ax_v.plot(mocap_time, mocap_vel[:, 0], color="tab:blue", linewidth=0.65, label="mocap v")[0]
    line_ref_w = ax_w.step(ref_time, reference[:, 4], where="post", color="khaki", linestyle="--", linewidth=0.9, label="ref omega")[0]
    line_mocap_w = ax_w.plot(mocap_time, mocap_vel[:, 1], color="goldenrod", linewidth=0.65, label="mocap omega")[0]
    velocity_lines = [line_ref_v, line_mocap_v, line_ref_w, line_mocap_w]
    if len(imu_time):
        line_gyro_z = ax_w.plot(
            imu_time,
            np.deg2rad(gyro_z[:, 0]),
            color="tab:purple",
            linewidth=0.45,
            alpha=0.8,
            label="imu gyro z",
        )[0]
        velocity_lines.append(line_gyro_z)
    ax_v.set_xlabel("time [s]")
    ax_v.set_ylabel("linear velocity [m/s]")
    ax_w.set_ylabel("angular velocity [rad/s]")
    ax_v.set_title("Velocity")
    ax_v.grid(True)
    ax_v.legend(handles=velocity_lines, loc="best")

    ax = axes[0, 2]
    ax_duty = ax.twinx()
    cmd_time, cmd_right = stair_series(command_time, wheel_cmd[:, 0], wheel_time[-1] if len(wheel_time) else None)
    _, cmd_left = stair_series(command_time, wheel_cmd[:, 1], wheel_time[-1] if len(wheel_time) else None)
    lines = [
        ax.step(cmd_time, cmd_right, where="post", color="tab:green", linestyle="--", linewidth=0.75, label="cmd right")[0],
        ax.plot(wheel_time, wheel_speeds[:, 0], color="tab:green", linewidth=0.8, label="meas right")[0],
        ax.step(cmd_time, cmd_left, where="post", color="tab:orange", linestyle="--", linewidth=0.75, label="cmd left")[0],
        ax.plot(wheel_time, wheel_speeds[:, 1], color="tab:orange", linewidth=0.8, label="meas left")[0],
    ]
    if len(duty_time):
        duty_end_time = wheel_time[-1] if len(wheel_time) else None
        duty_right_time, duty_right = stair_series(duty_time, duty_cycle[:, 0], duty_end_time)
        duty_left_time, duty_left = stair_series(duty_time, duty_cycle[:, 1], duty_end_time)
        lines.extend(
            [
                ax_duty.step(
                    duty_right_time,
                    duty_right,
                    where="post",
                    color="tab:green",
                    linestyle="-",
                    linewidth=0.2,
                    alpha=0.5,
                    label="duty right",
                )[0],
                ax_duty.step(
                    duty_left_time,
                    duty_left,
                    where="post",
                    color="tab:orange",
                    linestyle="-",
                    linewidth=0.2,
                    alpha=0.5,
                    label="duty left",
                )[0],
            ]
        )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("wheel speed [rad/s]")
    ax_duty.set_ylabel("duty cycle")
    ax.set_title("Wheel Speeds")
    ax.grid(True)
    ax.legend(handles=lines)

    for index, ax in enumerate(axes[1, :]):
        ax.step(ref_time, reference[:, index], where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")
        ax.plot(pose_time, measured_pose[:, index], color="tab:blue", linewidth=0.95, label="Measured")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(("x [m]", "y [m]", "theta [rad]")[index])
        ax.set_title(("x State", "y State", "theta State")[index])
        ax.grid(True)
        ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Standalone Pololu log summary plotter.")
    parser.add_argument("--log", default="Pololu Data/Experiments/2026_06_27 Johannes Encoder RAM/")
    parser.add_argument("--output", default=None)
    parser.add_argument("--clip-after-first-trajectory", default=True)
    parser.add_argument("--max-time", type=float, default=5)
    parser.add_argument("--mocap-diff-window", type=int, default=5)
    parser.add_argument("--mocap-min-dt", type=float, default=0.005)
    parser.add_argument("--no-mocap-stats", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log)
    if log_path.is_dir():
        output_dir = Path(args.output) if args.output is not None else log_path / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in log_files(log_path):
            output_path = output_dir / f"{path.stem}.pdf"
            plot_log(
                path,
                output_path,
                clip=args.clip_after_first_trajectory,
                max_time=args.max_time,
                mocap_diff_window=args.mocap_diff_window,
                mocap_min_dt=args.mocap_min_dt,
                print_mocap_stats=not args.no_mocap_stats,
            )
            print(output_path)
        return

    output_path = Path(args.output) if args.output is not None else log_path.with_suffix(".pdf")
    plot_log(
        log_path,
        output_path,
        clip=args.clip_after_first_trajectory,
        max_time=args.max_time,
        mocap_diff_window=args.mocap_diff_window,
        mocap_min_dt=args.mocap_min_dt,
        print_mocap_stats=not args.no_mocap_stats,
    )
    print(output_path)


if __name__ == "__main__":
    main()
