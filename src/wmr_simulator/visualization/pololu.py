from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_logged_summary(
    log,
    *,
    out_prefix: str = "pololu_log",
    out_dir: str | Path = "visualize",
) -> Path:
    plt = _plot_module()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    ref_time = np.asarray(log.reference.time_s, dtype=float)
    reference = np.asarray(log.reference.states, dtype=float)
    pose_time = np.asarray(log.pose.time_s, dtype=float)
    measured_pose = np.asarray(log.pose.states, dtype=float)
    command_time = np.asarray(log.pose.command_time_s, dtype=float)
    wheel_cmd = np.asarray(log.pose.wheel_cmd, dtype=float)
    wheel_time = np.asarray(log.wheel.time_s, dtype=float)
    wheel_speeds = np.asarray(log.wheel.speeds, dtype=float)
    odom_vel = np.asarray(log.wheel.vel_omega, dtype=float)
    duty_cycle = np.asarray(log.wheel.duty_cycle, dtype=float)
    mocap_vel_time, mocap_vel = _mocap_vel_omega(pose_time, measured_pose)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Pololu Log Summary ({out_prefix})", fontsize=16)

    ax_traj = axes[0, 0]
    ax_traj.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    ax_traj.plot(measured_pose[:, 0], measured_pose[:, 1], color="tab:blue", linewidth=1.2, label="Measured")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax_vel = axes[0, 1]
    ax_vel_omega = ax_vel.twinx()
    reference_speed = np.linalg.norm(reference[:, 3:5], axis=1)
    line_ref_v = ax_vel.step(ref_time, reference_speed, where="post", color="lightskyblue", linestyle="--", linewidth=0.9, label="ref v")[0]
    line_mocap_v = ax_vel.plot(
        mocap_vel_time,
        mocap_vel[:, 0],
        color="tab:blue",
        linewidth=0.65,
        label="mocap v",
    )[0]
    line_odom_v = ax_vel.plot(
        wheel_time,
        odom_vel[:, 0],
        color="navy",
        linestyle="--",
        linewidth=0.45,
        label="odom v",
    )[0]
    line_ref_w = ax_vel_omega.step(ref_time, reference[:, 5], where="post", color="khaki", linestyle="--", linewidth=0.9, label=r"ref $\omega$")[0]
    line_mocap_w = ax_vel_omega.plot(
        mocap_vel_time,
        mocap_vel[:, 1],
        color="goldenrod",
        linewidth=0.65,
        label=r"mocap $\omega$",
    )[0]
    line_odom_w = ax_vel_omega.plot(
        wheel_time,
        odom_vel[:, 1],
        color="darkgoldenrod",
        linestyle="--",
        linewidth=0.45,
        label=r"odom $\omega$",
    )[0]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True)
    ax_vel.legend(
        handles=[line_ref_v, line_mocap_v, line_odom_v, line_ref_w, line_mocap_w, line_odom_w],
        loc="best",
    )

    ax_wheels = axes[0, 2]
    cmd_time, cmd_right = _stair_series(command_time, wheel_cmd[:, 0], wheel_time[-1] if len(wheel_time) else None)
    _, cmd_left = _stair_series(command_time, wheel_cmd[:, 1], wheel_time[-1] if len(wheel_time) else None)
    line_cmd_right = ax_wheels.step(cmd_time, cmd_right, where="post", color="tab:green", linestyle="--", linewidth=0.75, label="cmd right")[0]
    line_meas_right = ax_wheels.plot(wheel_time, wheel_speeds[:, 0], color="tab:green", linewidth=1.2, label="meas right")[0]
    line_cmd_left = ax_wheels.step(cmd_time, cmd_left, where="post", color="tab:orange", linestyle="--", linewidth=0.75, label="cmd left")[0]
    line_meas_left = ax_wheels.plot(wheel_time, wheel_speeds[:, 1], color="tab:orange", linewidth=1.2, label="meas left")[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    ax_wheels.legend(handles=[line_cmd_right, line_meas_right, line_cmd_left, line_meas_left])

    labels = ("x [m]", "y [m]", "theta [rad]")
    titles = ("x State", "y State", "theta State")
    for index, ax in enumerate(axes[1, :]):
        ax.step(ref_time, reference[:, index], where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")
        ax.plot(pose_time, measured_pose[:, index], color="tab:blue", linewidth=1.2, label="Measured")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(labels[index])
        ax.set_title(titles[index])
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Log summary PDF saved at: {output_path}")
    return output_path

def _plot_motor_model_axes(
    ax_duty,
    time: np.ndarray,
    duty_cycle: np.ndarray,
    encoder_wheel_speeds: np.ndarray,
    model_wheel_speeds: np.ndarray | None,
):
    ax_speed = ax_duty.twinx()
    model_wheel_speeds = (
        np.full_like(encoder_wheel_speeds, np.nan)
        if model_wheel_speeds is None
        else np.asarray(model_wheel_speeds, dtype=float)
    )
    model_wheel_speeds = model_wheel_speeds[: len(time)]

    line_dc_l = ax_duty.step(
        time,
        duty_cycle[:, 1],
        where="post",
        color="tab:orange",
        linestyle="--",
        linewidth=0.8,
        label="DC left",
    )[0]
    line_enc_l = ax_speed.plot(
        time,
        encoder_wheel_speeds[:, 1],
        color="tab:orange",
        linestyle="--",
        linewidth=0.7,
        label="enc left",
    )[0]
    line_model_l = ax_speed.plot(
        time,
        model_wheel_speeds[:, 1],
        color="tab:orange",
        linestyle="-",
        linewidth=0.9,
        label="model left",
    )[0]
    line_dc_r = ax_duty.step(
        time,
        duty_cycle[:, 0],
        where="post",
        color="tab:green",
        linestyle="--",
        linewidth=0.6,
        label="DC right",
    )[0]
    line_enc_r = ax_speed.plot(
        time,
        encoder_wheel_speeds[:, 0],
        color="tab:green",
        linestyle="--",
        linewidth=0.7,
        label="enc right",
    )[0]
    line_model_r = ax_speed.plot(
        time,
        model_wheel_speeds[:, 0],
        color="tab:green",
        linestyle="-",
        linewidth=0.9,
        label="model right",
    )[0]

    _set_symmetric_ylim(ax_duty, duty_cycle)
    speed_values = [encoder_wheel_speeds]
    if model_wheel_speeds.size:
        speed_values.append(model_wheel_speeds)
    _set_symmetric_ylim(ax_speed, np.concatenate(speed_values, axis=0))
    ax_duty.set_xlabel("time [s]")
    ax_duty.set_ylabel("duty cycle")
    ax_speed.set_ylabel("wheel speed [rad/s]")
    ax_duty.set_title("Motor Model")
    ax_duty.grid(True)
    ax_duty.legend(
        handles=[line_dc_l, line_enc_l, line_model_l, line_dc_r, line_enc_r, line_model_r],
        loc="best",
    )


def _plot_module():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _mocap_vel_omega(time_s: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) < 2:
        return time_s, np.zeros((len(time_s), 2), dtype=float)

    dt = np.diff(time_s)
    delta_xy = np.diff(pose[:, :2], axis=0)
    theta = np.unwrap(pose[:, 2])
    heading = theta[1:]
    linear_velocity = (delta_xy[:, 0] * np.cos(heading) + delta_xy[:, 1] * np.sin(heading)) / dt
    angular_velocity = np.diff(theta) / dt
    return time_s[1:], np.column_stack([linear_velocity, angular_velocity])


def _stair_series(time: np.ndarray, values: np.ndarray, end_time: float | None) -> tuple[np.ndarray, np.ndarray]:
    if end_time is None or len(time) == 0 or end_time <= time[-1]:
        return time, values
    return np.concatenate([time, [end_time]]), np.concatenate([values, values[-1:]])


def _set_symmetric_ylim(ax, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    limit = np.nanmax(np.abs(values)) if values.size else 0.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    ax.set_ylim(-1.05 * limit, 1.05 * limit)


def _shade_rejected_intervals(ax, time: np.ndarray, weights: np.ndarray | None) -> None:
    if weights is None:
        return
    time = np.asarray(time, dtype=float)
    weights = np.asarray(weights, dtype=float)
    rejected_indices = np.flatnonzero(weights[: len(time)] <= 0.0)
    for index in rejected_indices:
        if index < 0 or index >= len(time):
            continue
        if index > 0:
            left = 0.5 * (time[index - 1] + time[index])
        elif len(time) > 1:
            left = time[index] - 0.5 * (time[index + 1] - time[index])
        else:
            left = time[index]

        if index + 1 < len(time):
            right = 0.5 * (time[index] + time[index + 1])
        elif index > 0:
            right = time[index] + 0.5 * (time[index] - time[index - 1])
        else:
            right = time[index]
        ax.axvspan(
            left,
            right,
            color="0.85",
            alpha=0.7,
            linewidth=0.0,
            zorder=0,
        )
