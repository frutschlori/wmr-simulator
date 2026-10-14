from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_logged_summary(
    log,
    *,
    out_prefix: str = "pololu_log",
    out_dir: str | Path = "visualize",
    imu_time_s: np.ndarray | None = None,
    imu_gyro_z: np.ndarray | None = None,
    show_gains: bool = False,
    gains: np.ndarray | None = None,
) -> Path:
    """Eight-panel overview of one log; ``imu_gyro_z`` [rad/s] is overlaid on
    the mocap omega. ``show_gains`` overlays the applied gains of a simulated
    log; ``gains`` ``(num_commands, num_gains)`` gives them for a recorded log."""
    plt = _plot_module()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    ref_time = np.asarray(log.reference.time_s, dtype=float)
    reference = np.array(log.reference.states, dtype=float, copy=True)
    pose_time = np.asarray(log.pose.time_s, dtype=float)
    measured_pose = np.array(log.pose.states, dtype=float, copy=True)
    command_time = np.asarray(log.pose.command_time_s, dtype=float)
    wheel_cmd = np.asarray(log.pose.wheel_cmd, dtype=float)
    wheel_time = np.asarray(log.wheel.time_s, dtype=float)
    wheel_speeds = np.asarray(log.wheel.speeds, dtype=float)

    # Keep plotted headings continuous while leaving the logged states intact.
    reference[:, 2] = np.unwrap(reference[:, 2])
    measured_pose[:, 2] = np.unwrap(measured_pose[:, 2])

    # Raw mocap (clean_states) as slim background lines for recorded logs.
    pose_twists = getattr(log.pose, "twists", None)
    clean_time = getattr(log.pose, "clean_time_s", None)
    clean_states = getattr(log.pose, "clean_states", None)
    if pose_twists is not None and clean_time is not None and clean_states is not None:
        raw_time = np.asarray(clean_time, dtype=float)
        raw_pose = np.array(clean_states, dtype=float, copy=True)
        raw_pose[:, 2] = np.unwrap(raw_pose[:, 2])
        mocap_vel_time = pose_time
        mocap_vel = np.asarray(pose_twists, dtype=float)[:, [0, 2]]
        raw_vel_time, raw_vel = _mocap_vel_omega(raw_time, raw_pose)
    else:
        raw_time = raw_pose = None
        mocap_vel_time, mocap_vel = _mocap_vel_omega(pose_time, measured_pose)
        raw_vel_time = raw_vel = None
    raw_style = dict(linewidth=0.4, alpha=0.7)

    reference_speed = np.linalg.norm(reference[:, 3:5], axis=1)
    reference_velocity = np.column_stack([reference_speed, reference[:, 5]])
    reference_acc_time, reference_acceleration = _time_derivative(ref_time, reference_velocity)
    mocap_acc_time, mocap_acceleration = _time_derivative(mocap_vel_time, mocap_vel)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(f"Pololu Log Summary ({out_prefix})", fontsize=16)

    ax_traj = axes[0, 0]
    ax_traj.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    if raw_pose is not None:
        ax_traj.plot(raw_pose[:, 0], raw_pose[:, 1], color="tab:blue", **raw_style, label="Measured (raw)")
    ax_traj.plot(measured_pose[:, 0], measured_pose[:, 1], color="tab:blue", linewidth=0.8, label="Measured")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax_vel = axes[0, 1]
    ax_vel_omega = ax_vel.twinx()
    line_ref_v = ax_vel.step(ref_time, reference_speed, where="post", color="lightskyblue", linestyle="--", linewidth=0.9, label="ref v")[0]
    velocity_handles_raw = []
    if raw_vel is not None:
        velocity_handles_raw.append(
            ax_vel.plot(raw_vel_time, raw_vel[:, 0], color="tab:blue", **raw_style, label="mocap v (raw)")[0]
        )
    line_mocap_v = ax_vel.plot(
        mocap_vel_time,
        mocap_vel[:, 0],
        color="tab:blue",
        linewidth=0.65,
        label="mocap v",
    )[0]
    line_ref_w = ax_vel_omega.step(ref_time, reference[:, 5], where="post", color="khaki", linestyle="--", linewidth=0.9, label=r"ref $\omega$")[0]
    if raw_vel is not None:
        velocity_handles_raw.append(
            ax_vel_omega.plot(raw_vel_time, raw_vel[:, 1], color="goldenrod", **raw_style, label=r"mocap $\omega$ (raw)")[0]
        )
    if imu_gyro_z is not None and imu_time_s is not None and len(imu_time_s):
        velocity_handles_raw.append(
            ax_vel_omega.plot(
                np.asarray(imu_time_s, dtype=float),
                np.asarray(imu_gyro_z, dtype=float),
                color="tab:purple",
                linewidth=0.5,
                alpha=0.7,
                label=r"IMU $\omega$",
            )[0]
        )
    line_mocap_w = ax_vel_omega.plot(
        mocap_vel_time,
        mocap_vel[:, 1],
        color="goldenrod",
        linewidth=0.65,
        label=r"mocap $\omega$",
    )[0]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True)
    ax_vel.legend(
        handles=[line_ref_v, line_mocap_v, line_ref_w, line_mocap_w, *velocity_handles_raw],
        loc="best",
    )

    ax_wheels = axes[0, 2]
    cmd_time, cmd_right = _stair_series(command_time, wheel_cmd[:, 0], wheel_time[-1] if len(wheel_time) else None)
    _, cmd_left = _stair_series(command_time, wheel_cmd[:, 1], wheel_time[-1] if len(wheel_time) else None)
    line_cmd_right = ax_wheels.step(cmd_time, cmd_right, where="post", color="tab:green", linestyle="--", linewidth=0.75, label="cmd right")[0]
    line_meas_right = ax_wheels.plot(wheel_time, wheel_speeds[:, 0], color="tab:green", linewidth=0.8, label="meas right")[0]
    line_cmd_left = ax_wheels.step(cmd_time, cmd_left, where="post", color="tab:orange", linestyle="--", linewidth=0.75, label="cmd left")[0]
    line_meas_left = ax_wheels.plot(wheel_time, wheel_speeds[:, 1], color="tab:orange", linewidth=0.8, label="meas left")[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds and Duty Cycles")
    ax_wheels.grid(True)
    wheel_legend_handles = [line_cmd_right, line_meas_right, line_cmd_left, line_meas_left]

    # Duty on a fixed [-1, 1] axis.
    duty_cycle = np.asarray(log.wheel.duty_cycle, dtype=float)
    duty_time = wheel_time[: len(duty_cycle)]
    ax_duty = ax_wheels.twinx()
    for column, label, color in ((0, "duty right", "tab:green"), (1, "duty left", "tab:orange")):
        wheel_legend_handles.append(
            ax_duty.plot(
                duty_time,
                duty_cycle[: len(duty_time), column],
                color=color,
                linestyle=":",
                linewidth=0.7,
                alpha=0.8,
                label=label,
            )[0]
        )
    ax_duty.set_ylim(-1.05, 1.05)
    ax_duty.set_ylabel("duty cycle [-]")

    ax_linear_acceleration = axes[0, 3]
    line_ref_linear_acceleration = ax_linear_acceleration.step(
        reference_acc_time,
        reference_acceleration[:, 0],
        where="post",
        color="tab:red",
        linestyle="--",
        linewidth=0.9,
        label="Reference",
    )[0]
    line_mocap_linear_acceleration = ax_linear_acceleration.plot(
        mocap_acc_time,
        mocap_acceleration[:, 0],
        color="tab:blue",
        linewidth=0.7,
        label="Mocap (smoothed)",
    )[0]
    ax_linear_acceleration.set_xlabel("time [s]")
    ax_linear_acceleration.set_ylabel("linear acceleration [m/s²]")
    ax_linear_acceleration.set_title("Linear Acceleration")
    ax_linear_acceleration.grid(True)
    ax_linear_acceleration.legend(handles=[line_ref_linear_acceleration, line_mocap_linear_acceleration])

    # Applied gains: motor gains here, outer gains on the state subplots.
    if gains is not None:
        gains_log = np.asarray(gains, dtype=float)
    else:
        gains_log = getattr(log.pose, "gains", None)
        gains_log = None if (not show_gains or gains_log is None) else np.asarray(gains_log, dtype=float)
    if gains_log is not None:
        ax_wheel_gains = ax_wheels.twinx()
        # Outside the duty axis, which already occupies the right spine.
        ax_wheel_gains.spines["right"].set_position(("axes", 1.12))
        for column, label, style in ((3, r"$k_p$ motor", "-"), (4, r"$k_i$ motor", "--")):
            wheel_legend_handles.append(
                ax_wheel_gains.step(
                    command_time,
                    gains_log[:, column],
                    where="post",
                    color="black",
                    linestyle=style,
                    linewidth=0.8,
                    label=label,
                )[0]
            )
        ax_wheel_gains.set_ylabel("motor gains [-]")
    ax_wheels.legend(handles=wheel_legend_handles)

    labels = ("x [m]", "y [m]", "theta [rad]")
    titles = ("x State", "y State", "theta State")
    gain_labels = (r"$k_x$", r"$k_y$", r"$k_\theta$")
    for index, ax in enumerate(axes[1, :3]):
        ax.step(ref_time, reference[:, index], where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")
        if raw_pose is not None:
            ax.plot(raw_time, raw_pose[:, index], color="tab:blue", **raw_style, label="Measured (raw)")
        ax.plot(pose_time, measured_pose[:, index], color="tab:blue", linewidth=0.95, label="Measured")
        legend_handles = ax.get_lines()[:]
        if gains_log is not None:
            gain_ax = ax.twinx()
            legend_handles.append(
                gain_ax.step(
                    command_time,
                    gains_log[:, index],
                    where="post",
                    color="black",
                    linewidth=0.8,
                    label=gain_labels[index],
                )[0]
            )
            gain_ax.set_ylabel(f"{gain_labels[index]} [-]")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(labels[index])
        ax.set_title(titles[index])
        ax.grid(True)
        ax.legend(handles=legend_handles)

    ax_angular_acceleration = axes[1, 3]
    line_ref_angular_acceleration = ax_angular_acceleration.step(
        reference_acc_time,
        reference_acceleration[:, 1],
        where="post",
        color="tab:red",
        linestyle="--",
        linewidth=0.9,
        label="Reference",
    )[0]
    line_mocap_angular_acceleration = ax_angular_acceleration.plot(
        mocap_acc_time,
        mocap_acceleration[:, 1],
        color="tab:blue",
        linewidth=0.7,
        label="Mocap (smoothed)",
    )[0]
    ax_angular_acceleration.set_xlabel("time [s]")
    ax_angular_acceleration.set_ylabel("angular acceleration [rad/s²]")
    ax_angular_acceleration.set_title("Angular Acceleration")
    ax_angular_acceleration.grid(True)
    ax_angular_acceleration.legend(handles=[line_ref_angular_acceleration, line_mocap_angular_acceleration])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Log summary PDF saved at: {output_path}")
    return output_path


def plot_logged_trajectories(
    logs,
    labels,
    *,
    out_prefix: str = "logged_trajectories",
    out_dir: str | Path = "visualize",
) -> Path:
    """One XY panel with every logged trajectory of an iteration (reference
    dashed, measured pose solid, each log annotated with its label)."""
    plt = _plot_module()
    from matplotlib.lines import Line2D

    from wmr_simulator.visualization.trajectories import decimate_path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9, 9))
    for index, (log, label) in enumerate(zip(logs, labels)):
        color = color_cycle[index % len(color_cycle)]
        reference = decimate_path(np.asarray(log.reference.states, dtype=float)[:, :2])
        measured = decimate_path(np.asarray(log.pose.states, dtype=float)[:, :2])
        ax.plot(reference[:, 0], reference[:, 1], color=color, linestyle="--", linewidth=1.0, alpha=0.9)
        ax.plot(measured[:, 0], measured[:, 1], color=color, linewidth=1.0)
        if not len(measured):
            continue
        ax.plot(measured[0, 0], measured[0, 1], marker="o", markersize=4, color=color)
        # Stagger the labels along the paths.
        anchor = measured[int(round((index + 1) / (len(logs) + 1) * (len(measured) - 1)))]
        ax.annotate(
            label,
            xy=(anchor[0], anchor[1]),
            xytext=(4, 4),
            textcoords="offset points",
            color=color,
            fontsize=8,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Logged Trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(
        handles=[
            Line2D([], [], color="0.3", linestyle="--", linewidth=1.0, label="Reference"),
            Line2D([], [], color="0.3", linewidth=1.0, label="Logged"),
        ],
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Logged trajectories PDF saved at: {output_path}")
    return output_path


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


def _time_derivative(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a scalar or column-wise signal on a possibly sparse time grid."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        finite = np.isfinite(values)
    else:
        finite = np.all(np.isfinite(values), axis=1)
    finite &= np.isfinite(time_s)
    if len(time_s) > 1:
        finite[1:] &= np.diff(time_s) > 0.0
    time_s = time_s[finite]
    values = values[finite]
    if len(time_s) < 2:
        return time_s, np.zeros_like(values)
    edge_order = 2 if len(time_s) >= 3 else 1
    return time_s, np.gradient(values, time_s, axis=0, edge_order=edge_order)


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
