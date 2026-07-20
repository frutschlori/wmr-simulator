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
    """Six-panel overview of one log; ``imu_gyro_z`` (rad/s, see
    pololu.log_loader.load_imu_gyro_z) is overlaid on the mocap omega.
    ``show_gains`` overlays the applied controller gains (``log.pose.gains``,
    simulated logs only) on the wheel-speed and state subplots. ``gains``
    (``(num_commands, num_gains)``, at the command timestamps) overrides that
    source -- used for recorded logs whose gains are reconstructed offline from
    the gain parametrization (pololu.gain_reconstruction)."""
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

    # Pololu logs carry Savitzky-Golay-smoothed poses in states and filter-
    # derivative twists; the raw (unsmoothed) mocap surviving outlier rejection
    # lives in clean_states/clean_time_s (a shorter stream than time_s). Show
    # that as slim background lines. Simulated logs (twists None) keep the
    # single-stream plot.
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
    reference_speed = np.linalg.norm(reference[:, 3:5], axis=1)
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
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    wheel_legend_handles = [line_cmd_right, line_meas_right, line_cmd_left, line_meas_left]

    # Applied controller gains along the run (time-varying under a gain
    # parametrization): motor PI gains here, outer gains on the state subplots.
    if gains is not None:
        gains_log = np.asarray(gains, dtype=float)
    else:
        gains_log = getattr(log.pose, "gains", None)
        gains_log = None if (not show_gains or gains_log is None) else np.asarray(gains_log, dtype=float)
    if gains_log is not None:
        ax_wheel_gains = ax_wheels.twinx()
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
    for index, ax in enumerate(axes[1, :]):
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

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Log summary PDF saved at: {output_path}")
    return output_path

def plot_run_comparison(
    log_groups,
    labels,
    *,
    out_prefix: str = "run_comparison",
    out_dir: str | Path = "visualize",
) -> Path:
    """Overlay groups of logs of the same reference: XY trajectory, the smoothed
    mocap body twists (v, omega), and the x/y/theta states over time.
    ``log_groups`` is one list of SimulationLogs (pololu.log_loader) per label;
    all runs of a group share one color and are drawn as thin lines so the
    run-to-run spread is visible. ``pose.twists`` carry the Savitzky-Golay
    filter-derivative velocities. The reference is drawn from the first log of
    the first group."""
    plt = _plot_module()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_traj, ax_v, ax_w = axes[0, :]
    state_axes = axes[1, :]

    reference = np.asarray(log_groups[0][0].reference.states, dtype=float)
    ref_time = np.asarray(log_groups[0][0].reference.time_s, dtype=float)
    ax_traj.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    reference_speed = np.linalg.norm(reference[:, 3:5], axis=1)
    ax_v.step(ref_time, reference_speed, where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")
    ax_w.step(ref_time, reference[:, 5], where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")
    for index, ax in enumerate(state_axes):
        ax.step(ref_time, reference[:, index], where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")

    run_style = dict(linewidth=0.6, alpha=0.8)
    for logs, label, color in zip(log_groups, labels, colors):
        for run_index, log in enumerate(logs):
            legend_label = label if run_index == 0 else None
            pose_time = np.asarray(log.pose.time_s, dtype=float)
            pose = np.asarray(log.pose.states, dtype=float)
            twists = np.asarray(log.pose.twists, dtype=float)
            ax_traj.plot(pose[:, 0], pose[:, 1], color=color, **run_style, label=legend_label)
            ax_v.plot(pose_time, twists[:, 0], color=color, **run_style, label=legend_label)
            ax_w.plot(pose_time, twists[:, 2], color=color, **run_style, label=legend_label)
            for index, ax in enumerate(state_axes):
                ax.plot(pose_time, pose[:, index], color=color, **run_style, label=legend_label)

    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_v.set_xlabel("time [s]")
    ax_v.set_ylabel("linear velocity [m/s]")
    ax_v.set_title("Linear Velocity (mocap, smoothed)")
    ax_w.set_xlabel("time [s]")
    ax_w.set_ylabel("angular velocity [rad/s]")
    ax_w.set_title("Angular Velocity (mocap, smoothed)")
    state_labels = ("x [m]", "y [m]", "theta [rad]")
    state_titles = ("x State", "y State", "theta State")
    for index, ax in enumerate(state_axes):
        ax.set_xlabel("time [s]")
        ax.set_ylabel(state_labels[index])
        ax.set_title(state_titles[index])
    for ax in axes.flat:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Comparison PDF saved at: {output_path}")
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
