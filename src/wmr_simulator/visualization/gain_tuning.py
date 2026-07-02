import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np


def plot_gain_tuning_summary(
    pipeline,
    init_log,
    tuned_log,
    out_prefix="gain_tuning_summary",
):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize", f"{out_prefix}.pdf")

    reference_time = np.asarray(tuned_log.reference.time_s, dtype=float)
    reference = np.asarray(tuned_log.reference.states, dtype=float)
    init_time = np.asarray(init_log.pose.time_s, dtype=float)
    init_pose = np.asarray(init_log.pose.true_states, dtype=float)
    tuned_time = np.asarray(tuned_log.pose.time_s, dtype=float)
    tuned_pose = np.asarray(tuned_log.pose.true_states, dtype=float)
    wheel_time = np.asarray(tuned_log.wheel.time_s, dtype=float)
    wheel_speeds = np.asarray(tuned_log.wheel.speeds, dtype=float)
    duty_cycle = np.asarray(tuned_log.wheel.duty_cycle, dtype=float)
    command_time = np.asarray(tuned_log.pose.command_time_s, dtype=float)
    wheel_cmd = np.asarray(tuned_log.pose.wheel_cmd, dtype=float)

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f"Gain Tuning Summary ({out_prefix})", fontsize=16)
    ax_traj = plt.subplot2grid((2, 10), (0, 0), colspan=1, fig=fig)
    ax_vel = plt.subplot2grid((2, 10), (0, 1), colspan=3, fig=fig)
    ax_wheels = plt.subplot2grid((2, 10), (0, 4), colspan=3, fig=fig)
    ax_motor = plt.subplot2grid((2, 10), (0, 7), colspan=3, fig=fig)
    state_axes = [
        plt.subplot2grid((2, 10), (1, 0), colspan=3, fig=fig),
        plt.subplot2grid((2, 10), (1, 3), colspan=4, fig=fig),
        plt.subplot2grid((2, 10), (1, 7), colspan=3, fig=fig),
    ]

    ax_traj.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    ax_traj.plot(init_pose[:, 0], init_pose[:, 1], color="tab:blue", linewidth=1.0, label="Initial")
    ax_traj.plot(tuned_pose[:, 0], tuned_pose[:, 1], color="tab:orange", linewidth=1.2, label="Tuned")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize="small")

    ax_vel_omega = ax_vel.twinx()
    tuned_vel_time, tuned_vel = _pose_vel_omega(tuned_time, tuned_pose)
    line_tuned_v = ax_vel.plot(tuned_vel_time, tuned_vel[:, 0], color="tab:blue", linewidth=1.0, label="tuned v")[0]
    line_tuned_w = ax_vel_omega.plot(
        tuned_vel_time,
        tuned_vel[:, 1],
        color="tab:purple",
        linewidth=1.0,
        label=r"tuned $\omega$",
    )[0]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True)
    ax_vel.legend(handles=[line_tuned_v, line_tuned_w], loc="best")

    cmd_time, cmd_right = _stair_series(command_time, wheel_cmd[:, 0], wheel_time[-1] if len(wheel_time) else None)
    _, cmd_left = _stair_series(command_time, wheel_cmd[:, 1], wheel_time[-1] if len(wheel_time) else None)
    line_cmd_right = ax_wheels.step(
        cmd_time,
        cmd_right,
        where="post",
        color="tab:green",
        linestyle="--",
        linewidth=0.7,
        label="cmd right",
    )[0]
    line_meas_right = ax_wheels.plot(
        wheel_time,
        wheel_speeds[:, 0],
        color="tab:green",
        linewidth=0.9,
        label="meas right",
    )[0]
    line_cmd_left = ax_wheels.step(
        cmd_time,
        cmd_left,
        where="post",
        color="tab:orange",
        linestyle="--",
        linewidth=0.7,
        label="cmd left",
    )[0]
    line_meas_left = ax_wheels.plot(
        wheel_time,
        wheel_speeds[:, 1],
        color="tab:orange",
        linewidth=0.9,
        label="meas left",
    )[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    ax_wheels.legend(handles=[line_cmd_right, line_meas_right, line_cmd_left, line_meas_left])

    _plot_tuned_motor_axes(ax_motor, wheel_time, duty_cycle, wheel_speeds)

    state_labels = ("x [m]", "y [m]", "theta [rad]")
    state_names = ("x", "y", "theta")
    for index, ax in enumerate(state_axes):
        ax.step(
            reference_time,
            reference[:, index],
            where="post",
            color="tab:red",
            linestyle="--",
            linewidth=0.9,
            label="Reference",
        )
        ax.plot(init_time, init_pose[:, index], color="tab:blue", linewidth=1.0, label="Initial")
        ax.plot(tuned_time, tuned_pose[:, index], color="tab:orange", linewidth=1.2, label="Tuned")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(state_labels[index])
        ax.set_title(f"{state_names[index]} State")
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf_filename, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Gain tuning summary PDF saved at: {pdf_filename}")


def plot_controller_tuning_errors(pipeline, init_log, tuned_log, out_prefix="ctrl_tuning"):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize", f"{out_prefix}_tracking_errors.pdf")

    reference_poses = np.asarray(pipeline.reference_states[:, :3])
    reference_pose_indices = np.arange(
        0,
        len(reference_poses) * pipeline.inner_steps_per_geometry_step,
        pipeline.inner_steps_per_geometry_step,
        dtype=int,
    )
    init_poses = np.asarray(init_log.pose.states)[reference_pose_indices]
    tuned_poses = np.asarray(tuned_log.pose.states)[reference_pose_indices]
    plot_len = min(len(reference_poses), len(init_poses), len(tuned_poses), len(pipeline.reference_time_grid))
    plot_time = np.asarray(pipeline.reference_time_grid[:plot_len])

    init_errors = init_poses[:plot_len] - reference_poses[:plot_len]
    tuned_errors = tuned_poses[:plot_len] - reference_poses[:plot_len]
    init_errors[:, 2] = (init_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    tuned_errors[:, 2] = (tuned_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x error [m]", "y error [m]", "theta error [rad]"]
    for idx, label in enumerate(labels):
        axes[idx].plot(plot_time, init_errors[:, idx], label="Initial gains", linewidth=1.2)
        axes[idx].plot(plot_time, tuned_errors[:, idx], label="Tuned gains", linewidth=1.2)
        axes[idx].set_ylabel(label)
        axes[idx].grid(True)
        axes[idx].legend()

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Pose Tracking Errors", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf_filename, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Controller tuning error PDF saved at: {pdf_filename}")


def _stair_series(time: np.ndarray, values: np.ndarray, end_time: float | None) -> tuple[np.ndarray, np.ndarray]:
    if end_time is None or len(time) == 0 or end_time <= time[-1]:
        return time, values
    return np.concatenate([time, [end_time]]), np.concatenate([values, values[-1:]])


def _pose_vel_omega(time_s: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) < 2:
        return time_s, np.zeros((len(time_s), 2), dtype=float)

    dt = np.diff(time_s)
    delta_xy = np.diff(pose[:, :2], axis=0)
    theta = np.unwrap(pose[:, 2])
    heading = theta[1:]
    linear_velocity = (delta_xy[:, 0] * np.cos(heading) + delta_xy[:, 1] * np.sin(heading)) / dt
    angular_velocity = np.diff(theta) / dt
    return time_s[1:], np.column_stack([linear_velocity, angular_velocity])


def _plot_tuned_motor_axes(
    ax_duty,
    time: np.ndarray,
    duty_cycle: np.ndarray,
    wheel_speeds: np.ndarray,
):
    ax_speed = ax_duty.twinx()
    line_dc_l = ax_duty.plot(
        time,
        duty_cycle[:, 1],
        color="#f2c14e",
        linewidth=0.7,
        label="DC left",
    )[0]
    line_speed_l = ax_speed.plot(
        time,
        wheel_speeds[:, 1],
        color="#f28e2b",
        linewidth=0.9,
        label="speed left",
    )[0]
    line_dc_r = ax_duty.plot(
        time,
        duty_cycle[:, 0],
        color="#59a14f",
        linewidth=0.7,
        label="DC right",
    )[0]
    line_speed_r = ax_speed.plot(
        time,
        wheel_speeds[:, 0],
        color="#1b7f3a",
        linewidth=0.9,
        label="speed right",
    )[0]

    _set_symmetric_ylim(ax_duty, duty_cycle)
    _set_symmetric_ylim(ax_speed, wheel_speeds)
    ax_duty.set_xlabel("time [s]")
    ax_duty.set_ylabel("duty cycle")
    ax_speed.set_ylabel("wheel speed [rad/s]")
    ax_duty.set_title("Motor")
    ax_duty.grid(True)
    ax_duty.legend(handles=[line_dc_l, line_speed_l, line_dc_r, line_speed_r], loc="best")


def _set_symmetric_ylim(ax, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    limit = np.nanmax(np.abs(values)) if values.size else 0.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    ax.set_ylim(-1.05 * limit, 1.05 * limit)
