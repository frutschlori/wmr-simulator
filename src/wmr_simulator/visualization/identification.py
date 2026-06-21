import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def plot_trajectory(
    pipeline,
    untuned_log=None,
    tuned_log=None,
    out_prefix="trajectory",
    out_path=None,
    theta_arrow_stride=20,
    theta_arrow_length=0.12,
    trajectory_linewidth=1.0,
    theta_arrow_width=0.0025,
    save_pdf: bool = True,
    show_plot: bool = False,
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    reference_states = np.asarray(pipeline.reference_states)
    full_reference_states = np.asarray(getattr(pipeline, "full_reference_states", pipeline.reference_states))
    def _extract_poses(sim_log):
        if sim_log is None:
            return None
        poses = np.asarray(sim_log.pose.states)
        return poses[:, :3]

    def _draw_theta_arrows(ax, poses, color):
        if poses is None or len(poses) == 0:
            return

        stride = max(1, int(theta_arrow_stride))
        arrow_poses = poses[::stride]
        if arrow_poses.shape[1] >= 5:
            dx = theta_arrow_length * arrow_poses[:, 3]
            dy = theta_arrow_length * arrow_poses[:, 4]
        else:
            dx = theta_arrow_length * np.cos(arrow_poses[:, 2])
            dy = theta_arrow_length * np.sin(arrow_poses[:, 2])
        ax.quiver(
            arrow_poses[:, 0],
            arrow_poses[:, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.8,
            width=theta_arrow_width,
        )

    tuned_poses = _extract_poses(tuned_log)
    untuned_poses = _extract_poses(untuned_log)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(
        full_reference_states[:, 0],
        full_reference_states[:, 1],
        linestyle="-",
        linewidth=trajectory_linewidth,
        color="red",
        label="Reference",
    )
    _draw_theta_arrows(ax, full_reference_states[:, :5], "red")
    if len(reference_states) > 0 and not np.array_equal(reference_states, full_reference_states):
        ax.plot(
            reference_states[:, 0],
            reference_states[:, 1],
            linestyle="-",
            linewidth=1.5 * trajectory_linewidth,
            color="blue",
            label="Selected",
        )
        _draw_theta_arrows(ax, reference_states, "blue")

    if untuned_poses is not None:
        ax.plot(
            untuned_poses[:, 0],
            untuned_poses[:, 1],
            linewidth=trajectory_linewidth,
            linestyle="-",
            color="blue",
            label="Initial",
        )
        _draw_theta_arrows(ax, untuned_poses, "blue")

    if tuned_poses is not None:
        ax.plot(
            tuned_poses[:, 0],
            tuned_poses[:, 1],
            linewidth=trajectory_linewidth,
            linestyle="-",
            color="orange",
            label="Tuned",
        )
        _draw_theta_arrows(ax, tuned_poses, "orange")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Comparison")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    if save_pdf:
        fig.savefig(output_filename, bbox_inches="tight", transparent=True)
        print(f"Trajectory PDF saved at: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_tracking_error_frame(
    pipeline,
    wheel_radius_values,
    base_diameter_values,
    tracking_error_surface,
    num_realizations,
    seed,
    out_path,
    hidden_params=None,
    init_params=None,
    z_min=None,
    z_max=None,
    x_label="Wheel radius [m]",
    y_label="Base diameter [m]",
    surface_label="Identification Loss",
    init_label="Initial parameter guess",
    hidden_label="Hidden parameters",
    min_label="Minimum sampled loss",
    theta_arrow_stride=20,
    theta_arrow_length=0.12,
    trajectory_linewidth=1.0,
    theta_arrow_width=0.0025,
):
    os.makedirs("visualize", exist_ok=True)

    reference_states = np.asarray(pipeline.reference_states)
    full_reference_states = np.asarray(getattr(pipeline, "full_reference_states", pipeline.reference_states))

    def _draw_theta_arrows(ax, poses, color):
        if poses is None or len(poses) == 0:
            return

        stride = max(1, int(theta_arrow_stride))
        arrow_poses = poses[::stride]
        if arrow_poses.shape[1] >= 5:
            dx = theta_arrow_length * arrow_poses[:, 3]
            dy = theta_arrow_length * arrow_poses[:, 4]
        else:
            dx = theta_arrow_length * np.cos(arrow_poses[:, 2])
            dy = theta_arrow_length * np.sin(arrow_poses[:, 2])
        ax.quiver(
            arrow_poses[:, 0],
            arrow_poses[:, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.8,
            width=theta_arrow_width,
        )

    wheel_radius_grid, base_diameter_grid = np.meshgrid(wheel_radius_values, base_diameter_values)
    surface = np.asarray(tracking_error_surface)

    def _surface_value(wheel_radius, base_diameter):
        radius_idx = int(np.argmin(np.abs(np.asarray(wheel_radius_values) - wheel_radius)))
        base_idx = int(np.argmin(np.abs(np.asarray(base_diameter_values) - base_diameter)))
        return float(surface[base_idx, radius_idx])

    fig = plt.figure(figsize=(10, 14))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    ax_traj = fig.add_subplot(grid[0, 0])
    ax_surface = fig.add_subplot(grid[1, 0], projection="3d")

    ax_traj.plot(
        full_reference_states[:, 0],
        full_reference_states[:, 1],
        linestyle="--",
        linewidth=0.8 * trajectory_linewidth,
        color="red",
        label="Complete",
    )
    _draw_theta_arrows(ax_traj, full_reference_states[:, :5], "red")
    if len(reference_states) > 0 and not np.array_equal(reference_states, full_reference_states):
        ax_traj.plot(
            reference_states[:, 0],
            reference_states[:, 1],
            linestyle="-",
            linewidth=1.5 * trajectory_linewidth,
            color="blue",
            label="Selected",
        )
        _draw_theta_arrows(ax_traj, reference_states, "blue")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory Comparison")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend()

    surf = ax_surface.plot_surface(
        wheel_radius_grid,
        base_diameter_grid,
        surface,
        cmap="viridis",
        linewidth=0,
        alpha=0.75,
    )

    min_index = np.unravel_index(np.argmin(surface), surface.shape)
    min_base = float(base_diameter_values[min_index[0]])
    min_radius = float(wheel_radius_values[min_index[1]])
    min_loss = float(surface[min_index])

    if init_params is not None:
        init_radius = float(init_params.wheel_radius)
        init_base = float(init_params.base_diameter)
        init_loss = _surface_value(init_radius, init_base)
        ax_surface.scatter(
            [init_radius],
            [init_base],
            [init_loss],
            color="white",
            edgecolors="black",
            s=60,
            depthshade=False,
            alpha=1,
            label=init_label,
        )
    if hidden_params is not None:
        hidden_radius = float(hidden_params.wheel_radius)
        hidden_base = float(hidden_params.base_diameter)
        hidden_loss = _surface_value(hidden_radius, hidden_base)
        ax_surface.scatter(
            [hidden_radius],
            [hidden_base],
            [hidden_loss],
            color="gold",
            edgecolors="black",
            marker="*",
            s=180,
            depthshade=False,
            alpha=1,
            label=hidden_label,
        )
    ax_surface.scatter(
        [min_radius],
        [min_base],
        [min_loss],
        color="red",
        edgecolors="black",
        marker="X",
        s=100,
        depthshade=False,
        alpha=1,
        label=min_label,
    )

    ax_surface.set_xlabel(x_label)
    ax_surface.set_ylabel(y_label)
    ax_surface.set_zlabel(surface_label)
    if z_min is not None and z_max is not None:
        ax_surface.set_zlim(z_min, z_max)
    ax_surface.view_init(elev=25, azim=-50)
    ax_surface.legend(loc="upper right")
    fig.colorbar(
        surf,
        ax=ax_surface,
        shrink=0.7,
        pad=0.1,
        label=surface_label,
        format=FormatStrFormatter("%.1f"),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")


def plot_reference_trajectories(
    training_reference_trajectories,
    validation_reference_trajectories=None,
    out_prefix="reference trajectories",
    theta_arrow_stride=30,
    theta_arrow_length=0.07,
    trajectory_linewidth=1.0,
    theta_arrow_width=0.0025,
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = os.path.join("visualize", f"{out_prefix}.pdf")

    training_reference_trajectories = np.asarray(training_reference_trajectories)
    validation_reference_trajectories = (
        None if validation_reference_trajectories is None else np.asarray(validation_reference_trajectories)
    )

    def _draw_theta_arrows(ax, poses, color):
        if poses is None or len(poses) == 0:
            return

        stride = max(1, int(theta_arrow_stride))
        arrow_poses = poses[::stride]
        dx = theta_arrow_length * np.cos(arrow_poses[:, 2])
        dy = theta_arrow_length * np.sin(arrow_poses[:, 2])
        ax.quiver(
            arrow_poses[:, 0],
            arrow_poses[:, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.8,
            width=theta_arrow_width,
        )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for idx, reference_states in enumerate(training_reference_trajectories):
        color = colors[idx % len(colors)]
        ax.plot(
            reference_states[:, 0],
            reference_states[:, 1],
            linestyle="-",
            linewidth=trajectory_linewidth,
            color=color,
        )
        _draw_theta_arrows(ax, reference_states[:, :3], color)

    if validation_reference_trajectories is not None:
        for idx, reference_states in enumerate(validation_reference_trajectories):
            color = colors[(idx + len(training_reference_trajectories)) % len(colors)]
            ax.plot(
                reference_states[:, 0],
                reference_states[:, 1],
                linestyle="--",
                linewidth=0.8 * trajectory_linewidth,
                color=color,
            )
            _draw_theta_arrows(ax, reference_states[:, :3], color)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=trajectory_linewidth, linestyle="-", label="Training"),
        Line2D([0], [0], color="black", linewidth=0.8 * trajectory_linewidth, linestyle="--", label="Validation"),
    ]

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Reference Trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Reference trajectories PDF saved at: {output_filename}")


def plot_system_id(
    pipeline,
    init_target_log,
    init_log,
    predicted_log,
    out_prefix="system_identification",
):
    target_log = pipeline.target_log

    plot_time = np.asarray(target_log.pose.time_s, dtype=float)
    reference_time = np.asarray(target_log.reference.time_s, dtype=float)
    wheel_time = np.asarray(target_log.wheel.time_s, dtype=float)
    command_time = np.asarray(target_log.pose.command_time_s, dtype=float)
    reference = np.asarray(target_log.reference.states, dtype=float)
    measurements = np.asarray(target_log.pose.states, dtype=float)
    true_measurements = np.asarray(target_log.pose.true_states, dtype=float)
    replay = np.asarray(predicted_log.pose.states, dtype=float)
    wheel_actual = np.asarray(target_log.wheel.speeds, dtype=float)
    duty_cycle = np.asarray(target_log.wheel.duty_cycle, dtype=float)
    wheel_cmd = np.asarray(target_log.pose.wheel_cmd, dtype=float)
    motor_model_wheel_speeds = np.asarray(
        pipeline.motor_wheel_speed_rollout(
            pipeline.estimated_params,
            target_log,
            window_length=getattr(pipeline, "window_length", None),
        ),
        dtype=float,
    )

    plot_len = min(len(plot_time), len(measurements), len(true_measurements))
    plot_time = plot_time[:plot_len]
    measurements = measurements[:plot_len]
    true_measurements = true_measurements[:plot_len]
    replay = replay[: max(plot_len - 1, 0)]
    window_starts = _window_start_indices(pipeline, plot_len, getattr(pipeline, "window_length", None))
    wheel_window_starts = _window_start_indices(pipeline, len(wheel_time), getattr(pipeline, "window_length", None))

    os.makedirs("visualize", exist_ok=True)
    output_path = os.path.join("visualize", f"{out_prefix}.pdf")

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f"Identification Summary ({out_prefix})", fontsize=16)
    ax_traj = plt.subplot2grid((2, 10), (0, 0), colspan=1, fig=fig)
    ax_vel = plt.subplot2grid((2, 10), (0, 1), colspan=3, fig=fig)
    ax_wheels = plt.subplot2grid((2, 10), (0, 4), colspan=3, fig=fig)
    ax_motor = plt.subplot2grid((2, 10), (0, 7), colspan=3, fig=fig)
    state_axes = [
        plt.subplot2grid((2, 10), (1, 0), colspan=3, fig=fig),
        plt.subplot2grid((2, 10), (1, 3), colspan=4, fig=fig),
        plt.subplot2grid((2, 10), (1, 7), colspan=3, fig=fig),
    ]

    measurement_label = "Measured" if getattr(pipeline, "uses_external_target_log", False) else "True"
    show_estimate_dots = not getattr(pipeline, "uses_external_target_log", False)
    ax_traj.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    ax_traj.plot(true_measurements[:, 0], true_measurements[:, 1], color="tab:blue", linewidth=1.2, label=measurement_label)
    if show_estimate_dots:
        ax_traj.scatter(measurements[:, 0], measurements[:, 1], color="tab:blue", marker=".", s=5, linewidths=0.0)
    _plot_replay_windows_xy(ax_traj, measurements, replay, window_starts)
    ax_traj.scatter(measurements[window_starts, 0], measurements[window_starts, 1], marker="x", s=32, linewidths=1.0, color="black")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize="small")

    ax_vel_omega = ax_vel.twinx()
    ref_speed = np.linalg.norm(reference[:, 3:5], axis=1)
    true_vel_time, true_vel = _pose_vel_omega(plot_time, true_measurements)
    line_ref_v = ax_vel.step(reference_time, ref_speed, where="post", color="tab:blue", linestyle="--", linewidth=0.8, label="ref v")[0]
    line_true_v = ax_vel.plot(true_vel_time, true_vel[:, 0], color="tab:blue", linewidth=1.0, label="true v")[0]
    line_model_v, line_model_w = _plot_windowed_velocity(ax_vel, ax_vel_omega, plot_time, measurements, replay, window_starts)
    line_ref_w = ax_vel_omega.step(reference_time, reference[:, 5], where="post", color="tab:purple", linestyle="--", linewidth=0.8, label=r"ref $\omega$")[0]
    line_true_w = ax_vel_omega.plot(true_vel_time, true_vel[:, 1], color="tab:purple", linewidth=1.0, label=r"true $\omega$")[0]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True)
    velocity_handles = [line_ref_v, line_true_v, line_model_v, line_ref_w, line_true_w, line_model_w]
    ax_vel.legend(handles=[handle for handle in velocity_handles if handle is not None], loc="best")

    cmd_time, cmd_right = _stair_series(command_time, wheel_cmd[:, 0], wheel_time[-1] if len(wheel_time) else None)
    _, cmd_left = _stair_series(command_time, wheel_cmd[:, 1], wheel_time[-1] if len(wheel_time) else None)
    line_cmd_right = ax_wheels.step(cmd_time, cmd_right, where="post", color="tab:green", linestyle="--", linewidth=0.6, label="cmd right")[0]
    line_meas_right = ax_wheels.plot(wheel_time, wheel_actual[:, 0], color="tab:green", linewidth=0.9, label="meas right")[0]
    line_cmd_left = ax_wheels.step(cmd_time, cmd_left, where="post", color="tab:orange", linestyle="--", linewidth=0.6, label="cmd left")[0]
    line_meas_left = ax_wheels.plot(wheel_time, wheel_actual[:, 1], color="tab:orange", linewidth=0.9, label="meas left")[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    ax_wheels.legend(handles=[line_cmd_right, line_meas_right, line_cmd_left, line_meas_left])

    _plot_windowed_motor_model_axes(
        ax_motor,
        wheel_time,
        duty_cycle,
        wheel_actual,
        motor_model_wheel_speeds,
        wheel_window_starts,
    )

    state_labels = ("x [m]", "y [m]", "theta [rad]")
    state_names = ("x", "y", "theta")
    for index, ax in enumerate(state_axes):
        ax.step(reference_time, reference[:, index], where="post", color="tab:red", linestyle="--", linewidth=0.9, label="Reference")
        ax.plot(plot_time, true_measurements[:, index], color="tab:blue", linewidth=1.2, label=measurement_label)
        if show_estimate_dots:
            ax.scatter(plot_time, measurements[:, index], color="tab:blue", marker=".", s=5, linewidths=0.0)
        _plot_replay_windows_state(ax, plot_time, measurements, replay, window_starts, index)
        _plot_window_start_markers(ax, plot_time, measurements[:, index], window_starts)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(state_labels[index])
        ax.set_title(f"{state_names[index]} State")
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"System ID summary PDF saved at: {output_path}")


def _window_start_indices(pipeline, plot_len: int, window_length: int | None) -> np.ndarray:
    num_intervals = max(plot_len - 1, 1)
    resolved_window_length = pipeline.resolve_replay_window_length(window_length, num_intervals)
    return np.arange(0, num_intervals, resolved_window_length, dtype=int)


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


def _plot_replay_windows_xy(
    ax,
    measurements: np.ndarray,
    replay: np.ndarray,
    window_starts: np.ndarray,
):
    for window_index, start_idx in enumerate(window_starts):
        _, segment_poses = _model_window_segment(measurements, replay, None, window_starts, window_index)
        if len(segment_poses) < 2:
            continue
        label = "Model" if window_index == 0 else None
        ax.plot(
            segment_poses[:, 0],
            segment_poses[:, 1],
            color="tab:orange",
            linewidth=1.0,
            label=label,
        )


def _plot_replay_windows_state(
    ax,
    plot_time: np.ndarray,
    measurements: np.ndarray,
    replay: np.ndarray,
    window_starts: np.ndarray,
    state_index: int,
):
    _plot_replay_windows_series(
        ax,
        plot_time,
        measurements,
        replay,
        window_starts,
        state_index,
        color="tab:orange",
        label="Model",
        linewidth=1.0,
    )


def _plot_windowed_motor_model_axes(
    ax_duty,
    time: np.ndarray,
    duty_cycle: np.ndarray,
    encoder_wheel_speeds: np.ndarray,
    model_wheel_speeds: np.ndarray,
    window_starts: np.ndarray,
):
    ax_speed = ax_duty.twinx()

    line_dc_l = ax_duty.plot(
        time,
        duty_cycle[:, 1],
        color="#f2c14e",
        linewidth=0.3,
        label="DC left",
    )[0]
    line_enc_l = ax_speed.plot(
        time,
        encoder_wheel_speeds[:, 1],
        color="#f28e2b",
        linewidth=0.9,
        label="enc left",
    )[0]
    line_dc_r = ax_duty.plot(
        time,
        duty_cycle[:, 0],
        color="#59a14f",
        linewidth=0.3,
        label="DC right",
    )[0]
    line_enc_r = ax_speed.plot(
        time,
        encoder_wheel_speeds[:, 0],
        color="#1b7f3a",
        linewidth=0.9,
        label="enc right",
    )[0]

    model_left = _plot_replay_windows_series(
        ax_speed,
        time,
        encoder_wheel_speeds,
        model_wheel_speeds,
        window_starts,
        1,
        color="#b85c00",
        label="model left",
        linewidth=0.9,
    )
    model_right = _plot_replay_windows_series(
        ax_speed,
        time,
        encoder_wheel_speeds,
        model_wheel_speeds,
        window_starts,
        0,
        color="#0b5d1e",
        label="model right",
        linewidth=0.9,
    )
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
        handles=[line_dc_l, line_enc_l, model_left, line_dc_r, line_enc_r, model_right],
        loc="best",
    )


def _plot_replay_windows_series(
    ax,
    time: np.ndarray,
    measurements: np.ndarray,
    replay: np.ndarray,
    window_starts: np.ndarray,
    value_index: int,
    color: str,
    label: str,
    linewidth: float,
):
    first_line = None
    for window_index, _ in enumerate(window_starts):
        segment_time, segment_values = _model_window_segment(
            measurements,
            replay,
            time,
            window_starts,
            window_index,
        )
        if len(segment_values) < 2:
            continue
        line = ax.plot(
            segment_time,
            segment_values[:, value_index],
            color=color,
            linewidth=linewidth,
            label=label if first_line is None else None,
        )[0]
        first_line = line if first_line is None else first_line
    return first_line


def _plot_windowed_velocity(
    ax_v,
    ax_w,
    time: np.ndarray,
    measurements: np.ndarray,
    replay: np.ndarray,
    window_starts: np.ndarray,
):
    first_v = None
    first_w = None
    for window_index, _ in enumerate(window_starts):
        segment_time, segment_poses = _model_window_segment(measurements, replay, time, window_starts, window_index)
        if len(segment_poses) < 2:
            continue
        vel_time, velocity = _pose_vel_omega(segment_time, segment_poses)
        line_v = ax_v.plot(
            vel_time,
            velocity[:, 0],
            color="tab:orange",
            linewidth=1.0,
            label="model v" if first_v is None else None,
        )[0]
        line_w = ax_w.plot(
            vel_time,
            velocity[:, 1],
            color="tab:red",
            linewidth=1.0,
            label=r"model $\omega$" if first_w is None else None,
        )[0]
        first_v = line_v if first_v is None else first_v
        first_w = line_w if first_w is None else first_w
    return first_v, first_w


def _model_window_segment(
    measurements: np.ndarray,
    replay: np.ndarray,
    plot_time: np.ndarray | None,
    window_starts: np.ndarray,
    window_index: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    start_idx = min(int(window_starts[window_index]), len(measurements) - 1)
    if window_index + 1 < len(window_starts):
        end_idx = min(int(window_starts[window_index + 1]), len(measurements) - 1)
    else:
        end_idx = len(measurements) - 1

    replay_segment = replay[start_idx:end_idx]
    poses = np.vstack([measurements[start_idx], replay_segment])
    if plot_time is None:
        return None, poses
    times = np.concatenate([plot_time[start_idx : start_idx + 1], plot_time[start_idx + 1 : end_idx + 1]])
    return times, poses


def _plot_window_start_markers(
    ax,
    plot_time: np.ndarray,
    values: np.ndarray,
    window_starts: np.ndarray,
    color: str = "black",
):
    ax.scatter(
        plot_time[window_starts],
        values[window_starts],
        marker="x",
        s=32,
        linewidths=1.0,
        color=color,
    )


def _set_symmetric_ylim(ax, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    limit = np.nanmax(np.abs(values)) if values.size else 0.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    ax.set_ylim(-1.05 * limit, 1.05 * limit)


def plot_loss_history(
    loss_history,
    validation_loss_history=None,
    hidden_loss_history=None,
    parameter_error_history=None,
    motor_loss_history=None,
    out_prefix="system_id",
):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize/", f"loss_{out_prefix}.pdf")

    steps = np.arange(1, len(loss_history) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    legend_handles = []
    ax.plot(steps, np.asarray(loss_history), 'b-', linewidth=2, label='Normalized Geometry Error')
    legend_handles.extend(ax.get_lines()[-1:])
    if motor_loss_history is not None and len(motor_loss_history) > 0:
        motor_steps = np.arange(1, len(motor_loss_history) + 1)
        ax.plot(motor_steps, np.asarray(motor_loss_history), color='C1', linewidth=2, label='Normalized Motor Error')
        legend_handles.extend(ax.get_lines()[-1:])
    if validation_loss_history is not None and len(validation_loss_history) > 0:
        validation_steps = np.arange(1, len(validation_loss_history) + 1)
        ax.plot(validation_steps, np.asarray(validation_loss_history), 'r-', linewidth=2, label='Validation')
        legend_handles.extend(ax.get_lines()[-1:])
    hidden_ax = None
    parameter_history = parameter_error_history if parameter_error_history is not None else hidden_loss_history
    if parameter_history is not None and len(parameter_history) > 0:
        hidden_steps = np.arange(1, len(parameter_history) + 1)
        hidden_ax = ax.twinx()
        hidden_ax.plot(hidden_steps, np.asarray(parameter_history), 'g-', linewidth=2, label='Parameter MSE [mm^2]')
        hidden_ax.set_yscale('log')
        legend_handles.extend(hidden_ax.get_lines()[-1:])
    ax.set_yscale('log')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss', color='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    if hidden_ax is not None:
        hidden_ax.set_ylabel('Parameter MSE [mm^2]', color='black')
        hidden_ax.tick_params(axis='y', colors='black')
        hidden_ax.spines['right'].set_color('black')
        hidden_ax.spines['top'].set_color('black')
    ax.set_title('Loss History')
    ax.grid(True)
    ax.legend(handles=legend_handles, loc='best')
    fig.tight_layout()
    fig.savefig(pdf_filename, bbox_inches='tight', transparent=True)
    plt.close(fig)

    print(f"Loss history PDF saved at: {pdf_filename}")


def plot_system_id_realization_sweep(
    num_realizations, tracking_losses, parameter_mse, out_prefix="si_realization_sweep", num_seeds=None):

    os.makedirs("visualize", exist_ok=True)
    output_filename = os.path.join("visualize", f"{out_prefix}.pdf")

    x_values = np.asarray(num_realizations, dtype=float)
    tracking_losses = np.asarray(tracking_losses)
    parameter_mse = np.asarray(parameter_mse)

    fig, ax1 = plt.subplots(1, 1, figsize=(8.5, 4.8))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        x_values,
        tracking_losses,
        color="tab:blue",
        marker="o",
        linewidth=2,
        label="Final tracking MSE",
    )[0]
    line2 = ax2.plot(
        x_values,
        parameter_mse,
        color="tab:orange",
        marker="s",
        linewidth=2,
        label="Final Parameter MSE [mm^2]",
    )[0]

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of replay realizations")
    ax1.set_ylabel("Final tracking loss", color="tab:blue")
    ax2.set_ylabel("Parameter MSE [mm^2]", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    if num_seeds is None:
        ax1.set_title("System ID Sensitivity to Replay Realizations")
    else:
        ax1.set_title(f"System ID Sensitivity to Replay Realizations ({num_seeds} seeds averaged)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(min(x_values), max(x_values))
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([str(int(value)) if value.is_integer() else str(value) for value in x_values])
    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc='center right')

    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved figure to: {output_filename}")
    plt.show()


def plot_tracking_error_surface(
    wheel_radius_values,
    base_diameter_values,
    tracking_error_surface,
    num_realizations,
    seed,
    hidden_params=None,
    init_params=None,
    out_prefix="si_tracking_error_surface",
    out_path=None,
    z_min=None,
    z_max=None,
    x_label="Wheel radius [m]",
    y_label="Base diameter [m]",
    surface_label="Identification Loss",
    title="Tracking Error Surface",
    init_label="Initial parameter guess",
    hidden_label="Hidden parameters",
    min_label="Minimum sampled loss",
    save_pdf: bool = True,
    show_plot: bool = False,
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    wheel_radius_grid, base_diameter_grid = np.meshgrid(wheel_radius_values, base_diameter_values)
    surface = np.asarray(tracking_error_surface)

    def _surface_value(wheel_radius, base_diameter):
        radius_idx = int(np.argmin(np.abs(np.asarray(wheel_radius_values) - wheel_radius)))
        base_idx = int(np.argmin(np.abs(np.asarray(base_diameter_values) - base_diameter)))
        return float(surface[base_idx, radius_idx])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        wheel_radius_grid,
        base_diameter_grid,
        surface,
        cmap="viridis",
        linewidth=0,
        alpha=0.75,
    )

    min_index = np.unravel_index(np.argmin(surface), surface.shape)
    min_base = float(base_diameter_values[min_index[0]])
    min_radius = float(wheel_radius_values[min_index[1]])
    min_loss = float(surface[min_index])

    if init_params is not None:
        init_radius = float(init_params.wheel_radius)
        init_base = float(init_params.base_diameter)
        init_loss = _surface_value(init_radius, init_base)
        ax.scatter(
            [init_radius],
            [init_base],
            [init_loss],
            color="white",
            edgecolors="black",
            s=60,
            depthshade=False,
            alpha=1,
            label=init_label,
        )
    if hidden_params is not None:
        hidden_radius = float(hidden_params.wheel_radius)
        hidden_base = float(hidden_params.base_diameter)
        true_loss = _surface_value(hidden_radius, hidden_base)
        ax.scatter(
            [hidden_radius],
            [hidden_base],
            [true_loss],
            color="gold",
            edgecolors="black",
            marker="*",
            s=180,
            depthshade=False,
            alpha=1,
            label=hidden_label,
        )
    ax.scatter(
        [min_radius],
        [min_base],
        [min_loss],
        color="red",
        edgecolors="black",
        marker="X",
        s=100, depthshade=False, alpha=1,
        label=min_label,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(surface_label)
    if z_min is not None and z_max is not None:
        ax.set_zlim(z_min, z_max)
    ax.set_title(title)
    ax.view_init(elev=10, azim=-157)
    ax.legend(loc="upper right")
    fig.colorbar(
        surf,
        ax=ax,
        shrink=0.7,
        pad=0.1,
        label=surface_label,
        format=FormatStrFormatter("%.1f"),
    )

    fig.tight_layout()
    if save_pdf:
        fig.savefig(output_filename, bbox_inches="tight")
        print(f"Saved figure to: {output_filename}")
    if show_plot:
        plt.show()
    plt.close(fig)
