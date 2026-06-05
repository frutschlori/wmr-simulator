import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
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
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    reference_states = np.asarray(pipeline.reference_states)
    full_reference_states = np.asarray(getattr(pipeline, "full_reference_states", pipeline.reference_states))
    estimator_filter_type = pipeline.estimator.filter_type

    def _extract_estimated_poses(sim_log):
        if sim_log is None:
            return None
        if estimator_filter_type == "dr":
            poses = np.asarray(sim_log.estimator_states.pose_meas)
        else:
            poses = np.asarray(sim_log.estimator_states.pose_hat)
        return poses[:, :3]

    def _extract_actual_poses(sim_log):
        if sim_log is None:
            return None
        poses = np.asarray(sim_log.robot_states.pose)
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

    tuned_poses = _extract_estimated_poses(tuned_log)
    untuned_poses = _extract_estimated_poses(untuned_log)
    tuned_actual_poses = _extract_actual_poses(tuned_log)
    untuned_actual_poses = _extract_actual_poses(untuned_log)

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
        if untuned_actual_poses is not None:
            ax.plot(
                untuned_actual_poses[:, 0],
                untuned_actual_poses[:, 1],
                linewidth=trajectory_linewidth,
                linestyle="-",
                color="blue",
                label="Initial actual",
            )
        ax.plot(
            untuned_poses[:, 0],
            untuned_poses[:, 1],
            linewidth=trajectory_linewidth,
            linestyle="--",
            color="blue",
            label="Initial estimated",
        )
        _draw_theta_arrows(ax, untuned_poses, "blue")

    if tuned_poses is not None:
        if tuned_actual_poses is not None:
            ax.plot(
                tuned_actual_poses[:, 0],
                tuned_actual_poses[:, 1],
                linewidth=trajectory_linewidth,
                linestyle="-",
                color="orange",
                label="Tuned actual",
            )
        ax.plot(
            tuned_poses[:, 0],
            tuned_poses[:, 1],
            linewidth=trajectory_linewidth,
            linestyle="--",
            color="orange",
            label="Tuned estimated",
        )
        _draw_theta_arrows(ax, tuned_poses, "orange")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Comparison")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Trajectory PDF saved at: {output_filename}")


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
        linewidth=trajectory_linewidth,
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
                linewidth=trajectory_linewidth,
                color=color,
            )
            _draw_theta_arrows(ax, reference_states[:, :3], color)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=trajectory_linewidth, linestyle="-", label="Training"),
        Line2D([0], [0], color="black", linewidth=trajectory_linewidth, linestyle="--", label="Validation"),
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
    show_markers: bool = False,
):
    target_log = pipeline.target_log
    is_real_experiment = getattr(pipeline, "uses_external_target_log", False)
    marker_kwargs = _line_marker_kwargs(show_markers)

    plot_len = min(
        target_log.robot_states.pose.shape[0],
        predicted_log.robot_states.pose.shape[0],
    )
    plot_time = _plot_time_for_pipeline(pipeline, plot_len)
    reference = _reference_states_for_time(pipeline, plot_time)
    reference_vel = _reference_vel_omega_for_time(pipeline, plot_time)
    measurements = _measurement_pose_series(pipeline, target_log)[:plot_len]
    replay = np.asarray(predicted_log.robot_states.pose)[:plot_len]
    measured_vel = np.asarray(target_log.robot_states.vel_omega)
    odom_vel = np.asarray(predicted_log.robot_states.vel_omega)
    velocity_time, measured_vel, odom_vel, reference_vel = _aligned_velocity_series(
        plot_time,
        measured_vel,
        odom_vel,
        reference_vel,
    )
    wheel_cmd = np.asarray(target_log.robot_states.wheel_cmd)[:plot_len]
    wheel_actual = np.asarray(target_log.robot_states.wheel_speeds)[:plot_len]
    estimates = None if is_real_experiment else _estimated_pose_series(pipeline, target_log)[:plot_len]

    window_starts = _window_start_indices(pipeline, plot_len, getattr(pipeline, "window_length", None))

    os.makedirs("visualize", exist_ok=True)
    output_path = os.path.join("visualize", f"{out_prefix}.pdf")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("System Identification Summary", fontsize=16)

    ax_traj = axes[0, 0]
    if reference is not None:
        ax_traj.plot(
            reference[:, 0],
            reference[:, 1],
            color="tab:red",
            linestyle="--",
            linewidth=1.3,
            label="Reference",
            **marker_kwargs,
        )
    ax_traj.plot(
        measurements[:, 0],
        measurements[:, 1],
        color="tab:blue",
        linewidth=1.2,
        label="Measured",
        **marker_kwargs,
    )
    if estimates is not None:
        ax_traj.scatter(estimates[:, 0], estimates[:, 1], s=2, color="tab:blue", alpha=0.45)
    _plot_replay_windows_xy(ax_traj, measurements, replay, window_starts, marker_kwargs)
    ax_traj.scatter(
        measurements[window_starts, 0],
        measurements[window_starts, 1],
        marker="x",
        s=32,
        linewidths=1.0,
        color="black",
    )
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax_vel = axes[0, 1]
    ax_vel_omega = ax_vel.twinx()
    line_meas_v = ax_vel.plot(
        velocity_time,
        measured_vel[:, 0],
        color="tab:blue",
        linewidth=1.2,
        label="mocap v" if is_real_experiment else "meas v",
        **marker_kwargs,
    )[0]
    line_meas_w = ax_vel_omega.plot(
        velocity_time,
        measured_vel[:, 1],
        color="tab:purple",
        linewidth=1.2,
        label=r"mocap $\omega$" if is_real_experiment else r"meas $\omega$",
        **marker_kwargs,
    )[0]
    line_odom_v = ax_vel.plot(
        velocity_time,
        odom_vel[:, 0],
        color="tab:blue",
        linestyle=":",
        linewidth=1.4,
        label="odom v",
        **marker_kwargs,
    )[0]
    line_odom_w = ax_vel_omega.plot(
        velocity_time,
        odom_vel[:, 1],
        color="tab:purple",
        linestyle=":",
        linewidth=1.4,
        label=r"odom $\omega$",
        **marker_kwargs,
    )[0]
    line_ref_v = None
    line_ref_w = None
    if reference_vel is not None:
        line_ref_v = ax_vel.plot(
            velocity_time,
            reference_vel[:, 0],
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            label="ref v",
            **marker_kwargs,
        )[0]
        line_ref_w = ax_vel_omega.plot(
            velocity_time,
            reference_vel[:, 1],
            color="tab:purple",
            linestyle="--",
            linewidth=1.2,
            label=r"ref $\omega$",
            **marker_kwargs,
        )[0]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity Tracking")
    ax_vel.grid(True)
    ax_vel.legend(
        handles=[
            handle
            for handle in [line_ref_v, line_meas_v, line_odom_v, line_ref_w, line_meas_w, line_odom_w]
            if handle is not None
        ],
        loc="best",
    )

    ax_wheels = axes[0, 2]
    line_cmd_right = ax_wheels.plot(
        plot_time,
        wheel_cmd[:, 0],
        color="tab:green",
        linestyle="--",
        linewidth=1.1,
        label="cmd right",
        **marker_kwargs,
    )[0]
    line_cmd_left = ax_wheels.plot(
        plot_time,
        wheel_cmd[:, 1],
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="cmd left",
        **marker_kwargs,
    )[0]
    line_meas_right = ax_wheels.plot(
        plot_time,
        wheel_actual[:, 0],
        color="tab:green",
        linewidth=1.2,
        label="actual right",
        **marker_kwargs,
    )[0]
    line_meas_left = ax_wheels.plot(
        plot_time,
        wheel_actual[:, 1],
        color="tab:orange",
        linewidth=1.2,
        label="actual left",
        **marker_kwargs,
    )[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    ax_wheels.legend(handles=[line_cmd_right, line_meas_right, line_cmd_left, line_meas_left])

    state_labels = ("x [m]", "y [m]", "theta [rad]")
    state_names = ("x", "y", "theta")
    for index, ax in enumerate(axes[1]):
        if reference is not None:
            ax.plot(plot_time, reference[:, index], "r--", linewidth=1.3, label="Reference", **marker_kwargs)
        ax.plot(plot_time, measurements[:, index], color="tab:blue", linewidth=1.2, label="Measured", **marker_kwargs)
        if estimates is not None:
            ax.scatter(plot_time, estimates[:, index], s=2, color="tab:blue", alpha=0.45)
        _plot_replay_windows_state(ax, plot_time, measurements, replay, window_starts, index, marker_kwargs)
        _plot_window_start_markers(ax, plot_time, measurements[:, index], window_starts)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(state_labels[index])
        ax.set_title(f"{state_names[index]} State")
        ax.grid(True)
        ax.legend()

    if show_markers:
        _set_line_widths(fig, 1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)

    print(f"System ID summary PDF saved at: {output_path}")


def _plot_time_for_pipeline(pipeline, plot_len: int) -> np.ndarray:
    if getattr(pipeline, "target_time_s", None) is not None:
        return np.asarray(pipeline.target_time_s, dtype=float)[:plot_len]
    return np.asarray(pipeline.sim_time_grid, dtype=float)[:plot_len]


def _reference_states_for_time(pipeline, plot_time: np.ndarray) -> np.ndarray | None:
    reference_states = getattr(pipeline, "reference_states", None)
    if reference_states is None:
        return None
    reference_states = np.asarray(reference_states)
    if reference_states.size == 0:
        return None

    if getattr(pipeline, "target_time_s", None) is None:
        return _pad_or_trim_reference(reference_states, len(plot_time))

    if getattr(pipeline, "uses_external_target_log", False):
        return _pad_or_trim_reference(reference_states, len(plot_time))

    ref_indices = np.rint(plot_time / pipeline.dt).astype(int)
    ref_indices = np.clip(ref_indices, 0, len(reference_states) - 1)
    return reference_states[ref_indices, :3]


def _reference_vel_omega_for_time(pipeline, plot_time: np.ndarray) -> np.ndarray | None:
    reference_states = getattr(pipeline, "reference_states", None)
    if reference_states is None:
        return None
    reference_states = np.asarray(reference_states)
    if reference_states.ndim != 2 or reference_states.shape[1] < 6:
        return None

    if getattr(pipeline, "target_time_s", None) is None or getattr(pipeline, "uses_external_target_log", False):
        reference_states = _pad_or_trim_full_reference(reference_states, len(plot_time))
    else:
        ref_indices = np.rint(plot_time / pipeline.dt).astype(int)
        ref_indices = np.clip(ref_indices, 0, len(reference_states) - 1)
        reference_states = reference_states[ref_indices]

    linear_speed = np.linalg.norm(reference_states[:, 3:5], axis=1)
    angular_speed = reference_states[:, 5]
    return np.column_stack([linear_speed, angular_speed])


def _aligned_velocity_series(
    plot_time: np.ndarray,
    measured_vel: np.ndarray,
    odom_vel: np.ndarray,
    reference_vel: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    measured_vel = np.asarray(measured_vel, dtype=float)
    odom_vel = np.asarray(odom_vel, dtype=float)
    reference_vel = None if reference_vel is None else np.asarray(reference_vel, dtype=float)

    if len(measured_vel) == len(plot_time) - 1:
        velocity_len = min(len(measured_vel), max(len(odom_vel) - 1, 0), len(plot_time) - 1)
        if reference_vel is not None:
            velocity_len = min(velocity_len, max(len(reference_vel) - 1, 0))
        velocity_time = plot_time[1 : velocity_len + 1]
        measured_vel = measured_vel[:velocity_len]
        odom_vel = odom_vel[1 : velocity_len + 1]
        reference_vel = None if reference_vel is None else reference_vel[1 : velocity_len + 1]
        return velocity_time, measured_vel, odom_vel, reference_vel

    velocity_len = min(len(measured_vel), len(odom_vel), len(plot_time))
    if reference_vel is not None:
        velocity_len = min(velocity_len, len(reference_vel))
    velocity_time = plot_time[:velocity_len]
    measured_vel = measured_vel[:velocity_len]
    odom_vel = odom_vel[:velocity_len]
    reference_vel = None if reference_vel is None else reference_vel[:velocity_len]
    return velocity_time, measured_vel, odom_vel, reference_vel


def _pad_or_trim_reference(reference_states: np.ndarray, length: int) -> np.ndarray:
    reference_states = reference_states[:, :3]
    if len(reference_states) >= length:
        return reference_states[:length]
    tail = np.tile(reference_states[-1], (length - len(reference_states), 1))
    return np.vstack([reference_states, tail])


def _pad_or_trim_full_reference(reference_states: np.ndarray, length: int) -> np.ndarray:
    if len(reference_states) >= length:
        return reference_states[:length]
    tail = np.tile(reference_states[-1], (length - len(reference_states), 1))
    return np.vstack([reference_states, tail])


def _measurement_pose_series(pipeline, sim_log) -> np.ndarray:
    return np.asarray(sim_log.robot_states.pose)


def _estimated_pose_series(pipeline, sim_log) -> np.ndarray:
    if pipeline.estimator.filter_type == "dr":
        return np.asarray(sim_log.estimator_states.pose_meas)
    return np.asarray(sim_log.estimator_states.pose_hat)


def _window_start_indices(pipeline, plot_len: int, window_length: int | None) -> np.ndarray:
    num_intervals = max(plot_len - 1, 1)
    resolved_window_length = pipeline.resolve_window_length(window_length)
    return np.arange(0, num_intervals, resolved_window_length, dtype=int)


def _plot_replay_windows_xy(
    ax,
    measurements: np.ndarray,
    replay: np.ndarray,
    window_starts: np.ndarray,
    marker_kwargs: dict[str, object],
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
            **marker_kwargs,
        )


def _plot_replay_windows_state(
    ax,
    plot_time: np.ndarray,
    measurements: np.ndarray,
    replay: np.ndarray,
    window_starts: np.ndarray,
    state_index: int,
    marker_kwargs: dict[str, object],
):
    for window_index, start_idx in enumerate(window_starts):
        segment_time, segment_poses = _model_window_segment(
            measurements,
            replay,
            plot_time,
            window_starts,
            window_index,
        )
        if len(segment_poses) < 2:
            continue
        label = "Model" if window_index == 0 else None
        ax.plot(
            segment_time,
            segment_poses[:, state_index],
            color="tab:orange",
            linewidth=1.0,
            label=label,
            **marker_kwargs,
        )


def _model_window_segment(
    measurements: np.ndarray,
    replay: np.ndarray,
    plot_time: np.ndarray | None,
    window_starts: np.ndarray,
    window_index: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    start_idx = min(int(window_starts[window_index]), len(replay) - 1)
    if window_index + 1 < len(window_starts):
        end_idx = min(int(window_starts[window_index + 1]), len(replay) - 1)
    else:
        end_idx = len(replay) - 1

    replay_segment = replay[start_idx + 1 : end_idx + 1]
    poses = np.vstack([measurements[start_idx], replay_segment])
    if plot_time is None:
        return None, poses
    times = np.concatenate([plot_time[start_idx : start_idx + 1], plot_time[start_idx + 1 : end_idx + 1]])
    return times, poses


def _plot_window_start_markers(ax, plot_time: np.ndarray, values: np.ndarray, window_starts: np.ndarray):
    ax.scatter(
        plot_time[window_starts],
        values[window_starts],
        marker="x",
        s=32,
        linewidths=1.0,
        color="black",
    )


def _line_marker_kwargs(show_markers: bool) -> dict[str, object]:
    if not show_markers:
        return {}
    return {
        "marker": "x",
        "markersize": 3.2,
        "markeredgewidth": 0.8,
    }


def _set_line_widths(fig, linewidth: float) -> None:
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(linewidth)


def plot_loss_history(loss_history, validation_loss_history=None, hidden_loss_history=None,
                      parameter_error_history=None, out_prefix="system_id"):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize/", f"loss_{out_prefix}.pdf")

    steps = np.arange(1, len(loss_history) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    legend_handles = []
    ax.plot(steps, np.asarray(loss_history), 'b-', linewidth=2, label='Tracking Error')
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
    ax.set_ylabel('Tracking MSE', color='black')
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
    ax.set_title(
        f"{title} (seed={seed}, replay realizations={num_realizations})"
    )
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
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved figure to: {output_filename}")
    plt.close(fig)
