import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


def plot_windowed_replay_trajectory(
    pipeline,
    closed_loop_log,
    replay_log,
    window_length=None,
    out_prefix="identification_windowed_trajectory",
    out_path=None,
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    reference_states = np.asarray(pipeline.reference_states)
    full_reference_states = np.asarray(getattr(pipeline, "full_reference_states", pipeline.reference_states))
    closed_loop_actual = np.asarray(closed_loop_log.robot_states.pose)
    if pipeline.estimator.filter_type == "dr":
        closed_loop_estimates = np.asarray(closed_loop_log.estimator_states.pose_meas)
        replay_estimates = np.asarray(replay_log.estimator_states.pose_meas)
    else:
        closed_loop_estimates = np.asarray(closed_loop_log.estimator_states.pose_hat)
        replay_estimates = np.asarray(replay_log.estimator_states.pose_hat)
    replay_actual = np.asarray(replay_log.robot_states.pose)

    resolved_window_length = pipeline.resolve_window_length(window_length)
    num_replay_intervals = max(len(replay_estimates) - 1, 1)
    window_start_indices = np.arange(0, num_replay_intervals, resolved_window_length)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(
        full_reference_states[:, 0],
        full_reference_states[:, 1],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Reference",
    )
    if len(reference_states) > 0 and not np.array_equal(reference_states, full_reference_states):
        ax.plot(
            reference_states[:, 0],
            reference_states[:, 1],
            color="red",
            linestyle="-",
            linewidth=1.5,
            label="Selected Reference",
        )
    ax.plot(
        closed_loop_actual[:, 0],
        closed_loop_actual[:, 1],
        color="blue",
        linestyle="-",
        linewidth=0.9,
        label="Closed-Loop Actual",
    )
    ax.scatter(
        closed_loop_estimates[:, 0],
        closed_loop_estimates[:, 1],
        color="blue",
        marker="x",
        s=3,
        alpha=1,
        linewidth=0.5,
        label="Closed-Loop Estimate",
    )
    for window_idx, start_idx in enumerate(window_start_indices):
        end_idx = min(start_idx + resolved_window_length, len(replay_actual))
        if end_idx <= start_idx:
            continue
        label = "Windowed Replay Actual" if window_idx == 0 else None
        window_actual = replay_actual[start_idx:end_idx].copy()
        window_actual[0] = closed_loop_estimates[start_idx]
        ax.plot(
            window_actual[:, 0],
            window_actual[:, 1],
            color="orange",
            linestyle="-",
            linewidth=0.9,
            label=label,
        )
    ax.scatter(
        replay_estimates[:, 0],
        replay_estimates[:, 1],
        color="orange",
        s=3,
        marker="x",
        alpha=1,
        linewidth=0.5,
        label="Windowed Replay Estimate",
    )
    ax.scatter(
        closed_loop_estimates[window_start_indices, 0],
        closed_loop_estimates[window_start_indices, 1],
        marker="x",
        s=36,
        linewidths=1.0,
        color="black",
        label="Window Start",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Identification Replay Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=False, facecolor="white")
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


def plot_system_id(pipeline, init_target_log, init_log, predicted_log, out_prefix="system_identification"):

    hidden_log = pipeline.target_log
    estimator_filter_type = pipeline.estimator.filter_type
    reference_states = pipeline.reference_states
    is_real_experiment = getattr(pipeline, "uses_external_target_log", False)
    target_label = "meas" if is_real_experiment else "hidden"

    plot_len = min(
        hidden_log.robot_states.pose.shape[0],
        init_target_log.robot_states.pose.shape[0],
        init_log.robot_states.pose.shape[0],
        predicted_log.robot_states.pose.shape[0],
    )
    if getattr(pipeline, "target_time_s", None) is not None:
        plot_time = np.asarray(pipeline.target_time_s)[:plot_len]
    else:
        plot_time = np.asarray(pipeline.sim_time_grid)[:plot_len]

    # Handle reference states
    refstates = None
    if reference_states is not None:
        if getattr(pipeline, "target_time_s", None) is not None and not getattr(pipeline, "uses_external_target_log", False):
            ref_indices = np.rint(plot_time / pipeline.dt).astype(int)
            ref_indices = np.clip(ref_indices, 0, len(reference_states) - 1)
            extended_ref_states = reference_states[ref_indices]
        elif len(plot_time) > len(reference_states):
            num_extra_steps = len(plot_time) - len(reference_states)
            last_ref_state = reference_states[-1]
            extended_ref_states = np.vstack([
                reference_states,
                np.tile(last_ref_state, (num_extra_steps, 1))
            ])
        else:
            extended_ref_states = reference_states[:len(plot_time)]

        ref_pos = np.array(extended_ref_states[:, 0:2])
        ref_th = np.array(extended_ref_states[:, 2])
        refstates = np.column_stack([ref_pos, ref_th])
        min_len = min(len(refstates), len(plot_time))
        refstates = refstates[:min_len]

    def _extract_state_series(sim_log):
        poses = np.asarray(sim_log.robot_states.pose)[:plot_len]
        if estimator_filter_type == "dr":
            est_pose = np.asarray(sim_log.estimator_states.pose_meas)[:plot_len]
        else:
            est_pose = np.asarray(sim_log.estimator_states.pose_hat)[:plot_len]
        return poses, est_pose

    poses_hidden, est_hidden = _extract_state_series(hidden_log)
    poses_init_hidden, est_init_hidden = _extract_state_series(init_target_log)
    poses_init, est_init = _extract_state_series(init_log)
    poses_pred, est_pred = _extract_state_series(predicted_log)

    # --- Prepare output path
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize/", f"{out_prefix}.pdf")

    with PdfPages(pdf_filename) as pdf:
        def _plot_state_tracking(hidden_actual, hidden_est, actual_guess, est_guess, title_suffix):
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            labels = ["x", "y", "theta"]

            for k in range(3):
                if refstates is not None:
                    axes[k].plot(plot_time, refstates[:, k], 'r-', label=f'Ref {labels[k]}')
                if is_real_experiment:
                    axes[k].plot(plot_time, hidden_actual[:, k], 'b--', label="Measured")
                    axes[k].plot(plot_time, actual_guess[:, k], 'c-.', label="Model")
                else:
                    axes[k].plot(plot_time, hidden_actual[:, k], 'b--', label=f'Actual {labels[k]} ({target_label})')
                    axes[k].plot(plot_time, hidden_est[:, k], 'g--', label=f'Est {labels[k]} ({target_label})')
                    axes[k].plot(plot_time, actual_guess[:, k], 'c-.', label=f'Actual {labels[k]} ({title_suffix})')
                    axes[k].plot(plot_time, est_guess[:, k], 'm-.', label=f'Est {labels[k]} ({title_suffix})')
                axes[k].set_ylabel(f'{["X Position", "Y Position", "Angle"][k]}')
                axes[k].legend()
                axes[k].grid(True)

            axes[-1].set_xlabel('Time [s]')
            fig.suptitle(f"State Tracking ({title_suffix})", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches='tight', transparent=True)
            plt.close(fig)

        _plot_state_tracking(poses_init_hidden, est_init_hidden, poses_init, est_init, "initial guess")
        _plot_state_tracking(poses_hidden, est_hidden, poses_pred, est_pred, "identified")

        d = pdf.infodict()
        d['Title'] = 'Differential Drive Simulation Results'
        d['Author'] = 'Wheeled Robot Simulator'
        d['Subject'] = 'State Tracking and Wheel Inputs'
        d['Keywords'] = 'diffdrive, simulation, robotics, control'

    print(f"Multi-page PDF saved at: {pdf_filename}")


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
