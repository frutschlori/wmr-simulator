import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
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
        linestyle="--",
        linewidth=trajectory_linewidth,
        color="red",
        label="Complete",
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
    init_params=None,
    x_label="Wheel radius [m]",
    y_label="Base diameter [m]",
    surface_label="Tracking loss",
    init_label="Initial parameter guess",
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
    ax_surface.view_init(elev=25, azim=-50)
    ax_surface.legend(loc="upper right")
    fig.colorbar(surf, ax=ax_surface, shrink=0.7, pad=0.1, label=surface_label)

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
    time = pipeline.sim_time_grid
    reference_states = pipeline.reference_states

    plot_time = np.asarray(time[1:])

    # Handle reference states
    refstates = None
    if reference_states is not None:
        if len(plot_time) > len(reference_states):
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
        poses = np.asarray(sim_log.robot_states.pose)[:len(plot_time)]
        if estimator_filter_type == "dr":
            est_pose = np.asarray(sim_log.estimator_states.pose_meas)[:len(plot_time)]
        else:
            est_pose = np.asarray(sim_log.estimator_states.pose_hat)[:len(plot_time)]
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
                axes[k].plot(plot_time, hidden_actual[:, k], 'b--', label=f'Actual {labels[k]} (hidden)')
                axes[k].plot(plot_time, hidden_est[:, k], 'g--', label=f'Est {labels[k]} (hidden)')
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


def plot_loss_history(loss_history, validation_loss_history=None, hidden_loss_history=None, out_prefix="system_id"):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize/", f"loss_{out_prefix}.pdf")

    steps = np.arange(1, len(loss_history) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    legend_handles = []
    ax.plot(steps, np.asarray(loss_history), 'b-', linewidth=2, label='Training')
    legend_handles.extend(ax.get_lines()[-1:])
    if validation_loss_history is not None and len(validation_loss_history) > 0:
        validation_steps = np.arange(1, len(validation_loss_history) + 1)
        ax.plot(validation_steps, np.asarray(validation_loss_history), 'r-', linewidth=2, label='Validation')
        legend_handles.extend(ax.get_lines()[-1:])
    hidden_ax = None
    if hidden_loss_history is not None and len(hidden_loss_history) > 0:
        hidden_steps = np.arange(1, len(hidden_loss_history) + 1)
        hidden_ax = ax.twinx()
        hidden_ax.plot(hidden_steps, np.asarray(hidden_loss_history), 'g-', linewidth=2, label='Hidden Parameter')
        # hidden_ax.set_yscale('log')
        legend_handles.extend(hidden_ax.get_lines()[-1:])
    # ax.set_yscale('log')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Tracking MSE', color='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    if hidden_ax is not None:
        hidden_ax.set_ylabel('Parameter MSE', color='black')
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


def plot_controller_tuning_errors(pipeline, init_log, tuned_log, out_prefix="ctrl_tuning"):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize", f"{out_prefix}_tracking_errors.pdf")

    reference_poses = np.asarray(pipeline.reference_states[:, :3])
    reference_vel = np.asarray(pipeline.reference_states[:, 3:5])
    reference_omega = np.asarray(pipeline.reference_states[:, 5])
    init_poses = np.asarray(init_log.robot_states.pose)
    tuned_poses = np.asarray(tuned_log.robot_states.pose)
    init_vel_omega = np.asarray(init_log.robot_states.vel_omega)
    tuned_vel_omega = np.asarray(tuned_log.robot_states.vel_omega)
    plot_len = min(
        len(reference_poses),
        len(reference_vel),
        len(reference_omega),
        len(init_poses),
        len(tuned_poses),
        len(init_vel_omega),
        len(tuned_vel_omega),
        len(pipeline.sim_time_grid) - 1,
    )
    plot_time = np.asarray(pipeline.sim_time_grid[1:plot_len + 1])

    reference_poses = reference_poses[:plot_len]
    reference_vel = reference_vel[:plot_len]
    reference_omega = reference_omega[:plot_len]
    init_poses = init_poses[:plot_len]
    tuned_poses = tuned_poses[:plot_len]
    init_vel_omega = init_vel_omega[:plot_len]
    tuned_vel_omega = tuned_vel_omega[:plot_len]

    init_errors = init_poses - reference_poses
    tuned_errors = tuned_poses - reference_poses
    init_errors[:, 2] = (init_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    tuned_errors[:, 2] = (tuned_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    reference_speed = np.sqrt(reference_vel[:, 0] ** 2 + reference_vel[:, 1] ** 2)
    init_speed_errors = init_vel_omega[:, 0] - reference_speed
    tuned_speed_errors = tuned_vel_omega[:, 0] - reference_speed
    init_omega_errors = init_vel_omega[:, 1] - reference_omega
    tuned_omega_errors = tuned_vel_omega[:, 1] - reference_omega

    with PdfPages(pdf_filename) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ["x error [m]", "y error [m]", "theta error [rad]"]

        for idx, label in enumerate(labels):
            axes[idx].plot(plot_time, init_errors[:, idx], label="Initial gains", linewidth=2)
            axes[idx].plot(plot_time, tuned_errors[:, idx], label="Tuned gains", linewidth=2)
            axes[idx].set_ylabel(label)
            axes[idx].grid(True)
            axes[idx].legend()

        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(r"Pose Errors $(\hat x - x_{ref})$", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight", transparent=True)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(plot_time, init_speed_errors, label="Initial gains", linewidth=2)
        axes[0].plot(plot_time, tuned_speed_errors, label="Tuned gains", linewidth=2)
        axes[0].set_ylabel("linear velocity error [m/s]")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(plot_time, init_omega_errors, label="Initial gains", linewidth=2)
        axes[1].plot(plot_time, tuned_omega_errors, label="Tuned gains", linewidth=2)
        axes[1].set_ylabel("angular velocity error [rad/s]")
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle("Velocity Errors $(v_{ctrl} - v_{ref})$", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight", transparent=True)
        plt.close(fig)

    print(f"Controller tuning error PDF saved at: {pdf_filename}")


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
        label="Final Parameter MSE",
    )[0]

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of replay realizations")
    ax1.set_ylabel("Final tracking loss", color="tab:blue")
    ax2.set_ylabel("Parameter MSE", color="tab:orange")
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
    x_label="Wheel radius [m]",
    y_label="Base diameter [m]",
    surface_label="Tracking loss",
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
    ax.set_title(
        f"{title} (seed={seed}, replay realizations={num_realizations})"
    )
    ax.view_init(elev=25, azim=-50)
    ax.legend(loc="upper right")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label=surface_label)

    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved figure to: {output_filename}")
    plt.close(fig)
