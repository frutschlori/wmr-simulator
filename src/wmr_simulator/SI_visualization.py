import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def plot_system_id(pipeline, init_target_log, init_log, predicted_log, out_prefix="plot_trajectory"):

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


def plot_loss_history(loss_history, out_prefix="plot_trajectory"):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize/", f"{out_prefix}_loss.pdf")

    steps = np.arange(1, len(loss_history) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(steps, np.asarray(loss_history), 'b-', linewidth=2, label='Loss')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss History')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf_filename, bbox_inches='tight', transparent=True)
    plt.close(fig)

    print(f"Loss history PDF saved at: {pdf_filename}")


def plot_system_id_realization_sweep(
    num_realizations, tracking_losses, parameter_mse, out_prefix="si_realization_sweep", num_seeds=None
):
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
    hidden_params,
    init_params,
    num_realizations,
    seed,
    out_prefix="si_tracking_error_surface",
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = os.path.join("visualize", f"{out_prefix}.pdf")

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

    hidden_radius = float(hidden_params.wheel_radius)
    hidden_base = float(hidden_params.base_diameter)
    init_radius = float(init_params.wheel_radius)
    init_base = float(init_params.base_diameter)
    true_loss = _surface_value(hidden_radius, hidden_base)
    init_loss = _surface_value(init_radius, init_base)
    min_index = np.unravel_index(np.argmin(surface), surface.shape)
    min_base = float(base_diameter_values[min_index[0]])
    min_radius = float(wheel_radius_values[min_index[1]])
    min_loss = float(surface[min_index])

    ax.scatter(
        [init_radius],
        [init_base],
        [init_loss],
        color="white",
        edgecolors="black",
        s=60, depthshade=False, alpha=1,
        label="Initial parameter guess",
    )
    ax.scatter(
        [hidden_radius],
        [hidden_base],
        [true_loss],
        color="gold",
        edgecolors="black",
        marker="*",
        s=180, depthshade=False, alpha=1,
        label="Hidden parameters",
    )
    ax.scatter(
        [min_radius],
        [min_base],
        [min_loss],
        color="red",
        edgecolors="black",
        marker="X",
        s=100, depthshade=False, alpha=1,
        label="Minimum sampled loss",
    )

    ax.set_xlabel("Wheel radius [m]")
    ax.set_ylabel("Base diameter [m]")
    ax.set_zlabel("Tracking loss")
    ax.set_title(
        f"Tracking Error Surface (seed={seed}, replay realizations={num_realizations})"
    )
    ax.view_init(elev=18, azim=-145)
    ax.legend(loc="upper right")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="Tracking loss")

    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved figure to: {output_filename}")
    plt.show()
