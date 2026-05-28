import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


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
