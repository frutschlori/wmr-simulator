import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np


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
