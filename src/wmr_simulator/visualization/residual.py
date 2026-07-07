"""Plots for the learned residual dynamics model (residual_model.residual)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wmr_simulator.residual_model.residual import RESIDUAL_OUTPUT_DIM, TARGET_LABELS

TWIST_LABELS = (r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$\omega$ [rad/s]")


def plot_training_history(
    history: dict,
    *,
    out_dir: str | Path = "visualize",
    out_name: str = "residual_training_loss.pdf",
    ylabel: str = "MSE (normalized targets)",
) -> Path:
    """Train/validation loss curves."""
    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.semilogy(epochs, history["train_loss"], label="train")
    if np.isfinite(history["validation_loss"]).any():
        ax.semilogy(epochs, history["validation_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = out_dir / out_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    split_name: str,
    *,
    out_dir: str | Path = "visualize",
) -> Path:
    """Predicted vs target residual channels: time series (top) and scatter (bottom)."""
    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)
    fig, axes = plt.subplots(2, RESIDUAL_OUTPUT_DIM, figsize=(13, 7))
    sample_index = np.arange(len(targets))
    for channel in range(RESIDUAL_OUTPUT_DIM):
        ax = axes[0, channel]
        ax.plot(sample_index, targets[:, channel], label="target", lw=0.8)
        ax.plot(sample_index, predictions[:, channel], label="predicted", lw=0.8, alpha=0.8)
        ax.set_title(TARGET_LABELS[channel])
        ax.set_xlabel("sample")
        ax.grid(True, alpha=0.3)
        if channel == 0:
            ax.legend()

        ax = axes[1, channel]
        ax.scatter(targets[:, channel], predictions[:, channel], s=3, alpha=0.4)
        limits = np.array([targets[:, channel].min(), targets[:, channel].max()])
        ax.plot(limits, limits, "k--", lw=1)
        ax.set_xlabel("target")
        ax.set_ylabel("predicted")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Residual model predictions ({split_name})")
    fig.tight_layout()
    output_path = out_dir / f"residual_predictions_{split_name}.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_rollout_comparison(
    *,
    dataset: dict,
    predicted_residual: np.ndarray,
    corrected_twist: np.ndarray,
    nominal_poses: np.ndarray,
    corrected_poses: np.ndarray,
    out_prefix: str,
    out_dir: str | Path = "visualize",
) -> list[Path]:
    """Nominal vs residual-augmented open-loop rollout against the mocap log.

    Saves three figures: XY pose trajectory, body twist time series, and the
    residual predictions vs measured residual targets over time.
    """
    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)
    time_s = dataset["time_s"]
    measured_twist = dataset["measured_twist"]
    nominal_twist = dataset["nominal_twist"]
    output_paths = []

    # Pose trajectory comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(dataset["pose_states"][:, 0], dataset["pose_states"][:, 1], "k-", lw=1.5, label="mocap")
    ax.plot(nominal_poses[:, 0], nominal_poses[:, 1], "--", lw=1.2, label="nominal rollout")
    ax.plot(corrected_poses[:, 0], corrected_poses[:, 1], "-", lw=1.2, label="residual rollout")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_paths.append(out_dir / f"{out_prefix}_trajectory.pdf")
    fig.savefig(output_paths[-1])
    plt.close(fig)

    # Body twist comparison
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for channel, ax in enumerate(axes):
        ax.plot(time_s, measured_twist[:, channel], "k-", lw=0.8, label="measured (mocap)")
        ax.plot(time_s, nominal_twist[:, channel], "--", lw=0.8, label="nominal")
        ax.plot(time_s, corrected_twist[:, channel], "-", lw=0.8, alpha=0.8, label="nominal + residual")
        ax.set_ylabel(TWIST_LABELS[channel])
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    output_paths.append(out_dir / f"{out_prefix}_twist.pdf")
    fig.savefig(output_paths[-1])
    plt.close(fig)

    # Residual predictions over time
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for channel, ax in enumerate(axes):
        ax.plot(time_s, dataset["targets"][:, channel], "k-", lw=0.8, label="measured residual")
        ax.plot(time_s, predicted_residual[:, channel], "-", lw=0.8, alpha=0.8, label="predicted residual")
        ax.set_ylabel(TARGET_LABELS[channel])
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    output_paths.append(out_dir / f"{out_prefix}_residuals.pdf")
    fig.savefig(output_paths[-1])
    plt.close(fig)

    return output_paths


def plot_closed_loop_rollout(
    sim_log,
    *,
    out_prefix: str,
    out_dir: str | Path = "visualize",
) -> Path:
    """Closed-loop simulation (residual-augmented dynamics) vs its reference.

    Left: XY trajectory of the reference and the simulated (true) robot pose.
    Right: forward velocity and angular rate of the robot against the
    reference's velocity profile.
    """
    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)

    reference = np.asarray(sim_log.reference.states, dtype=float)
    reference_time = np.asarray(sim_log.reference.time_s, dtype=float)
    poses = np.asarray(sim_log.pose.true_states, dtype=float)
    vel_omega = np.asarray(sim_log.wheel.vel_omega, dtype=float)
    wheel_time = np.asarray(sim_log.wheel.time_s, dtype=float)

    fig, (ax_xy, ax_v, ax_w) = plt.subplots(
        1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1, 1]}
    )
    ax_xy.plot(reference[:, 0], reference[:, 1], "k--", lw=1.2, label="reference")
    ax_xy.plot(poses[:, 0], poses[:, 1], "-", lw=1.2, label="closed loop (residual)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend()

    reference_speed = np.linalg.norm(reference[:, 3:5], axis=1)
    ax_v.step(reference_time, reference_speed, where="post", color="k", ls="--", lw=1.0, label="ref $v$")
    ax_v.plot(wheel_time, vel_omega[:, 0], lw=0.9, label="sim $v_x$")
    ax_v.set_xlabel("time [s]")
    ax_v.set_ylabel("forward velocity [m/s]")
    ax_v.grid(True, alpha=0.3)
    ax_v.legend()

    ax_w.step(reference_time, reference[:, 5], where="post", color="k", ls="--", lw=1.0, label=r"ref $\omega$")
    ax_w.plot(wheel_time, vel_omega[:, 1], lw=0.9, label=r"sim $\omega$")
    ax_w.set_xlabel("time [s]")
    ax_w.set_ylabel("angular velocity [rad/s]")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend()

    fig.suptitle("Closed-loop simulation with residual model")
    fig.tight_layout()
    output_path = out_dir / f"{out_prefix}.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _ensure_dir(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_module():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt
