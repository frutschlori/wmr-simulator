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
    segment_lengths: list[int] | None = None,
) -> Path:
    """Predicted vs target residual channels: time series (top) and scatter (bottom).

    The training set concatenates samples from several independent experiment
    logs; the step from one log's end to the next log's start is never
    predicted. ``segment_lengths`` (per-log sample counts, summing to
    ``len(targets)``) draws each log as its own solid line so those boundaries
    are visibly broken rather than joined by a spurious connecting segment.
    """
    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)
    fig, axes = plt.subplots(2, RESIDUAL_OUTPUT_DIM, figsize=(2.6 * RESIDUAL_OUTPUT_DIM, 7))
    if segment_lengths is None or len(segment_lengths) == 0:
        segment_lengths = [len(targets)]
    boundaries = np.cumsum(segment_lengths)[:-1]
    for channel in range(RESIDUAL_OUTPUT_DIM):
        ax = axes[0, channel]
        start = 0
        for seg_index, seg_len in enumerate(segment_lengths):
            stop = start + seg_len
            index = np.arange(start, stop)
            first = seg_index == 0
            ax.plot(index, targets[start:stop, channel], color="C0", lw=0.8,
                    label="target" if first else None)
            ax.plot(index, predictions[start:stop, channel], color="C1", lw=0.8, alpha=0.8,
                    label="predicted" if first else None)
            start = stop
        for boundary in boundaries:
            ax.axvline(boundary - 0.5, color="0.6", lw=0.6, ls=":")
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
    fig, axes = plt.subplots(RESIDUAL_OUTPUT_DIM, 1, figsize=(10, 3 * RESIDUAL_OUTPUT_DIM), sharex=True)
    axes = np.atleast_1d(axes)
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


def plot_gate_map(
    model,
    features: np.ndarray,
    *,
    out_dir: str | Path = "visualize",
    out_name: str = "residual_gate_map.pdf",
) -> Path:
    """Expert-gate map over the operating envelope.

    Left: the training operating points [v_nom, omega_nom] colored by which
    expert wins the gate (argmax over the K expert weights), with the k-means
    centers overlaid. Right: the residual *coverage* (1 - null-expert weight)
    on a grid -- where it drops to zero the residual is switched off (unseen
    regime), the out-of-distribution safety the ensemble provides.
    """
    from wmr_simulator.residual_model.residual import gate_weights

    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)
    features = np.asarray(features, dtype=float)

    weights = np.asarray(gate_weights(model, features))  # (N, K)
    winner = weights.argmax(axis=1)
    centers = np.asarray(model.input_mean) + np.asarray(model.centers) * np.asarray(model.input_std)
    num_experts = weights.shape[1]

    fig, (ax_experts, ax_cov) = plt.subplots(1, 2, figsize=(13, 5))

    scatter = ax_experts.scatter(
        features[:, 0], features[:, 1], c=winner, s=4, alpha=0.5, cmap="tab10", vmin=0, vmax=9
    )
    ax_experts.scatter(centers[:, 0], centers[:, 1], c="k", marker="x", s=80, label="centers")
    ax_experts.set_xlabel(r"$v_{nom}$ [m/s]")
    ax_experts.set_ylabel(r"$\omega_{nom}$ [rad/s]")
    ax_experts.set_title(f"Active expert ({num_experts} experts)")
    ax_experts.legend()
    ax_experts.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax_experts, label="expert index")

    # Coverage grid: 1 - null weight = sum of the expert weights.
    margin_x = 0.15 * (np.ptp(features[:, 0]) + 1e-6)
    margin_y = 0.15 * (np.ptp(features[:, 1]) + 1e-6)
    grid_v = np.linspace(features[:, 0].min() - margin_x, features[:, 0].max() + margin_x, 120)
    grid_w = np.linspace(features[:, 1].min() - margin_y, features[:, 1].max() + margin_y, 120)
    mesh_v, mesh_w = np.meshgrid(grid_v, grid_w)
    grid_features = np.column_stack([mesh_v.ravel(), mesh_w.ravel()])
    coverage = np.asarray(gate_weights(model, grid_features)).sum(axis=1).reshape(mesh_v.shape)
    contour = ax_cov.contourf(mesh_v, mesh_w, coverage, levels=np.linspace(0, 1, 11), cmap="viridis")
    ax_cov.scatter(features[:, 0], features[:, 1], c="w", s=1, alpha=0.15)
    ax_cov.scatter(centers[:, 0], centers[:, 1], c="r", marker="x", s=80)
    ax_cov.set_xlabel(r"$v_{nom}$ [m/s]")
    ax_cov.set_ylabel(r"$\omega_{nom}$ [rad/s]")
    ax_cov.set_title("Residual coverage (0 = off / OOD)")
    fig.colorbar(contour, ax=ax_cov, label="coverage")

    fig.tight_layout()
    output_path = out_dir / out_name
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


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
