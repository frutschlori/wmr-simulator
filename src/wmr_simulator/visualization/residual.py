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
    """Predicted vs target residual channels: time series (top) and scatter
    (bottom); ``segment_lengths`` (per-log sample counts) breaks the lines
    between logs."""
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


TWIST_ONLY_LABEL = "Residual (twist only)"
TWIST_INPUT_LABEL = "Residual (twist + input)"


def plot_gate_map(
    model,
    features: np.ndarray,
    *,
    out_dir: str | Path = "visualize",
    out_name: str = "residual_gate_map.pdf",
) -> Path:
    """Expert-gate map over the operating envelope.

    Left: the training points [v_nom, omega_nom] colored by the winning expert,
    with the k-means centers. Right: the residual coverage (1 - null-expert
    weight) over the command offset ``[v_cmd - v_nom, omega_cmd - omega_nom]``
    at the mean training state, with the training offsets overlaid.
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

    # Coverage grid over the command offset at the mean training state.
    offset = features[:, 2:4] - features[:, 0:2]
    mean_state = features[:, 0:2].mean(axis=0)
    margin_x = 0.25 * (np.ptp(offset[:, 0]) + 1e-6)
    margin_y = 0.25 * (np.ptp(offset[:, 1]) + 1e-6)
    grid_v = np.linspace(offset[:, 0].min() - margin_x, offset[:, 0].max() + margin_x, 120)
    grid_w = np.linspace(offset[:, 1].min() - margin_y, offset[:, 1].max() + margin_y, 120)
    mesh_v, mesh_w = np.meshgrid(grid_v, grid_w)
    grid_offset = np.column_stack([mesh_v.ravel(), mesh_w.ravel()])
    grid_features = np.hstack([np.tile(mean_state, (len(grid_offset), 1)), mean_state + grid_offset])
    coverage = np.asarray(gate_weights(model, grid_features)).sum(axis=1).reshape(mesh_v.shape)
    contour = ax_cov.contourf(mesh_v, mesh_w, coverage, levels=np.linspace(0, 1, 11), cmap="viridis")
    ax_cov.scatter(offset[:, 0], offset[:, 1], c="w", s=1, alpha=0.15)
    ax_cov.set_xlabel(r"$v_{cmd} - v_{nom}$ [m/s]")
    ax_cov.set_ylabel(r"$\omega_{cmd} - \omega_{nom}$ [rad/s]")
    ax_cov.set_title(
        f"Residual coverage vs command offset\n"
        f"(0 = off / OOD; state fixed at mean "
        f"[{mean_state[0]:.2f} m/s, {mean_state[1]:.2f} rad/s])"
    )
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
    nominal_sim_log=None,
    measured_log=None,
    out_dir: str | Path = "visualize",
) -> Path:
    """Closed-loop simulation (residual-augmented dynamics) vs its reference.

    Left: XY paths; right: forward velocity and yaw rate. ``measured_log`` (the
    recorded run of the same reference) and ``nominal_sim_log`` (the same loop
    without the residual) are overlaid, with the position RMSE against the
    mocap poses in the legend.
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
    pose_time = np.asarray(sim_log.pose.time_s, dtype=float)
    ax_xy.plot(poses[:, 0], poses[:, 1], "-", lw=1.2,
               label=f"closed loop (residual){_mocap_rmse_suffix(pose_time, poses, measured_log)}")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)

    reference_speed = np.linalg.norm(reference[:, 3:5], axis=1)
    ax_v.step(reference_time, reference_speed, where="post", color="k", ls="--", lw=1.0, label="ref $v$")
    ax_v.plot(wheel_time, vel_omega[:, 0], lw=0.9, label="sim $v_x$ (residual)")
    ax_v.set_xlabel("time [s]")
    ax_v.set_ylabel("forward velocity [m/s]")
    ax_v.grid(True, alpha=0.3)

    ax_w.step(reference_time, reference[:, 5], where="post", color="k", ls="--", lw=1.0, label=r"ref $\omega$")
    ax_w.plot(wheel_time, vel_omega[:, 1], lw=0.9, label=r"sim $\omega$ (residual)")
    ax_w.set_xlabel("time [s]")
    ax_w.set_ylabel("angular velocity [rad/s]")
    ax_w.grid(True, alpha=0.3)

    if nominal_sim_log is not None:
        nominal_poses = np.asarray(nominal_sim_log.pose.true_states, dtype=float)
        nominal_vel_omega = np.asarray(nominal_sim_log.wheel.vel_omega, dtype=float)
        nominal_time = np.asarray(nominal_sim_log.wheel.time_s, dtype=float)
        nominal_pose_time = np.asarray(nominal_sim_log.pose.time_s, dtype=float)
        ax_xy.plot(
            nominal_poses[:, 0], nominal_poses[:, 1], "-", color="C2", lw=1.2, alpha=0.85,
            label="closed loop (nominal)"
            f"{_mocap_rmse_suffix(nominal_pose_time, nominal_poses, measured_log)}",
        )
        ax_v.plot(nominal_time, nominal_vel_omega[:, 0], "-", color="C2", lw=0.9, alpha=0.85,
                  label="sim $v_x$ (nominal)")
        ax_w.plot(nominal_time, nominal_vel_omega[:, 1], "-", color="C2", lw=0.9, alpha=0.85,
                  label=r"sim $\omega$ (nominal)")

    if measured_log is not None:
        measured_poses = np.asarray(measured_log.pose.states, dtype=float)
        pose_time = np.asarray(measured_log.pose.time_s, dtype=float)
        ax_xy.plot(measured_poses[:, 0], measured_poses[:, 1], "-", color="C3", lw=1.2,
                   alpha=0.85, label="measured (mocap)")
        if measured_log.pose.twists is not None:
            measured_twist = np.asarray(measured_log.pose.twists, dtype=float)
            ax_v.plot(pose_time, measured_twist[:, 0], "-", color="C3", lw=0.9, alpha=0.85,
                      label="measured $v_x$")
            ax_w.plot(pose_time, measured_twist[:, 2], "-", color="C3", lw=0.9, alpha=0.85,
                      label=r"measured $\omega$")

    for ax in (ax_xy, ax_v, ax_w):
        ax.legend(fontsize=8)

    fig.suptitle("Closed-loop simulation with residual model vs measured log")
    fig.tight_layout()
    output_path = out_dir / f"{out_prefix}.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _mocap_rmse_suffix(sim_time: np.ndarray, sim_poses: np.ndarray, measured_log) -> str:
    """`"  (0.123 m)"` -- position RMSE of a simulated run against the mocap
    poses, interpolated onto the simulation's pose times."""
    if measured_log is None:
        return ""
    sim_time = np.asarray(sim_time, dtype=float)
    measured_time = np.asarray(measured_log.pose.time_s, dtype=float)
    measured = np.asarray(measured_log.pose.states, dtype=float)
    overlap = (sim_time >= measured_time[0]) & (sim_time <= measured_time[-1])
    if not overlap.any():
        return ""
    interpolated = np.column_stack(
        [np.interp(sim_time[overlap], measured_time, measured[:, axis]) for axis in (0, 1)]
    )
    error = np.sqrt(np.mean(np.sum((sim_poses[overlap, :2] - interpolated) ** 2, axis=1)))
    return f"  ({error:.3f} m)"


def _ensure_dir(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_module():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt
