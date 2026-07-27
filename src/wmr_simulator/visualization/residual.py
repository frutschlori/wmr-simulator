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


TWIST_ONLY_LABEL = "Residual (twist only)"
TWIST_INPUT_LABEL = "Residual (twist + input)"


def plot_open_loop_model_comparison(
    entries: list[dict],
    *,
    twist_entry: dict | None = None,
    out_prefix: str = "residual_open_loop_comparison",
    out_dir: str | Path = "visualize",
) -> list[Path]:
    """Open-loop rollouts of two residual descriptors against mocap.

    Replays each log's recorded duty cycles through the nominal model and
    through both residual variants -- the state-only descriptor
    ``[v_nom, omega_nom]`` and the (state, action) descriptor that adds the
    commanded twist -- and integrates each twist series into a trajectory.
    Open loop, so the controller is out of the picture and what is left is the
    dynamics model's own accumulated drift.

    ``entries`` is one dict per log with keys ``name``, ``measured_poses``,
    ``nominal_poses``, ``twist_only_poses``, ``twist_input_poses``; each pose
    array is (N, 3). ``measured_poses`` should be the mocap pose track
    (``dataset["pose_states"]``) so the legend RMSE is against the measurement
    itself; integrating the measured twist instead only agrees with it to ~5 mm.
    Optional ``twist_entry`` adds a body-twist time series figure for a single
    log (keys ``name``, ``time_s`` and the matching ``*_twist`` arrays, each
    (N, 3)).
    """
    plt = _plot_module()
    out_dir = _ensure_dir(out_dir)
    output_paths = []

    def rmse(poses, reference):
        return float(np.sqrt(np.mean(np.sum((poses[:, :2] - reference[:, :2]) ** 2, axis=1))))

    # Trajectory grid, one panel per log.
    columns = min(len(entries), 2)
    rows = int(np.ceil(len(entries) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 5.5 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, entry in zip(flat_axes, entries):
        measured = entry["measured_poses"]
        series = (
            (measured, "k-", 1.6, "Mocap"),
            (entry["nominal_poses"], "--", 1.2, "Nominal (no residual)"),
            (entry["twist_only_poses"], "-", 1.2, TWIST_ONLY_LABEL),
            (entry["twist_input_poses"], "-", 1.4, TWIST_INPUT_LABEL),
        )
        for poses, style, width, label in series:
            suffix = "" if label == "Mocap" else f"  ({rmse(poses, measured):.3f} m)"
            ax.plot(poses[:, 0], poses[:, 1], style, lw=width, label=f"{label}{suffix}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(entry["name"])
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    for ax in flat_axes[len(entries):]:
        ax.set_visible(False)
    fig.suptitle("Open-loop rollout vs mocap (position RMSE in legend)")
    fig.tight_layout()
    output_paths.append(out_dir / f"{out_prefix}_trajectory.pdf")
    fig.savefig(output_paths[-1])
    plt.close(fig)

    if twist_entry is None:
        return output_paths

    # Body twist time series for one log; v_y is omitted (neither model
    # produces a lateral channel -- see residual_model.residual).
    channels = (0, 2)
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 3.2 * len(channels)), sharex=True)
    time_s = twist_entry["time_s"]
    for ax, channel in zip(np.atleast_1d(axes), channels):
        ax.plot(time_s, twist_entry["measured_twist"][:, channel], "k-", lw=0.8, label="Mocap")
        ax.plot(time_s, twist_entry["nominal_twist"][:, channel], "--", lw=0.8,
                label="Nominal (no residual)")
        ax.plot(time_s, twist_entry["twist_only_twist"][:, channel], "-", lw=0.8, alpha=0.8,
                label=TWIST_ONLY_LABEL)
        ax.plot(time_s, twist_entry["twist_input_twist"][:, channel], "-", lw=0.9, alpha=0.9,
                label=TWIST_INPUT_LABEL)
        ax.set_ylabel(TWIST_LABELS[channel])
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(ncol=2, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("time [s]")
    fig.suptitle(f"Body twist, open loop -- {twist_entry['name']}")
    fig.tight_layout()
    output_paths.append(out_dir / f"{out_prefix}_twist.pdf")
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

    One panel per half of the (state, action) descriptor.

    Left: the training operating points [v_nom, omega_nom] colored by which
    expert wins the gate (argmax over the K expert weights), with the k-means
    centers overlaid.

    Right: the residual *coverage* (1 - null-expert weight) -- where it drops to
    zero the residual is switched off (unseen regime), the out-of-distribution
    safety the ensemble provides. Swept over the *command offset*
    ``[v_cmd - v_nom, omega_cmd - omega_nom]`` at the mean training state, i.e.
    over how hard the wheel loop is accelerating. That is where the interesting
    OOD structure lives: the state half is densely covered inside the driven
    envelope, so a steady-state slice would be uniformly 1 and say nothing. The
    training data's own command offsets are overlaid.
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

    # Coverage grid (1 - null weight = sum of the expert weights) over the
    # command offset at the mean training state -- see the docstring.
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

    Left: XY trajectory of the reference and the simulated (true) robot pose.
    Right: forward velocity and angular rate of the robot against the
    reference's velocity profile.

    ``measured_log`` is the Pololu log whose reference the simulation tracked
    (same controller, same gains -- see
    ``residual.simulate_closed_loop_on_log_reference``). Overlaying it turns the
    figure from "does the loop track the reference" into the question that
    actually matters: does the *simulated* robot deviate from the reference the
    way the real one did. Mocap poses and the Savitzky-Golay twists are the
    measured series; the log's own reference is plotted too, since the sim
    resamples it and the two need not coincide sample for sample.

    ``nominal_sim_log`` is the same closed loop run with the residual switched
    off. It is the control the measured overlay needs: closeness to mocap only
    counts as evidence for the residual if the residual-free sim is *further*
    away, and a residual can just as well push the simulated robot past the real
    one. Position RMSE against the mocap poses goes in the legend.
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
    """`"  (0.123 m)"` -- position RMSE of a simulated run against the mocap poses.

    The two runs sample at different times (sim at its fixed pose step, mocap at
    whatever the tracker delivered), so the measured track is linearly
    interpolated onto the simulation's pose times, restricted to the window both
    cover.
    """
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
