"""Figures for one alternating gain/trajectory tuning run."""

import os

import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.visualization.animation import create_gif_from_png_frames
from wmr_simulator.visualization.trajectories import plot_trajectory_set


GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor")


def _plot_gain_lines(ax, xs, gains, gain_names=GAIN_NAMES, marker=None):
    """One line per gain vs ``xs``, on a shared ``ax``.

    Linear y-axis, not log: the gains span less than a decade once the block
    is solved, and log turns kimotor's decay towards 0 into a dive off the
    bottom that dominates the panel and hides what the other four gains do.
    Shared between the joint-tuning history plot and the cross-iteration
    pipeline-progress plot (visualization.pipeline_progress), which have
    different x-axes (rounds vs. iterations) but the same gain-panel styling.
    """
    for index, name in enumerate(gain_names):
        ax.plot(xs, gains[:, index], linewidth=1.2, label=name, marker=marker)
    ax.set_ylabel("gain")
    ax.legend(fontsize=7, ncol=2)


def plot_joint_tuning_history(history: dict, out_prefix: str = "joint_tuning_history", out_path=None):
    """Four panels: the two block losses, the gains, and the constraint state.

    The two losses share an x-axis but never a y-axis: they are different
    objectives in different units and the loop never sums them.
    """
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    gain_loss = np.asarray(history["gain_loss_pre"], dtype=float)
    trajectory_loss = np.asarray(history["trajectory_loss_pre"], dtype=float)
    gains = np.asarray(history["gains"], dtype=float)
    fim = np.asarray(history["fim_loss"], dtype=float)
    violation = np.asarray(history["max_fractional_violation"], dtype=float)
    rounds = np.arange(len(gain_loss))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # Both gain curves, because the gap between them *is* the moving-target
    # effect: training is scored on the trajectories being optimized, which
    # change every round, so only validation is comparable across rounds and
    # only validation selects the gains that ship.
    axes[0, 0].plot(rounds, gain_loss, color="C0", linewidth=1.2, alpha=0.5, label="training")
    if "gain_validation_loss" in history:
        validation_loss = np.asarray(history["gain_validation_loss"], dtype=float)
        axes[0, 0].plot(rounds, validation_loss, color="C1", linewidth=1.4, label="validation")
        best_round = history.get("best_gain_round")
        if best_round is not None:
            axes[0, 0].axvline(best_round, color="C3", linestyle="--", linewidth=1.0, label="best (shipped)")
    axes[0, 0].set_title("Gain-tuning loss (warm-start rounds are blank)")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(rounds, trajectory_loss, color="C1", linewidth=1.2)
    axes[0, 1].set_title("Trajectory objective  log(A-opt) + penalty")
    axes[0, 1].set_ylabel("objective")

    _plot_gain_lines(axes[1, 0], rounds, gains)
    axes[1, 0].set_title("Controller gains")

    axes[1, 1].plot(rounds, np.mean(fim, axis=1), color="C2", linewidth=1.2, label="mean A-optimality")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("trace(FIM^-1)")
    violation_axis = axes[1, 1].twinx()
    violation_axis.plot(
        rounds, np.max(violation, axis=1), color="C3", linewidth=1.0, label="max violation"
    )
    violation_axis.set_ylabel("max fractional violation")
    axes[1, 1].set_title("Information and constraint state")

    for axis in axes.ravel():
        axis.set_xlabel("Round")
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Joint tuning history PDF saved at: {output_filename}")
    return output_filename


def plot_joint_tuning_trajectories(
    result,
    out_prefix: str = "joint_tuning_trajectories",
    out_path=None,
    reference_states=None,
    realizations=None,
    start_offsets=None,
    title: str = "Jointly Optimized Trajectories",
):
    """The final trajectories, rolled out under the final gains and the final
    start offsets -- i.e. under the conditions the design was scored on, not
    under the stock controller.

    ``reference_states`` / ``realizations`` / ``start_offsets`` override the
    run's own, which is how a held-out panel is drawn with the same gains and
    the same code. ``start_offsets`` is (T, R, 3), one bundle per trajectory:
    each trajectory designs its own starts."""
    pipeline = result.trajectory_pipeline
    realizations = result.realizations if realizations is None else realizations
    reference_batch = result.reference_states if reference_states is None else reference_states
    start_offsets = result.start_offsets if start_offsets is None else start_offsets

    closed_loop_poses = make_closed_loop_rollout(pipeline)(
        reference_batch,
        result.gains,
        realizations.robot_keys,
        realizations.estimator_keys,
        start_offsets,
    )
    return plot_trajectory_set(
        reference_batch,
        closed_loop_poses,
        out_prefix=out_prefix,
        out_path=out_path,
        title=title,
    )


def make_closed_loop_rollout(pipeline):
    """One jitted ``(references, gains, robot_keys, estimator_keys, offsets) ->
    (T, R, N, 3)`` rollout batch, built once and reused.

    ``gains`` is an argument rather than a closure so the animation can reuse a
    single compiled function across every frame: the gains change each round but
    the shapes do not, and re-jitting per frame costs more than the rollouts.
    """
    import jax

    def rollout(reference_states, gains, robot_key, estimator_key, start_offset):
        return pipeline.simulation.run_closed_loop(
            pipeline.nominal_physical_params(),
            controller_gains=gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            wheel_speed_log_source="estimated",
            reference_states=reference_states,
            initial_pose=reference_states[0, :3] + start_offset,
        ).pose.true_states

    return jax.jit(
        jax.vmap(
            jax.vmap(rollout, in_axes=(None, None, 0, 0, 0)),
            in_axes=(0, None, None, None, 0),
        )
    )


def _frame_annotation(snapshot) -> str:
    """Round number, that round's gains, and the held-out score behind them.

    The validation loss is on the frame because "tracking improves" is a claim
    about the *held-out* set: the trajectories in the picture are the training
    set and they are being optimized to be hard, so the picture alone cannot
    make that case.
    """
    values = [float(value) for value in snapshot.gains]
    loss = float(snapshot.validation_loss)
    loss_text = "warm start (no gain step)" if not np.isfinite(loss) else f"{loss:.6f}"
    # Two short gain lines rather than one long one: a 60-character monospace
    # line is most of the figure's width.
    return "\n".join(
        (
            f"round {snapshot.round_index}",
            f"validation loss {loss_text}",
            "  ".join(f"{name}={value:.3g}" for name, value in zip(GAIN_NAMES[:3], values[:3])),
            "  ".join(f"{name}={value:.3g}" for name, value in zip(GAIN_NAMES[3:], values[3:])),
        )
    )


def save_joint_tuning_trajectory_trace(
    result,
    out_prefix: str = "joint_tuning_rounds",
    frames_dir: str | None = None,
    gif_path: str | None = None,
    frame_duration: float = 0.2,
    stop_duration: float = 1.5,
):
    """Animate the per-round trace: one frame per recorded round, each the same
    figure :func:`plot_joint_tuning_trajectories` draws -- every trajectory's
    reference plus one closed-loop rollout per realization from its own offset
    start, with the start-pose arrows -- but under *that round's* gains,
    control points and offsets.

    Needs ``run_joint_tuning(..., trajectory_trace_stride=N)``; the trace is
    opt-in and empty otherwise. The axes are pinned to the union of every
    frame's extent so the curves morph instead of the view zooming around them.
    """
    import jax

    snapshots = getattr(result, "trajectory_trace", ())
    if not snapshots:
        raise ValueError(
            "No per-round trace on this result. Run run_joint_tuning with "
            "trajectory_trace_stride > 0."
        )

    os.makedirs("visualize", exist_ok=True)
    if frames_dir is None:
        frames_dir = os.path.join("visualize", f"{out_prefix}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    pipeline = result.trajectory_pipeline
    realizations = result.realizations
    rollout_batch = make_closed_loop_rollout(pipeline)
    reference_from_control_points = jax.jit(
        jax.vmap(pipeline.reference_states_from_control_points)
    )

    print(f"Rendering {len(snapshots)} joint-tuning round frames into {frames_dir}")
    frames = []
    for snapshot in snapshots:
        reference_states = reference_from_control_points(
            jax.numpy.asarray(snapshot.control_points, dtype=jax.numpy.float32)
        )
        closed_loop_poses = rollout_batch(
            reference_states,
            jax.numpy.asarray(snapshot.gains, dtype=jax.numpy.float32),
            realizations.robot_keys,
            realizations.estimator_keys,
            jax.numpy.asarray(snapshot.start_offsets, dtype=jax.numpy.float32),
        )
        frames.append(
            (
                snapshot,
                np.asarray(reference_states, dtype=float),
                np.asarray(closed_loop_poses, dtype=float),
            )
        )

    # One view for the whole animation, over references and rollouts alike: a
    # frame that autoscales makes a stationary curve look like it moved.
    all_x = np.concatenate(
        [reference[..., 0].ravel() for _, reference, _ in frames]
        + [poses[..., 0].ravel() for _, _, poses in frames]
    )
    all_y = np.concatenate(
        [reference[..., 1].ravel() for _, reference, _ in frames]
        + [poses[..., 1].ravel() for _, _, poses in frames]
    )
    x_margin = max(0.05 * float(np.ptp(all_x)), 0.05)
    y_margin = max(0.05 * float(np.ptp(all_y)), 0.05)
    axis_limits = (
        (float(all_x.min()) - x_margin, float(all_x.max()) + x_margin),
        (float(all_y.min()) - y_margin, float(all_y.max()) + y_margin),
    )

    for snapshot, reference_states, closed_loop_poses in frames:
        plot_trajectory_set(
            reference_states,
            closed_loop_poses,
            out_path=os.path.join(frames_dir, f"frame_{snapshot.round_index:05d}.png"),
            title="Joint Tuning",
            axis_limits=axis_limits,
            annotation=_frame_annotation(snapshot),
            verbose=False,
        )

    if gif_path is None:
        gif_path = os.path.join("visualize", f"{out_prefix}.gif")
    return create_gif_from_png_frames(
        frames_dir=frames_dir,
        frame_time=frame_duration,
        stop_time=stop_duration,
        gif_path=gif_path,
    )
