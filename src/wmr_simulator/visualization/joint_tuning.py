"""Figures for one alternating gain/trajectory tuning run."""

import os

import matplotlib.pyplot as plt
import numpy as np

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
    import jax

    pipeline = result.trajectory_pipeline
    realizations = result.realizations if realizations is None else realizations
    reference_batch = result.reference_states if reference_states is None else reference_states
    start_offsets = result.start_offsets if start_offsets is None else start_offsets

    def rollout(reference_states, robot_key, estimator_key, start_offset):
        return pipeline.simulation.run_closed_loop(
            pipeline.nominal_physical_params(),
            controller_gains=result.gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            wheel_speed_log_source="estimated",
            reference_states=reference_states,
            initial_pose=reference_states[0, :3] + start_offset,
        ).pose.true_states

    closed_loop_poses = jax.jit(
        jax.vmap(jax.vmap(rollout, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, 0))
    )(
        reference_batch,
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
