"""Figures for one alternating gain/trajectory tuning run."""

import os

import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.visualization.trajectories import plot_trajectory_set


GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor")


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

    axes[0, 0].plot(rounds, gain_loss, color="C0", linewidth=1.2)
    axes[0, 0].set_title("Gain-tuning loss (warm-start rounds are blank)")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(rounds, trajectory_loss, color="C1", linewidth=1.2)
    axes[0, 1].set_title("Trajectory objective  log(A-opt) + penalty")
    axes[0, 1].set_ylabel("objective")

    for index, name in enumerate(GAIN_NAMES):
        axes[1, 0].plot(rounds, gains[:, index], linewidth=1.2, label=name)
    axes[1, 0].set_title("Controller gains")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("gain")
    axes[1, 0].legend(fontsize=7, ncol=2)

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


def plot_benchmark_comparison(
    records,
    group_by: str = "mode",
    metrics=(
        ("held_out_loss_random", "Held-out tuning loss", False),
        ("held_out_rmse_random", "Held-out pose RMSE [m]", False),
        ("plateau_ky_width", "ky plateau width (share of range)", False),
        ("a_criterion", "A-optimality  trace(FIM^-1)", True),
        ("max_violation", "Max fractional limit violation", False),
        ("s_per_round", "Seconds per round", False),
    ),
    out_prefix: str = "joint_tuning_benchmark",
    out_path=None,
    title: str = "Joint tuning benchmark",
):
    """One panel per headline metric, one bar per group, individual seeds
    overplotted as dots.

    The dots are the point: with 2-3 seeds a bar alone hides whether a 2%
    difference in means is larger than the seed spread, which for several of
    these metrics it is not.
    """
    from wmr_simulator.joint_tuning.benchmark import comparison_row

    rows = [comparison_row(record) for record in records]
    if not rows:
        raise ValueError("No records to plot.")
    groups = []
    for row in rows:
        label = str(row[group_by])
        if label not in groups:
            groups.append(label)

    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    num_metrics = len(metrics)
    num_columns = min(3, num_metrics)
    num_rows = int(np.ceil(num_metrics / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(4.2 * num_columns, 3.4 * num_rows))
    axes = np.atleast_1d(axes).ravel()

    positions = np.arange(len(groups), dtype=float)
    for axis, (field, label, log_scale) in zip(axes, metrics):
        values_per_group = [
            np.asarray(
                [row[field] for row in rows if str(row[group_by]) == group and row[field] is not None],
                dtype=float,
            )
            for group in groups
        ]
        means = [float(np.nanmean(values)) if values.size else np.nan for values in values_per_group]
        axis.bar(positions, means, width=0.6, color="C0", alpha=0.35, edgecolor="C0")
        for position, values in zip(positions, values_per_group):
            if values.size:
                axis.plot(
                    np.full(values.shape, position), values, "o", color="C1", markersize=5, zorder=3
                )
        axis.set_xticks(positions)
        axis.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
        axis.set_title(label, fontsize=9)
        axis.grid(True, axis="y", alpha=0.4)
        if log_scale:
            axis.set_yscale("log")
    for axis in axes[num_metrics:]:
        axis.axis("off")

    fig.suptitle(f"{title} — bars are group means, dots individual seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Benchmark comparison PDF saved at: {output_filename}")
    return output_filename


def plot_joint_tuning_evaluation(
    result,
    held_out_reference_states,
    held_out_realizations,
    out_prefix: str = "joint_tuning",
    out_dir: str | None = None,
):
    """The two figures a tuned run is judged on: the trajectories it was tuned
    *on* and the held-out ones it was scored on, both rolled out under the final
    gains.

    Same gains, same drawing code, different conditions -- so a gap between the
    two panels is generalization and nothing else.
    """
    training = plot_joint_tuning_trajectories(
        result,
        out_prefix=f"{out_prefix}_training_trajectories",
        out_path=None if out_dir is None else os.path.join(out_dir, f"{out_prefix}_training_trajectories.pdf"),
    )
    validation = plot_joint_tuning_trajectories(
        result,
        reference_states=held_out_reference_states,
        realizations=held_out_realizations,
        title="Held-out Trajectories under the Tuned Gains",
        out_prefix=f"{out_prefix}_validation_trajectories",
        out_path=None if out_dir is None else os.path.join(out_dir, f"{out_prefix}_validation_trajectories.pdf"),
    )
    return training, validation


def plot_joint_tuning_trajectories(
    result,
    out_prefix: str = "joint_tuning_trajectories",
    out_path=None,
    reference_states=None,
    realizations=None,
    title: str = "Jointly Optimized Trajectories",
):
    """The final trajectories, rolled out under the final gains and the final
    start offsets -- i.e. under the conditions the design was scored on, not
    under the stock controller.

    ``reference_states`` / ``realizations`` override the run's own, which is how
    the held-out panel is drawn with the same gains and the same code."""
    import jax

    pipeline = result.trajectory_pipeline
    realizations = result.realizations if realizations is None else realizations
    reference_batch = result.reference_states if reference_states is None else reference_states

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
        jax.vmap(jax.vmap(rollout, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None))
    )(
        reference_batch,
        realizations.robot_keys,
        realizations.estimator_keys,
        realizations.start_offsets,
    )
    return plot_trajectory_set(
        reference_batch,
        closed_loop_poses,
        out_prefix=out_prefix,
        out_path=out_path,
        title=title,
    )
