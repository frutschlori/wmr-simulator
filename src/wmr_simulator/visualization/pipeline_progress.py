"""Cross-iteration summary of a full active-learning pipeline run.

One figure, three panels:

- a grouped bar chart with five gain groups (kx, ky, kth, kpmotor, kimotor),
  one bar per iteration at each group; gains are the controller gains deployed
  to record each iteration (see ``evaluate_pipeline_progress``);
- two loss panels laid out exactly like the gain-tuning loss-history plot
  (``visualization.identification.plot_loss_history``): panel 1 carries the
  tracking loss (left) and velocity-tracking loss (right); panel 2 carries the
  total objective (left) and input-delta loss (right). Solid = closed-loop sim
  of the recording controller, dashed = the actual recorded run, one point per
  iteration.

The per-iteration records (gains + sim/real loss terms) are computed by
``active_learning.progress.evaluate_pipeline_progress`` and only rendered here.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.gain_tuning.optimizers import _GAIN_NAMES


def _plot_metric_pair(ax, right_ax, records, left_key, right_key, left_color, right_color, left_label, right_label):
    """Plot one sim/real metric on ``ax`` (left) and another on ``right_ax``
    (twin), solid for sim and dashed for real, one point per iteration."""
    handles = []
    xs = [rec["index"] for rec in records]
    for axis, key, color, label in (
        (ax, left_key, left_color, left_label),
        (right_ax, right_key, right_color, right_label),
    ):
        sim = [rec["sim"][key] for rec in records]
        real = [rec["real"][key] for rec in records]
        handles.append(axis.plot(xs, sim, color=color, linestyle="-", linewidth=1.9, label=f"Sim {label}")[0])
        handles.append(axis.plot(xs, real, color=color, linestyle="--", linewidth=1.7, label=f"Real {label}")[0])
    return handles


def plot_pipeline_progress(records, experiment_root, out_path=None, out_prefix="pipeline_progress"):
    """Render the pipeline-progress figure from evaluated ``records``.

    ``records`` is the list returned by
    ``active_learning.progress.evaluate_pipeline_progress``. The figure is
    written to ``out_path`` if given, else to
    ``<experiment_root>/visualize/<out_prefix>.pdf``.
    """
    experiment_root = Path(experiment_root)
    gain_records = [rec for rec in records if rec["gains"] is not None]
    loss_records = [rec for rec in records if rec["sim"] is not None and rec["real"] is not None]
    if not gain_records and not loss_records:
        raise ValueError(f"Nothing to plot for {experiment_root}: no tuned gains and no recorded runs.")

    if out_path is None:
        out_dir = experiment_root / "visualize"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{out_prefix}.pdf"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Pipeline Progress: {experiment_root.name}", fontsize=16)
    ax_gains = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
    ax_track = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax_total = plt.subplot2grid((2, 2), (1, 1), fig=fig)

    # --- grouped gain bar chart (no grid) -----------------------------------
    if gain_records:
        indices = [rec["index"] for rec in gain_records]
        gains = np.vstack([rec["gains"] for rec in gain_records])
        n_iter, n_gains = gains.shape
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, n_iter))
        group_x = np.arange(n_gains)
        bar_width = 0.8 / n_iter
        for row, (index, color) in enumerate(zip(indices, colors)):
            offset = (row - (n_iter - 1) / 2.0) * bar_width
            ax_gains.bar(group_x + offset, gains[row], width=bar_width, color=color, label=f"it {index:02d}")
        ax_gains.set_xticks(group_x)
        ax_gains.set_xticklabels(list(_GAIN_NAMES))
        ax_gains.set_ylabel("gain value [-]")
        ax_gains.set_title("Controller gains over iterations")
        ax_gains.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize="small", title="iteration")
    else:
        ax_gains.text(0.5, 0.5, "no tuned gains", ha="center", va="center", transform=ax_gains.transAxes)
        ax_gains.set_title("Controller gains over iterations")

    # --- loss panels (sim solid, real dashed) --------------------------------
    if loss_records:
        loss_indices = [rec["index"] for rec in loss_records]

        # Panel 1: tracking (left, C0) + velocity tracking (right, C1).
        velocity_ax = ax_track.twinx()
        handles = _plot_metric_pair(
            ax_track, velocity_ax, loss_records,
            "tracking", "velocity_tracking", "C0", "C1", "tracking", "velocity tracking",
        )
        ax_track.set_ylabel("Tracking loss", color="C0")
        velocity_ax.set_ylabel("Velocity tracking loss", color="C1")
        ax_track.tick_params(axis="y", colors="C0")
        velocity_ax.tick_params(axis="y", colors="C1")
        ax_track.spines["left"].set_color("C0")
        velocity_ax.spines["right"].set_color("C1")
        ax_track.set_xlabel("iteration")
        ax_track.set_xticks(loss_indices)
        ax_track.set_title("Tracking loss over iterations")
        ax_track.legend(handles=handles, loc="best", fontsize="small")

        # Panel 2: total objective (left, C2) + input-delta (right, C3).
        input_delta_ax = ax_total.twinx()
        handles = _plot_metric_pair(
            ax_total, input_delta_ax, loss_records,
            "total", "input_delta", "C2", "C3", "total", "input delta",
        )
        ax_total.set_ylabel("Total objective", color="C2")
        input_delta_ax.set_ylabel("Input delta loss", color="C3")
        ax_total.tick_params(axis="y", colors="C2")
        input_delta_ax.tick_params(axis="y", colors="C3")
        ax_total.spines["left"].set_color("C2")
        input_delta_ax.spines["right"].set_color("C3")
        ax_total.set_xlabel("iteration")
        ax_total.set_xticks(loss_indices)
        ax_total.set_title("Objective over iterations")
        ax_total.legend(handles=handles, loc="best", fontsize="small")
    else:
        for ax in (ax_track, ax_total):
            ax.text(0.5, 0.5, "no recorded runs", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)
