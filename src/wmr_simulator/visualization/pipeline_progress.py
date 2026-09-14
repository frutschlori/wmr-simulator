"""Cross-iteration summary of a full active-learning pipeline run.

One figure, a gain panel and a yaw-ringing panel over a 2x3 grid of loss
panels:

- the gain history (one line per gain: kx, ky, kth, kpmotor, kimotor),
  x = iteration, styled like the joint-tuning gain-history plot
  (``visualization.joint_tuning.plot_joint_tuning_history``, whose gain panel
  it shares a helper with); gains are the *static* controller gains deployed to
  record each iteration -- a gain parametrization scales them per sample, so
  its base gains are not a number worth plotting (see ``_static_gains``);
- one small panel per plotted loss term (``PLOTTED_TERMS``: every weighted term
  of the objective except the input-energy one, six of them, which is what the
  2x3 grid is sized from), evaluated on the fixed-baseline benchmark runs (one
  point = run mean). A panel each rather than pairs sharing a twin axis: the
  terms span orders of magnitude, so any two of them on one pair of axes leaves
  the smaller unreadable. Solid = closed-loop sim of the recording controller,
  dashed = the actual recorded run, one point per iteration, with a shaded band
  over the individual runs behind the recorded mean -- the runs are chained, so one of them diverging is a result rather
  than scatter an average may quietly absorb;
- the yaw ringing of the recorded runs (``baseline_runs.run_yaw_ringing``), the
  smoothness metric beside the tracking terms, as the run mean with the same
  band over the individual runs. Recorded runs only: the JAX plant cannot ring.

The per-iteration records (gains + sim/real loss terms) are computed by
``active_learning.progress.evaluate_pipeline_progress`` and only rendered here.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.active_learning.progress import TERM_NAMES
from wmr_simulator.gain_tuning.optimizers import _GAIN_NAMES
from wmr_simulator.visualization.joint_tuning import _plot_gain_lines

# Every loss term gets a panel except the input energy: it is a regularizer on
# how hard the controller drives the motors, not a measure of how the iteration
# did, and it is weighted 0 in the shipped defaults. The remaining six are what
# the 2x3 grid below is sized for.
PLOTTED_TERMS = tuple(name for name in TERM_NAMES if name != "input")


def _plot_metric(ax, records, key, color, label):
    """Plot one sim/real loss term on ``ax``, solid for sim and dashed for
    real, one point per iteration.

    The recorded curve is the mean over an iteration's benchmark runs; the band
    behind it spans the individual runs, so an iteration whose runs disagree
    cannot pass for one whose runs agree.
    """
    handles = []
    xs = [rec["index"] for rec in records]
    sim = [rec["sim"][key] for rec in records]
    real = [rec["real"][key] for rec in records]
    # marker="o": a pipeline can be a single iteration, where a bare line draws
    # nothing at all (the same reason the gain panel carries markers).
    handles.append(
        ax.plot(xs, sim, color=color, linestyle="-", marker="o", markersize=3.5, linewidth=1.9, label=f"Sim {label}")[0]
    )
    handles.append(
        ax.plot(xs, real, color=color, linestyle="--", marker="o", markersize=3.5, linewidth=1.7, label=f"Real {label}")[0]
    )
    runs = [[run[key] for run in rec["real_runs"]] for rec in records]
    if any(len(values) > 1 for values in runs):
        lower = [min(values) if values else np.nan for values in runs]
        upper = [max(values) if values else np.nan for values in runs]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.15, linewidth=0)
    return handles


def plot_pipeline_progress(records, experiment_root, out_path=None, out_prefix="pipeline_progress"):
    """Render the pipeline-progress figure from evaluated ``records``.

    ``records`` is the list returned by
    ``active_learning.progress.evaluate_pipeline_progress``. The figure is
    written to ``out_path`` if given, else to
    ``<experiment_root>/visualize/<out_prefix>.pdf``.
    """
    experiment_root = Path(experiment_root)
    records = [{**rec, "real_runs": rec.get("real_runs") or []} for rec in records]
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

    # Three rows: the gain history across the top, then the loss terms in a
    # 2x3 grid below it. The panels are small on purpose -- the terms differ by
    # orders of magnitude, so one axis each is what makes them readable at all.
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Pipeline Progress: {experiment_root.name} (baseline benchmark)", fontsize=16)
    ax_gains = plt.subplot2grid((3, 3), (0, 0), colspan=2, fig=fig)
    ax_ringing = plt.subplot2grid((3, 3), (0, 2), fig=fig)
    loss_axes = [
        plt.subplot2grid((3, 3), (1 + index // 3, index % 3), fig=fig)
        for index in range(len(PLOTTED_TERMS))
    ]

    # --- gain history line plot ---------------------------------------------
    if gain_records:
        indices = [rec["index"] for rec in gain_records]
        gains = np.vstack([rec["gains"] for rec in gain_records])
        # marker="o": unlike the joint-tuning round history (hundreds of
        # points), a pipeline has one point per iteration and can be as short
        # as a single iteration, where a bare line draws nothing at all.
        _plot_gain_lines(ax_gains, indices, gains, _GAIN_NAMES, marker="o")
        ax_gains.set_xlabel("iteration")
        ax_gains.set_xticks(indices)
        ax_gains.set_title("Static controller gains over iterations")
    else:
        ax_gains.text(0.5, 0.5, "no static gains", ha="center", va="center", transform=ax_gains.transAxes)
        ax_gains.set_title("Static controller gains over iterations")

    # --- one loss panel per weighted term (sim solid, real dashed) -----------
    if loss_records:
        loss_indices = [rec["index"] for rec in loss_records]
        for index, (ax, term) in enumerate(zip(loss_axes, PLOTTED_TERMS)):
            label = term.replace("_", " ")
            color = f"C{index}"
            handles = _plot_metric(ax, loss_records, term, color, label)
            ax.set_ylabel(f"{label} loss", color=color, fontsize="small")
            ax.tick_params(axis="y", colors=color, labelsize="small")
            ax.tick_params(axis="x", labelsize="small")
            ax.spines["left"].set_color(color)
            ax.set_xlabel("iteration", fontsize="small")
            ax.set_xticks(loss_indices)
            ax.set_title(f"{label} loss".capitalize(), fontsize="medium")
            ax.legend(handles=handles, loc="best", fontsize="x-small")
    else:
        for ax in loss_axes:
            ax.text(0.5, 0.5, "no benchmark runs", ha="center", va="center", transform=ax.transAxes)

    # --- yaw ringing of the recorded runs ------------------------------------
    ringing_records = [
        (rec["index"], [value for value in rec.get("yaw_ringing_runs") or [] if value is not None])
        for rec in records
    ]
    ringing_records = [(index, values) for index, values in ringing_records if values]
    if ringing_records:
        xs = [index for index, _ in ringing_records]
        ax_ringing.plot(
            xs, [np.mean(values) for _, values in ringing_records],
            color="black", linestyle="--", marker="o", markersize=3.5, linewidth=1.7, label="Real yaw ringing",
        )
        ax_ringing.fill_between(
            xs, [min(values) for _, values in ringing_records], [max(values) for _, values in ringing_records],
            color="black", alpha=0.15, linewidth=0,
        )
        ax_ringing.set_xticks(xs)
        ax_ringing.legend(loc="best", fontsize="x-small")
    else:
        ax_ringing.text(0.5, 0.5, "no IMU yaw rate in the runs", ha="center", va="center", transform=ax_ringing.transAxes)
    ax_ringing.set_xlabel("iteration", fontsize="small")
    ax_ringing.set_ylabel("yaw ringing [rad/s]", fontsize="small")
    ax_ringing.tick_params(labelsize="small")
    ax_ringing.set_title("Yaw ringing (high-passed IMU yaw rate RMS)", fontsize="medium")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)
