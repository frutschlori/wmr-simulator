"""Cross-iteration summary of a full active-learning pipeline run.

One figure: the static gains per iteration, one panel per weighted loss term
(``PLOTTED_TERMS``) on the benchmark runs (solid: simulation of the recording
controller, dashed: the recorded runs, with a band over the runs), and the yaw
ringing of the recorded runs (worst shape, with a band over the other
shapes). The records come from ``active_learning.progress``.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.active_learning.progress import TERM_NAMES
from wmr_simulator.controller import GAIN_NAMES

# Every loss term except the input energy gets a panel (2x3 grid).
PLOTTED_TERMS = tuple(name for name in TERM_NAMES if name != "input")


def _plot_gain_lines(ax, xs, gains, gain_names=GAIN_NAMES, marker=None):
    """One line per gain vs ``xs`` on ``ax``."""
    for index, name in enumerate(gain_names):
        ax.plot(xs, gains[:, index], linewidth=1.2, label=name, marker=marker)
    ax.set_ylabel("gain")
    ax.legend(fontsize=7, ncol=2)


def _plot_metric(ax, records, key, color, label):
    """Plot one sim/real loss term on ``ax`` (sim solid, recorded dashed with
    a band over the runs), one point per iteration."""
    handles = []
    xs = [rec["index"] for rec in records]
    sim = [rec["sim"][key] for rec in records]
    real = [rec["real"][key] for rec in records]
    # Markers, so a single iteration is visible.
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

    # Gain history on top, the loss terms in a 2x3 grid below.
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
        _plot_gain_lines(ax_gains, indices, gains, marker="o")
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
    ringing_records = []
    for rec in records:
        shape_medians = [
            float(np.median(scored))
            for values in (rec.get("yaw_ringing_by_shape") or {}).values()
            if (scored := [value for value in values if value is not None])
        ]
        if shape_medians:
            ringing_records.append((rec["index"], shape_medians))
    if ringing_records:
        xs = [index for index, _ in ringing_records]
        ax_ringing.plot(
            xs, [max(medians) for _, medians in ringing_records],
            color="black", linestyle="--", marker="o", markersize=3.5, linewidth=1.7,
            label="Real yaw ringing, worst shape",
        )
        ax_ringing.fill_between(
            xs, [min(medians) for _, medians in ringing_records], [max(medians) for _, medians in ringing_records],
            color="black", alpha=0.15, linewidth=0,
        )
        ax_ringing.set_xticks(xs)
        ax_ringing.legend(loc="best", fontsize="x-small")
    else:
        ax_ringing.text(0.5, 0.5, "no IMU yaw rate in the runs", ha="center", va="center", transform=ax_ringing.transAxes)
    ax_ringing.set_xlabel("iteration", fontsize="small")
    ax_ringing.set_ylabel("yaw ringing [rad/s]", fontsize="small")
    ax_ringing.tick_params(labelsize="small")
    ax_ringing.set_title("Yaw ringing (band-passed IMU yaw-rate error RMS)", fontsize="medium")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)
