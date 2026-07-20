"""Cross-iteration summary of a full active-learning pipeline run.

Reads the per-iteration ``results/gains.yaml`` files written by the
``tune-gains`` stage and plots how the five controller gains and the tuning
losses evolve across the pipeline iterations. One figure, three panels:

- a grouped bar chart with five gain groups (kx, ky, kth, kpmotor, kimotor),
  one bar per iteration at each group (log y-axis, since kx is ~1e-3 while the
  others are O(10));
- final training loss per iteration;
- final validation loss per iteration.

The loss panels mirror the gain-tuning loss-history plots but collapse each
iteration's optimization to its single converged value (one point per
iteration).
"""

import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import yaml

from wmr_simulator.gain_tuning.optimizers import _GAIN_NAMES

# Small positive floor so a disabled (0.0) gain is still visible on the log axis.
_LOG_FLOOR = 1e-4


def _load_iteration_results(experiment):
    """Return per-iteration gains/losses for iterations that were tuned.

    Skips iterations without a ``results/gains.yaml`` (e.g. a freshly recorded
    iteration whose gains have not been tuned yet).
    """
    iterations = []
    for index in experiment.iteration_indices():
        gains_path = experiment.paths(index).gains_result
        if not gains_path.is_file():
            continue
        with gains_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        gains = data.get("gains")
        if gains is None:
            continue
        iterations.append(
            {
                "index": index,
                "gains": np.asarray(gains, dtype=float),
                "final_loss": data.get("final_loss"),
                "final_validation_loss": data.get("final_validation_loss"),
            }
        )
    return iterations


def plot_pipeline_progress(experiment, out_path=None, out_prefix="pipeline_progress"):
    """Plot gain and loss progression across a pipeline's iterations.

    ``experiment`` is a loaded ``active_learning.experiment.Experiment``. The
    figure is written to ``out_path`` if given, else to
    ``<experiment root>/visualize/<out_prefix>.pdf``.
    """
    iterations = _load_iteration_results(experiment)
    if not iterations:
        raise ValueError(
            f"No tuned iterations (results/gains.yaml) found under {experiment.root}."
        )

    indices = [it["index"] for it in iterations]
    gains = np.vstack([it["gains"] for it in iterations])  # (n_iter, n_gains)
    n_iter, n_gains = gains.shape

    if out_path is None:
        out_dir = experiment.root / "visualize"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{out_prefix}.pdf"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, n_iter))

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Pipeline Progress: {experiment.root.name}", fontsize=16)
    ax_gains = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
    ax_train = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax_val = plt.subplot2grid((2, 2), (1, 1), fig=fig)

    # --- grouped gain bar chart ---------------------------------------------
    group_x = np.arange(n_gains)
    total_width = 0.8
    bar_width = total_width / n_iter
    gains_plot = np.clip(gains, _LOG_FLOOR, None)
    for row, (index, color) in enumerate(zip(indices, colors)):
        offset = (row - (n_iter - 1) / 2.0) * bar_width
        ax_gains.bar(
            group_x + offset,
            gains_plot[row],
            width=bar_width,
            color=color,
            label=f"it {index:02d}",
        )
    ax_gains.set_yscale("log")
    ax_gains.set_xticks(group_x)
    ax_gains.set_xticklabels(list(_GAIN_NAMES))
    ax_gains.set_ylabel("gain value [-] (log)")
    ax_gains.set_title("Controller gains over iterations")
    ax_gains.grid(True, axis="y", which="both", alpha=0.3)
    ax_gains.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize="small", title="iteration"
    )

    # --- loss panels ---------------------------------------------------------
    for ax, key, title, color in (
        (ax_train, "final_loss", "Final training loss", "tab:blue"),
        (ax_val, "final_validation_loss", "Final validation loss", "tab:orange"),
    ):
        xs = [it["index"] for it in iterations if it[key] is not None]
        ys = [float(it[key]) for it in iterations if it[key] is not None]
        if ys:
            ax.plot(xs, ys, marker="o", color=color, linewidth=1.2)
            if all(y > 0 for y in ys):
                ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.set_xticks(indices)
        ax.grid(True, alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)
