"""Identified robot parameters across an active-learning experiment.

One 2x2 figure, a panel per physical parameter (wheel radius, wheelbase, max
wheel speed, motor time constant), each plotting that
parameter's joint fit against the iteration index. The per-log fits that went
into each joint fit are drawn as faint dots behind it, so an iteration whose
logs disagreed cannot pass for one whose logs agreed.

Logs the outlier screen *dropped* are deliberately not drawn: they did not
enter any joint fit, and a single rejected log is far enough out to compress
the panel by an order of magnitude and hide the trend the figure exists to
show. They are in the iteration's own
``visualization.identification.plot_identification_log_parameters`` figure,
which is where the screening decision belongs.

The records come from ``active_learning.identified_parameters`` and are only
rendered here.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

from wmr_simulator.active_learning.identified_parameters import PARAMETER_SPECS


def plot_identified_parameters(records, experiment_root, out_path=None, out_prefix="identified_parameters"):
    """Render the identified-parameter history from collected ``records``.

    ``records`` is the list returned by
    ``active_learning.identified_parameters.collect_identified_parameters``.
    The figure is written to ``out_path`` if given, else to
    ``<experiment_root>/visualize/<out_prefix>.pdf``.
    """
    experiment_root = Path(experiment_root)
    records = list(records)
    if not records:
        raise ValueError(f"Nothing to plot for {experiment_root}: no iteration has identification results.")

    if out_path is None:
        out_dir = experiment_root / "visualize"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{out_prefix}.pdf"
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    indices = [record["index"] for record in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    fig.suptitle(f"Identified parameters: {experiment_root.name}", fontsize=16)

    for panel, (name, label, scale) in enumerate(PARAMETER_SPECS):
        ax = axes.flat[panel]

        # Per-log spread first, so the joint fit draws on top of it.
        for record in records:
            for entry in record["per_log"]:
                if entry["excluded"]:
                    continue
                ax.plot(
                    record["index"],
                    scale * entry["params"][name],
                    ".",
                    color="tab:blue",
                    markersize=6,
                    alpha=0.45,
                    zorder=2,
                )

        values = [scale * record["params"][name] for record in records]
        ax.plot(indices, values, "-o", color="tab:green", linewidth=1.8, markersize=5, zorder=3)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if panel >= 2:
            ax.set_xlabel("iteration")

    # One x tick per iteration; a long experiment would otherwise get
    # fractional iteration numbers from the default locator.
    axes.flat[0].set_xticks(indices)

    handles = [plt.Line2D([], [], color="tab:green", marker="o", linewidth=1.8, label="joint fit")]
    if any(not entry["excluded"] for record in records for entry in record["per_log"]):
        handles.append(
            plt.Line2D([], [], color="tab:blue", marker=".", linestyle="none", alpha=0.6, label="per-log fit")
        )
    axes.flat[0].legend(handles=handles, loc="best", fontsize="small")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)
