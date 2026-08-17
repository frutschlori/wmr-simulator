"""Every iteration's runs of one held-out baseline reference, in one figure.

One panel per controller variant (static gains left, gain parametrization
right), sharing axes so the two are read against each other, with one colour per
iteration and every run of that iteration drawn as a thin line of it. The
reference is the same dashed black curve in both panels -- that is the whole
point of a baseline: it did not move, so a difference between two colours is a
difference in the controller.

The legend carries each iteration's mean position RMSE against the reference,
because the eye cannot rank two overlapping bands of runs but that number can.

The records are collected by ``active_learning.baseline_runs`` and only rendered
here.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np


def _iteration_label(record, variant: str) -> str:
    runs = record.runs.get(variant, [])
    errors = [run.tracking_rmse for run in runs if run.tracking_rmse is not None]
    label = f"iteration {record.index}"
    if errors:
        label += f" (RMSE {np.mean(errors):.3f} m, {len(runs)} run{'s' if len(runs) > 1 else ''})"
    return label


def _plot_variant_panel(ax, records, variant, colors) -> None:
    # The runs themselves are thin and translucent so an overlapping band stays
    # readable, which makes them illegible as legend keys -- hence proxy handles.
    handles = []
    reference = next((rec.reference for rec in records if rec.reference is not None), None)
    if reference is not None:
        handles.append(
            ax.plot(
                reference[:, 0],
                reference[:, 1],
                color="black",
                linestyle="--",
                linewidth=1.1,
                label="reference",
                zorder=1,
            )[0]
        )

    for record in records:
        runs = record.runs.get(variant, [])
        if not runs:
            continue
        for run in runs:
            ax.plot(
                run.poses[:, 0],
                run.poses[:, 1],
                color=colors[record.index],
                linewidth=0.9,
                alpha=0.75,
                zorder=2,
            )
        handles.append(
            plt.Line2D([], [], color=colors[record.index], linewidth=1.8,
                       label=_iteration_label(record, variant))
        )
    if len(handles) == (1 if reference is not None else 0):
        ax.text(0.5, 0.5, "no runs recorded", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(handles=handles, loc="best", fontsize="small", framealpha=0.9)


def plot_baseline_runs(records, panels, out_path, shape=None):
    """Render the cross-iteration baseline comparison figure.

    ``records`` is one shape's list of
    ``active_learning.baseline_runs.BaselineIterationRuns``, ``panels`` the
    ``(variant, title)`` pairs to draw side by side (see
    ``baseline_runs.variant_panels``).
    """
    if not records:
        raise ValueError("No baseline runs to plot.")
    if not panels:
        raise ValueError("No controller variants to plot.")

    out_path = Path(out_path)
    os.makedirs(out_path.parent or ".", exist_ok=True)

    colors = {
        record.index: plt.get_cmap("tab10")(position % 10)
        for position, record in enumerate(records)
    }
    fig, axes = plt.subplots(
        1, len(panels), figsize=(6.5 * len(panels), 6.0), sharex=True, sharey=True, squeeze=False
    )
    for ax, (variant, title) in zip(axes[0], panels):
        _plot_variant_panel(ax, records, variant, colors)
        ax.set_title(title)

    title = "Baseline runs over iterations"
    fig.suptitle(f"{title}: {shape}" if shape else title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)
