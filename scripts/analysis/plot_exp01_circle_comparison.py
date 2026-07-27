"""Compare the circle runs recorded during experiment exp01.

The experiment directories contain binary SD-card logs.  This script decodes
them into a temporary directory, overlays every direct ``TR*`` recording in
each circle directory, and writes a two-panel PDF.  The same iteration color
is used in both panels; iteration 1 is the manual-tuning baseline.

Run from the repository root:

    python scripts/analysis/plot_exp01_circle_comparison.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

# The analysis script must also run headlessly (for example from a terminal or
# automated experiment post-processing), where the macOS GUI backend aborts.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.pololu.decode_binary import decode_file
from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log


ITERATION_PATTERN = re.compile(r"iteration_(\d+)$")
GAIN_MLP_DIRECTORY_NAMES = ("with gain MLP", "with_gain_MLP")
SERIES_TITLES = ("x position", "y position", "heading", "linear velocity", "angular velocity")
SERIES_Y_LABELS = ("x [m]", "y [m]", r"$\theta$ [rad]", "v [m/s]", r"$\omega$ [rad/s]")


def _iteration_directories(experiment_dir: Path) -> list[tuple[int, Path]]:
    directories = []
    for path in experiment_dir.glob("iteration_*"):
        match = ITERATION_PATTERN.fullmatch(path.name)
        if match and path.is_dir():
            directories.append((int(match.group(1)), path))
    return sorted(directories)


def _circle_directory(iteration_dir: Path, gain_mlp: bool) -> Path | None:
    data_dir = iteration_dir / "data"
    if not gain_mlp:
        circle_dir = data_dir / "circle"
        return circle_dir if circle_dir.is_dir() else None

    for name in GAIN_MLP_DIRECTORY_NAMES:
        circle_dir = data_dir / name / "circle"
        if circle_dir.is_dir():
            return circle_dir
    return None


def _load_circle_logs(circle_dir: Path, temporary_dir: Path) -> list:
    """Decode and load the direct circle recordings in ``circle_dir``.

    Nested directories are deliberately excluded: they belong to separate
    recording sets rather than the circle repetitions for this comparison.
    """
    logs = []
    for binary_path in sorted(path for path in circle_dir.glob("TR*") if path.is_file()):
        csv_path = temporary_dir / f"{circle_dir.parent.parent.name}_{circle_dir.parent.name}_{binary_path.name}.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            decoded = decode_file(str(binary_path), str(csv_path))
        if not decoded:
            print(f"Skipping unreadable recording: {binary_path}")
            continue
        try:
            logs.append(load_pololu_traj_control_log(csv_path, clip_after_first_trajectory=True))
        except ValueError as error:
            print(f"Skipping {binary_path}: {error}")
    return logs


def _label(iteration: int) -> str:
    return "manual tune" if iteration == 1 else f"iteration {iteration}"


def _plot_trajectory_panel(ax, log_groups: list[tuple[int, list]], colors: dict[int, tuple]) -> None:
    reference = next((logs[0].reference.states for _, logs in log_groups if logs), None)
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        ax.plot(
            reference[:, 0],
            reference[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="reference",
            zorder=1,
        )

    for iteration, logs in log_groups:
        for run_index, log in enumerate(logs):
            pose = np.asarray(log.pose.states, dtype=float)
            ax.plot(
                pose[:, 0],
                pose[:, 1],
                color=colors[iteration],
                linewidth=0.85,
                alpha=0.72,
                label=_label(iteration) if run_index == 0 else None,
                zorder=2,
            )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")


def _reference_series(log_groups: list[tuple[int, list]]) -> tuple[np.ndarray, np.ndarray] | None:
    for _, logs in log_groups:
        if logs:
            return (
                np.asarray(logs[0].reference.time_s, dtype=float),
                np.asarray(logs[0].reference.states, dtype=float),
            )
    return None


def _plot_state_velocity_row(
    axes,
    log_groups: list[tuple[int, list]],
    colors: dict[int, tuple],
    series_indices: tuple[int, ...],
) -> None:
    reference = _reference_series(log_groups)
    if reference is not None:
        reference_time, reference_states = reference
        reference_values = (
            reference_states[:, 0],
            reference_states[:, 1],
            np.unwrap(reference_states[:, 2]),
            np.linalg.norm(reference_states[:, 3:5], axis=1),
            reference_states[:, 5],
        )
        for ax, values in zip(axes, (reference_values[index] for index in series_indices)):
            ax.step(reference_time, values, where="post", color="black", linestyle="--", linewidth=0.85, zorder=1)

    for iteration, logs in log_groups:
        for log in logs:
            pose_time = np.asarray(log.pose.time_s, dtype=float)
            pose = np.asarray(log.pose.states, dtype=float)
            twists = np.asarray(log.pose.twists, dtype=float)
            values = (pose[:, 0], pose[:, 1], np.unwrap(pose[:, 2]), twists[:, 0], twists[:, 2])
            for ax, series in zip(axes, (values[index] for index in series_indices)):
                ax.plot(pose_time, series, color=colors[iteration], linewidth=0.7, alpha=0.45, zorder=2)


def _plot_time_series_comparison(
    static_groups: list[tuple[int, list]],
    gain_mlp_groups: list[tuple[int, list]],
    colors: dict[int, tuple],
    output_path: Path,
    series_indices: tuple[int, ...],
) -> Path:
    """Plot selected pose/twist series with one controller type per row."""
    fig, axes = plt.subplots(2, len(series_indices), figsize=(4 * len(series_indices), 8.5), sharex="col", squeeze=False)
    _plot_state_velocity_row(axes[0], static_groups, colors, series_indices)
    _plot_state_velocity_row(axes[1], gain_mlp_groups, colors, series_indices)

    for column, series_index in enumerate(series_indices):
        title = SERIES_TITLES[series_index]
        y_label = SERIES_Y_LABELS[series_index]
        axes[0, column].set_title(title)
        for row in range(2):
            axes[row, column].set_xlabel("time [s]")
            axes[row, column].set_ylabel(y_label)
            axes[row, column].grid(True, alpha=0.35)

    legend_handles = [
        plt.Line2D([], [], color="black", linestyle="--", linewidth=0.85, label="reference"),
        *[
            plt.Line2D([], [], color=colors[iteration], linewidth=1.5, label=_label(iteration))
            for iteration in sorted(colors)
        ],
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=True)
    fig.text(0.006, 0.69, "Static Gains", rotation="vertical", va="center", ha="left", fontsize=13)
    fig.text(0.006, 0.29, "Gain MLP", rotation="vertical", va="center", ha="left", fontsize=13)
    fig.tight_layout(rect=(0.02, 0.08, 1, 1))
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def plot_comparison(experiment_dir: Path, output_path: Path, *, view: str = "trajectory") -> Path:
    """Create the static-gains / gain-MLP circle comparison PDF."""
    iteration_dirs = _iteration_directories(experiment_dir)
    groups = {False: [], True: []}

    with tempfile.TemporaryDirectory(prefix="exp01_circle_logs_") as temp_name:
        temporary_dir = Path(temp_name)
        for iteration, iteration_dir in iteration_dirs:
            for gain_mlp in (False, True):
                circle_dir = _circle_directory(iteration_dir, gain_mlp)
                if circle_dir is None:
                    continue
                logs = _load_circle_logs(circle_dir, temporary_dir)
                if logs:
                    groups[gain_mlp].append((iteration, logs))

    if not groups[False] and not groups[True]:
        raise ValueError(f"No readable circle recordings found under {experiment_dir}")

    loaded_iterations = sorted({iteration for controller_groups in groups.values() for iteration, _ in controller_groups})
    colors = {
        iteration: plt.get_cmap("tab10")(index % 10)
        for index, iteration in enumerate(loaded_iterations)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if view in {"state-velocity", "poses", "velocities"}:
        series_indices = {
            "state-velocity": (0, 1, 2, 3, 4),
            "poses": (0, 1, 2),
            "velocities": (3, 4),
        }[view]
        return _plot_time_series_comparison(groups[False], groups[True], colors, output_path, series_indices)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    _plot_trajectory_panel(axes[0], groups[False], colors)
    _plot_trajectory_panel(axes[1], groups[True], colors)
    axes[0].set_title("Static Gains")
    axes[1].set_title("Gain MLP")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("Pololu Data/Experiments/exp01"),
        help="Experiment directory containing iteration_XX folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Pololu Data/Experiments/exp01/visualize/circle_trajectory_comparison.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--view",
        choices=("trajectory", "state-velocity", "poses", "velocities"),
        default="trajectory",
        help="Plot XY trajectories, the combined 2x5 plot, poses, or velocities.",
    )
    args = parser.parse_args()
    output_path = plot_comparison(args.experiment_dir, args.output, view=args.view)
    print(f"Circle trajectory comparison saved at: {output_path}")


if __name__ == "__main__":
    main()
