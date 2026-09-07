"""Overlay every iteration's runs of each held-out baseline reference.

One figure per baseline shape, written into an iteration's visualize dir --
the same output the gains stage produces, regenerated from the experiment as
it stands now (e.g. after dropping further benchmark recordings into it).

Example:
    python scripts/plot_baseline_runs.py "Pololu Data/exp1"
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from pathlib import Path

from wmr_simulator.active_learning.baseline_runs import collect_baseline_runs, variant_panels
from wmr_simulator.active_learning.experiment import Experiment
from wmr_simulator.visualization.baseline_runs import plot_baseline_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Path to the experiment root (contains experiment.yaml).")
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration whose visualize dir the figures go into (default: the last one).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <experiment>/iteration_XX/visualize).",
    )
    args = parser.parse_args()

    experiment = Experiment.load(args.experiment)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        indices = experiment.iteration_indices()
        if not indices:
            print("Experiment has no iterations.")
            return 1
        index = args.iteration if args.iteration is not None else indices[-1]
        if index not in indices:
            print(f"Experiment has no iteration {index} (has {indices}).")
            return 1
        out_dir = experiment.paths(index).visualize_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_baseline_runs(experiment)
    if not collected:
        print("No baseline runs recorded in this experiment.")
        return 1
    for shape, records in collected.items():
        out_path = plot_baseline_runs(
            records, variant_panels(records), out_dir / f"baseline_runs_{shape}.pdf", shape=shape
        )
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
