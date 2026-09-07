"""Plot gain/loss progression across a pipeline experiment's iterations.

Example:
    python scripts/plot_pipeline_progress.py "Pololu Data/Experiments/exp05"
    python scripts/plot_pipeline_progress.py "Pololu Data/exp1" \
        --benchmark-dir fast_circle_static --static
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

from wmr_simulator.active_learning.experiment import Experiment
from wmr_simulator.active_learning.progress import evaluate_pipeline_progress
from wmr_simulator.visualization.pipeline_progress import plot_pipeline_progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Path to the experiment root (contains experiment.yaml).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: <experiment>/visualize/pipeline_progress.pdf).",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help=(
            "data/ subdirectory the benchmark runs are read from, for experiments whose "
            "recordings sit under a shape name of their own (e.g. fast_circle_static) "
            "instead of benchmark/ or benchmark_static/."
        ),
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help=(
            "The runs were driven on each iteration's static gains, not on its deployed "
            "controller, so simulate the sim curve with those static gains and no gain "
            "parametrization."
        ),
    )
    args = parser.parse_args()

    experiment = Experiment.load(args.experiment)
    records = evaluate_pipeline_progress(
        experiment, benchmark_dir=args.benchmark_dir, static_controller=args.static
    )
    out_path = plot_pipeline_progress(records, experiment.root, out_path=args.out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
