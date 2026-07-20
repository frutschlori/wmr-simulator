"""Plot gain/loss progression across a pipeline experiment's iterations.

Example:
    python scripts/plot_pipeline_progress.py "Pololu Data/Experiments/exp05"
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

from wmr_simulator.active_learning.experiment import Experiment
from wmr_simulator.visualization.pipeline_progress import plot_pipeline_progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Path to the experiment root (contains experiment.yaml).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: <experiment>/visualize/pipeline_progress.pdf).",
    )
    args = parser.parse_args()

    experiment = Experiment.load(args.experiment)
    out_path = plot_pipeline_progress(experiment, out_path=args.out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
