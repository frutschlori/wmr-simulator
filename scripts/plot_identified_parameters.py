"""Plot the identified robot parameters across a pipeline experiment's iterations.

Example:
    python scripts/plot_identified_parameters.py "Pololu Data/exp02"
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

from wmr_simulator.active_learning.experiment import Experiment
from wmr_simulator.active_learning.identified_parameters import collect_identified_parameters
from wmr_simulator.visualization.identified_parameters import plot_identified_parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Path to the experiment root (contains experiment.yaml).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: <experiment>/visualize/identified_parameters.pdf).",
    )
    args = parser.parse_args()

    experiment = Experiment.load(args.experiment)
    records = collect_identified_parameters(experiment)
    out_path = plot_identified_parameters(records, experiment.root, out_path=args.out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
