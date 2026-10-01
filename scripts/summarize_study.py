"""Collect a study phase's runs into summary.csv, benchmark_runs.csv and summary.md.

    python scripts/summarize_study.py "Pololu Data/thesis_ch5/phase1_tuner"
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.active_learning.study import summarize_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase_dir", help="Phase directory of a study (<study-root>/<phase>).")
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Processes scoring runs the launcher has not scored yet (default: half the CPUs).",
    )
    args = parser.parse_args(argv)
    for path in summarize_study(args.phase_dir, max_workers=args.max_workers):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
