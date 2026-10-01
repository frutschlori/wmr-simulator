"""Run a multi-configuration, multi-seed study of the active-learning loop.

Every (configuration, seed) of the spec runs as its own MuJoCo experiment, up
to --max-parallel at once under a memory guard, under <study-root>/<phase>/<configuration>_seed<k>/ with a
study_manifest.yaml (code version, dirty diff, full experiment.yaml, wall
clock). Rerunning resumes: finished runs are skipped and interrupted stages are
redone (active_learning.study).

    python scripts/run_study.py studies/thesis_ch5/phase1_tuner.yaml
    python scripts/summarize_study.py "Pololu Data/thesis_ch5/phase1_tuner"
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.active_learning.study import launch_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", help="Study spec yaml (studies/...).")
    parser.add_argument("--study-root", default="Pololu Data/thesis_ch5", help="Directory holding the phases.")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these configurations of the spec.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Runs executed side by side.")
    parser.add_argument(
        "--memory-per-run-gb", type=float, default=6.0,
        help="Peak memory budget of one run; a run starts only when MemAvailable covers it.",
    )
    parser.add_argument(
        "--memory-floor-gb", type=float, default=3.0,
        help="MemAvailable kept free; below it the youngest run is stopped and requeued.",
    )
    args = parser.parse_args(argv)
    launch_study(
        args.spec, args.study_root, only=args.only, max_parallel=args.max_parallel,
        memory_per_run_gb=args.memory_per_run_gb, memory_floor_gb=args.memory_floor_gb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
