"""Run a multi-configuration, multi-seed study of the active-learning loop.

Every (configuration, seed) of the spec runs as its own MuJoCo experiment, one
after the other, under <study-root>/<phase>/<configuration>_seed<k>/ with a
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
    args = parser.parse_args(argv)
    launch_study(args.spec, args.study_root, only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
