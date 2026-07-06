"""Thin CLI wrapper: iterative active-learning loop (identify -> residual ->
tune gains -> next experiment), organized in resumable stages per iteration.

All logic lives in wmr_simulator.active_learning (experiment.py for the
directory layout, stages.py for the stages, cli.py for the argument parsing).

Typical usage:
    python scripts/run_active_learning.py init --experiment experiments/exp01
    python scripts/run_active_learning.py run --experiment experiments/exp01
    # ... collect robot data onto the SD card, copy logs into data/ ...
    python scripts/run_active_learning.py run --experiment experiments/exp01
    python scripts/run_active_learning.py status --experiment experiments/exp01

Set JAX_PLATFORMS=gpu to run the optimization stages on the GPU.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.active_learning.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
