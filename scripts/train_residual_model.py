"""Thin CLI wrapper: train the residual dynamics model from Pololu logs.

All logic lives in wmr_simulator.models.residual (see train_from_logs);
plotting lives in wmr_simulator.visualization.residual.

Example:
    python scripts/train_residual_model.py \
        --problem problems/pololu_gains.yaml \
        --log-dir "Pololu Data/Experiments/2026_07_01" \
        --out models/residual_pololu.pkl
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.models.residual import train_main

if __name__ == "__main__":
    train_main()
