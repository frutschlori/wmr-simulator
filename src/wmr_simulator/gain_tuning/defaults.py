"""Default gain-tuning hyperparameters shared by the standalone tuning script
and the active-learning pipeline.

``scripts/run_gain_tuning.py`` uses these as its CLI argument defaults, and the
active-learning ``tune-gains`` stage can opt into them wholesale (see the
``use_standalone_gain_tuning_defaults`` experiment flag) instead of the
experiment's own ``gain_tuning`` block. Keeping them here means the two entry
points never drift and there is nothing to port between them.

The keys mirror the ``gain_tuning`` block of the experiment config; structural
inputs the pipeline owns (seed, trajectory directory, residual model) are not
part of this set.
"""

from __future__ import annotations

GAIN_TUNING_DEFAULTS: dict = {
    "steps": 500,
    "learning_rate": 1e-4,
    "num_realizations": 1,
    "num_lhs_points": 500,
    "num_adam_optimizations": 3,
    "validation_split": 0.2,
    "velocity_tracking_weight": 0.5,
    "input_weight": 0.0,
    "input_delta_weight": 1.0,
    "gain_delta_weight": 0.0,
    "k_min_stab": 1e-3,
    "k_max_stab": 50.0,
    "k_max_rest": 20.0,
    "gain_parametrization": None,  # None -> follow problem yaml
    "static_pretune": True,
    "static_pretune_steps": 500,
    "static_pretune_learning_rate": 1e-4,
    # Narrow the static LHS presearch to a +/- band around the problem's current
    # gains (0 -> full [k_min_stab, k_max_stab] range); refines across iterations.
    "presearch_relative_range": 0.0,
    # Warm-start the gain parametrization from the problem's gain_parametrization
    # (e.g. the previous iteration's trained schedule) instead of the identity.
    "warm_start_schedule": True,
}
