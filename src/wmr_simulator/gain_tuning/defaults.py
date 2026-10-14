"""Default gain-tuning hyperparameters of the active-learning ``tune-gains``
stage (see the ``use_standalone_gain_tuning_defaults`` experiment flag).

The keys mirror the ``gain_tuning`` block of the experiment config.
"""

from __future__ import annotations

GAIN_TUNING_DEFAULTS: dict = {
    "steps": 160,
    "learning_rate": 3e-4,
    # Noise realizations per trajectory in every objective evaluation.
    "num_realizations": 4,
    # Rollout start pose offset, one per realization: uniform in a disk of
    # init_offset_radius [m] and in +/-init_offset_angle [rad] (both 0: off).
    "init_offset_radius": 0.2,
    "init_offset_angle": 0.3,
    # Latin-hypercube presearch points (0: off).
    "num_lhs_points": 0,
    # "bfgs" or "adam". BFGS ignores learning_rate and splits `steps` into
    # restarts of 40 line-search steps.
    "optimizer": "bfgs",
    # Number of presearch candidates that are refined.
    "num_adam_optimizations": 3,
    "validation_split": 0.2,
    # Drop training rollouts whose loss at the initial gains exceeds this
    # multiple of the median (0: off).
    "outlier_loss_factor": 10.0,
    # Loss weights. Position in m^2, heading as 2 - 2cos, velocities normalized
    # by v_max / omega_max, input delta in duty^2 per step.
    "position_tracking_weight": 1.0,
    "heading_tracking_weight": 1.0,
    "linear_velocity_tracking_weight": 0.25,
    "angular_velocity_tracking_weight": 1.0,
    "input_weight": 0.0,
    "input_delta_weight": 1.0,
    "omega_delta_weight": 0.0,
    "gain_delta_weight": 0.0,
    "k_min_stab": 1e-3,
    "k_max_stab": 20.0,
    "k_max_rest": 20.0,
    "gain_parametrization": None,  # None -> follow problem yaml
    # Tune a static controller next to the parametrized one.
    "static_tune": True,
    "static_tune_steps": 160,
    "static_tune_learning_rate": 3e-4,
    # Relative band of the presearch around the init gains (0: full range).
    "presearch_relative_range": 0.0,
    # Warm-start the gain schedule from the problem's (previous iteration's) one.
    "warm_start_schedule": True,
}
