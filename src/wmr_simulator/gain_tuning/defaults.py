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
    # Noise realizations averaged into every objective evaluation. With 1 the
    # optimizer fits a single measurement-noise draw: measured over seeds 0-3 on
    # the same 16 training trajectories, the per-draw ranking noise (~15% of the
    # loss) exceeds the spread between candidate gain vectors (~7%), and the
    # winning gains swing by orders of magnitude with the seed alone.
    "num_realizations": 8,
    # Randomized rollout start pose, one draw per noise realization: uniform in a
    # disk of init_offset_radius [m] and uniform over +/-init_offset_angle [rad].
    # Starting exactly on the reference leaves ~1 cm of tracking error, so the
    # loss is nearly flat in kx/ky (measured: kx has negative curvature at the
    # optimum, ky's whole range moves the loss by ~5%, one noise sigma). The
    # defaults match how the robot is actually placed by hand across the
    # exp04/exp05 logs: 31-100 mm and up to 9.5 deg. Both 0 disables.
    "init_offset_radius": 0.2,
    "init_offset_angle": 0.2,
    "num_lhs_points": 500,
    "num_adam_optimizations": 3,
    "validation_split": 0.2,
    "velocity_tracking_weight": 3.0,
    "input_weight": 0.0,
    "input_delta_weight": 1.0,
    # Penalty on step-to-step change in the robot yaw rate (normalized by
    # omega_max); discourages gains that oscillate omega. 0 disables.
    "omega_delta_weight": 0.0,
    "gain_delta_weight": 0.0,
    "k_min_stab": 1e-3,
    "k_max_stab": 50.0,
    "k_max_rest": 20.0,
    "gain_parametrization": None,  # None -> follow problem yaml
    # Run an independent static-gain tuning (LHS + multistart Adam, no
    # parametrization) next to the parametrized one, as a comparison baseline.
    "static_tune": True,
    "static_tune_steps": 500,
    "static_tune_learning_rate": 1e-4,
    # Narrow each run's LHS presearch to a +/- band around its init gains
    # (0 -> full [k_min_stab, k_max_stab] range); refines across iterations.
    "presearch_relative_range": 0.0,
    # Warm-start the gain parametrization from the problem's gain_parametrization
    # (e.g. the previous iteration's trained schedule) instead of the identity.
    "warm_start_schedule": True,
}
