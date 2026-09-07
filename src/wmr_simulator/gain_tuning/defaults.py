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
    "steps": 160,
    "learning_rate": 3e-4,
    # Noise realizations averaged into every objective evaluation. Each
    # trajectory receives independent child keys from the run's frozen bundle,
    # so even R=1 spans one noise stream per trajectory; R>1 adds independent
    # realizations within every trajectory as well.
    "num_realizations": 4,
    # Randomized rollout start pose, one draw per noise realization: uniform in a
    # disk of init_offset_radius [m] and uniform over +/-init_offset_angle [rad].
    # Starting exactly on the reference leaves ~1 cm of tracking error, so the
    # loss is nearly flat in kx/ky (measured: kx has negative curvature at the
    # optimum, ky's whole range moves the loss by ~5%, one noise sigma). The
    # defaults match how the robot is actually placed by hand across the
    # exp04/exp05 logs: 31-100 mm and up to 9.5 deg. Both 0 disables.
    "init_offset_radius": 0.05,
    "init_offset_angle": 0.2,
    # 0 = presearch off, which is the default now that the refinement optimizer
    # is BFGS. The LHS sweep exists to hand Adam a good starting basin; measured
    # on trajectory_exports/gain_optimized_current, BFGS with no presearch
    # reaches held-out 0.009392 against 0.009390 with 64 LHS points, so the
    # sweep buys nothing. Raise it again if a run looks basin-trapped.
    "num_lhs_points": 0,
    # Refinement optimizer: "bfgs" or "adam". BFGS ignores `learning_rate` (its
    # line search sets the step length) and reads `steps` as a total inner
    # budget split into restarts of 40 line-search-bounded steps. It reaches
    # kimotor = 0 by gradient, which sits on the box boundary and is out of
    # Adam's reach at any learning rate that keeps the other four stable.
    "optimizer": "bfgs",
    # Number of refinement multistarts, i.e. how many of the best presearch
    # candidates get refined. Inert in the default configuration: with the
    # presearch off there is only the one init-gain candidate to start from, so
    # `num_starts = min(this, num_candidates)` collapses to 1. It still binds
    # when the presearch is re-enabled or when `init_gains` is passed as a
    # batch. (Name predates the optimizer being selectable.)
    "num_adam_optimizations": 3,
    "validation_split": 0.2,
    # Drop a training rollout from the objective when its loss at the initial
    # gains exceeds this multiple of the median rollout loss. One diverging
    # rollout out of 32 was measured at 86% of the whole training loss, which
    # stalls the BFGS line search and makes the tuner return the stock gains
    # unchanged. Applied to training only (the held-out score keeps its
    # outliers) and frozen at the initial gains, so the objective stays smooth.
    # 0 disables.
    "outlier_loss_factor": 20.0,
    # Loss weights
    "position_tracking_weight": 1.5,
    "heading_tracking_weight": 1.0,
    "velocity_tracking_weight": 1.0,
    "input_weight": 0.0,
    "input_delta_weight": 0.0,
    "omega_delta_weight": 1.5, # Penalty on step-to-step change in the robot yaw rate, discourages gains that oscillate omega
    "gain_delta_weight": 1e-5,
    "k_min_stab": 1e-2,
    "k_max_stab": 20.0,
    "k_max_rest": 20.0,
    "gain_parametrization": None,  # None -> follow problem yaml
    # Run an independent static-gain tuning (LHS + multistart Adam, no
    # parametrization) next to the parametrized one, as a comparison baseline.
    "static_tune": True,
    "static_tune_steps": 160,
    "static_tune_learning_rate": 3e-4,
    # Narrow each run's LHS presearch to a +/- band around its init gains
    # (0 -> full [k_min_stab, k_max_stab] range); refines across iterations.
    "presearch_relative_range": 0.0,
    # Warm-start the gain parametrization from the problem's gain_parametrization
    # (e.g. the previous iteration's trained schedule) instead of the identity.
    "warm_start_schedule": True,
}
