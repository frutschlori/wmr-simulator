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
    "outlier_loss_factor": 10.0,
    # Loss weights. Set 2026-09-09 by scoring the sim's loss terms against what
    # the MuJoCo plant actually does: 267 gain vectors driven on circle_fast
    # (15 runs each) against their unweighted loss components on three
    # iterations' tuning sets. Rank correlation of each term with plant pose
    # RMSE -- position +0.17, heading +0.23, linear velocity +0.56, angular
    # velocity +0.49, input +0.42, omega_delta +0.14 -- i.e. the previous
    # weighting spent its three largest weights on its three *worst*
    # predictors. Re-tuning six archived iterations and benchmarking the result
    # on the plant, median pose RMSE / rollouts over 0.3 m out of 90:
    # old weights 0.082 / 10, these 0.063 / 1.
    "position_tracking_weight": 1.0,
    # Measured, not an oversight: heading tracking on the designed tuning set is
    # a near-zero predictor of plant behaviour (+0.05 and +0.12 on two of the
    # three iterations). Raising it back is what pushes kth up, and the plant
    # wants kth <= 8. Do not restore it to 1.0 without re-measuring.
    "heading_tracking_weight": 0.2,
    # Velocity tracking is split by channel: [v, omega] are each normalized by
    # their own limit (v_max, omega_max) before being squared, so these two are
    # relative weights on commensurate errors. Separate because the linear and
    # angular channels are tracked by different gains (kx against ky/kth and
    # the inner loop) and because omega_max scales the angular channel. These
    # two are the only terms that stay informative about the plant on *every*
    # iteration's design, which is why they now carry the objective.
    "linear_velocity_tracking_weight": 1.0,
    "angular_velocity_tracking_weight": 1.0,
    # Small but nonzero: duty magnitude correlates +0.42 with plant pose RMSE
    # (saturation is what the aggressive gains buy), and at 0 the objective has
    # nothing pricing actuator authority at all.
    "input_weight": 0.05,
    "input_delta_weight": 0.0,
    # Penalty on the step-to-step change in the robot yaw rate. It was 1.5 --
    # the largest weight in the set -- on the theory that it prices the theta
    # ringing seen on hardware. Measured, it cannot: it reads
    # `predicted_log.wheel.vel_omega`, the *true* plant twist off a PT1, which
    # is smooth by construction here. Same four gain vectors, sim vs MuJoCo
    # alpha_rms: 2.6/11.4, 5.8/16.1, 5.9/45.4, 5.1/22.2 -- the sim is 4-8x
    # smaller and, over kth 12/20/stock, completely flat where the plant
    # separates 4x. Kept small rather than 0 so it still damps the one thing it
    # can see; making it price real ringing needs phase lag in the wheel loop,
    # not a larger weight here.
    "omega_delta_weight": 0.1,
    "gain_delta_weight": 1e-4,
    "k_min_stab": 1e-3,
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
