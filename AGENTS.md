# wmr-simulator

JAX-based differentiable simulator for a small two-wheeled (differential-drive)
Pololu robot. Research goal: controller gain tuning via optimal design of
experiments. The master loop:

plan informative trajectory (FIM/DoE) → run on robot (SD card, manual data
transfer) → decode logs → system identification → fit residual model → plan
tuning trajectories → gain tuning in sim → export gains/trajectories to robot →
repeat until converged.

## Repo map

- `src/wmr_simulator/simulation.py` — closed-loop rollout pipeline
- `src/wmr_simulator/robot.py` — nominal kinematic model (+ burnout/traction
  limit; the limit saturates *smoothly* (`burnout.soft_clip`), because a hard
  `clip`'s derivative w.r.t. `a_slip_max` is zero-then-jump and turns the
  trajectory-optimization FIM objective into a staircase)
- `src/wmr_simulator/residual_model/` — learned residual dynamics: spectral-normed expert ensemble + OOD gate (Equinox)
- `src/wmr_simulator/identification/` — system ID, FIM
- `src/wmr_simulator/gain_tuning/`, `gain_parametrization/` — tuning objective/optimizer, controller-gain parametrizations
- `src/wmr_simulator/joint_tuning/` — alternating gain + trajectory optimization
  (`run_joint_tuning`, `scripts/run_joint_tuning.py`). Its own package because
  the existing direction is `trajectory_optimization → gain_tuning.objectives`
  and this loop also needs `gain_tuning.pipeline`. Two Adam optimizers (the
  blocks live in incompatible search spaces), built once and threaded through;
  `warm_start_rounds` is just the first N rounds with the gain block skipped, so
  the trajectory Adam state carries into the alternating phase. The two
  objectives are never summed and never compared. `start_offset_mode`
  (default `optimize`) decides
  whether the rollout start offsets are a frozen draw (`random`), a
  deterministic spread (`static`), or decision variables of the *trajectory*
  block (`optimize`, `optimize-heading`, `optimize-displacement`) — free offsets
  are smoothly squashed into the feasible set rather than clipped, since
  FIM-optimized offsets are driven to the boundary. They are per *trajectory*
  ((T, R, 3), as in the standalone batch designer and as the gain tuner reads
  them back), never one shared bundle — a shared one could not survive the
  round trip through an export directory. `mode` picks the
  interleaving: `alternating` (the scheme) or `sequential` (the baseline — the
  same two blocks and the *same per-block step counts*, run one after the
  other, so a difference between them is never a difference in budget).

  Neither block's own loss measures progress for the *pair* — each is scored
  against the other block's current iterate, so the gain loss answers a
  different question every round (measured: the gain block descends its own
  objective on 97% of steps while the alternation raises it on 84% of rounds).
  So the loop keeps a **frozen scoring set** (the trajectories as of the round
  the gain block first ran), returns the **best** iterate on it (`gains`; the
  last iterate is `final_gains`), and stops only on two-sided stagnation of that
  score (`has_stagnated`). The standalone one-sided rule read a *rising* loss as
  convergence and quit at round 150 with a −0.137 "improvement".

  **At the shipped default the trust region never engages** (measured
  2026-08-08). `pipeline.py:827` sets
  `step_radius = trust_radius / trajectory_steps_per_round`, and at
  `tspr = 1` Adam's update norm is `lr*sqrt(dim)` = `1e-3*sqrt(220)` = 0.0148,
  below the 2e-2 radius — `--trust-radius -1` reproduces the default run
  bit-for-bit. It only binds at `tspr >= 2` or a larger learning rate. So the
  paragraph below is true *for those settings*, not for the default 40:1 run.
  A corollary: **raising `--trajectory-steps-per-round` subdivides the same
  capped distance rather than travelling further** — `tspr` 3 and 5 give
  identical results to 5 s.f. The real knobs are `--trust-radius` and
  `--trajectory-learning-rate`.

  **The trust region is what makes the pair stable** (`--trust-radius`,
  negative disables; `--trust-gain-loss-increase`). Without it the loop runs
  cleanly for ~190 rounds — gain loss flat to +8%, trajectory objective
  descending monotonically, constraint violation *falling* — and is then
  destroyed by a single round: gain loss 0.0109 → 0.0502, `kth` 5.85 → 9.8, no
  recovery. Tightening `constraint_violation_tolerance` 5x only moves the event
  from round 189 to 210.

  The event is a **coupled basin jump**, not either block misbehaving: both
  objectives worsen in the same round (trajectory objective went *uphill*
  −9.7359 → −8.8878, violation spiking to 17x tolerance), which neither block
  does alone. The gain objective has two minima — `kth` ≈ 6 at loss 0.0105 and
  `kth` ≈ 8–10 at loss 0.037 — and once a trajectory move tips the gain solver
  into the second, the changed gains change the FIM, which kicks the trajectory
  block, which lands where the old gains score 4x worse. So the step is capped
  in L2 and then **judged by what it did to the gain block** at fixed gains;
  judging by the trajectory objective would not work, since that objective
  improves through the jump. This constrains the *coupling*, not either block.

  Measured with the trust region on (BFGS gain block, `g_tol` 0.005, 5/1):
  rounds 0–165 are the stable regime — gain loss 0.0100 → 0.0108 (+8%),
  trajectory objective −9.616 → −9.765 descending monotonically, violation
  0.0154 → 0.0076 *falling*, radius never binding. The cap then starts binding
  at ~round 175, the radius collapses 2e-2 → 1e-6, and `trust_stall_rounds`
  ends the run at 219 with `converged_reason = "trust region collapsed"`.
  Final gain loss 0.0115 against **0.0629 without the trust region** — and the
  trajectory objective only gives up −9.76 vs −10.26, which was never real
  progress since it came with 5.5x worse gains. Note the collapse *oscillates*
  (accept grows 1.5x, reject shrinks 0.5x), which is why the stall counter
  tracks the radius sitting at its floor and not consecutive rejections.
  `trust_gain_loss_increase` 0.02 and 0.10 give bit-identical runs: the rejected
  steps are discrete jumps well over 10%, not marginal trades.

  **Next thing to try if the hard cap proves too blunt** (not implemented): a
  **proximal / PALM-style term** on the trajectory block,
  `+ rho/2 * ||theta - theta_prev||^2` added to `trajectory_objective`, where
  `theta` is the trajectory decision vector and `theta_prev` its value at the
  start of the round. Same goal by a smooth route, with real convergence theory
  for alternating minimization behind it, and it degrades gracefully where a
  hard radius either binds or does not. It would *replace* the radius, not
  complement it — but keep the gain-loss acceptance test either way, since that
  is what encodes "stay informative, don't become untrackable". Note a proximal
  term does not move the fixed points, only the path; it prevents the *jump*
  because a large discrete move becomes expensive, which is exactly the failure
  above.

  The gain block used to also carry a periodic **presearch probe** to cross the
  `kimotor` basin, since `kimotor` sits 0.5 optimizer-space units from 0 in sqrt
  space and Adam needs ~1000 steps to make that trip. **Removed 2026-08-06**: a
  quasi-Newton step is not size-limited by a learning rate, so `--gain-solver
  bfgs` reaches `kimotor=0` by gradient in a single round and the probe has no
  reason to exist. Don't re-add it.

  **The loop does converge** (measured 2026-08-05, 700 rounds; an earlier claim
  that the trajectory block "runs away with no fixed point" was wrong and was
  based on reading `gain_loss_pre` — the *moving*-set loss — as if it were a
  progress measure. On the frozen scoring set the whole 700-round run moves
  1.31x, not the 12x that series suggested). The constraint term is a quadratic
  penalty, not a barrier, so an equilibrium exists where marginal information
  equals marginal penalty, and both blocks reach it: at `g_tol` 0.005 the
  violation settles at 0.04 and the scoring loss, gains and FIM are all flat
  from round 350 to 700.

  Every run shows one **basin transition at ~180 trajectory steps**, robust to
  learning rate, inner-step counts and `g_tol`: the FIM term rises briefly, then
  falls fast, `kth` moves 5.2 → 9.1 and `kx` drops. This is the joint
  optimization working, not an instability — a more informative design regime
  with the gains re-tuned to it. It looks like a failure only on the frozen
  scoring set, which is the round-0 design and is therefore biased toward the
  gains that suited it. **The scoring set answers "did the loop stay stable",
  never "are these gains better"**; nothing in the pipeline answers the latter
  yet, and it needs a neutral held-out trajectory set.

  Balance between the blocks is set by the *ratio* of their progress, not the
  absolute rates. `gain_steps_per_round` / `trajectory_steps_per_round` are
  inner steps per round; raising **both** equally is a no-op (5/5 reproduces 1/1
  step-for-step). Asymmetric 5/1 plus `g_tol` 0.005 is what settles the loop.

  **Validation vs training** (2026-08-05). The loop's own trajectories are its
  *training* set; the gains that ship are the best iterate on a held-out
  **validation** set (`--validation-trajectories`, default
  `trajectory_exports/validation_trajectories`, read by
  `joint_tuning/validation.py` — a separate reader from
  `load_reference_states_exports` because a validation set is deliberately
  ragged in length, which the batched reader rejects — as of 2026-08-08 all six
  files are in fact 101 samples, so the set is *not* currently ragged; keep the
  reader anyway, the constraint is intentional). Without a directory the
  fallback is the training design frozen at the first gain round: stationary,
  but biased toward the gains that suited it.

  **The set is effectively a one-trajectory benchmark** (measured 2026-08-08
  over 18 tuning runs): five of the six curves score within a few percent for
  *every* gain vector produced — `50ms_turbo` 0.0117-0.0133, the three
  `baseline_*` flat, `circle_fast` 0.0085-0.0153 — and all discrimination comes
  from `lemniscate_fast` (0.069 -> 0.55). Report the median beside the mean or
  the set reads far noisier than it is, and add harder curves before trusting
  it to separate two good gain vectors.

  **Cold start is much worse than warm start** (measured 2026-08-08, seed 0,
  10 trajectories x 5 control points x 4 realizations). Held-out validation:
  warm default **0.02717**, cold `wsr=0` **0.19649** (7.2x worse), against
  anchors of 0.51950 for stock gains and 0.03138 for the standalone BFGS gain
  tuner on the same fixed design. The warm start is load-bearing — cold start
  is also 6.3x worse than not running the joint loop at all. Mechanism: on
  straight lines the gain block's *training* loss is 0.0062 (the easiest
  tracking problem there is) while its validation loss is 0.5723 — textbook
  overfitting to an uninformative design. The first-round gains land in the
  `kth ~ 8-9` basin and never leave, because each round re-solves from the
  current gain values (cold runs end at `kth` 8.2-8.8, warm at 3.69). More
  rounds do not rescue it: 1000 rounds self-terminated at 317 on a collapsed
  trust region, still at 0.10222. Validation is monotone in how informative the
  design is when the gain block first runs: −7.52 -> 0.19649, −8.0 -> 0.10652,
  −9.55 -> 0.03331, warm −10.29 -> 0.02717.

  **The gain block reaches its conditional optimum in one round and the
  alternation never improves on it.** Warm start's best validation is round 1
  of 250; `wsr=500`'s is round 500, i.e. the first round the gain block ran.
  Every configuration then drifts slightly *up*. What the loop buys is entirely
  in how informative the trajectories are before the first gain step. The
  ~180-step basin transition is a warm-start phenomenon and does not appear at
  cold start, which descends smoothly and monotonically.

  `--save-round-GIF` / `--round-trace-stride` animate the trajectories and
  their closed-loop rollouts round by round (opt-in; `trajectory_trace_stride`
  defaults to 0 so the default run's memory profile is unchanged).
  `--warm-start-trajectories ""` requests a cold start; it used to raise
  `FileNotFoundError`, so the experiment was not expressible from the CLI.

  **Early stopping is off by default** (`--convergence-rel-tol` < 0). The rule
  is sound but no stagnation rule fits this loop: the validation loss is flat to
  5 s.f. for ~150 rounds and *then* the basin transition happens, so the rule
  stopped at round 100, ~90 rounds early. Use a fixed round budget.

  **The gain block is BFGS, via optimistix** (`joint_tuning/gain_solvers.py`).
  There is no solver choice and no `--gain-solver` flag — one solver, measured.
  The earlier
  "L-BFGS is 8x and loses at equal wall clock" finding was measured on the
  **trajectory** block and does not carry over: that block is high-dimensional
  and vmapped, so every trajectory waits for the worst line search, while the
  gain block is 5-D, where a dense inverse-Hessian approximation costs nothing.

  **Adam was removed 2026-08-06 — do not re-add it.** On the gain block with
  trajectories fixed (initial loss 0.0700): Adam 0.01532 in 52 s / 2000 steps,
  BFGS 0.01310 in 0.7 s, `cg-fr` 0.00994 in 4.5 s. Worse than slow, Adam's step
  is size-limited by its learning rate, so it cannot reach `kimotor = 0` inside
  any sane budget — that is what the deleted presearch probe existed to paper
  over. `lbfgs` and `cg-pr` went too: at a 200-step budget they reach 0.04807
  and 0.04400, barely leaving the start (Polak-Ribière's automatic restart keeps
  collapsing the search to steepest descent here). `cg-fr` was removed last: it
  reaches 0.00994, but so does BFGS measured the way the loop actually uses it —
  a fresh bounded solve every round — with 5 restarts x 40 steps giving 0.009939
  and the same gains to 5 s.f. in 3.2 s against CG's 4.5 s.

  Two integration facts that bite. One optimistix `solver.step` is one
  *line-search trial*, not one accepted update, so a round is a bounded
  `optx.minimise` and **the inner budget is not a step count**: BFGS at a budget
  of 5 does not leave the initial point *at all*. `resolve_steps_per_round`
  therefore defaults to 40 and refuses anything under 15 — shipping that footgun
  once cost a full run that returned the stock gains unchanged. **40 gain steps
  to 1 trajectory step** is the measured balance: the *ratio* is the knob
  (raising both equally is a no-op — 5/5 reproduces 1/1 step-for-step), and the
  asymmetry is what stops the trajectory block outrunning the gains. A full
  default run is **97 s** — 250 rounds budgeted, self-terminated at 209 by the
  trust region, 0.46 s/round. Second, `minimise` rebuilds the Hessian approximation each round,
  so curvature is not carried across rounds; `JointState.gain_opt_state` is a
  placeholder and only the trajectory optimizer carries Adam state.

  `warm_start_trajectories_dir` / `--warm-start-trajectories` starts the loop
  from a directory of designed trajectory pickles instead of designing its own.
  The exports carry the **control points** as well as the states
  (`reference_states_export_payload`), so the loop picks up the trajectory
  block's decision variables directly and resumes on exactly the curve the
  export left off on — nothing is reconstructed. The design's start offsets are
  adopted the same way. The directory *sets* `num_trajectories`,
  `num_realizations` and `num_control_points` (all properties of the design, not
  of this run) and forces `warm_start_rounds` to 0. Without the argument the
  loop designs from scratch, which is what `warm_start_rounds` pays for.
  Exports missing either the offsets or the control points are rejected.
- `src/wmr_simulator/planner.py` — reference trajectories for the fixed
  waypoints behind `planner:` in the problem yamls: a cvxpy min-jerk quintic
  spline that *interpolates* them, entry point `compute_reference_trajectory`
  (returns `(reference_states, traj)`). Standalone — the trajectory optimizer
  does not use it, and neither imports the other.
- `src/wmr_simulator/trajectory_optimization/` — FIM optimization over its own
  curve parametrization: control points are the decision variables, and two
  bases are selectable (`curve_parametrization`) — `bspline.py`, a clamped
  uniform cubic B-spline, and `bezier.py`, a global Bernstein curve.
  `curves.py` owns everything the two share (clamping, the pins, the
  reference-state assembly, the `CurvePlan` base and the registry);
  `parametrization.py` holds the s-curve/linear time scaling shared inside the
  subpackage.
  `objective_mode` selects the FIM design parameters (robot params vs gains);
  it says nothing about the curve. What the two modes do to the *curve* is
  carried by `pinned_positions` and `pin_start_heading`: identification pins
  **only the start** control point (the spline is clamped, so it *is* the
  curve's start; `clamp_control_points` writing it in is what zeroes its
  gradient) and holds the second control point on the start-heading ray, gain
  tuning pins nothing. The **goal control point is a free decision variable**
  (2026-08-08), bounded only by the environment-box clip every control point
  already gets — only the start pose has to be placed by hand on the real
  robot, so only it is boundary data.

  **The design point is always the static gains, and the designer always drives
  the static controller** (fixed 2026-09-06). No rollout in
  `TrajectoryOptimizationPipeline` passes `schedule_params`, so the gain
  parametrization is off everywhere in here — but the design point used to come
  straight off the problem yaml's `controller.gains`, which in active learning
  are the *parametrized* run's base gains (`finalize` writes those into
  `robot_config.yaml`, `write_iteration_problem` copies them on). Those are not
  a controller anybody runs: the network absorbs whatever scale they take, so
  they drift far from the effective gains (measured on `real02`: base
  `kth` 50 / `kpmotor` 21.5 against effective 9.25 / 0.0), and being large they
  flatten the designs — see the `dull-designs-follow-runaway-gains` memory,
  where either a high `kth` or a high `kpmotor` costs ~3x in total heading
  change on its own. Designing at them while driving the static controller
  designs for a robot that does not exist.

  `TrajectoryOptimizationPipeline(controller_gains=)` now names the design point
  explicitly (None keeps the problem yaml's, which is only the same thing when
  no parametrization has been trained against them), and both active-learning
  designers pass `_static_design_gains(paths)` — the gains of
  `robot_config_static_gains.yaml`, i.e. the same file `stage_tune_gains` warm
  starts its *static* run from and the same controller `ROBOTCFG_static.CFG`
  ships. A missing file means there is no separate static controller for the
  iteration (iteration 1, or an experiment with the parametrization off) and the
  problem's own gains already are the static ones. Constructing the pipeline
  with no explicit gains on a problem whose parametrization is *trained*
  (theta != 0 — the stock yamls ship an enabled one at theta = 0, which is the
  static controller exactly) prints a warning, since that is the flaw and it is
  otherwise silent; it is not an error, because pointing
  `run_trajectory_optimization_tuning.py` at an iteration's problem yaml is a
  legitimate thing to do. Joint tuning is unaffected —
  both of its blocks are static-gain-only already, and its design point is the
  gain iterate.

  Note what is *not* fixed by this: the gain tuner still rolls out with the
  schedule **on** (`gain_tuning/objectives.py`), so the designer and the tuner
  it feeds still see different controllers. Designing at the scheduled gains
  instead would make the design point trajectory-dependent (the factors depend
  on tracking error), which is a real change, not a wiring fix.

  **Identification replay must stay windowed** (`--window-length 50`, not
  full-sequence; 2026-08-08). The replay integrates *estimated* wheel speeds
  open loop, and the encoder noise surviving the low-pass is autocorrelated
  over `wheel_lp_tau`, so an un-windowed replay is a heading random walk:
  replay-vs-truth pose RMSE 0.0067 +/- 0.0009 m windowed against
  0.083 +/- 0.048 m un-windowed over 10 seeds (the +/-58% scatter is the
  signature). The FIM is `J^T J` and assumes i.i.d. measurement noise, so it
  reads that drift as information — un-windowed `trace(FIM^-1)` looks 205x
  "better" while the measured `wheel_radius` SNR *falls* (0.13 -> 0.08), and
  fitting `(r, L)` to an un-windowed replay drives `base_diameter` to **2x**
  its true value, absorbing drift into a physical parameter. Window resets are
  what keep the residual stationary. Note the identification *fit* defaults
  (`run_identification_pololu.py`, `active_learning/experiment.py:85`) are
  still un-windowed — same argument applies, unexamined, highest-value
  follow-up.

  **The wheel low-pass is a drift suppressor here, not a drift source.** Its
  own contribution is a *bounded* 2.5 cm path lag that closes to 0.00003 m
  final error (a group delay, not a drift); turning it off makes un-windowed
  replay **2.7x worse** (0.107 -> 0.290 m). The lag is real — an MSE-vs-shift
  sweep minimizes at ~53 ms, exactly `wheel_lp_tau` — but advancing the speeds
  by it recovers only 3%, and the firmware has the same lag, so **do not add
  phase-lead compensation to the sim replay**. Firmware timing re-verified
  2026-08-08 against `pololu-rs/firmware/src/inner_controller.rs`
  (`fc = 3 Hz -> tau = 0.05305 s`, and it logs `omega_*_lp`, so real
  `omega_*_meas` are already filtered).

  **Real logs do not replay well un-windowed either** (0.0094 m windowed vs
  0.154 m full-sequence over 8 exp04/exp05 logs) — so "it fits real data but
  not sim" is a false premise. The identification plot looks good because its
  params were *fitted to that log*; the trajectory-design summary plot replays
  at fixed nominal params. Different questions.

  There is **no way to switch the encoder low-pass off** any more (2026-08-08):
  no `--wheel-lp-tau` flag, no constructor kwarg on
  `TrajectoryOptimizationPipeline` or `run_joint_tuning`. It is read once from
  the problem yaml. `TrajectoryOptimizationPipeline.wheel_lp_tau` survives as a
  read-only mirror. (Several toy/legacy yamls still carry `wheel_lp_tau: 0.0` —
  `curve`, `empty`, `empty_noise_and_slip`, `oval`, `problem_hidden`,
  `straight` — none reachable from a current pipeline default.)

  The objective (`objectives.trajectory_objective`, 2026-08-03) is
  `fim_objective_term + constraint_loss / constraint_violation_tolerance`, not
  the raw sum. For the default A-optimality the first term is
  `log(trace(FIM^-1))`; the log makes the FIM term's gradient the *relative*
  change in the criterion, so it is scale-free with no state to carry — which
  alternating optimization needs, because the FIM's absolute scale moves
  whenever its design point (the gains) does.
  `constraint_violation_tolerance` (default 0.05) is the single scale knob: a
  `g_tol` fractional over-limit then costs a `g_tol` fractional loss of
  information. Because the objective normalizes itself, the optimizers no
  longer carry a round-0 `loss_scale` (which a legitimately negative loss
  cannot express anyway).

  `criterion` selects the design criterion (`--criterion`, default
  `a-optimality`; also `d-optimality` = `-logdet(FIM)` and `e-optimality`).
  A and E are variances and get the log; D is *already* a log-determinant and
  must not get a second one — `-logdet(FIM)` is signed, so `log` of it is NaN
  exactly when the design is informative. All criteria read off the FIM
  **factor** (`fim.py`), never an assembled FIM.

  **FIM parameter scaling is relative for four gains and a pinned constant for
  `kimotor`** (`fim_parameter_scaling`). Relative scaling -- the column of
  parameter `p` scaled by `p` itself -- makes the criterion measure information
  about a *fractional* change, which matches the log space the first four gains
  are searched in. `kimotor` is searched in sqrt space over `[0, k_max_rest]`
  because it is the one gain allowed to be exactly 0, and scaling it by its own
  value blanks its FIM column as the tuner drives it toward 0: measured at stock
  gains with `ki` swept, A-optimality went 2.1e-3 -> 4.24 and cond(FIM)
  175 -> 1.5e8 between `ki=5` and `ki=0`. A floor (`GAIN_FIM_SCALING_FLOOR`,
  `k_min_stab`) only bounds how bad it gets -- the column is still dead. So
  `kimotor` gets a *constant* scale (`pipeline.kimotor_fim_scale`,
  `--kimotor-fim-scale` on the tuning designer, default 5.0; None/negative ties
  it to the design point, which falls back to `k_max_rest` at `kimotor = 0`).

  The constant's *size* sets how much of `trace(FIM^-1)` lives in the `kimotor`
  direction, and that share is what buys **curvature**, because tight turns are
  what excite the motor loop. Scaling by the search range `k_max_rest = 20`
  instead -- the first attempt -- dropped the share to 3-5% and cost **28% of
  the designed curves' mean |κ|** over 500 steps (2.275 -> 1.645); the control
  points moved *further* and the paths got *longer*, i.e. long sweeping arcs
  instead of tight wiggles, because the budget went to `kx/ky/kth`, which prefer
  large tracking excursions. If designed trajectories ever look unexpectedly
  flat, check this share first. Note the share depends on the design *point* as
  well as the constant: at the nominal `kimotor = 5` it is ~38%, but at
  `kimotor = 0` with the same constant 5.0 it is 3.1% -- still 19x what the
  tied fallback gives (0.16%), and the setting the designs were judged good on.

  **Removing the scaling outright was tried and reverted** (2026-08-11, kept on
  branch `backup/remove-fim-scaling`). The argument for it was sound on paper --
  the five gains sit inside one order of magnitude, and unscaled columns are
  finite at `kimotor = 0`, which deletes every special case at once. Measured on
  the stock problem it also looked like a no-op: `cond(FIM)` 176 -> 177, arc
  3.663 -> 3.914 m, mean |κ| 1.640 -> 1.772. **The designs nevertheless look
  clearly worse by eye**, which is the acceptance test that counts here; the
  aggregate shape statistics did not capture it. Removing it also cost
  identification mode conditioning when `a_slip_max` is in the design
  parameters (1.2e9 -> 4.2e13, past what the float32 factor path is comfortable
  with) -- another reason `--no-fim-a-slip-max` is right there.

  **There is no `kimotor_floor`** (added and removed 2026-08-11): raising the
  *plant* gain off 0 for the design was a way to keep the tied scale usable, and
  the pinned scale solves the same problem without designing against a
  controller the robot will not run. Don't reintroduce it.

  **Removal re-tested 2026-09-06 and rejected again, this time with numbers
  that separate.** Stage settings (5 CPs, 10 trajectories, lr 0.03, 50 steps,
  optimized offsets), `real02` iteration-2 plant, total heading change per
  design, relative+pinned vs unscaled:

  | design point | relative+pinned | unscaled |
  |---|---|---|
  | stock `[4.5, 6, 12, 2.5, 5]` | 4.11 | 3.79 |
  | `real02` it02 static `[2.78, 4.79, 6.90, 2.16, 0]` | 3.47 (3.88 at 250 steps, 3.67 seed 1) | 2.34 (1.82, 2.27) |
  | `real02` it04 static `[4.12, 12.3, 17.0, 0.20, 0]` | 3.66 | 4.60 |
  | July it03 static `[1.17, 1.67, 10.5, 2.36, 0]` | 1.40 | 0.36 |
  | July it05 static `[1.10, 1.39, 10.8, 2.50, 0]` | 1.45 | 0.44 |

  A tie at stock (which is where the 2026-08-11 revert was judged, and why it
  looked like a no-op on paper), a win for unscaled only at the runaway
  `kpmotor = 0.2` point, and a 1.5-4x loss at every sane static-tuned point --
  at the July gains the unscaled designs are nearly straight lines. Relative
  scaling asks for equal *fractional* accuracy, so it keeps investing in the
  numerically small gains (`kx`, `ky`, `kpmotor`), which is what buys the
  turns; in raw units the criterion satisfies itself on the large-`kth`
  column. So the scaling stays, pinned `kimotor` constant and all. Note also
  that the July static gains give dull designs under *either* scaling on this
  designer (1.4 vs the 3.5-4 of the other points): tiny `kx`/`ky` are a dull
  design point in their own right.

  **The encoder low-pass is on here too** (2026-08-03). The pipeline used to
  zero `estimator.wheel_lp_tau` for FIM conditioning; measured, LP-on is both
  better conditioned and more informative in either mode, and the sim's filter
  timing matches the firmware exactly — see the
  `gain-fim-wheel-lp-stiffness` memory. `wheel_lp_tau=` / `--wheel-lp-tau`
  overrides it for the ablation only. One consequence for **identification**
  mode: with the filter on, a dull trajectory never excites the traction limit,
  so `a_slip_max` is near-unobservable at the initial curve (marginal
  information 5915 → 0.21) — prefer `--no-fim-a-slip-max` there unless slip is
  expected.

  Why a B-spline here (2026-08-02, replacing first a global Bezier, then a
  piecewise-quintic *interpolating* spline): the basis is **approximating**, so
  a control point is a soft handle rather than a point the curve is dragged
  through. An interpolating basis turns a control-point move `d` over knot
  spacing `h` into curvature `~d/h^2`, so the motion constraints bind after a
  few centimetres and the optimizer returns near-straight trajectories. Local
  support and a partition-of-unity basis also avoid the high-degree Bernstein
  conditioning problems the global Bezier had. C^2 continuity, the convex-hull
  property (so clipping control points into the environment box really does
  bound the curve), and clamped endpoints all come for free from the basis.
  `--num-control-points` is the stiffness knob, openly: 6-8 is the useful range,
  more gives finer detail but a stiffer, choppier objective. Limitations: the
  curve does not pass through its control points (hence the separate `planner.py`
  for waypoints that must be hit), and the basis matmuls need
  `precision=HIGHEST` or XLA's default silently costs ~2^-13 relative.

  **The global Bezier is back as a second option, and it is the tuning
  designer's default** (2026-08-10). `--curve-parametrization bezier|bspline`
  on `run_trajectory_optimization_tuning.py`; `curve_parametrization=` on
  `TrajectoryOptimizationPipeline` and `run_joint_tuning`, both still defaulting
  to `bspline`, as does the identification designer
  (`run_trajectory_optimization.py`, which has no flag). It is only the basis:
  clamping, the decision vector, the start-offset tail, the pins, the FIM and
  the constraint penalty are all basis-independent and live in
  `curves.py`, so a pipeline takes either without anything downstream noticing.
  `bspline.py` and `bezier.py` are now *only* their basis matrices.
  Both are linear maps from control points to sampled curve, so both are
  precomputed float64 matrices applied at `precision=HIGHEST` -- the Bezier is
  *not* the old per-step `comb` evaluation in float32, which is the one thing
  worth keeping from the rewrite.

  `--num-control-points` means different things per basis, so its default
  follows the parametrization (`curves.DEFAULT_CONTROL_POINT_COUNTS`): **11
  (order 10) for the Bezier, 5 for the B-spline**. A B-spline control point is a
  local handle over three knot spans; a Bezier control point is global and the
  count *is* the polynomial degree.

  Measured at the shipped defaults (10 trajectories, 500 steps, seed 0, same
  wall clock -- 8.5 it/s either way): the Bezier designs are **1.7x mean |kappa|
  and 3.0x max |kappa| over 23% less arc length** (arc 2.38 vs 3.09 m,
  mean |kappa| 36.7 vs 21.6, max 480 vs 159) -- tighter, wigglier curves, which
  is what the parametrization was brought back for. At the time both banks still
  came out as long diagonals across the environment box with a hook at one end,
  which was read as structural -- see the s-curve/curvature-constraint argument
  under "Designs being homogeneous" below, where `--time-scaling linear` is the
  knob that changes it. **Caveat: that comparison was run on the
  firmware-aligned EKF, which is now known to flatten the designs under *either*
  basis** (see the EKF section). The relative Bezier-vs-B-spline numbers should
  survive, since both bases saw the same plant, but the "character is unchanged"
  conclusion was drawn on a plant that made everything dull and is worth
  re-measuring on the reverted one.

  **The Bezier ships a stabilization term, and so does the B-spline as of
  2026-08-10** (the sentence used to read "and the B-spline does not"; see the
  tangent-floor entry below, which is the `differentiable-simulator` version of
  the same term). A global
  basis can stall the curve mid-path -- drive `|dpos/ds|` to ~0 in the interior
  while the endpoints stay put -- and heading and yaw rate are both read off the
  tangent there. `bezier.tangent_floor_loss` (`--tangent-floor-weight`, default
  1.0, 0 disables) is the mean squared *fractional* shortfall of `|dpos/ds|`
  below `1e-2`, so it is bounded in [0, 1] whatever the problem's scale and one
  weight keeps its meaning against a log-scale FIM term. It is sampled uniformly
  in `s`, **not** on the time grid: under the s-curve scaling the time samples
  cluster near `s = 0` and `s = 1`, leaving the middle of the curve -- where a
  global basis actually stalls -- barely observed. The pipeline adds it via
  `stabilization_loss_from_control_points`, which returns 0 for the B-spline, so
  that path's objective is bit-identical to before. It is *complementary* to the
  `TANGENT_FLOOR_FRACTION` guard in `curves.reference_from_derivatives`, which
  is on for both bases: the guard keeps the gradient finite at a stalled curve,
  the loss pushes the optimizer off it.

  **The B-spline stalls too, at 6+ control points, and the objective used to
  pay it to** (2026-08-10, `differentiable-simulator`; `bspline.py` here, the
  same term lives on the Bezier in `curves.py`/`bezier.py` on
  `wip/tuning-trajectory-optimization`). Three or four control points bunch into
  a ~0.05-0.16 m cluster, `|dpos/ds|` drops to 0.013 against a mean of 2.7, and
  the curve creeps for ~0.5 s and then reverses -- a near-cusp, the "straight
  line with a sharp edge" in a design plot. Measured over 30 designs: 0/10 at 5
  control points, 1/10 at 6, 1/10 at 8.

  It is **not** a numerical artifact but a genuine optimum. Blending a stalled
  control polygon back to an evenly spread one over the same route costs **7.5x
  in `trace(FIM^-1)`** (7.4e-5 -> 5.6e-4, i.e. 2.0 log units), monotonically,
  while the constraint term rises 0.047 -- 2% of what the FIM pays. Same
  "untrackable is informative" pathology the divergence gate was written for,
  leaking through a hole: that gate was position-only (`measurements[..., :2]`)
  and a stalled reference barely moves, so it sat at 0.1375 m against the 0.15 m
  tolerance and stayed open. (The gate has since been removed; the tangent-floor
  loss is now the only thing pricing this, which is another reason it has to be
  a real constraint rather than a nudge.)

  **What blinds the motion limits is the floored `dtheta_ds` denominator, not
  the fallback heading.** Measured by toggling the guard's two halves
  independently: unfloored, the constraint term is **1868**; as shipped, 0.048;
  swapping the `arctan2` fallback in or out changes it by *exactly nothing*,
  because `constraint_loss_from_reference_states` never reads `theta`. At a
  stall `nsq` is 1.8e-4 against a floor of 7.5e-4, so reported `|omega|` is 3.5
  rad/s where the geometry demands 31.8, and `alpha` -- a finite difference of
  that same damped `omega` -- inherits it: 0.90x `alpha_max` reported against
  14.3x demanded, at a 0.2 mm turn radius. The limits would have crushed the
  stall on their own.

  **Do not penalize nearly-coincident control points instead** -- measured, it
  does not separate. The tightest 3-point cluster in the whole set (0.029 m) is
  healthy; a 0.063 m one stalls. Proximity is fine if the cluster is roughly
  collinear and only kills the tangent when it doubles back. `min |dpos/ds|`
  separates cleanly by 7x (0.0134/0.0139 stalled, >= 0.099 healthy), which is
  what the term reads.

  `bspline.tangent_floor_loss` is therefore **`smooth_positive_max(g)**2` on the
  fractional shortfall, the same form as every limit in `constraints.py`, not
  the Bezier's mean** -- a stall covers ~20 of 101 samples, so a mean dilutes a
  93% shortfall to 0.11 against a 2.0 FIM reward, where the smooth max keeps the
  0.93. The pipeline divides it by the same `constraint_violation_tolerance`, so
  a fractional tangent shortfall costs what an equal fractional over-limit
  costs; a fully stalled curve lands at ~24 and every healthy design at exactly
  0 (bit-identical objective to before the term existed).

  **The floor is relative and stated against the guard's own scale**
  (`--min-tangent-fraction`, default `DEFAULT_MIN_TANGENT_FRACTION` 0.025, 0
  disables). The guard fires at `sqrt(TANGENT_FLOOR_FRACTION)` = 1% of the
  curve's `rms(|dpos/ds|)` (now named `GUARD_TANGENT_FRACTION`), so pinning the
  loss to the same rms makes the 2.5x margin a constant of the code rather than
  of the problem -- no environment size or path length can quietly let the
  optimizer reach the fallback tangent, and a test asserts the ordering.
  Absolute floors do not survive this: 0.15 was right at rms 6.3 and waves
  through a 10x-scaled stall at `min|t|` 0.143. Note `sim_time` is **not** a
  variable here at all -- `|dpos/ds|` is `B1 @ control_points`, metres per unit
  `s`, and the time scaling lives entirely in `s_dot`. The floor is
  `stop_gradient`-wrapped, or shrinking the whole curve into a shorter duller
  path would pay the term off as readily as fixing the stall. The pathology is
  also more faithfully *relative*: what breaks the reference is one stretch
  being far slower than the rest, not the curve being slow in metres.

  **A stalled design is refused at export** (`assert_curve_is_exportable` in
  `save_reference_states_pickle`, the choke point every designed export goes
  through, `active_learning/stages.py` included; `tangent_diagnostics` returns
  the same numbers without raising). Wherever the guard fired, the exported
  states carry the constant `[1, 0]` fallback heading and a ~166 deg step in one
  50 ms sample, and nothing downstream can recognize it -- the gain tuner rolls
  out against it and `pololu/reference_exporter` puts it on the robot. The batch
  path in `run_trajectory_optimization_tuning.py` pre-checks every design before
  creating the export directory, so a bad run lists all offenders instead of
  dying half-written. Baseline exports carry no control points and are not
  checked, correctly -- they are not optimizer output.

  **Rejected: replacing the `[1, 0]` fallback with the last valid tangent.** It
  is unreachable now (the loss already costs 5.6-6.8 objective units at the
  guard threshold, against ~2 available from the FIM), it would still be
  discontinuous at the threshold crossing rather than correct, and it routes a
  stalled sample's gradient onto an upstream sample instead of the current clean
  zero. Cost was never the obstacle -- a `lax.cummax` forward-fill is a few
  lines.

  **Trajectory exports now record `curve_parametrization`.** The same `(K, 2)`
  control-point array is a different curve under a different basis, so without
  it a joint-tuning warm start pointed at a Bezier design would silently resume
  as a B-spline. `load_reference_states_exports` surfaces it (None for exports
  written before the field existed) and rejects a directory mixing two bases;
  `run_joint_tuning` adopts it the same way it adopts the design's trajectory,
  realization and control-point counts.

  The optimizer is Adam (`--learning-rate`, 1e-2), with a stopping rule that
  ends a run once the loss has improved by less than `CONVERGENCE_REL_TOL` over
  the last `CONVERGENCE_WINDOW` steps, so `--opt-steps` is a safe upper bound
  rather than an exact setting. Each trajectory returns the *best* point it
  visited, not its last: Adam spends ~7% of steps going uphill. 250 steps
  reaches the same best-of-batch as 1500 did, so that is the default. Batched runs vmap a single trajectory's update
  so each keeps its own optimizer state. L-BFGS with a line search was tried and
  reverted: on this objective it costs ~8x per step (the averaged FIM makes the
  line search work harder, and under vmap every trajectory waits for the worst)
  and at equal wall clock Adam reaches a better loss.

  In gain-tuning mode the FIM is an **expectation over a shared `Realizations`
  bundle** (`gain_tuning/objectives.py`: noise key pairs + start-pose offsets,
  `num_realizations` of each, defaulting to `GAIN_TUNING_DEFAULTS`). It is drawn
  once and reused at every evaluation, and the per-realization measurements are
  stacked so the stacked factor's FIM is the sum of the per-rollout FIMs. The
  same bundle is what `optimize_controller_gains` scores gains on, so a
  trajectory designed to be informative about the gains is informative about
  them *under the conditions the tuner sees* — `joint_tuning` hands one object
  to both. Common random numbers keep the objective deterministic, and
  `plot_trajectory_batch` draws one closed-loop run per realization so the
  figure shows the conditions the design was scored under.
  Without this the rollout starts on the reference, the tracking
  errors are ~0, and since the Kanayama law multiplies kx and ky by exactly those
  errors their FIM columns nearly vanish (measured: kx marginal information 359
  with no offset vs 1.9e6 with). It costs ~1.3x per step — the extra rollouts
  vectorize almost for free. Identification mode does not do this: there the
  start is where the robot is physically placed.

  **The start offsets are optional decision variables here too**
  (2026-08-03). `start_offset_mode` / `--start-offset-mode` (gain-tuning mode
  only; identification raises) picks `random` (the frozen draw), `static`, or
  the `optimize*` modes, which append the free offsets to the decision vector
  as `[ravel(control_points), ravel(free_offsets)]` and let Adam move them
  alongside the control points — same squash and mask as the joint loop, via
  `start_offsets.py`, which owns every way an offset comes into being (random
  draw, static spread, squashed decision variable). It sits in this subpackage
  because the offsets are part of the *experiment being designed*; the gain
  tuner is downstream and either reads designed offsets off the pickles or
  calls the sampler here. In a batched run every
  trajectory gets its own offsets (they are scored, exported and plotted per
  trajectory); `clamp_decision_variables` leaves the tail alone, since the
  squash already keeps it feasible. The offsets get their own pace via
  `--offset-displacement-step-factor` / `--offset-heading-step-factor`
  (multipliers on the update, so effectively `factor * learning_rate` for those
  coordinates only; the control points always step at 1.0 and their tuned rate
  is untouched). This is not cosmetic — the free offsets are unitless
  pre-squash against a 0.2 m / 0.5 rad scale, and at factor 1 they crawl:
  measured over 3 trajectories x 60 steps, best objective 1/1 −8.98,
  10/20 −9.33, 30/60 −9.75, with the offsets only reaching the disk/angle
  boundary (where the FIM wants them) at the larger factors.

  **The offsets and the control points ship with the curve** (2026-08-03/04).
  A gain-tuning-mode export
  puts the design's `start_offsets` (R, 3) into the trajectory pickle, and every
  batch export puts its `control_points` (K, 2) there too — the sampled states
  are a lossy record of a design (they are the basis applied to the control
  points on one time grid), so anything that wants to keep *optimizing* the
  trajectory needs the decision variables themselves. Consumers read this
  payload by key and ignore what they do not know, so adding fields is safe for
  the reference exporter, the gain tuner and the identification analysis alike.
  (`reference_states_export_payload`, read back by
  `load_reference_states_exports` for the joint loop's warm start;
  `ControllerTuningPipeline` keeps its own reader rather than importing this
  subpackage, which would reverse the `trajectory_optimization → gain_tuning`
  direction — the two must stay in step on the payload format;
  identification-mode exports carry none,
  and both `run_trajectory_optimization_tuning.py` and `run_joint_tuning.py`
  write their *final* offsets, which under an optimizing `start_offset_mode`
  are decision variables). `ControllerTuningPipeline`
  reads them back per trajectory (T, R, 3) and the tuner rolls out on exactly
  those starts instead of drawing its own — so `--num-realizations` is then set
  by the design, not by the CLI, and a trajectory is tuned on under the
  conditions it was designed to be informative under. All-or-none per
  directory: a mixed set raises. Without offsets in the pickles nothing
  changes (every trajectory shares the tuner's own draw).
  The gain-tuning figures draw the whole family: `summary_gain_tuning`'s
  trajectory panel, `summary_training` and `summary_validation` show one
  rollout per realization from its own offset start, each with a start-pose
  arrow (`visualization/trajectories.draw_start_pose_arrow`) because a dot
  hides the heading, which is a design variable too. `plot_trajectory_set`
  no longer draws control points at all — with N trajectories x R realizations
  the figure was already full. The single-run time-series
  panels are realization 0, also started at its offset.
- `src/wmr_simulator/pololu/` — log decoding/loading, spline pose smoothing, JSN/ROBOTCFG export
- `src/wmr_simulator/active_learning/` — staged iteration CLI (`scripts/run_active_learning.py`)
- `src/wmr_simulator/mujoco_sim/` — MuJoCo stand-in for the real robot: hidden
  plant + a port of the firmware's control stack, duty in and SD-card-format
  binary logs out (`.Codex/mujoco_deployment_plan.md`, which is the state of
  record). It is the only reader of `models/pololu_hidden.yaml` /
  `models/pololu_calibrated.xml`, so the pipeline still has parameters to
  identify — and now a ground truth to be scored against.

  **An exact clock made the setpoint index truncate, and every MuJoCo log
  before 2026-09-07 carries the artifact.** `SetpointFinder`'s
  `index = int(t / dt)` floors an *exact* multiple of `dt` down by a float
  epsilon (`0.15 / 0.05 = 2.9999999999999996`), and the port's clock puts every
  outer tick on exactly such a multiple — so every ~5th tick re-issued the
  previous setpoint and the next one jumped two states: **46 repeats + 46
  double-steps per 340 outer ticks**. The reference *timestamps* stay a perfect
  50 ms, so it shows up as a reference whose content advances at 0/50/100 ms —
  a jittery staircase in `x_des`/`v_ff`. The firmware does the same f32 floor
  (`setpoint/mod.rs`) and would have the same bug on a perfect clock, but its
  `t` is *measured elapsed* time (`(Instant::now() - start).as_millis()`), which
  is always at least ~1 ms past the deadline, and +1 ms removes every mismatch.
  So this was a side effect of "the clock is exact", not a modelled deviation.
  Fixed by `TICK_EPSILON_S` (1e-9): the tick resolves *forward*, as a late one
  does on the robot.

  Measured, mid-run reference repeats (identical `x_des`/`y_des` while
  `v_ff > 0`): MuJoCo `test01`/`test02` **13.8%** (840/6100 each) against
  **0.4%** on `2026_08_19/real02` (20/5281) and **0.96%** on `2026_07_27`
  (54/5646); after the fix a MuJoCo run is 0.3%, and its apparent/commanded
  reference speed (`Δpos/Δt` over `v_ff`) goes from p5 0.00 / p95 2.27 to
  0.72 / 1.31 — the real log's numbers to 2 d.p. It was not cosmetic: over
  4 seeds on an `exp01`-style identification trajectory, tracking RMSE
  0.0319 -> 0.0270 m, max deviation 0.0867 -> 0.0634 m, peak duty
  0.822 -> 0.554, i.e. the double-jump was a real disturbance the controller
  had to chase. Anything measured on a MuJoCo log written before this date —
  benchmark RMSE, identification fits, residual targets — carries it.

  **The XML was rebuilt from scratch 2026-08-20** against the manufacturer's
  drawing (`models/3pi-plus-chassis-kit-dimensions.pdf`, 1:1 vector art) and its
  STEP assembly (`models/3pi-plus-chassis-kit/`), plus measurements on the
  actual robot. Everything before that date was a plausible stand-in whose
  chassis size, wheel width, caster size and position, ground clearance, mass,
  CoM and frictions were guesses tuned against symptoms. The XML header states
  every source and flags the numbers that are not read off the drawing.

  Highlights: 96 mm body (was 89), 12.7 mm caster ball 40.4 mm back (was 11 mm),
  171 g weighed (was 135), CoM 2.5 mm behind the axle and 15 mm up, an explicit
  low-friction nose skid for the front lip, frictions set from material pairs,
  a near-rigid floor, and CAD meshes for the rendering.

  Rules the rebuild established, each of which cost a measurement to find:

  - **The contact set is declared, never incidental.** Anything touching the
    floor needs an explicit `<pair>`. MuJoCo combines geom friction as the
    elementwise MAXIMUM, so the front contact (then the chassis rim) silently
    ran at mu 1.0, grippier than the tires. And with no `<pair>` it *averages*
    solref, so stiffening only the floor gave a wheel contact of (0.0115 1.5) —
    the mean with the wheel geom's untouched default — and barely moved the
    penetration. Same trap, opposite direction.
  - **The wheel collision cylinder is narrower than the tire (3.0 vs 7.5 mm),
    and that is physics, not a fudge.** MuJoCo reduces a cylinder on a plane to
    a two-point line contact at the rim edges, so the half-length *is* the scrub
    lever, and the real tire is crowned rather than flat-tread. Fraction of a
    commanded wheel-speed split achieved in an open-loop turn: 7 mm → 24%,
    3.8 mm → 79%, 3.0 mm → 92%, 2.2 mm → 106% (over-rotates). Note the CAD
    cannot pick the value: the sub-10 µm band is below the tessellation's own
    ±16 µm noise, and a silicone tire deforms under load anyway, so the
    geometric bracket is 3–5.5 mm and the choice is made on turn fidelity.
  - **The resting attitude is a check, not a setting.** Every 0.1 mm of caster
    drop is 0.14 deg of pitch on the 40.4 mm lever, so "rests on the caster,
    very nearly level, nose a shade up" pins the geometry tightly. A test
    asserts it, along with the mass budget and the CoM.
  - **Never set the motor from a free-running or straight-line wheel-speed
    rise.** That measurement is dominated by vehicle inertia reflected through
    the wheel and is extremely slip-sensitive; the PT1 `time_constant` the
    pipeline identifies is a *lumped* stand-in for motor dynamics plus loading
    over real driving. Like-for-like — the identification pipeline run on four
    closed-loop plant logs — the plant identifies at 0.19-0.20 s across a 1.5x
    range of stall torque, so it does not discriminate at all. An earlier
    version of this file set the stall torque from exactly that bad evidence.
  - **The plant needs 1.5x the datasheet stall torque, on a torque budget.**
    Instrumenting the actuator over the baselines at stock gains: `circle_slow`
    peaks at 15.5 mN-m, `circle_medium` 23.5, `circle_fast` 44.4, against a
    datasheet stall of 24.5. `circle_medium` sits *on* the ceiling, so every
    transient saturates — 6/6 seeds diverge at 45% duty saturation at 1.0x and
    1.2x, 0/6 at 0% saturation at 1.5x. Ruled out as causes, each measured:
    `omega_max` (230/250/262 identical), contact-patch width (7/5/3.8 mm all
    diverge at 1.0x), and tire mu (0.5/0.7/0.9, none rescues it). Why the factor
    is 1.5 is unresolved; a stall-torque measurement on the hardware settles it.
    Read that mu result narrowly — it says mu cannot fix a *torque* shortfall,
    not that mu does nothing. On the current plant mu is the sharpest knob there
    is on `circle_fast` (0.023 m / 0-of-8 at 0.9 to 0.618 m / 8-of-8 at 0.60),
    and 0.70 is what the shipped value is now set from.

  **Rotation markers are on the caster, not the wheels** (2026-08-21). The two
  red `*_wheel_marker` boxes are gone and `caster_marker` — a small red dot at
  the bottom of the ball, `contype`/`conaffinity` 0 and `mass="0"` — took their
  place. The wheels never needed one: the CAD wheel mesh has a spoked hub that
  reads as rotation on its own. The caster did, because a smooth sphere on a
  smooth floor gives the eye nothing to track, and whether it is *rolling* or
  dragging is the one thing that body exists to get right (see the skid warning
  in the XML). It sits at the contact patch, which is what a camera under the
  floor looks straight up at. It stands 0.5 mm proud because a dot fully inside
  an opaque sphere cannot be seen and a flush one vanishes at grazing angles.

  Note what this fixed on the way past: the wheel markers **had mass**. They
  declared none, so MuJoCo derived 0.0108 g each from its default density —
  0.3% of the wheel, and 0.6% of spin inertia that `_wheel_spin_inertia` does
  not know about, since it subtracts only the *collision* cylinder's share from
  the armature. Each wheel body is now exactly the declared 3.6 g with its CoM
  on the axle and symmetric inertia, and the robot exactly the weighed 171.0 g.
  The caster is untouched (1.5 g, same inertia) and the contact set is still
  the four declared geoms. A visual geom that silently weighs something is the
  general trap here — declare `mass="0"` on anything added for the camera.

  **Where the plant's limits now sit**, open loop, against the traction limit
  `mu*g` = 8.8 m/s²: a turn entered at speed tracks below ~0.5x and spins out
  above ~0.8x, and a full-duty launch from rest *jumps* rather than slips (27%
  of ticks with the wheels unloaded, 1% mean slip) — which is what the hardware
  is observed to do.

  **The rebuild's one piece of external validation was spurious, and the tire
  friction now carries it instead** (2026-09-07). The claim was: on the real
  robot the stock gains diverge on `circle_fast`, and the plant reproduces that
  6/6 while tracking `circle_slow`/`circle_medium` cleanly. The plant did — but
  *because of the setpoint truncation bug* (see the tick-epsilon entry above).
  With that fixed, `circle_fast` at stock gains tracks at median **0.021 m,
  0/6**, and re-running with the old indexing monkeypatched back in restores the
  0.891 m / 6-of-6. So the one behaviour that could falsify the model was being
  reproduced by an artifact.

  The behaviour is real, and there is more data on it than the note assumed:
  **seven real `circle_fast` runs at exactly the stock ROBOTCFG**
  (`2026_08_19/real02/iteration_01/data/benchmark_circle_fast` ×4,
  `2026_07_27/exp01/iteration_01/data/circle` ×3 — both configs are
  `kx/ky/kth = 4.5/6/12`, `kp/ki_inner = 0.01/0.02`, `wheel_max 250`) give pose
  RMSE 0.119-0.429 m, **median 0.190, 2 of 7 past 0.3 m**. So the robot is
  *marginal* there, not reliably divergent. Tire `mu` is what sets this and it
  is a sharp, well-ordered knob — see the friction block in
  `models/pololu_hidden.yaml`, now **0.70** rather than the handbook 0.9,
  which puts the plant at median 0.169 / 3-of-8. That value is the one thing in
  the model fitted to behaviour rather than to a drawing, it is called out as
  such in the yaml, and it does not close the whole gap (`lemniscate_fast` is
  still ~4x too good).

  The general lesson stands and is now doubly earned: a hardware behaviour the
  plant either reproduces or does not is the only thing that can falsify this
  model — *and* it is worth checking that the plant reproduces it for the right
  reason. Get more of these; every other agreement above is a match to a drawing
  or a static measurement, not to how the robot *drives*. `scripts/convert_chassis_mesh.py` regenerates the render
  meshes from the STEP; it needs `cadquery-ocp` and `trimesh`, deliberately not
  project dependencies, and the meshes are committed.

    Wired into the loop
  as the `simulate-deployment` stage: `run --simulate-deployment` fills the
  waiting iteration's `data/`, and `init --mujoco --iterations N` lets `run`
  carry an experiment through N iterations unattended. The runs are chained
  repeats of the *bridged* trajectory — only the first is placed by hand, the
  rest start where the last bridge left the robot (unless it ended further than
  `mujoco_deployment.divergence_radius` = 0.2 m from the start, in which case it
  is placed by hand again, the same rule the benchmark stage chains under), and
  the loader clips the bridge back off. `GAINMLP.JSN` next to the config is
  honoured, evaluated by `pololu.gain_mlp_exporter.reference_forward` (the numpy
  mirror the firmware crate's golden fixtures come from) rather than a fourth
  copy of the network.

  **A bridged run ending within a millimetre of the start is normal, not a
  bug** (measured 2026-08-17 on `exp01`). The bridge's terminal min-jerk creep
  drops `v_ff` to 0.1 mm/s over the last couple of seconds, so `kx * x_e` has
  seconds at near-zero speed to null the *along-track* error — both stock and
  tuned controllers converge to a fixed point the chain then repeats to within
  tenths of a millimetre. Only body-x stays regulated there: `w = w_ff +
  v_ff * (ky*y_e + kth*sin th_e)` is multiplied by `v_ff`, so the lateral and
  heading errors freeze wherever the last moving phase left them, and the two
  gain sets simply split the residual differently — iteration 1 (stock,
  `kth/ky` = 2.0) froze at 0.2 mm lateral but **1.9° of heading**, iteration 2
  (tuned, `kth/ky` = 7.7) at 3.3 mm lateral and 0.06°, with chained tracking
  RMSE 0.017 m against 0.024 m. So a smaller printed *position* offset is not a
  better controller; it is the same convergence with the residual moved into
  the heading. `ki_inner` and the gain MLP were both ruled out as causes.

  The same plant also drives the **benchmark** stage, which is what makes the
  pipeline-progress plot mean anything: every iteration drives one *fixed*
  baseline reference (`benchmark.trajectory` in experiment.yaml, default
  `trajectory_exports/baselines/circle_fast.JSN`) under its own deployed
  controller, so a difference between two iterations' points is a difference in
  the controller and nothing else. The iteration's own identification and
  tuning trajectories cannot answer that — they are redesigned every iteration.
  `run` records it before identifying or tuning; the logs go to
  `data/benchmark[_static]/` (subdirectories, so the identify and residual
  stages, which only look one level deep, never see them) and the ground truth to
  `results/benchmark.yaml` (never into `data/`, which decode-logs would try to
  decode and which would leak the hidden plant besides).

  **Both controller options an iteration ships are driven, on paired seeds**
  (2026-08-17). Run `i` of each gets the *same* seed, so the hand placement and
  the sensor noise are common random numbers and the difference between the two
  sets is the controller and nothing else.

  - **Iteration 1 records only `data/benchmark_static/`**, driven from its own
    `ROBOTCFG.CFG` where it lives. Its gains *are* the stock static gains and its
    exported parametrization is exactly the identity — the init's output layer is
    zero, so every factor is `clip(1 + 0, ...)` = 1.0, asserted bit-exactly in
    `test_iteration_ones_parametrization_is_the_identity` — so those runs *are*
    the static-gain baseline. Nothing tuned exists to compare them against: the
    first tuned parametrization ships in iteration 2, so benchmarking anything
    "tuned" in iteration 1 would be benchmarking the stock controller twice.
    Iteration 1 therefore contributes a colour to the static panel only.
  - **From iteration 2** the iteration also carries `ROBOTCFG_static.CFG`, which
    `finalize` writes whenever the previous tuning ran an independent static tune
    beside a parametrized one — so its presence *is* the question "was the
    parametrization worth it". The deployed (parametrized) controller then gets
    `data/benchmark/` and the static side switches to those static-tuned gains.
  - An experiment with the parametrization off never gets a second variant in any
    iteration, correctly: its deployed controller is static throughout.

  The static side is always the gains some run converged to, never the
  parametrized run's base gains with the network taken away: base gains under a
  schedule are not a controller anybody tuned (they absorb whatever scale the
  factors leave them, over `[0, bound]`), so driving
  them bare would benchmark an artifact rather than the alternative on offer.

  `results/benchmark.yaml` is keyed by variant
  (`variants.{parametrized,static}`), and re-running the stage adds a missing
  variant without re-recording the one it has (the runs are chained, so
  re-recording half a set produces one that never happened). `ROBOTCFG_static.CFG`
  is driven from a *staged copy* under `benchmark/static/`, because
  `mujoco_sim.firmware.FirmwareConfig.from_file` picks up `GAINMLP.JSN` from
  beside the config the way `sdlog.rs` does — driven from the iteration root it
  would silently carry the network it is the baseline for. (Iteration 1 needs no
  staging: there the identity network beside the config *is* the static
  controller.) `iteration_status` reports `benchmark` done only when *every*
  variant has its runs, or `run` would skip a half-recorded comparison.
  `progress.py` reads `benchmark` first and falls back to `benchmark_static`,
  which is what keeps it on the *deployed* controller in every case — including
  iteration 1, which would otherwise lose its progress point.

  **The comparison figure is `visualize/baseline_runs_<shape>.pdf`**, written at
  the end of `tune-gains` (best-effort, beside `pipeline_progress.pdf`) from
  *every* iteration up to that one, so each iteration keeps the comparison as it
  stood when it finished. 1x2: static gains left, gain parametrization right,
  one colour per iteration, at most `MAX_RUNS_PER_ITERATION` = 5 runs each
  (repeats of one reference — past a handful the extra lines only thicken the
  same band), the fixed reference dashed in both panels, and each iteration's
  mean position RMSE in the legend. A single-variant experiment gets a single
  panel rather than an empty second one.
  `active_learning/baseline_runs.py` collects, `visualization/baseline_runs.py`
  renders — the same split `progress.py` / `visualization/pipeline_progress.py`
  use, and `progress.py` now reads its logs through
  `baseline_runs.load_run_logs`. It also reads the hand-recorded real-robot
  layout: `data/<shape>/` is a static-gain recording and
  `data/with gain MLP/<shape>/` (either spelling) a parametrized one, so the
  archived experiments plot without a separate script. Runs directly in `data/`
  are identification data and are never treated as baselines.

  The runs are chained the way a repeat is chained on the robot: only the first
  is placed by hand, each later one starts wherever the previous ended — unless
  the previous run finished further than `benchmark.divergence_radius` (0.3 m)
  from the reference's start point, in which case it is placed by hand again,
  which is what a person standing next to the robot would do. Chaining needs
  the reference to end where it begins; a self-closing one (both shipped
  baselines are) is driven as it is, anything else gets a wait + bridge path
  appended first (`pololu.bridge_exporter`, the same mechanism the
  identification trajectory is repeated with).

  **Benchmark numbers must be medians over seeds, never one run.** The stock
  gains sit close to the edge on these baselines, so a single seed's hand
  placement decides whether a run recovers or diverges, and single-seed
  comparisons flip sign on plant changes that do nothing. Measured 2026-08-20
  over 6 seeds, stock gains `[4.5, 6.0, 12.0, 2.5, 5.0]`, median pose RMSE with
  the count of runs over 0.3 m. **The 2026-08-20 columns are superseded**: they
  were measured with the setpoint truncation bug, which is what made
  `circle_fast` diverge at all. Current plant = tick fix + tire mu 0.70:

  | baseline | rebuilt plant (buggy setpoints) | current plant | real robot |
  |---|---|---|---|
  | `circle_slow` | 0.024, 0/6 | 0.012, 0/6 | — |
  | `circle_medium` | 0.045, 0/6 | 0.018, 0/6 | — |
  | `circle_fast` | 0.641, 6/6 | 0.169, 3/8 | 0.190, 2/7 |

  So `circle_fast` remains the one that discriminates, but for a different
  reason than before: it is now the trajectory that sits *on* the traction limit
  rather than the one the reference glitch broke. Iteration 1's benchmark on it
  is a set of marginal runs, some of which diverge, which is what the robot
  does. Switch `benchmark.trajectory` to `circle_medium` or `circle_slow` for a
  benchmark every iteration can track.

  The older figures recorded here (1.06 m / 48%, 1.14 m / 46%, 0.30 m / 6%,
  2026-08-17) do not reproduce on any current plant and should be ignored.

  **The rebuild's own trap, worth recognising elsewhere: an honest number can
  expose a dishonest one.** Putting the measured 171 g in (up from a guessed
  135 g) made `circle_medium` diverge on 3 of 6 seeds, because the loaded
  wheel-speed rise is dominated by vehicle inertia reflected through the wheel,
  `(m*r^2/2)/kv`, and at the datasheet stall torque that came out at 0.34 s
  logged — against the 0.16-0.18 s the identification pipeline converges to on
  real logs. The plant was twice as sluggish as the robot it stands in for. The
  fix was to calibrate `motor.stall_torque` against that identified time
  constant (1.5x datasheet) rather than to walk the mass back; why the factor is
  1.5 is unresolved and a stall-torque measurement on the hardware would settle
  it. The general lesson: when one guessed parameter is replaced by a
  measurement, the parameters it interacts with have to be re-derived, not left
  alone — a model tuned against symptoms hides its errors in whichever knob is
  still free.

  **`scripts/render_deployment_video.py` films one deployment**
  (`mujoco_sim/render.py`). Point `--trajectory` at any trajectory inside an
  experiment — a designed `.pkl`, a bridged or unbridged `.JSN` — and it walks
  up to that iteration's `ROBOTCFG.CFG` and drives it under the controller and
  identified parameters it was deployed with. `--gains parametrized|static|both`
  picks between the deployed `ROBOTCFG.CFG` + `GAINMLP.JSN` and the
  `ROBOTCFG_static.CFG` baseline, which is *staged into a scratch directory*
  for the same reason the benchmark stage stages it — `FirmwareConfig.from_file`
  picks up a `GAINMLP.JSN` sitting beside the config. An iteration that ships
  only one controller (iteration 1, or the parametrization off) resolves both
  variants to it and says so.

  **`--gains both` puts the two controllers in one video on the same seed**, so
  the hand placement and the sensor noise are common random numbers and the
  difference between the two traces is the controller and nothing else — the
  same pairing the benchmark stage records the variants under. A MuJoCo scene
  has one robot, so only one run can be the rendered chassis: the parametrized
  one is driven live and the static one is replayed beside it from its recorded
  pose track (`record_pose_track`, the same `FrameObserver` cadence). Both are
  real deployments; which carries the body is a rendering choice. The replayed
  robot is a **ring** around where its chassis is, not a disc — a good
  comparison keeps the two runs nearly on top of each other, and a disc at the
  chassis's own radius is simply hidden under the white body from above, which
  is the case the viewer most needs to see. Trace colour is per *variant*
  (`VARIANT_RGBA`: parametrized orange, static pink) in single-variant videos
  too, so a colour means the same thing across every video of a run. A
  single-controller iteration is refused rather than drawn twice.

  **`--layout split` renders several cameras of the same run into one frame,
  and MuJoCo needs no help doing it** (2026-08-21). A `Renderer` draws whatever
  `mjData` it is handed, so a pane is just another renderer over the same
  physics step: one `_ViewPane` per `View`, tiled into a numpy frame buffer by
  `pane_rects`. There is no second library stitching videos afterwards and
  therefore no way for the panes to drift out of step — the run is still one
  `run_deployment`, and the split screen costs one render per pane and nothing
  else. It composes with everything: `--gains both` draws the replay marker in
  every pane, `--slowmo-factor` is unchanged.

  `pane_rects` puts the first view in the left half and stacks the rest down
  the right, tiling the frame **exactly** — at 1920x1080 a 960x1080 overview
  beside two 960x540 chase panes, and 960x1080 twice with one chase angle — so
  nothing is scaled or letterboxed and no pixel is left uninitialized (a test
  asserts the cover is exactly 1 everywhere, at sizes that do not divide
  evenly). A static pane is framed against its *own* aspect ratio, since a
  960x1080 overview is a different shot from a 1920x1080 one. All three chase
  settings — `--follow-angle`, `--follow-distance`, `--follow-elevation` — are
  **per pane**: each takes one value that every chase pane shares, or one value
  per angle (`per_chase_view` broadcasts; a count that is neither is a typo and
  raises rather than zipping short). So a split screen can hold two genuinely
  different shots — close in from behind beside wide and abeam from under the
  floor — not the same shot at two bearings. The terminal line names all three per
  view, since two chase views can differ only in azimuth or only in stand-off.
  A single-camera layout takes the first of each list and says so, since the
  shipped defaults are a split screen's. Panes carry no caption of their own —
  nothing is drawn over a pane but the legend, at the top-left of the composed
  frame.

  **The floor is see-through from below, so `--follow-elevation -90` watches
  the caster roll.** MuJoCo renders a `plane` one-sided: a camera under it looks
  straight up through the floor at the caster ball, both tires and their contact
  patches, with the grid still faintly drawn over them. Nothing had to be made
  transparent. The flag states the chase camera's elevation outright instead of
  letting it follow from `--follow-height` / `--follow-distance` (0 level with
  the robot, positive above, negative below); `--follow-distance` is then the
  straight-line range. This is the only view of the contact geometry there is —
  a top-down shot cannot show it and even the flattest above-floor chase camera
  only catches the caster edge-on.

  The camera is described by `View` alone — a frozen dataclass carrying the mode
  and both cameras' knobs, with the derived quantities (`elevation_above_lookat`,
  `follow_range`, `line_scale`) as properties so the height and the elevation
  can never be set to disagree. `render_deployment` and
  `render_deployment_comparison` take `views=`; they no longer take
  `elevation` / `azimuth` / `camera_mode` / `follow_*`, which would have been a
  second way to say the same thing.

  **The legend is composited onto the frames, not added to the scene** — a
  MuJoCo scene carries no text. `_legend_overlay` draws it once with Pillow
  (nothing in it changes during a run) and `_composite` alpha-blends the patch
  into each rendered frame; its font and padding scale off the frame height, so
  it stays legible at 640x360 and does not swell at 4K. Entries are in draw
  order and cover exactly what is drawn, so a single-variant video gets two rows
  and a comparison three. This is why `pillow`, `imageio` and `imageio-ffmpeg`
  are now in `[project.dependencies]`: all three already arrived transitively
  via matplotlib/moviepy, but package code imports them, and relying on a
  transitive install is the bug the cvxpy/optimistix entry above records.

  It is **not** a re-simulation for the camera: `DeploymentVideoRecorder` is a
  `run_deployment` `observer` (called with `(tick, plant)` after every physics
  step), so there is one copy of the control loop and the video cannot drift
  from what `simulate-deployment` records. The run's `TRxx` goes to a temp
  directory unless `--log-dir` is given — a video is not data collection, and a
  stray log in an iteration's `data/` is identification data nobody asked for.
  The overlay is MuJoCo decor capsules: the reference in full from frame one,
  the trace appended per frame from the plant's *true* pose (ground truth, so
  it never leaves the picture). The overview camera is static and framed to fit
  the whole reference — `--elevation` (default 80, just off vertical so the chassis
  and its heading stay readable) and `--azimuth` (default 90, so +x is right
  and +y is up as in every `visualize/` plot). No residual model: the residual
  lives in the JAX simulator and nothing in the MuJoCo plant can consume one.
  ~8 s to render a 17 s run at 1280x720.

  There is also a `scripts/render_mujoco_video.py`, which is a different thing —
  constant open-loop duty on the bare XML, no firmware, no reference.
- `problems/*.yaml` — problem/robot configs (YAML schema changes must fan out to all files)
- `scripts/` — thin CLI wrappers; logic lives in the package
- `tests/` — pytest suite (fast, CPU)

Data: experiment logs in `Pololu Data/Experiments/<date>/.../TRxx.csv` (mind the
`.csv`), trained models in `models/*.pkl`, plots in `visualize/*.pdf`, trajectory
exports in `trajectory_exports/`.

Inside an iteration, `data/*.csv` are the **identification** logs (several
diverse trajectories per iteration, fitted jointly) and `data/<name>/*.csv` are
baseline comparison runs — `data/benchmark/` and `data/benchmark_static/`
written by the benchmark stage, or `data/circle` / `data/with gain MLP/circle`
for the hand-recorded runs of the real-robot experiments;
`active_learning/progress.py` scores the deployed controller's set and
`active_learning/baseline_runs.py` collects all of them for the cross-iteration
comparison figure.

**Every `data/` subdirectory recording gets a log summary plot, and the scan is
experiment-wide.** `run` calls `stages.plot_run_directory_logs` each time it
proceeds (also on its own as `plot-run-logs`), which walks *every* iteration's
`data/` for subdirectories holding `TR*` recordings and writes one
`plot_logged_summary` per run into `visualize/logs/<the same subdirectory>/`,
beside the identification logs' own plots -- `data/benchmark/` ->
`visualize/logs/benchmark/`, `data/with gain MLP/circle/` ->
`visualize/logs/with gain MLP/circle/`. It has to be
experiment-wide because benchmark logs arrive *after* the iteration that
recorded them was finalized (a person copies them off the SD card later), and
because the decode-logs stage only plots the identification logs sitting
directly in `data/`. Binaries are decoded into a temp directory, never into the
experiment, and a run whose plot already exists is skipped, so a rescan of a
finished experiment costs nothing. The gain overlay follows the same
variant rule `baseline_runs` uses: `benchmark/` and the `with gain MLP/` tree
are replayed through the iteration's gain parametrization, every other
subdirectory against `robot_config_static_gains.yaml` (never the problem yaml's
base gains, which under a schedule are not a controller anybody drove). The identify stage
only looks one level deep. It runs
per log first, drops logs whose parameters break away from the batch median
(`identification.outlier_z_threshold`, `identification/outliers.py`), and then
fits the rest jointly — per-log normalized losses averaged, so a long or fast
log cannot dominate. `identification.identify_a_slip_max: false` holds the
traction limit at the robot config's value (burnout still active in the
rollout) for runs too slow to excite it; an unobservable parameter otherwise
scatters across logs and still ships its winning value downstream.

**`identify` also writes `visualize/identified_parameters.pdf`**, a cross-
iteration 2x2 of wheel radius, wheelbase, max wheel speed and motor time
constant against the iteration index — the loop's answer to "are the
identified parameters settling?". Regenerated from *every* iteration up to the
current one and written into that iteration's own `visualize/`, best-effort, the
same pattern `pipeline_progress.pdf` and `baseline_runs_<shape>.pdf` follow at
the end of `tune-gains`. `active_learning/identified_parameters.py` collects
(straight out of each `results/identification.yaml`, no rollouts, so it is
instant), `visualization/identified_parameters.py` renders, and
`scripts/plot_identified_parameters.py` runs the pair over a finished
experiment. The per-log fits
behind each joint fit are drawn as faint dots, but the logs the outlier screen
**dropped are deliberately left out**: one rejected log sits far enough out to
compress a panel by an order of magnitude (measured on `exp02` iteration 1:
wheel radius 12.2-17.7 mm against a 16.0-16.4 mm trend) and hide what the figure
exists to show.
The screening decision belongs to the iteration's own
`plot_identification_log_parameters`. Nothing here reads
`models/pololu_hidden.yaml` — the pipeline never sees the true parameters, so
neither does its own progress plot.

## Current focus: nominal-operation pipeline + residual overhaul (2026-07)

Goal for now: good tracking during *nominal* (non-drifting) operation and a
sim that makes the gain optimizer pick sensible, transferable gains. Extreme
regimes (drift) are deferred. Priorities: stable closed loop > accuracy >
simplicity.

Residual model (rewritten 2026-07-24, details + results in memory
`residual-expert-ensemble-overhaul`): minimal Sym2Real-style additive residual
(arXiv:2509.15412). Inputs are the (state, action) pair
`[v_nom, omega_nom, v_cmd, omega_cmd]` (`RESIDUAL_FEATURE_NAMES`, extended from
state-only 2026-07-26); outputs `[delta_v, delta_omega]` (no lateral channel).
Combination is additive and direct — the old first-order lag (`residual_tau`),
the `*_filt` feature memory, the slip gate, the tanh output bound, and the
multi-step rollout stage are all **removed**, along with the
`twist_filtered`/`residual_state` fields on `DiffDriveState`. `ResidualEnsemble`
is K small tanh MLPs, each weight spectrally normalized at forward time
(`spectral_norm_cap`), blended by a k-means gate over the full descriptor with
a null "zero expert" so the residual smoothly → 0 out of distribution.
Single-step training only; ~10 s for a full run. Train CLI defaults to
`exp04`+`exp05` (all iterations, gathered recursively). Old `models/*.pkl` are
the previous checkpoint format and will not load — retrain.

Do **not** add wheel speeds to the residual inputs: given `r` and
`base_diameter` they are a bijective linear map of the twist
(`ur = (2v + L omega)/2r`), i.e. the same numbers in another basis, and they
measurably add nothing. The commanded twist (kinematics of
`max_wheel_speed * duty`) is the half that carries new information — it is what
lets a memoryless residual express *transient* errors (motor time constant,
delay, backlash), which scale with command minus state.

How much the command half is worth depends on how wrong the nominal model is
(measured 2026-07-27, controlled 2×2 over the 30-log exp04+exp05 pool, same
split/seed/epochs). Held-out `domega` RMSE [baseline → twist-only →
twist+input]: **global params 0.3160 → 0.2936 → 0.2805; per-log params
0.3200 → 0.2978 → 0.2901** — the command half's extra contribution falls from
+58% over twist-only to +35%. Mechanism: the stock `time_constant` is 0.120 s
while every identified iteration sits at 0.16–0.18 s, and a too-fast motor lag
is a pure transient error ∝ (command − state), exactly what the command half
encodes. A linear fit of `domega` on `omega_cmd - omega_nom` alone explains
R² = 0.152 under global params but only 0.074 per-log. So the old
"0.265 → 0.240" headline was substantially a proxy for a mis-identified motor
lag; keep the command half (it still wins open loop on every log), but do not
expect it to pay for a nominal model that is already identified per log.

Beware per-sample RMSE as the yardstick here: per-log params *raise* the
residual target RMS (`domega` 0.281 → 0.299, `dv` 0.0384 → 0.0398) while
*lowering* integrated open-loop pose error (mean over 4 mixed-split logs
0.197 → 0.129 m nominal, 0.090 → 0.079 m with the residual). Global params
leave a smaller but systematic error that integrates; per-log leaves a larger,
less biased one.

Honest open-loop verdict on the **6 held-out logs** of the seed-0 file split,
scored against the mocap pose track (`visualize/residual_open_loop_comparison_*.pdf`,
rebuilt 2026-07-27): mean 0.167 m nominal → 0.151 twist-only → **0.137
twist+input**, i.e. ~18%, not the ~40% a train-heavy log selection suggests.
Both models *hurt* on the two exp04 logs (it01 TR03 0.148 → 0.175, it02 TR01
0.142 → 0.173) and help most on exp05 it05 TR00 (0.222 → 0.080) and it03 TR04
(0.116 → 0.050). Score this figure on validation logs only — three of its four
earlier panels were training logs.

Open modeling question (2026-07-24): the sim does **not** penalize high motor
`k_i` because the wheel-speed PI loop feeds back a first-order-lag (PT1) motor,
and a PT1+PI closed loop is stable for *all* `k_p, k_i > 0` — no phase-margin
loss, so high `k_i` just tracks faster in sim while it oscillates omega on the
real robot. Candidate fixes (see the residual memory / discussion): add
realistic phase lag to the wheel loop (measurement delay or 2nd-order motor) so
high `k_i` loses phase margin in sim too; and/or a yaw-rate smoothness penalty
in the gain-tuning objective (`omega_delta_weight`, implemented 2026-07-24,
alongside the existing `input_delta_weight` duty-rate penalty; 0 disables).
NOT the fix: the bare `+ kth*th_e` term
on omega (controller.py line 70, kept removed) — it is not in the canonical
Kanayama tracking law (line 71: `w = w_d + v_d*(ky*y_e + kth*sin th_e)`), is not
scaled by `v_d` so it spins at low speed, and breaks the Lyapunov guarantee;
don't add it to sim or firmware.

Interaction to watch: the error-MLP gain parametrization effectively learns the
*inverse* of the residual — intended, but the two nets in a loop can co-adapt
into a marginally-stable pair (wobbly braking with the MLP on where the static
controller is smooth). Keep both spectral-norm caps modest and the MLP output
bound tight.

**Every optimizer can now design/tune on the residual-augmented plant, and it is
opt-in everywhere** (2026-08-17). `TrajectoryOptimizationPipeline(residual_model=)`
and `run_joint_tuning(residual_model=)` take a checkpoint loaded by
`residual_model.load_residual_model`, exactly as `run_gain_tuning_experiment`
already did; `--residual-model PATH` (default None = nominal) exists on
`run_trajectory_optimization.py`, `run_trajectory_optimization_tuning.py` and
`run_joint_tuning.py`. One constructor argument covers a whole designer because
every rollout in it goes through `simulation.run_closed_loop`, which falls back
to the pipeline's model. Note the **identification replay is unaffected** — the
residual reaches the closed-loop log the design is built from, not
`replay_simulation_log`, which has no residual path at all; that is the coherent
split (the residual is part of the plant generating the data, the replay is the
nominal model whose parameters are being fitted).

The joint loop hands it to **both** blocks — `TrajectoryOptimizationPipeline`
*and* `ControllerTuningPipeline`. They have to roll out the same plant, the same
invariant the existing `wheel_lp_tau` / `gain_wheel_lp_tau` provenance pair
records; a run's `gains.yaml` carries `residual_model:` next to them.

In active learning the two designers get a flag each, because they sit on
opposite sides of residual training (`plan-id-trajectory` -> ... -> `identify` ->
`train-residual` -> `plan-tuning-trajectories` -> `tune-gains`):
`identification_trajectory.use_residual_model` (**default false**) can only use
the *previous* iteration's `results/residual_model.pkl`, and
`tuning_trajectories.use_residual_model` (**default true**) uses *this*
iteration's, freshly fitted to the logs the iteration recorded. The asymmetry is
the point: the tuning design is scored on the plant `tune-gains` will roll out,
while the identification FIM's design parameters are the nominal robot
parameters being identified and the previous residual corrects a plant that has
since been re-identified. A missing checkpoint designs nominal and says so
rather than raising (iteration 1 has no predecessor; an experiment with
`use_residual_model` off never trains one).

**Unmeasured**: whether designing on the residual plant produces better designs
or better gains. All that has been checked is that it changes the objective, so
the residual really is in the plant — identification design loss
-12.4954 -> -12.5264, tuning design -7.1123 -> -7.1792, and the joint loop's
gains move (`ky` 20.4 -> 35.1 over 3 rounds). Turning a flag off reproduces the
nominal run bit-for-bit.

**A residual fitted to saturated logs destroys gain tuning, and `tune-gains` has
its own flag for exactly that** (`gain_tuning.use_residual_model`, default true;
needs the top-level `use_residual_model` as well). Measured 2026-08-17 on a
MuJoCo `exp01` iteration 1: at the problem's own gains, 0/40 rollouts diverged on
the nominal plant with either the standalone or the active-learning trajectory
set, and **26/40 diverged with the residual in the plant** — same trajectories,
same problem yaml, worst deviation 0.10 m -> 2.18 m. Swapping the trajectory set
or the problem yaml changed nothing; the residual alone accounted for all of it.
This is the failure to suspect first when the active-learning loop tunes far
worse than `run_gain_tuning.py`, whose `--residual-model` defaults to None.

The mechanism, worth recognizing in any future checkpoint. The iteration's
identification logs are recorded with **stock gains** on an aggressive
FIM-designed trajectory, so duty saturates and the robot slips. The nominal
kinematic model cannot explain that, so it all lands in the residual *target*:
measured `domega` target RMS **4.24 rad/s** against the ~0.28-0.30 of the real
exp04/exp05 logs, with `omega_nom` reaching ±12 rad/s (`omega_max` 5) and
`omega_cmd` ±75. The ensemble then fits almost none of it (R² 0.15 on `domega`,
0.43 on `dv`) and what it does learn is a large speed-dependent yaw bias —
Δω +0.92 rad/s at v=1.5, +1.63 at v=2.0, with Δv about -30% of v. No controller
survives that. The OOD gate does **not** save you here: the huge `omega_cmd`
spread inflates `input_std` to 10.74, every sample collapses into one cluster,
and the gate reads 0.99-1.00 across the whole envelope (p1 = 0.963 on its own
training data). So check the *target* RMS and the gate spread on a new
checkpoint, not just its validation loss.

**The tuning designer's stage and `run_trajectory_optimization_tuning.py` must be
kept in step by hand** — the stage builds `TrajectoryOptimizationPipeline`
directly, so any knob it does not pass silently falls back to a *pipeline*
default that differs from the script's CLI default. Four had drifted (fixed
2026-08-17, now `tuning_trajectories` config keys): `start_offset_mode`
(`optimize` vs the pipeline's `random`), `constraint_component_weights`
(alpha 0.5 vs 1.0), `constraint_smooth_max_beta` (10.0 vs 20.0), and the export
passing `control_points`. The beta one is the substantive one and the direction
is counterintuitive: `smooth_positive_max` is `logsumexp(beta*g)/beta`, which
overshoots the true max by up to `log(num_samples)/beta`, so a **lower** beta
holds the design further inside the limits — 0.50 of margin at beta=10 against
0.25 at beta=20. At the pipeline default the stage's designs ran at up to
**0.94x `alpha_max`**, against 0.47-0.69x from the script, i.e. straight into the
divergence regime recorded above (<= 0.75x tracks, >= 1.03x diverges). Passing
all four brings it to 0.49-0.76x. Not passing `control_points` also meant
`save_reference_states_pickle` never ran `assert_curve_is_exportable`, so a
tangent-stalled design could ship out of the loop where the script refuses it;
the stage now pre-checks the whole batch before writing any of it, as the script
does. The exported `start_offsets` were also the *shared* bundle repeated for
every trajectory rather than the per-trajectory designed ones.

**Trajectory-design motion limits are separable from the plant's**
(`TrajectoryOptimizationPipeline(motion_limits=)`, a partial robot-config block
overriding the problem yaml's `v_max` / `a_max` / `a_max_lateral` / `omega_max` /
`alpha_max`). It moves only the bar the *constraint term* holds the curve under;
the robot, the controller and the gain-parametrization feature scale all still
read the yaml. `identification_trajectory.motion_limits` uses it, because the two
designs are driven under different conditions: a tuning trajectory is rolled out
in sim from a small start offset, while the identification trajectory is placed
by hand and driven open loop on the robot, so it is worth designing inside a
gentler envelope than the hardware's nominal one.

**Per-log configs (do not use the global problem YAML for log comparisons).**
Every active-learning log lives in `<experiment>/<iteration_XX>/data/TRxx.csv`,
and that iteration folder's `problem.yaml` holds the config that was actually
running when it was recorded: tuned base gains, the error-MLP `theta` exported
to `GAINMLP.JSN`, and the *identified* robot/estimator params (verified to match
the folder's `robot_config.yaml`). These differ substantially per iteration —
tuned `kx` is ~0.003 where stock `problems/pololu_gains.yaml` has 4.5, and the
motor `time_constant` moves 0.12 → 0.16–0.18 s — so comparing a log against the
global YAML simulates a controller that never recorded it (it produces spurious
overdrive: duty 0.93 vs 0.76, v 2.14 vs 1.65 m/s). `problem_path_for_log` /
`robot_params_for_log` resolve this; `build_residual_dataset` (via
`train_from_logs`) and `simulate_closed_loop_on_log_reference` both use them,
with the global problem only as a fallback for logs outside an iteration folder.

**Start a sim-vs-log comparison at the log's first mocap pose**, not at the
reference start (`run_closed_loop(initial_pose=...)`, passed by
`simulate_closed_loop_on_log_reference`). The robot is placed by hand: measured
start offsets are 31–100 mm and up to 9.5° across the exp04/exp05 logs, a large
fraction of the 0.05–0.09 m tracking RMSE being compared, and driving that
offset out is part of what the real run's error contains. Closed-loop sim-vs-
mocap over 4 logs, reference-start → measurement-start init: nominal
0.0653 → 0.0592 m, residual 0.0568 → 0.0480 m; the residual's edge over nominal
widens from −13% to −19%, so the unfair init was *understating* it.

Known-bad data: `exp05/iteration_04` was identified from a single run and its
params are broken (`max_wheel_speed` 217 rad/s vs 233–240 elsewhere) — open-loop
pose error on its log goes 0.11 m → 0.54 m versus the stock params. Re-identify
or exclude it; it single-handedly flips aggregate validation metrics.

Sibling repo: robot firmware at `~/Desktop/Uni/MA/Code/pololu-rs` — trajectory
JSN limits live in `firmware/src/trajectory_reading.rs` (`MAX_POINTS`, scratch
size) and are mirrored in `pololu/reference_exporter.py`. The error-MLP gain
parametrization exports to `GAINMLP.JSN` via `pololu/gain_mlp_exporter.py`
(spectral norm baked in); the firmware side is `firmware/libs/gain_mlp`, whose
capacity limits are mirrored in the exporter and whose golden test fixtures
come from `scripts/export_gain_mlp.py --golden`.

## Controller and gain config

This branch runs the **Kanayama** tracking law only (`controller.py`), feeding
the wheel-speed PI inner loop. A dynamic-feedback-linearization controller (a
1:1 port of the firmware's `dynamic_feedback_control`) exists on a **separate
branch**: it needed a lot of rewiring to select between laws and tracked
poorly, so it was deliberately taken back out here. Not a concern for current
work — don't re-add it or its `controller.type` switch, and treat any leftover
reference to it (`dynamic_feedback_gains`, a 9-gain vector, `GAIN_NAMES`,
`ACTIVE_GAIN_INDICES`, `FIRMWARE_GAIN_SLICE`) as stale.

Gains are one flat 5-vector under `controller.gains`:
`[kx, ky, kth, kpmotor, kimotor]` — the same five the firmware knows, so
ROBOTCFG/GAINMLP export the whole vector and `error_mlp.NUM_GAINS` is 5.
Result yamls (`results/gains.yaml`, `models/tuned_gains.yaml`) hold that same
vector.

**The standalone tuner refines with BFGS and no presearch** (2026-08-08).
`--optimizer` is `bfgs` (default) | `adam`, and `num_lhs_points` defaults to
**0**. BFGS ignores `--learning-rate` (its line search sets the step length) and
reads `--steps` as a total inner budget split into restarts of 40
line-search-bounded steps — restarts, not one long solve, because a fresh
Hessian approximation escapes flat directions the previous one baked in
(measured: 5x40 reaches 0.009939 where a single 200-step solve stalls at
0.013097).

Measured on `trajectory_exports/gain_optimized_current`, static gains, held-out
validation loss:

| config | validation |
|---|---|
| **bfgs, no presearch** | **0.009392** |
| bfgs + LHS 64 | 0.009390 |
| adam 500 + LHS 64 | 0.009600 |
| adam 500, no presearch | 0.010648 |

**These four numbers are stale as of commit `821e569`** (found 2026-08-08).
The real static baseline on `gain_optimized_current` is validation
**0.0139441** with gains `[2.241, 3.588, 4.581, 2.961, 0]`, and the scheduled
run is **0.008741** — both matching the committed `models/tuned_gains.yaml`
bit-for-bit. Note `kth ~ 4.6` is a *third* cluster, distinct from both basins
recorded below. The table's *ordering* (BFGS beats Adam, presearch buys
nothing) was not re-measured and is not in question; only the absolute values.

**BFGS stalls outright about 1 run in 18** (~6%, measured over 18 tuning runs
2026-08-08): it returns the problem's `controller.gains` bit-for-bit with all
restarts spent without leaving the start, and a held-out score 17x worse. The
training loss equals the initial candidate's to 5 s.f., which makes it cheap to
assert on — do that, because a stalled run silently ships stock gains and looks
like a success. This is the concrete failure mode behind "raise
`num_lhs_points` if a run looks basin-trapped".

**The stall is one diverging rollout, and it is now caught automatically**
(measured 2026-08-09). `outlier_loss_factor` / `--outlier-loss-factor`
(default **10.0**, 0 disables) drops any *training* rollout whose loss at the
initial gains exceeds that multiple of the median rollout's.

Why it was needed: at the pre-alignment EKF with `noise_angle` 0.01 / seed 0,
one rollout of 32 (training trajectory 4, realization 2) scored 1.287 against a
0.0075 median -- **86% of the whole training loss**. It dominated the gradient,
the backtracking Armijo line search never found a sufficient decrease, every
restart burned all 40 line-search-bounded steps, and the tuner returned
`controller.gains` bit-for-bit while reporting success. Turning the rule on
fixed that run outright: training 0.06232 -> 0.00832, held-out
0.01582 -> **0.00703**, 1.2 -> 10 it/s. It is a safety net, not a crutch --
at the current aligned defaults, where nothing stalls, on/off is
0.006761 vs 0.006757 held-out.

Design points that matter. The mask is computed **once, at the initial gains,
and frozen**: deciding per evaluation would make the objective discontinuous
exactly where a rollout crosses the threshold, which is worse for a line search
than the outlier. It is per **(trajectory, realization)**, not per trajectory,
so one bad start offset does not discard three good rollouts. It applies to
**training only** -- validation outliers are reported and kept, or the held-out
score would flatter itself. `weighted_realization_mean` (objectives.py) is the
weighted reduction; `per_rollout_losses` / `rollout_outlier_weights` /
`_report_rollout_outliers` (optimizers.py) build and print the mask.

**Why that rollout diverged is actuator saturation, not bad luck, and not a
monotone noise effect.** Swept over `noise_angle` at `enc = 0.01`, stock gains,
the same rollout: fine at 0.005 (pose RMSE 0.067 m) and 0.006 (0.101), **broken
across 0.007-0.018** (0.42-0.67 m), fine again at 0.02 (0.063) and 0.03
(0.069). A *band*, sharply bounded at both ends. In every broken case `|duty|`
hits exactly 1.000 for 6-27% of the run; in every good case it stays <= 0.79.
The rollout runs at `|v|` ~ 2.08 m/s against `v_max` 2.5 with stock `kth` 12,
i.e. near the edge of actuator authority, and the mocap jitter is what tips it
over. Below the band too little jitter reaches the controller to trigger it;
above it the (pre-alignment, i.e. current) Kalman gain had fallen enough to filter the
high-frequency content out despite the larger raw noise. Decisively: the *tuned*
gains track that same rollout at **0.019 m** RMSE with max duty 0.653 -- the
trajectory is fine, the stock gains are not, and the outlier was what blocked
the solver from reaching the gains that fix it.

**The gain-tuning summary figures used to draw one shared noise realization**
(fixed 2026-08-17). `visualization/gain_tuning.rollout_realizations` called
`run_closed_loop` without `robot_key` / `estimator_key`, so every rollout in
`summary_gain_tuning` / `summary_training` / `summary_validation` fell back to
the pipeline's single `target_*_key`: the R "realizations" of a trajectory
differed only in their *start offset*, all on one noise draw the tuner never
scored. The objective meanwhile gives every **(trajectory, realization)** pair
its own key (`split_realization_keys_by_trajectory`, namespace 0 training / 1
validation). The figures now take the same keys, via
`realization_keys_for_set(realizations, num_trajectories, key_namespace)` --
`realizations` is a required argument of the summary plotters and the keys a
required argument of `rollout_realizations`, so no call site can fall back
silently. Split over the *whole* set and then sliced, since the keys a
trajectory was scored under depend on how many trajectories the set has (a
`max_trajectories` figure must not re-split the first few).

This was purely cosmetic -- no gains, designs or losses move -- but the figure
was **optimistic**, which is worth knowing when reading old plots: on an
active-learning tuning set at the stock gains, shared key against the
objective's own keys, pose RMSE 0.0357 vs 0.0376 m and max heading error
0.285 vs 0.425 rad. It is also what made a `summary_training.pdf` look like a
much better-tracking controller than the *same* trajectories under the
trajectory designer's `plot_trajectory_batch` (which always passed
per-realization keys). If the two ever disagree again, check the keys before
suspecting the designs: with the keys matched, the two rollout paths agree to
every printed digit.

**The firmware-alignment of the EKF was tried and reverted** (2026-08-09/10,
reverted 2026-08-10). The attempt: split the conflated
`noise_pos` / `noise_angle` into *injected sensor noise* (config) and the
*filter's assumed* `Q` / `R` (module constants `FIRMWARE_Q` / `FIRMWARE_R`
copied from `pololu-rs/firmware/src/ekf.rs::default_at_origin`,
`Q = diag(1e-3, 1e-3, 1e-2)`, `R = diag(1e-4, 1e-4, 1e-3)`, steady-state
`K = 0.916` against the sim's `K_xy = 0.50` / `K_theta = 0.39`); drop the
input-noise term `Lx M Lx^T`; propagate x,y on the midpoint heading
`th + 0.5*w*dt` (the firmware's "stabilized prediction"); re-value
`pololu_gains.yaml` to the measured mocap floor (`noise_angle` 0.02 -> 0.004).

**It made the tuning trajectory designer emit uniformly dull curves**, and that
was not worth chasing further, so the whole EKF/noise half of commit `dad9757`
was reverted: `estimator.py` is back to `proc_pos_std` / `proc_theta_std` for
`Q`, `noise_pos` / `noise_angle` for `R`, the `Lx M Lx^T` input-noise term
(hence `slip_r_var` / `slip_l_var` are live again), the entry-heading
prediction, and the yamls carry `proc_*` again. `alpha_max` in
`pololu_gains.yaml` went back to 20.0 and `models/tuned_gains.yaml` to its
pre-alignment values, and the two constraint knobs the commit had retuned went
back as well: `DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE` 0.15 -> **0.05** and the
tuning designer's `--constraint-alpha-weight` 1.0 -> **0.5**, both of which
visibly improved the designs on their own. What was *kept* from that commit: the
identification loss no longer weighting encoder residuals by the slip variance
(`identification/losses.py`), `outlier_loss_factor` in gain tuning, the
divergence gate in trajectory optimization (itself removed later, see below), and the pyproject/uv changes.
Everything below about the aligned filter is therefore a record of what was
measured, not of what the code does. The alignment is in git history at
`dad9757` if it is ever revisited -- but see the next paragraph before doing
so: the revert is what brought the good curves back, confirmed.

**Confirmed by the revert: the EKF/noise change was the main cause of the dull
tuning trajectories** (2026-08-10). The designer produces diverse, curvy designs
again on the old plant. Two smaller contributors were reverted alongside it and
each improved the designs in its own right -- the constraint-violation tolerance
back to 0.05 (0.15 buys limit violations too cheaply) and the tuning designer's
`--constraint-alpha-weight` back to 0.5 -- but the EKF/noise change is the one
that dominated. The **divergence gate stayed on across the revert**, so it is
not what was suppressing shape (it was removed later the same day for an
unrelated reason -- see the divergence-gate entry below). The control that settles the filter's role:
under the aligned filter the **Bezier**
parametrization -- the one that reliably gives diverse, high-curvature curves --
came out looking essentially the same as the B-spline, i.e. equally dull. A
plant change that flattens the design objective under *both* bases is a plant
problem, not a parametrization problem. So if the firmware alignment is ever
attempted again, expect to have to solve this first, and treat the design
diversity of the tuning designer as the acceptance test for it.

Two implementation notes worth keeping if it is ever redone: `Q`, `R`, `I3`
must live in `__init__`, not `get_init_state` -- that runs inside every jitted
rollout, so assigning them there leaves them holding leaked tracers (the
reverted `estimator.py` has this bug again). And the measured true mocap noise,
off the exp04/exp05 raw tracks against a short Savitzky-Golay fit over 14 logs,
is **0.001 m** and **0.004 rad**; tracking was flat over `noise_angle`
0.002-0.015 and `enc_angle_noise` 0.005-0.02 on feasible designs (median pose
RMSE 0.058-0.066 m, zero divergences), so the noise level was never the
delicate knob it was suspected to be.

**Do not fix the EKF to use the low-passed wheel speeds** — the firmware's
odometry task (`firmware/src/odometry.rs`) feeds `ekf.predict` from the raw
counts and the sim matches this. Overturned 2026-09-03 and reinstated
2026-09-06: `odometry.rs` was deleted in `pololu-rs@4469e0a` and the inner loop
published `write_odom` from `omega_*_lp` for two months, which is the state the
`real02` logs were recorded in; the task has been restored. See the odometry
entry under "Controller and gain config".

**Divergence is a design-feasibility problem, not a noise problem and not a
speed problem** (measured 2026-08-10, 10 designs x 4 realizations at the tuned
gains `[2.32, 7.90, 9.46, 3.10, 0]`). Sorting rollouts by peak reference speed
gives a perfectly *clean* result the wrong way round: the **fastest** designs
(2.12-2.25 m/s) track at **0.02-0.06 m** RMSE with 0% duty saturation, while the
only divergent ones ran at ~1.2 m/s. The actual predictor is angular
acceleration against `alpha_max`:

| `|alpha|max / alpha_max` | rollouts diverging |
|---|---|
| 1.37, 1.10 | 4/4 each |
| 1.03 | borderline (RMSE 0.10-0.16) |
| <= 0.75 (7 designs) | 0/4, RMSE 0.02-0.06 |

Every design over the limit diverges, every design under it tracks. So "most
rollouts diverge" is the design optimizer emitting curves the robot physically
cannot turn -- the B-spline control points collapse into near-coincident
clusters (adjacent gaps of 0.02-0.10 m against 1.2-1.5 m elsewhere,
collinearity ratio down to 0.04) and the corner between them demands more yaw
acceleration than the robot has.

**REMOVED 2026-08-10: the divergence gate.** What follows is the record of
what it did and why it was taken back out; the code is in git at `4aca319`.

It gated each realization out of the FIM by a smooth logistic in its max
deviation from the reference (`divergence_tolerance`, 0.15 m), because an
ungated FIM *rewards* the control-point collapse above: a rollout the
controller cannot track has an enormous Jacobian w.r.t. the gains, so the
criterion reads divergence as information. Against no gate (10 trajectories,
500 steps, same seed) the loss history went from a dip to -13, a **spike to
+125** and a final -3.0 -- worse than the -8.4 it started at -- to a monotone
descent. The tolerance had to be *tight*: at 0.5 m the optimizer dove into the
spurious reward before the gate closed on it (a plunge
**-10.3 -> -15.3 -> -9.3** in a few steps, the "cliff then recovery" signature
in a loss history), where 0.15 ended *better*, -11.9 vs -10.3 over 500 steps.

**Why it went:** the gate was `stop_gradient`-wrapped by construction -- it
decided *whether* a rollout counted and was deliberately not something the
optimizer could push on -- so it moved the loss while contributing nothing to
the gradient. Measured on an instrumented 500-step single-trajectory run
(6 control points, s-curve, optimized offsets, lr 2e-3): the run is monotone
with **0% uphill steps** and roughness 0.003-0.007 for 450 steps, then in the
last 50 the gate closes on three of four rollouts (weight sum 3.44 -> 1.14),
`|delta gate|` jumps **50x** (0.0009 -> 0.0475), roughness jumps **17x** to
0.056, 8% of steps go uphill and the total *rises* 0.31. The FIM normalizer
divides by `sqrt(sum w)`, so the criterion's effective scale shifts as rollouts
drop out as well.

**Confirmed by removing it**: the batched 10-trajectory run at the current
settings is smooth end to end, with none of the fuzz from ~step 350 that
prompted the investigation. Note the single-trajectory instrumented run above
*understated* the effect -- it put the onset at 450 and confined the roughness
to the last 50 steps -- so trust the batched run here.

Two things had also changed since the gate was added, which is what made
removing it reasonable: `bspline.tangent_floor_loss` now blocks the
control-point-collapse route to spurious information *with* a gradient, and
`constraint_violation_tolerance` was retuned. **If the +125 spike or the
cliff-and-recovery signature comes back in a loss history, this is the first
thing to reconsider** -- the mechanism it was written for is real and was never
disproved, only re-covered from a different direction. But note the +125
measurement predates the tangent-floor loss, so it is not evidence about the
current objective.

**"Designs being homogeneous is not caused by the EKF/noise change" was wrong
-- OVERTURNED 2026-08-10 by the revert.** Recorded here because the proxy that
produced it looks convincing and should not be trusted again. Shape
discrimination -- the relative spread of the FIM term over a fixed bank of 24
random control-point sets -- measured **0.38 with the aligned filter against
0.44 with the old one**, a 15% difference and not a collapse, and lowering the
injected noise moved it the wrong way (`noise_angle` 0.004 -> 0.0002 gave 0.30).
That was read as exonerating the filter. It did not: reverting the EKF restored
the curves outright. The proxy scores the criterion over a bank of *random*
control points, which is not where the optimizer spends its time -- a criterion
can still separate random shapes while its gradient no longer leads anywhere
interesting from the initial design. Measure what the optimizer actually
produces (arc length, mean and max `|kappa|` over a real run), not how the
criterion ranks a random bank.

The other suspect ruled out at the time, `alpha_max`, does still stand as a
negative result: halving it 20 -> 10 *raises* mean `|kappa|` 4.32 -> 6.46 and
its spread 0.90 -> 1.24, so it is not a dullness knob in either direction.
Independently of the filter, the designs on this problem tend to share a long
diagonal across the environment box with a hook at one end, and that much is
structural: the s-curve time scaling peaks the speed mid-path and takes it to 0
at both ends,
while the curvature constraints scale with speed (`a_lat = v^2 kappa`,
`alpha ~ d(v kappa)/dt`), so curvature is only affordable near the ends.
`--time-scaling linear` visibly changes the character (mean `|kappa|`
6.46 -> 7.71, arc 3.37 -> 2.45, loops and scallops instead of diagonals) and is
the first knob to reach for; `--num-control-points 8` barely moves it
(4.63, spread 0.84). Part of the lost "diversity" is also the fix working: the
wild degenerate shapes in older runs were the untrackable ones.

**Dull designs follow the design point** (measured 2026-09-06 on
`2026_08_19/real02`). The archived tuning designs are curvy in iteration 1
(total heading change per design 3.95 rad) and flat from iteration 2 on
(0.7-1.4 rad, peak yaw rate 0.3-0.75 rad/s) under identical designer settings;
the only thing that changed was the gains the FIM was built around -- the
*parametrized run's base gains*, which `finalize` writes into the next
iteration's `problem.yaml`. Re-running
the stage's designer on the iteration-2 plant with gains swapped, total heading
change per design:

| design point | sum abs(dtheta) [rad] |
|---|---|
| stock `[4.5, 6, 12, 2.5, 5]` | 3.96 |
| stock with `kimotor = 0` | 3.31 |
| stock with `kth = 36` | 1.21 |
| parametrized base gains of it02 `[4.59, 5.86, 36.4, 6.55, 0]` | 1.31 (1.54 at 250 steps) |
| those with `kimotor = 5` | 1.45 |
| those with `kth = 12` (`kpmotor` 6.55 stays) | 1.42 |
| those with `kpmotor = 2.5` (`kth` 36 stays) | 0.78 |

Either a high `kth` or a high `kpmotor` flattens the designs on its own: the
loop is then so stiff that curvature stops revealing the gains, and what is
left informative is the start-offset transient, so the designer buys straight
diagonals with a hook near the start. `kimotor = 0` costs ~15% (the pinned
`kimotor_fim_scale` is doing its job), the 50-step budget little. The fix is
in the code since `e3ee7ab`: the designer takes `controller_gains=` and the
stage passes the **static-tune gains** (`_static_design_gains`,
`robot_config_static_gains.yaml`), never a parametrized run's base gains,
which are not a controller anybody runs. `tuning_trajectories.kimotor_fim_scale`
(default **5.0**) stays, because the static gains still ship `kimotor = 0` and
a tied scale would fall back to `k_max_rest` = 20 there (share of
`trace(FIM^-1)` ~0.16%); removing the scaling was re-tested the same day and
rejected (see the trajectory-optimization section).
Nothing can push `kimotor` back off 0 once the tuner puts it there, because the
sqrt-space parametrization has zero derivative at 0 (see "A sqrt-space gain
cannot be initialized at 0"); harmless if 0 is the accepted answer.

So the LHS presearch buys nothing once the refiner is BFGS: it exists to hand
Adam a starting basin, and BFGS does not need one — it lands `kimotor` at
*exactly* 0 by gradient, which is the value the 4000-point presearch used to be
required to find (via `_with_motor_zero_variants`). A whole default run is 35 s
including JAX startup and plotting. Raise `num_lhs_points` again only if a run
looks basin-trapped; the gain objective *is* multimodal (two basins measured,
`kth` ~ 6 at loss 0.0105 and `kth` ~ 8-10 at 0.037), BFGS just starts in the
right one from the stock gains.

**BFGS trains the gain parametrization too**, not just the base gains: the
trainable vector is one concatenated `[gain_values(5), parametrization_flat(num_w)]`,
so `--gain-schedule` puts the whole thing through the same solver. For the
`error_mlp` in `problems/pololu_gains.yaml` (one hidden layer of 16, all 5 gains
scheduled) `num_w = 213`, i.e. a 218-D BFGS solve — a dense inverse Hessian is
~48k entries there, so the usual "quasi-Newton does not scale" objection does
not bite at this size. Measured: held-out validation **0.00655 scheduled vs
0.00939 static**, and 97% of the 230 saved schedule entries are non-zero with
`||theta|| = 23.1`, so the MLP is genuinely trained and not sitting at identity.

**The gain-tuning summary's "Initial" curve is the controller the run started
from** (fixed 2026-09-06; found 2026-09-03). `run_gain_tuning_experiment` used
to build `init_hidden_log` / `init_model_log` with **no `schedule_params`** and
the problem's gains, and the problem's gains are the *parametrized* run's base
gains (`finalize` writes those, not `static_gains`) -- so the blue "Initial"
trace was base gains with their network taken away, a controller nobody ran.
Scored on each iteration's own training set, bare / deployed total loss: July
0.0076/0.0060, 0.0125/0.0063, 0.0337/0.0126, 0.0124/0.0055 (a real schedule,
factors 0.6-1.6, so the figure was nearly honest); `real02` 0.129/0.0041,
0.336/0.0022, 0.095/0.0035 (the network was undoing the base `kpmotor`, so the
bare gains oscillate). It is why every `real02` summary made its predecessor
look like it was ringing. Now `init_gains` / `init_schedule_params` on the
result name the starting controller -- the problem's gains under the
warm-started schedule when `schedule_enabled and warm_start_schedule`, else
the static gains (`static_init_gains` from the stage, i.e. the previous
iteration's `robot_config_static_gains.yaml`, or the problem's own) -- and
every "Initial" family (`summary_gain_tuning`, `summary_training`,
`summary_validation`, `ctrl_tuning_tracking_errors`) rolls those out. Same
class of figure-only bug as the shared-noise-key one above.

**The loop is not failing to load the previous iteration's gains, and it is not
collapsing on its own training set.** Scored on one fixed neutral set
(`real02/iteration_04`'s trajectories and plant): stock 0.00742, then the
shipped controllers of iterations 2/3/4 at 0.00437 / 0.00319 / 0.00333
parametrized and 0.00574 / 0.00450 / 0.00477 static. Real improvement, then a
plateau after iteration 3 — while the robot got worse. Each iteration's own
`summary_*` figures are scored on *its own* redesigned trajectory set, so they
are not comparable across iterations; use a fixed set.

**Watch the identified `base_diameter` across iterations.** 2026_08_19/`real02`
drifts 84.2 -> 87.7 -> 89.7 -> 89.7 -> 89.4 mm (+6.6%) while 2026_07_27 stayed
at 82.4-84.0. The per-log fits within each iteration are tight (88.9-90.5 mm at
iteration 3), so this is a consistent bias, not scatter — the signature the
un-windowed replay entry predicts (`identification.window_length: null` in both
experiments). A 6.6% wheelbase overestimate inflates the commanded wheel-speed
differential for a given yaw rate, i.e. it is a systematic yaw error in exactly
the channel that was observed oscillating. 2026_08_19 also added
`identification_trajectory.motion_limits`, which made the identification runs
much gentler (`omega_cmd` p95 67 vs 90 rad/s, peak duty 0.40 vs 0.54) and so
less informative about the wheelbase.

**Watch the base gains under the schedule** (raised 2026-08-08, **confirmed
disastrous on the robot 2026-08-19**). The parametrized run returns
`[4.37, 6.68, 41.19, 12.63, 0]` where the static run returns
`[2.36, 5.14, 6.35, 2.89, 0]` — `kth` 6.5x higher and `kpmotor` 4.4x. The factor
is `clip(1 + mlp(z), 0, bound)` with the *bound* (not the factor) clipped to
>= 1, so the effective `kth` ranges over `[0, ~206]` at `bound: 5`: the network
can switch a gain off as readily as scale it up. (An earlier note here claimed
the factor was bounded below at 1 and therefore that the effective `kth` was
`>= 41`; that was wrong.)

What the real experiment measured (`Pololu Data/archive/Experiments/2026_08_19`,
`real02`): the network's job had become **undoing `kpmotor`**, not scheduling
it. Replaying the shipped `GAINMLP.JSN` over the iteration's own logs, the
applied `kpmotor` factor sits at the *lower clip* for most of the run —
iteration 3 base 21.49, applied p5/p50/p95 = 0.215/0.292/3.07 (`kp_inner`
0.00095/0.00129/0.0135); iteration 4 base 16.30, applied 0.163/0.188/4.28. So
the effective inner-loop gain swings **~25x within a single run**, which is
exactly the "wobbly braking with the MLP on" co-adaptation warned about under
the residual section, and it is what produced the theta oscillation on hardware.
Compare the 2026_07_27 run, where the same network was a genuine schedule:
factors 0.6-1.6 and `kp_inner` 0.0096-0.0179, i.e. a modulation around 1.

The failure needs both halves — a base gain the sim does not care about (see the
`kpmotor` degeneracy below) and a solver that will actually travel to the box
edge. Adam at `lr` 1e-4 never got there; BFGS does.

**`kpmotor` has no interior optimum in this sim, and BFGS finds that out**
(measured 2026-08-19 on `real02/iteration_04`'s own tuning set, static gains,
the other four held at the tuned values). Held-out loss against `kpmotor`:

| `kpmotor` | 0.01 | 0.05 | 0.20 | 0.5 | 1 | 2.5 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| validation | **0.00268** | 0.00269 | 0.00278 | 0.00310 | 0.00385 | 0.00727 | 0.0237 | 0.0638 | 0.1218 |

Monotone toward the lower box bound, i.e. the sim's preferred inner loop is
**pure feedforward with the P feedback switched off**. `kth` and `ky` are not
like this — both have clean interior minima on the same run (`kth` 17, `ky`
12.3) — so this is specific to the inner loop, and it is the same modelling gap
recorded under "Open modeling question": the wheel-speed loop closes a PT1 motor
with a feedforward term that already delivers the right duty, so feedback buys
nothing in sim and costs nothing in phase margin, while on the robot `kp_inner`
is exactly the gain that sets phase margin and therefore whether theta
oscillates. `omega_delta_weight` does not price it (1.2 and 1.5 were both in
use).

Over 2026_08_19/`real02` the static side's `kp_inner` collapsed
0.0093 -> 0.0033 -> **0.00088** (1/33 of the shipped default) while the
parametrized side's ran the other way, 0.010 -> 0.028 -> **0.095**, with
`ktheta` pinned on the search-box bound in three of four iterations (36.4, then
50.0 = `k_max_stab`, then 20.0). Measured tracking RMSE on `circle_fast`, real
robot, static side: 0.151 (stock) -> **0.087** -> 0.138 -> 0.109 -> 0.332 — best
after one round, then deteriorating, while the sim's held-out loss improved
monotonically. So the sim and the robot disagree in *sign* along `kpmotor`.

**Neither the optimizer nor the 2026_07_27 objective weights explain it**
(measured 2026-09-03; an earlier entry here claiming Adam's small learning rate
acted as an implicit trust region was wrong). Re-tuning two `real02` iterations
under five configurations, `kpmotor` (static run's in brackets):

| config | it02 (sane init 6.55) | it04 (runaway init 16.3) |
|---|---|---|
| BFGS, no presearch (shipped) | 21.2 [0.74] | 26.2 [0.20] |
| Adam 500 @ 1e-4 + LHS 500, band 0.3 (2026_07_27 search) | **5.42** [1.07] | 26.2 [0.26] |
| BFGS + LHS 500, band 0.3 | - | 26.2 [0.26] |
| BFGS, 2026_07_27 objective weights | - | 36.5 [0.20] |
| Adam, 2026_07_27 objective weights | - | 28.7 [0.23] |

So the search *is* a contributing factor at the fork point — from a sane start
BFGS lands 4x higher than Adam in one round — but neither has a fixed point,
and once the iterate is in the runaway region every configuration agrees. The
2026_07_27 objective (`velocity_tracking_weight` 0.2, `omega_delta_weight` 0,
`num_realizations` 1, no start offsets, `k_max_stab` 40) is *worse*, not better.
Whatever kept 2026_07_27's gains sane, it was neither the search nor the loss
weights.

**Two candidate mechanisms were tested and both failed**, so do not reach for
them again without new evidence. (a) The feedforward mismatch: the robot's
`wheel_max` is 250 while the plant identifies 227-238, so the firmware's
feedforward is 5-10% weak, where the sim's is exact. Forcing the sim's
controller to divide by 250 against a 229.83 plant moves the `kpmotor`
validation sweep by <4% and leaves it monotone. (b) The EKF lag (see the
odometry entry below): making the sim's EKF predict on the low-passed wheel
speeds, as the firmware does, makes *high* `kpmotor` clearly worse (train
0.0072 -> 0.0270 at `kpmotor` 2) but still leaves no interior optimum.

What is left is a **missing disturbance, not a missing lag**: the sim's plant is
exactly the inverse of the controller's feedforward (`compute_duty` and
`robot.step` are both handed `robot_params`, and there is no way to give the
controller a mismatched model — `est_params` reaches only the estimator), so
there is nothing for inner-loop feedback to reject, while `input_delta_weight`
and `omega_delta_weight` charge it for amplifying encoder noise. Cost without
benefit, hence the slide to the bound.

The practical options, in order: (a) **drop `kpmotor`/`kimotor` from the tuned
set** and ship the hand-tuned firmware values (0.0288 / 0.048) — the sim cannot
inform them, and every controller that has ever worked on this robot had
`kp_inner` in 0.0096-0.0179; (b) give them their own tight box (`k_max_stab`
bounds all four log-space gains jointly, so this needs a per-gain box);
(c) model a disturbance the feedforward cannot cancel (per-wheel motor-gain
asymmetry, load-dependent gain) — untested, and the expensive route.

**`ktheta` on the box bound is a reportable failure, not a result.** Three
`real02` iterations returned `k_max_stab` (or the value it was lowered to) to
7 significant figures. A gain vector with a component on the boundary means the
box, not the objective, chose it; the tuner should say so.

**`odometry.rs` is back, and the EKF predicts on raw wheel speeds again**
(fixed 2026-09-06; found 2026-09-03). `pololu-rs@4469e0a` deleted the odometry
task as "mostly duplicate" of the inner loop — both derive a wheel speed from
the same counters — and had the inner loop publish `write_odom` from
`omega_l_lp`/`omega_r_lp` instead. It is not duplicate: the PI feedback wants
the 3 Hz / 53 ms low-passed speed and the EKF must not have it. For two months
the robot's pose estimate — the one `kx`, `ky`, `kth` and the gain MLP all
close around — carried that filter, while `estimator.py` predicted on the raw
`u_hat`. `4469e0a` is dated 2026-07-06, so **both** archived real experiments
(`2026_07_27` and `2026_08_19`/`real02`) were recorded in that state — it is
not what separates them, but it was wrong in both.

The restored task is the original minus its dead reckoning, plus the inner
loop's stall guard: it publishes `(v, w)` from raw counts at 100 Hz, and skips
a tick whose measured `dt` is under half nominal, because `Ticker::every`
yields missed ticks back to back after an SD-write stall and one encoder count
over a microsecond interval would go straight into `ekf.predict` as a huge
twist. The old task's `x`/`y`/`theta` dead reckoning was **not** restored: it
fed nothing (`trajectory_control.rs` and `goto.rs` read only `odom.v`/`odom.w`)
and was never even decoded into a CSV column, while restoring it would widen
the tag-8 log record from 2 floats back to 5 and break every decoder against
every log recorded since July. `OdomPose` therefore stays `{v, w}` and the log
format is untouched.

Still unaligned, deliberately: the firmware predicts at **20 Hz** (the outer
loop, reading the latest 100 Hz odometry sample) against the sim's 100 Hz, and
`Q`/`R` are the firmware's explicit constants rather than the sim's
`proc_*`/`noise_*` — mirroring those in JAX was tried and made the design
objective far too sensitive to simulated mocap noise (see the EKF-alignment
entry above). One consequence of the 20 Hz read worth knowing: the EKF now sees
a *decimated* raw sample rather than an anti-aliased one, which is noisier in
the encoder band than the low-passed version was. That is the trade the raw
feed buys the lag back for, and it is what the sim does.

**`wheel_max` is now exported to `ROBOTCFG.CFG`** (fixed 2026-09-06).
`robot_config_values` wrote `wheel_radius` and `wheel_base` from the identified
params and divided `kp_inner`/`ki_inner` by the identified `max_wheel_speed` —
but left `wheel_max` at the template's 250.0. Every shipped config in both
archived experiments says `wheel_max=250` against an identified 210-238, so the
firmware's feedforward `omega_cmd / wheel_max` was 5-16% weak. Visible in the
logs: reconstructing the inner loop from the logged command/measurement stream,
the integrator sits at a standing **+0.030 duty** in every stock-gain log of
both experiments, which is exactly the deficit at cruise. It is one value
serving two roles firmware-side — the feedforward divisor in
`inner_controller.rs` and the wheel-speed clamp in `trajectory_control.rs` —
and the sim uses `max_wheel_speed` for both as well, so exporting it aligns
both. It propagates into `mujoco_sim.FirmwareConfig` for free, which is why the
MuJoCo deployment's feedforward is now exact too; a hand-written CFG can still
disagree and the firmware believes the file, which is what
`test_a_hand_written_wheel_max_still_wins_over_the_identified_gain` pins.

**A BFGS stall shipped silently through a real iteration.** `real02/iteration_05`
returned its input gains to 6 s.f. (`[4.145187, 11.139698, 20.0, 16.297483, 0]`
in, `[4.145187, 11.139697, 19.999996, 16.297485, 0]` out) — the
"returns the problem's `controller.gains` bit-for-bit" signature recorded above,
at `outlier_loss_factor` 20.0. Assert on it in `tune-gains`: an iteration that
re-deploys its own input is a wasted robot session.

`num_adam_optimizations` (the multistart count) is **inert in this default
configuration**: `num_starts = min(it, num_candidates)`, and with the presearch
off there is only the one init-gain candidate. It still binds when the
presearch is re-enabled or when `init_gains` is passed as a batch.

**The inner-loop D-term was re-tested and rejected 2026-08-08 — do not re-add
it.** From an init of `kdmotor = 0.25`, BFGS drove it to *exactly* 0 in all 6
runs, and validation moved +0.006% against a seed spread **3403x larger**. It
is not a dead gradient: a pure-evaluation sweep at the tuned gains rises
monotonically — 0.01394 (kd=0) -> 0.01474 (0.01) -> 0.02377 (0.05) -> 0.2089
(0.25) -> 1.642 (1.0) — so 0 is a strongly preferred boundary optimum. The
sim's known optimism about aggressive inner-loop gains (PT1 + PI is stable for
all positive gains) cuts *in favour* of this verdict: a model biased toward
tolerating the D-term still rejected it. The firmware already has the slot
(`kd_inner`, `inner_controller.rs:113`, shipped at 0.0) — keeping it 0 is now a
measured choice. Implementation preserved on branch
`experiment/motor-d-term-rejected` if it ever needs revisiting; note
`GAINMLP.JSN` is hard-capped at `NUM_GAINS = 5` firmware-side, so any change to
the gain-vector length breaks that export.

**A sqrt-space gain cannot be initialized at 0.** `gain = k_max_rest * v^2` has
derivative identically zero at `v = 0`, so a gain started at exactly 0 is
frozen there forever and any "is it used?" experiment on it is vacuous.
`kimotor` only escapes this because it starts at 5.0 and descends *to* 0. Give
any future zero-allowed gain a positive nominal value in the problem yaml.

Gain-tuning search space (`gain_tuning/optimizers.py`): the first four gains
must stay strictly positive for stability, so they are searched in **log**
space over `[k_min_stab, k_max_stab]` — scale-free resolution across decades,
and they can never reach 0. `kimotor` is the one gain allowed to be exactly 0
(integral action off), which log space cannot express, so it uses **sqrt**
space over `[0, k_max_rest]`, which also concentrates resolution near 0.

## Commands

- Tests: `uv run pytest -q` (pytest is a uv dev dependency; whole suite
  ~2.9 min, 222 tests). No flags, no environment: the project is CPU-only as of
  2026-08-15, so a bare `uv run` is correct and re-syncing is harmless.
- **Dependencies are fully declared as of 2026-08-08** — `cvxpy` (imported by
  `planner.py`) and `optimistix` (imported by `gain_tuning/optimizers.py` and
  `joint_tuning/gain_solvers.py`) were both missing from `pyproject.toml`;
  `cvxpy` was absent entirely and `optimistix` was in the `dev` group despite
  being package runtime code. Both are now in `[project.dependencies]`, so a
  fresh `uv sync` imports `planner`, `simulation` and `gain_tuning.pipeline`.
- **This project is CPU-only (2026-08-15). There is no `cuda` extra any more,
  and `--extra cuda` / `UV_NO_SYNC=1` are no longer needed anywhere** — a plain
  `uv run` is the whole story, and `uv sync` no longer mutates anything a run
  depends on. The `[project.optional-dependencies] cuda = ["jax[cuda13]"]` entry
  was removed and `uv sync` uninstalled `jax-cuda13-*` plus 15 `nvidia-*`
  wheels. Nothing here wanted a GPU: the long sequential `lax.scan`s (rollout,
  replay, residual training) are launch-bound, so the GPU measured *slower* —
  see the `residual-training-gpu-slower` memory. The removal also ended a real
  failure mode: with the GPU stack installed, a driver that XLA cannot use fails
  the *entire* suite at setup (`JaxRuntimeError: ptxas too old` for CC 6.1, all
  222 tests), which is what the CPU pin in every script was quietly papering
  over. Note the same `uv sync` also picked up `cvxpy` 1.7.4 -> 1.9.2 (plus
  `osqp`, `scs`); `cvxpy` is `planner.py`'s solver, and `test_planner.py` passes
  on it.
- **`[build-system] requires` must track the installed uv.** It was pinned at
  `uv_build>=0.9.5,<0.10.0` against uv 0.11.3, which warns
  `does not contain the current uv version` on every build; now
  `>=0.11.3,<0.12.0`.
- **The root `.venv` is a shared mutable resource.** Its editable-install
  `.pth` points at whichever checkout last ran `uv sync`/`uv pip install -e .`,
  so a bare `python` (or `uv run --active`) in a git worktree can silently
  import `wmr_simulator` from a *different* worktree. When working outside the
  main checkout, either set `PYTHONPATH=<that worktree>/src` (it precedes
  site-packages) or build a worktree-local venv with
  `env -u VIRTUAL_ENV uv sync`, and assert on `wmr_simulator.__file__` before
  trusting any measurement.
- Quick syntax check after refactors: `python -m compileall -q src scripts tests`
- Ad-hoc snippets and smoke runs: prefix `JAX_PLATFORMS=cpu` (scripts set it
  themselves, heredocs don't)
- Long-running scripts (`run_identification_pololu.py`,
  `run_trajectory_optimization.py`, `run_gain_tuning.py`,
  `train_residual_model.py`, `generate_baseline_reference.py`) take minutes:
  wrap in explicit `timeout N` and reduce `--steps`/`--opt-steps`/`--epochs`
  for smoke tests. See `.Codex/skills/verify` for canonical smoke invocations.

## Conventions

- **AI assistants must never add co-authorship or any other AI attribution to
  commit messages.** They are also strictly forbidden from pushing to remotes
  or creating pull requests.
- Everything must stay fully differentiable in JAX (gain tuning optimizes
  through the rollout). Stack: JAX + Equinox + Optax.
- Residual model operates on body-frame twist, not global poses.
- IMU gyro values in logs are in **degrees** — convert with deg2rad.
- Spline-smoothed mocap velocities (pololu.pose_smoothing) are the canonical
  velocities for identification and plots; raw finite differences are noisy.
- A parameter initialized to `0` in YAML/CLI means "disabled".
- `base_diameter` is the *effective* wheelbase (absorbs tire scrub); there is no
  separate gamma parameter.

## Style

- No backwards-compatibility shims, deprecated-key warnings, or migration code —
  configs are updated in place.
- Don't add CLI flags or options speculatively; only what was asked.
- Keep scripts thin wrappers; put logic in the package, visualization in
  `wmr_simulator/visualization/`.
