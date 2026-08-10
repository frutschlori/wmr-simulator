"""Alternating optimization of controller gains and their tuning trajectories.

The sequential pipeline designs trajectories that are informative about the
*stock* gains and then tunes the gains on those fixed trajectories. By the time
the tuner has converged the trajectories are no longer informative about the
gains it converged to -- ``kx``/``ky`` land on a plateau where the whole
plausible range moves the tuning loss by a few percent. This loop instead
alternates:

    1. warm start: step the trajectories against the initial gains
    2. repeat: step the gains on the current trajectories, then step the
       trajectories against the just-stepped gains

Two Adam optimizers, never one. The blocks live in incompatible search spaces
(gains in the log/sqrt reparametrization clipped to [0, 1]; control points in
metres projected onto the environment box), so a single optimizer would need
per-block learning rates anyway -- which two optimizers give for free, while
each block keeps its moments across rounds in which the other moved. Both are
built once, before the loop, and threaded through: re-initializing per round
would re-pay Adam's bias-correction warmup every round.

The two objectives are never summed and never compared. That is the main
simplicity argument for alternating over a weighted-sum joint objective, and it
is why each side keeps its own normalization (the gain loss raw as the tuner
uses it, the trajectory loss intrinsically normalized by
:func:`trajectory_optimization.objectives.trajectory_objective`).

Neither block's own loss is a usable progress measure for the *pair*: each is
evaluated against the other block's current iterate, so the gain loss is scored
on trajectories that changed since the last round and consecutive values are not
comparable. Measured over 350 rounds the gain block descends its own objective
on 97% of its steps while the alternation raises it on 84% of rounds. Two
consequences, both handled here.

**Training and validation.** The loop's own trajectories are its *training* set:
they are decision variables and they move every round, so ``gain_loss_pre`` is a
score on a problem that no longer exists a round later. The gains that ship are
instead the best iterate on a **validation** set --
``validation_trajectories_dir``, a fixed, neutral collection the loop never
optimizes (:mod:`joint_tuning.validation`). Without one the fallback is the
training trajectories frozen at the round the gain block first ran: stationary,
which is enough to keep best-iterate selection honest, but biased towards the
gains that suited that particular design, so it can say "the loop stayed
stable" and never "these gains are better".

**Early stopping is off by default** (``convergence_rel_tol`` < 0). The rule
itself is sound -- two-sided (:func:`has_stagnated`), because the standalone
one-sided form reads a *rising* loss as convergence and used to quit this loop
at round 150 on a -0.137 "improvement" -- but no stagnation rule fits this
loop's shape. Every run plateaus with the validation loss flat to five
significant digits, then goes through a basin transition at ~180 trajectory
steps where the trajectory block finds a more informative regime and the gains
follow it (``kth`` 5.2 -> 9.1). Measured, the rule stopped at round 100, about
90 rounds before the transition. Telling "finished" from "between events" is not
something this rule can do, so the default is a fixed round budget.

**The trust region is what makes the pair stable** (``trust_radius``, negative
disables). Without it the loop runs beautifully for ~190 rounds -- the gain loss
flat to +8%, the trajectory objective descending monotonically, the constraint
violation *falling* -- and is then destroyed by a single round: the gain loss
jumps 0.0109 -> 0.0502, ``kth`` 5.85 -> 9.8, and it never recovers. Tightening
``constraint_violation_tolerance`` 5x only moves the event from round 189 to 210.

The event is a *coupled* basin jump, not either block misbehaving: both
objectives worsen in the same round (the trajectory objective went uphill
-9.7359 -> -8.8878 with the violation spiking to 17x tolerance), which neither
block does on its own. The gain objective has two minima -- ``kth`` ~ 6 at loss
0.0105 and ``kth`` ~ 8-10 at loss 0.037 -- and once a trajectory move tips the
gain solver into the second one, the changed gains change the FIM, which kicks
the trajectory block, which lands somewhere the old gains score 4x worse.

So the step is capped in L2 and then *judged by what it did to the gain block*:
a trajectory move that raises the gain loss at fixed gains by more than
``trust_gain_loss_increase`` is rejected and the radius halves; otherwise the
radius grows back towards its initial value. Judging by the trajectory
objective would not work -- that objective improves through the jump. This
deliberately constrains the **coupling** rather than either block, because
neither block is individually at fault.

Not implemented, and the obvious alternative if the hard cap proves too blunt:
a **proximal (PALM-style) term** on the trajectory block, ``+ rho/2 * ||theta -
theta_prev||^2`` added to ``trajectory_objective``. Same goal by a smooth route,
with actual convergence theory for alternating minimization behind it, and it
degrades gracefully where a hard radius either binds or does not. It would
replace (not complement) the radius; keep the gain-loss acceptance test either
way, since that is what encodes "stay informative, do not become untrackable".

The loop itself is a Python ``for`` over two jitted step functions rather than
one ``lax.scan``: per-round gradient work is ~0.4 s while the Python overhead is
under a millisecond, and in exchange every diagnostic is a list append and a
mid-loop NaN is a print away instead of a re-trace.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.objectives import (
    Realizations,
    closed_loop_objective_terms,
    make_realizations,
)
from wmr_simulator.gain_tuning.optimizers import (
    clip_optimizer_values,
    controller_gains_from_optimizer_values,
    controller_gains_to_optimizer_values,
)
from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline, resolve_gain_robot_params
from wmr_simulator.joint_tuning.gain_solvers import (
    make_gain_stepper,
    resolve_steps_per_round,
)
from wmr_simulator.joint_tuning.validation import load_validation_trajectories
from wmr_simulator.trajectory_optimization.start_offsets import (
    START_OFFSET_MODE_OPTIMIZE,
    START_OFFSET_MODE_RANDOM,
    START_OFFSET_MODE_STATIC,
    inverse_squash_start_offsets,
    sample_initial_pose_offset_batch,
    normalize_start_offset_mode,
    resolve_start_offsets,
    start_offset_mask,
    static_start_offsets,
)
from wmr_simulator.trajectory_optimization.constraints import (
    constraint_loss_components_from_reference_states,
    constraint_loss_from_reference_states,
)
from wmr_simulator.trajectory_optimization.objectives import (
    DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
    DEFAULT_CRITERION,
    fim_loss,
    fim_objective_term,
    normalize_criterion,
)
from wmr_simulator.trajectory_optimization.optimizers import (
    CONVERGENCE_REL_TOL,
    CONVERGENCE_WINDOW,
    relative_improvement,
)
from wmr_simulator.trajectory_optimization.pipeline import (
    OBJECTIVE_MODE_GAIN_TUNING,
    TrajectoryOptimizationPipeline,
    load_reference_states_exports,
)


GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor")
CONSTRAINT_COMPONENT_NAMES = ("v", "a", "lateral", "omega", "alpha")

# How the two blocks are interleaved. ``alternating`` is the scheme this package
# exists for; ``sequential`` is the baseline it has to beat -- the same two
# blocks, the same shared realizations and the same per-block step counts, but
# run one after the other, so the trajectories are designed against the initial
# gains only and never see the gains the tuner converges to.
MODE_ALTERNATING = "alternating"
MODE_SEQUENTIAL = "sequential"
MODES = (MODE_ALTERNATING, MODE_SEQUENTIAL)

def normalize_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(f"Unsupported joint tuning mode '{mode}'. Expected one of {list(MODES)}.")
    return mode


def _round_schedule(mode: str, num_rounds: int, warm_start_rounds: int) -> list[tuple[bool, bool]]:
    """``(run_gain_block, run_trajectory_block)`` per round.

    Both modes take exactly ``num_rounds`` trajectory steps and
    ``num_rounds - warm_start_rounds`` gain steps, so a comparison between them
    is a comparison of the *interleaving* and not of the budget.
    """
    gain_steps = max(0, num_rounds - warm_start_rounds)
    if mode == MODE_ALTERNATING:
        return [(index >= warm_start_rounds, True) for index in range(num_rounds)]
    return [(False, True)] * num_rounds + [(True, False)] * gain_steps


class JointState(NamedTuple):
    """Everything the loop carries between rounds.

    ``gain_opt_state`` is a placeholder: the gain block re-solves from scratch
    each round (see :mod:`joint_tuning.gain_solvers`), so only the trajectory
    optimizer actually carries state between rounds.

    ``gain_values`` are optimizer-space (log/sqrt) values, not gains, and
    ``free_offsets`` are the unconstrained pre-squash start-offset variables --
    both blocks' search spaces, not their physical readings.

    The offsets are per *trajectory*, matching the standalone designer (every
    trajectory in a batch optimizes its own starts and exports them) and the
    gain tuner downstream, which reads a (T, R, 3) bundle off the pickles. A
    single shared bundle could not survive a round trip through an export
    directory, which is what the warm start is.
    """

    gain_values: jax.Array              # (5,)
    gain_opt_state: optax.OptState
    decision_variables: jax.Array       # (T, 2K)
    free_offsets: jax.Array             # (T, R, 3)
    trajectory_opt_state: optax.OptState


class JointRoundSnapshot(NamedTuple):
    """One round of the alternation, kept only when ``trajectory_trace_stride > 0``.

    The decision variables and the resolved offsets, not the sampled states: the
    states are the basis applied to the control points on one time grid, so
    keeping the control points costs (K, 2) per trajectory instead of (N, 8) and
    the renderer rebuilds the curve through the same pipeline the loop used.
    ``gains`` are physical gains, and ``validation_loss`` is the held-out score
    behind them, so a frame can be annotated with the number that actually
    selects what ships (NaN on a warm-start round, where the gain block did not
    run and no score was taken).
    """

    round_index: int
    control_points: np.ndarray          # (T, K, 2)
    start_offsets: np.ndarray           # (T, R, 3)
    gains: np.ndarray                   # (5,)
    validation_loss: float


class JointTuningResult(NamedTuple):
    # The best iterate scored on the frozen scoring set, not the last one. On a
    # pair of blocks that keep moving each other's objective, the last iterate
    # is a tail sample; this is the one that ships.
    gains: jax.Array                    # (5,)
    final_gains: jax.Array              # (5,) the loop's last iterate
    best_gain_score: float              # validation loss behind `gains`
    control_points: jax.Array           # (T, K, 2)
    reference_states: jax.Array         # (T, N, 8)
    start_offsets: jax.Array            # (T, R, 3)
    # The root noise bundle. Its own ``start_offsets`` are the initial draw the
    # per-trajectory offsets above started from; the offsets the run ended on
    # are ``start_offsets``, and that is what ships and what anything scoring
    # the result should roll out from.
    realizations: Realizations
    initial_decision_variables: jax.Array       # (T, 2K) before any step
    warm_start_decision_variables: jax.Array    # (T, 2K) when the gain block first ran
    state: JointState                   # both Adam states, as the loop left them
    trajectory_pipeline: TrajectoryOptimizationPipeline
    gain_pipeline: ControllerTuningPipeline
    # Per-round snapshots, empty unless ``trajectory_trace_stride > 0``. Opt-in
    # so the default run's memory profile is unchanged.
    trajectory_trace: tuple
    history: dict
    config: dict
    timing: dict


def has_stagnated(reference_score: float, score: float, tolerance: float) -> bool:
    """Has the scoring loss stopped *moving* over a window?

    Two-sided on purpose. The standalone trajectory optimizer's rule asks
    ``improvement <= tolerance``, which also fires when the loss is getting
    rapidly worse -- fine there, where a rising loss means a diverging learning
    rate and stopping is the right outcome, but wrong here: this loop's gain
    score rises whenever the trajectory block makes the tracking problem harder,
    and that is a run still in motion, not a converged one. Rises are handled by
    keeping the best iterate; only genuine flatness ends the run.
    """
    return abs(float(relative_improvement(reference_score, score))) <= tolerance


def _max_fractional_violation(components: jax.Array) -> jax.Array:
    """Largest smooth-max fractional over-limit behind a component vector.

    Each component is ``weight * g**2`` with ``g`` the smooth-max fractional
    violation clipped at 0, and the joint loop leaves the component weights at
    1, so the violation reads straight back off the square root.
    """
    return jnp.sqrt(jnp.max(jnp.maximum(components, 0.0), axis=-1))


def run_joint_tuning(
    problem_path: str,
    *,
    mode: str = MODE_ALTERNATING,
    num_rounds: int = 250,
    warm_start_rounds: int = 50,
    num_trajectories: int = 8,
    num_control_points: int = 7,
    num_realizations: int = int(GAIN_TUNING_DEFAULTS["num_realizations"]),
    trajectory_learning_rate: float = 1e-3,
    # Inner steps each block takes per round, i.e. how close each gets to its
    # *conditional* optimum before handing the problem over. At 1/1 the loop is
    # pure Jacobi-style alternation and every step of one block lands on an
    # objective the other block just perturbed; raising these turns it into
    # block-coordinate descent proper, where each handover happens at a point
    # the block had time to settle on. The ratio matters more than the counts:
    # the block that outruns the other wants *fewer* inner steps, not more.
    gain_steps_per_round: int | None = None,
    trajectory_steps_per_round: int = 1,
    # Trust region on the trajectory block, in decision-variable L2 units.
    # `trust_radius` caps a round's movement; a round whose trajectory move
    # raises the gain block's own loss by more than `trust_gain_loss_increase`
    # (relative) is *rejected* and the radius halves, otherwise the radius grows
    # back towards its initial value. Negative radius disables the whole
    # mechanism. See the module docstring for what it is defending against.
    trust_radius: float = 2e-2,
    trust_gain_loss_increase: float = 0.02,
    trust_shrink: float = 0.5,
    trust_grow: float = 1.5,
    # A collapsed radius means every meaningful trajectory step hurts the gain
    # block: the pair is at the edge of its stable region and further rounds
    # achieve nothing. Stopping there is a genuine local-optimality signal for
    # the *coupled* problem, unlike the loss-stagnation rule above, which cannot
    # tell a plateau from a pause. Counted on the radius sitting at its floor,
    # not on consecutive rejections: the observed collapse *oscillates* (accept
    # grows 1.5x, reject shrinks 0.5x, net decay), so a consecutive-rejection
    # counter resets constantly and would never fire. 0 disables.
    trust_stall_rounds: int = 20,
    validation_trajectories_dir: str | None = None,
    # Negative disables early stopping, which is the default here on purpose.
    # Every run of this loop has the same shape: a long plateau where the
    # validation loss is flat to 5 significant digits, then a basin transition
    # at ~180 trajectory steps where the trajectory block finds a more
    # informative regime and the gains re-tune to it (kth 5.2 -> 9.1). *Any*
    # stagnation rule quits in the plateau -- measured, the old default stopped
    # at round 100, about 90 rounds before the transition. A stopping rule needs
    # to know the difference between "finished" and "between events", and this
    # one cannot, so the honest setting is a fixed round budget.
    convergence_rel_tol: float = -1.0,
    convergence_window: int = CONVERGENCE_WINDOW,
    # Keep a snapshot of the trajectory block's decision variables, its offsets,
    # the round's gains and its validation score every N rounds, for the
    # per-round animation. 0 (the default) keeps nothing, so a normal run's
    # memory profile is untouched; a stride is required rather than optional
    # because a 250-round run is 250 closed-loop rollout batches to render.
    trajectory_trace_stride: int = 0,
    warm_start_trajectories_dir: str | None = None,
    start_offset_mode: str = START_OFFSET_MODE_OPTIMIZE,
    init_offset_radius: float = float(GAIN_TUNING_DEFAULTS["init_offset_radius"]),
    init_offset_angle: float = float(GAIN_TUNING_DEFAULTS["init_offset_angle"]),
    # Tighter than the standalone designer's DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE.
    # There the constraint only has to hold one design in the feasible set; here
    # it also has to keep the trajectory block from running away from the gain
    # block, and it was not doing that -- measured over 350 rounds the max
    # fractional violation climbed 0.02 -> 0.20, i.e. 4x the 0.05 tolerance,
    # buying a 3x better FIM with trajectories the gains could no longer track.
    constraint_violation_tolerance: float = 0.02,
    constraint_smooth_max_beta: float = 20.0,
    criterion: str = DEFAULT_CRITERION,
    time_scaling: str = "s-curve",
    seed: int = 0,
    k_min_stab: float = float(GAIN_TUNING_DEFAULTS["k_min_stab"]),
    k_max_stab: float = float(GAIN_TUNING_DEFAULTS["k_max_stab"]),
    k_max_rest: float = float(GAIN_TUNING_DEFAULTS["k_max_rest"]),
    velocity_tracking_weight: float = float(GAIN_TUNING_DEFAULTS["velocity_tracking_weight"]),
    input_weight: float = float(GAIN_TUNING_DEFAULTS["input_weight"]),
    input_delta_weight: float = float(GAIN_TUNING_DEFAULTS["input_delta_weight"]),
    omega_delta_weight: float = float(GAIN_TUNING_DEFAULTS["omega_delta_weight"]),
    verbose: bool = True,
) -> JointTuningResult:
    """Alternate gain and trajectory Adam steps on one shared realization bundle.

    ``warm_start_rounds`` is not a separate stage: it is simply the first N
    rounds with the gain block skipped. The trajectory optimizer's Adam state
    therefore carries straight into the alternating phase instead of being
    rebuilt.

    ``warm_start_trajectories_dir`` replaces that phase with a set of designed
    trajectories read off disk: the exports carry the control points and the
    start offsets they were designed under, and both are adopted as the loop's
    own, so the alternation begins exactly where a standalone
    trajectory-optimization run left off. ``warm_start_rounds`` is then 0 -- the
    trajectories are already warm, and spending rounds re-warming them against
    the initial gains is exactly the cost this argument exists to avoid. The
    directory also *sets* ``num_trajectories``, ``num_realizations`` and
    ``num_control_points``: they are properties of the design being loaded, not
    of this run.

    The gains returned are the best iterate on the frozen scoring set, not the
    last one.
    """
    if num_rounds < 0:
        raise ValueError("num_rounds must be non-negative.")
    if warm_start_rounds < 0:
        raise ValueError("warm_start_rounds must be non-negative.")
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive.")
    if num_control_points < 4:
        raise ValueError("num_control_points must be >= 4 (cubic B-spline).")
    if num_realizations <= 0:
        raise ValueError("num_realizations must be positive.")
    if trajectory_steps_per_round < 1:
        raise ValueError("trajectory_steps_per_round must be >= 1.")
    if warm_start_trajectories_dir is None and warm_start_rounds >= num_rounds and num_rounds > 0:
        raise ValueError(
            f"warm_start_rounds={warm_start_rounds} >= num_rounds={num_rounds}: the gain block "
            "would never run and the gains would come back exactly as they went in."
        )
    if convergence_window < 1:
        raise ValueError("convergence_window must be positive.")
    if trajectory_trace_stride < 0:
        raise ValueError("trajectory_trace_stride must be non-negative (0 disables the trace).")
    gain_steps_per_round = resolve_steps_per_round(gain_steps_per_round)
    if not 0.0 < trust_shrink < 1.0 < trust_grow:
        raise ValueError("Need 0 < trust_shrink < 1 < trust_grow.")
    if trust_gain_loss_increase < 0.0:
        raise ValueError("trust_gain_loss_increase must be non-negative.")
    trust_region_enabled = trust_radius > 0.0
    max_trust_radius = trust_radius
    min_trust_radius = 1e-6

    validation_trajectories = (
        None if validation_trajectories_dir is None
        else load_validation_trajectories(validation_trajectories_dir)
    )
    start_offset_mode = normalize_start_offset_mode(start_offset_mode)
    mode = normalize_mode(mode)
    criterion = normalize_criterion(criterion)

    warm_start = (
        None if warm_start_trajectories_dir is None
        else load_reference_states_exports(warm_start_trajectories_dir)
    )
    if warm_start is not None:
        if warm_start.start_offsets is None:
            raise ValueError(
                f"The trajectories in {warm_start_trajectories_dir} carry no start offsets. "
                "Only a gain-tuning-mode design can warm-start this loop: its offsets are the "
                "conditions its FIM was averaged over, and without them the design says nothing "
                "about the gains."
            )
        if warm_start.control_points is None:
            raise ValueError(
                f"The trajectories in {warm_start_trajectories_dir} carry no control points. "
                "The warm start continues optimizing the curve, so it needs the decision "
                "variables themselves; re-export the design."
            )
        # All three counts are properties of the design being loaded, not of
        # this run: the curve's control points, its trajectories, its
        # realizations.
        num_trajectories = int(warm_start.reference_states.shape[0])
        num_realizations = int(warm_start.start_offsets.shape[1])
        num_control_points = int(warm_start.control_points.shape[1])
        warm_start_rounds = 0

    schedule = _round_schedule(mode, num_rounds, warm_start_rounds)

    construct_start = time.time()
    # One bundle, built once, handed to both sides: the trajectory designer's FIM
    # and the gain tuner's objective then average over the *same* noise draws and
    # start poses, which is the whole point of designing the trajectories against
    # the gains.
    realizations = make_realizations(
        jax.random.PRNGKey(seed),
        jax.random.PRNGKey(seed + 1),
        num_realizations,
        init_offset_radius,
        init_offset_angle,
    )

    trajectory_pipeline = TrajectoryOptimizationPipeline(
        problem_path,
        time_scaling=time_scaling,
        objective_mode=OBJECTIVE_MODE_GAIN_TUNING,
        realizations=realizations,
        criterion=criterion,
    )
    gain_pipeline = ControllerTuningPipeline(
        problem_path,
        robot_params=resolve_gain_robot_params(problem_path, None, None),
        seed=seed,
    )
    # Build the B-spline basis eagerly: constructing it inside a jit trace would
    # stage the time grid into a tracer (see TrajectoryOptimizationPipeline._spline_plan).
    trajectory_pipeline._spline_plan(num_control_points, trajectory_pipeline.time_scaling)

    limits = trajectory_pipeline.motion_limits()
    constraint_weights = trajectory_pipeline.constraint_weights()
    dt = trajectory_pipeline.problem.dt

    mask = start_offset_mask(start_offset_mode)
    # (T, R, 3): every trajectory gets its own start-pose bundle, as in the
    # standalone batch designer. A warm start adopts the loaded design's, which
    # is the whole point of shipping them with the curve.
    # Same semantics as TrajectoryOptimizationPipeline.batch_frozen_start_offsets:
    # only `random` gives each trajectory its own draw, because there the draw is
    # the design; the other modes broadcast one bundle, which `static` keeps and
    # the optimizing modes move away from independently per trajectory.
    if warm_start is not None:
        frozen_offsets = jnp.asarray(warm_start.start_offsets, dtype=jnp.float32)
    elif start_offset_mode == START_OFFSET_MODE_RANDOM:
        frozen_offsets = sample_initial_pose_offset_batch(
            jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), 5814), seed),
            num_trajectories,
            num_realizations,
            init_offset_radius,
            init_offset_angle,
        )
    else:
        frozen_offsets = jnp.broadcast_to(
            static_start_offsets(num_realizations, init_offset_radius, init_offset_angle)
            if start_offset_mode == START_OFFSET_MODE_STATIC
            else realizations.start_offsets,
            (num_trajectories, num_realizations, 3),
        )
    # The optimizing modes start *at* the frozen offsets rather than at zero, so
    # every mode begins under the same conditions and only their evolution differs.
    initial_free_offsets = jax.vmap(
        lambda offsets: inverse_squash_start_offsets(
            offsets, init_offset_radius, init_offset_angle
        )
    )(frozen_offsets)

    def offsets_from_free(free_offsets):
        """(T, R, 3) free variables -> (T, R, 3) feasible offsets."""
        return jax.vmap(
            lambda free, frozen: resolve_start_offsets(
                free, frozen, mask, init_offset_radius, init_offset_angle
            )
        )(free_offsets, frozen_offsets)

    def realizations_from_offsets(offsets):
        """The root bundle re-pointed at one trajectory's offsets."""
        return realizations._replace(start_offsets=offsets)

    def reference_states_from_decision_variables(decision_variables):
        return trajectory_pipeline.reference_states_from_control_points(
            trajectory_pipeline.control_points_from_decision_variables(decision_variables)
        )

    # ---------------------------------------------------------------- trajectory
    def trajectory_terms(decision_variables, gains, start_offsets):
        """One trajectory's objective plus the decomposition, in one rollout set.

        This is ``trajectory_objective`` written out so the FIM and constraint
        terms fall out as aux instead of costing a second rollout batch; the
        tests pin it against ``fim_loss_from_control_points``.
        """
        reference_states = reference_states_from_decision_variables(decision_variables)
        fim_factor = trajectory_pipeline.compute_fim_factor(
            reference_states=reference_states,
            gains=gains,
            realizations=realizations_from_offsets(start_offsets),
        )
        fim_term = fim_loss(fim_factor, criterion)
        components = constraint_loss_components_from_reference_states(
            reference_states=reference_states,
            dt=dt,
            limits=limits,
            weights=constraint_weights,
            smooth_max_beta=constraint_smooth_max_beta,
        )
        component_vector = jnp.stack([components[name] for name in CONSTRAINT_COMPONENT_NAMES])
        constraint_term = jnp.sum(component_vector)
        # The stabilization term is part of the objective, not an extra: the
        # standalone designer adds it inside fim_loss_from_control_points, and
        # the pinning test compares the two.
        stabilization_term = trajectory_pipeline.stabilization_loss_from_control_points(
            trajectory_pipeline.control_points_from_decision_variables(decision_variables),
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            constraint_violation_tolerance=constraint_violation_tolerance,
        )
        total = (
            fim_objective_term(fim_factor, criterion)
            + constraint_term / constraint_violation_tolerance
            + stabilization_term
        )
        return total, (fim_term, constraint_term, component_vector)

    def trajectory_loss(trajectory_params, gains):
        decision_variables, free_offsets = trajectory_params
        totals, aux = jax.vmap(trajectory_terms, in_axes=(0, None, 0))(
            decision_variables, gains, offsets_from_free(free_offsets)
        )
        # Mean over trajectories: with per-trajectory offsets they are fully
        # independent problems, and Adam's per-coordinate normalization makes
        # the 1/T factor irrelevant to every coordinate anyway.
        return jnp.mean(totals), (totals,) + aux

    trajectory_optimizer = optax.adam(trajectory_learning_rate)

    @jax.jit
    def trajectory_step(trajectory_params, optimizer_state, gains, trust_radius):
        (loss_pre, aux), grads = jax.value_and_grad(trajectory_loss, has_aux=True)(
            trajectory_params, gains
        )
        updates, next_optimizer_state = trajectory_optimizer.update(
            grads, optimizer_state, trajectory_params
        )
        # Trust region: cap the *whole round's* movement, both blocks of the
        # decision vector together, at `trust_radius` in L2. Scaling the update
        # rather than clipping it per coordinate keeps the step's direction --
        # a shorter version of the move Adam wanted, not a different move.
        update_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(updates))
        )
        scale = jnp.minimum(1.0, trust_radius / jnp.maximum(update_norm, 1e-12))
        updates = jax.tree_util.tree_map(lambda leaf: leaf * scale, updates)
        decision_variables, free_offsets = optax.apply_updates(trajectory_params, updates)
        next_trajectory_params = (
            jax.vmap(trajectory_pipeline.clamp_decision_variables)(decision_variables),
            free_offsets,
        )
        loss_post, _ = trajectory_loss(next_trajectory_params, gains)
        return (
            next_trajectory_params, next_optimizer_state, loss_pre, loss_post, aux, update_norm
        )

    @jax.jit
    def trajectory_eval(trajectory_params, gains):
        """The trajectory diagnostics without a step, for rounds the schedule
        skips the trajectory block (sequential mode's gain phase). The history
        stays rectangular; the post-step entry is NaN so the uphill fraction
        never counts a round in which nothing moved."""
        return trajectory_loss(trajectory_params, gains)

    # --------------------------------------------------------------------- gains
    def gain_loss(gain_values, decision_variables, free_offsets):
        gains = controller_gains_from_optimizer_values(
            gain_values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
        )
        offsets = offsets_from_free(free_offsets)
        reference_trajectories = jax.vmap(reference_states_from_decision_variables)(
            decision_variables
        )

        def terms_for_reference(reference_states, start_offsets):
            return closed_loop_objective_terms(
                gain_pipeline,
                gains,
                realizations.robot_keys,
                realizations.estimator_keys,
                velocity_tracking_weight=velocity_tracking_weight,
                input_weight=input_weight,
                input_delta_weight=input_delta_weight,
                omega_delta_weight=omega_delta_weight,
                reference_states=reference_states,
                initial_pose_offsets=start_offsets,
            )

        return jnp.sum(
            jnp.mean(jax.vmap(terms_for_reference)(reference_trajectories, offsets), axis=0)
        )

    # `gain_steps_per_round` caps a bounded inner solve, not a step count: the
    # line search would otherwise make "one step" mean nothing. See
    # joint_tuning.gain_solvers.
    gain_optimizer_init, gain_step = make_gain_stepper(gain_loss, gain_steps_per_round)
    # The gain loss on whatever training trajectories it is handed, no step.
    gain_step_score = jax.jit(gain_loss)

    # ----------------------------------------------------------- validation
    # Its own frozen noise bundle, independent of the training realizations:
    # the validation set exists to be a different draw of the same question, and
    # sharing the training keys would make a gain vector that overfits this run's
    # particular noise look good on both. The start offsets come from the files.
    if validation_trajectories is None:
        validation_score = None
    else:
        validation_realizations = make_realizations(
            jax.random.PRNGKey(seed + 7717),
            jax.random.PRNGKey(seed + 7718),
            int(validation_trajectories[0].start_offsets.shape[0]),
            init_offset_radius,
            init_offset_angle,
        )

        def _validation_term(reference_states, start_offsets):
            def score(gain_values):
                gains = controller_gains_from_optimizer_values(
                    gain_values,
                    k_min_stab=k_min_stab,
                    k_max_stab=k_max_stab,
                    k_max_rest=k_max_rest,
                )
                return jnp.sum(
                    closed_loop_objective_terms(
                        gain_pipeline,
                        gains,
                        validation_realizations.robot_keys,
                        validation_realizations.estimator_keys,
                        velocity_tracking_weight=velocity_tracking_weight,
                        input_weight=input_weight,
                        input_delta_weight=input_delta_weight,
                        omega_delta_weight=omega_delta_weight,
                        reference_states=reference_states,
                        initial_pose_offsets=start_offsets,
                    )
                )
            # One jit per trajectory: they have different sample counts by
            # design, so a single traced function would re-trace anyway.
            return jax.jit(score)

        _validation_terms = [
            _validation_term(
                jnp.asarray(trajectory.reference_states, dtype=jnp.float32),
                jnp.asarray(trajectory.start_offsets, dtype=jnp.float32),
            )
            for trajectory in validation_trajectories
        ]

        def validation_score(gain_values):
            """Mean over the held-out trajectories. Unweighted: they are chosen
            to be diverse, so weighting by length or difficulty would let the
            longest curve decide what 'better gains' means."""
            return float(
                np.mean([float(term(gain_values)) for term in _validation_terms])
            )

    # ---------------------------------------------------------------------- init
    if warm_start is None:
        decision_variables = jnp.stack(
            trajectory_pipeline.initial_decision_variable_candidates(
                num_control_points=num_control_points,
                num_trajectories=num_trajectories,
                seed=seed,
            ),
            axis=0,
        )
    else:
        # The design's own control points, straight off the pickles: they are
        # the trajectory block's decision variables, so the loop resumes on
        # exactly the curve the export left off on.
        decision_variables = jax.vmap(
            lambda control_points: jnp.ravel(
                trajectory_pipeline.clamp_control_points(control_points)
            )
        )(jnp.asarray(warm_start.control_points, dtype=jnp.float32))
    gain_values = controller_gains_to_optimizer_values(
        trajectory_pipeline.controller_gains,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
    )
    trajectory_params = (decision_variables, initial_free_offsets)
    state = JointState(
        gain_values=gain_values,
        gain_opt_state=gain_optimizer_init(gain_values),
        decision_variables=decision_variables,
        free_offsets=initial_free_offsets,
        trajectory_opt_state=trajectory_optimizer.init(trajectory_params),
    )
    construct_seconds = time.time() - construct_start

    history = {
        name: []
        for name in (
            "gain_loss_pre", "gain_loss_post", "trajectory_loss_pre", "trajectory_loss_post",
            "fim_loss", "log_fim_loss", "constraint_loss", "max_fractional_violation", "gains",
            "constraint_components", "gain_validation_loss", "trust_radius",
        )
    }
    if verbose:
        print(
            f"Joint tuning [{mode}]: {len(schedule)} rounds "
            f"({sum(run_trajectory for _, run_trajectory in schedule)} trajectory steps, "
            f"{sum(run_gain for run_gain, _ in schedule)} gain steps), "
            f"{num_trajectories} trajectories x {num_control_points} control points, "
            f"{num_realizations} realizations, start offsets '{start_offset_mode}', "
            f"wheel_lp_tau {trajectory_pipeline.wheel_lp_tau:.4g} s."
        )
        print(
            f"  trajectory learning rate {trajectory_learning_rate:.3g}; "
            f"constraint tolerance {constraint_violation_tolerance:.3g}; "
            f"inner steps per round: trajectory {trajectory_steps_per_round}, "
            f"gain {gain_steps_per_round}."
        )
        if warm_start is not None:
            print(
                f"  warm start: {num_trajectories} designed trajectories from "
                f"{warm_start_trajectories_dir}, with their control points and start offsets "
                "(no warm-start rounds)."
            )

    loop_start = time.time()
    converged_at_round = None
    converged_reason = None
    window_reference_score = None
    gain_steps_taken = 0
    warm_start_decision_variables = state.decision_variables
    rejected_rounds = 0
    rounds_at_min_radius = 0
    trajectory_trace: list[JointRoundSnapshot] = []

    def take_snapshot(round_index, state, gains, gain_score):
        return JointRoundSnapshot(
            round_index=int(round_index),
            control_points=np.asarray(
                jax.vmap(trajectory_pipeline.control_points_from_decision_variables)(
                    state.decision_variables
                ),
                dtype=float,
            ),
            start_offsets=np.asarray(offsets_from_free(state.free_offsets), dtype=float),
            gains=np.asarray(gains, dtype=float),
            validation_loss=float(gain_score),
        )

    # The stationary yardstick, frozen at the round the gain block first runs
    # (after any warm start, so it is a designed set and not the raw initial
    # candidates). Every "is this gain vector better" question in the loop is
    # asked against these trajectories and offsets, never against the moving
    # ones, so the answers are comparable across rounds.
    scoring_decision_variables = None
    scoring_free_offsets = None
    best_gain_values = state.gain_values
    best_gain_score = float("inf")
    best_gain_round = None
    for round_index, (alternating, run_trajectory) in enumerate(schedule):
        if alternating and gain_steps_taken == 0:
            # The trajectories as the gain block first sees them: everything
            # after this point is what the alternation itself did.
            warm_start_decision_variables = state.decision_variables
            scoring_decision_variables = state.decision_variables
            scoring_free_offsets = state.free_offsets
        if alternating:
            # The stepper owns the inner budget; `gain_loss_pre`/`_post`
            # bracket everything this block did before handing over.
            (gain_values, gain_opt_state, gain_loss_pre, gain_loss_post) = gain_step(
                state.gain_values,
                state.gain_opt_state,
                state.decision_variables,
                state.free_offsets,
            )
        else:
            gain_values, gain_opt_state = state.gain_values, state.gain_opt_state
            gain_loss_pre = gain_loss_post = jnp.nan

        gains = controller_gains_from_optimizer_values(
            gain_values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
        )
        if run_trajectory:
            trajectory_params = (state.decision_variables, state.free_offsets)
            trajectory_opt_state = state.trajectory_opt_state
            step_radius = (
                trust_radius / max(trajectory_steps_per_round, 1)
                if trust_region_enabled else jnp.inf
            )
            for inner in range(trajectory_steps_per_round):
                (
                    trajectory_params,
                    trajectory_opt_state,
                    inner_pre,
                    trajectory_loss_post,
                    (per_trajectory_loss, fim_terms, constraint_terms, component_vectors),
                    update_norm,
                ) = trajectory_step(
                    trajectory_params, trajectory_opt_state, gains, step_radius
                )
                if inner == 0:
                    trajectory_loss_pre = inner_pre

            # The acceptance test, and the whole point of the mechanism: the
            # trajectory block's job is to stay informative about the gains, not
            # to make the tracking problem unsolvable. So the step is judged by
            # what it did to the *gain* block's objective at fixed gains. A move
            # that raises it sharply is the coupled basin jump this defends
            # against -- measured, one such round took the gain loss 0.0109 ->
            # 0.0502 and the loop never recovered. Judging the step by the
            # trajectory objective instead would accept it happily: that
            # objective improved through the jump.
            if trust_region_enabled and alternating:
                gain_loss_before = (
                    float(gain_loss_post) if np.isfinite(float(gain_loss_post))
                    else float(gain_step_score(
                        gain_values, state.decision_variables, state.free_offsets
                    ))
                )
                gain_loss_after = float(
                    gain_step_score(gain_values, trajectory_params[0], trajectory_params[1])
                )
                relative_rise = (gain_loss_after - gain_loss_before) / max(
                    abs(gain_loss_before), 1e-12
                )
                if relative_rise > trust_gain_loss_increase:
                    # Reject: keep the trajectories *and* the optimizer state, so
                    # the rejected direction is not baked into Adam's moments and
                    # re-proposed at full strength next round.
                    trajectory_params = (state.decision_variables, state.free_offsets)
                    trajectory_opt_state = state.trajectory_opt_state
                    trajectory_loss_post = trajectory_loss_pre
                    trust_radius = max(trust_radius * trust_shrink, min_trust_radius)
                    rejected_rounds += 1
                    if verbose:
                        print(
                            f"  round {round_index}: trajectory step rejected "
                            f"(gain loss +{relative_rise:.1%}), trust radius -> {trust_radius:.3g}"
                        )
                else:
                    trust_radius = min(trust_radius * trust_grow, max_trust_radius)
                    rounds_at_min_radius = 0
        else:
            trajectory_params = (state.decision_variables, state.free_offsets)
            trajectory_opt_state = state.trajectory_opt_state
            trajectory_loss_pre, (
                per_trajectory_loss,
                fim_terms,
                constraint_terms,
                component_vectors,
            ) = trajectory_eval(trajectory_params, gains)
            trajectory_loss_post = jnp.nan
        state = JointState(
            gain_values=gain_values,
            gain_opt_state=gain_opt_state,
            decision_variables=trajectory_params[0],
            free_offsets=trajectory_params[1],
            trajectory_opt_state=trajectory_opt_state,
        )

        # The comparable number in the loop, and the one that selects what
        # ships. `gain_loss_pre` (the *training* loss) is scored on trajectories
        # that moved last round, so it answers a different question every time.
        # With a validation directory this is a genuinely held-out set; without
        # one it falls back to the training trajectories frozen at the round the
        # gain block first ran, which is stationary but biased towards the gains
        # that suited that design.
        if alternating:
            gain_score = (
                validation_score(gain_values) if validation_score is not None
                else float(
                    gain_step_score(gain_values, scoring_decision_variables, scoring_free_offsets)
                )
            )
            if gain_score < best_gain_score:
                best_gain_score = gain_score
                best_gain_values = gain_values
                best_gain_round = round_index
        else:
            gain_score = float("nan")

        history["trust_radius"].append(float(trust_radius) if trust_region_enabled else float("nan"))
        history["gain_validation_loss"].append(gain_score)
        history["gain_loss_pre"].append(float(gain_loss_pre))
        history["gain_loss_post"].append(float(gain_loss_post))
        history["trajectory_loss_pre"].append(float(trajectory_loss_pre))
        history["trajectory_loss_post"].append(float(trajectory_loss_post))
        history["fim_loss"].append(np.asarray(fim_terms, dtype=float))
        history["log_fim_loss"].append(np.asarray(fim_terms, dtype=float))
        history["constraint_loss"].append(np.asarray(constraint_terms, dtype=float))
        history["max_fractional_violation"].append(
            np.asarray(_max_fractional_violation(component_vectors), dtype=float)
        )
        history["constraint_components"].append(np.asarray(component_vectors, dtype=float))
        history["gains"].append(np.asarray(gains, dtype=float))

        if trajectory_trace_stride > 0 and round_index % trajectory_trace_stride == 0:
            trajectory_trace.append(take_snapshot(round_index, state, gains, gain_score))

        if not np.isfinite(float(trajectory_loss_pre)):
            print(f"  round {round_index}: trajectory loss is not finite; stopping.")
            break
        if alternating and not np.isfinite(float(gain_loss_pre)):
            print(f"  round {round_index}: gain loss is not finite; stopping.")
            break

        if verbose and (
            round_index % max(1, len(schedule) // 20) == 0 or round_index == len(schedule) - 1
        ):
            gain_text = "warm start" if not alternating else f"{float(gain_loss_pre):.6f}"
            print(
                f"  round {round_index:>4}: gain loss {gain_text}, "
                f"trajectory loss {float(trajectory_loss_pre):.6f}, "
                f"mean fim_loss {float(jnp.mean(fim_terms)):.4e}, "
                f"max violation {float(jnp.max(_max_fractional_violation(component_vectors))):.4f}"
            )

        if trust_region_enabled:
            # "At the floor" with slack for one growth step, so an oscillating
            # collapse (grow, reject, grow, reject) still counts as collapsed.
            at_floor = trust_radius <= min_trust_radius * trust_grow
            rounds_at_min_radius = rounds_at_min_radius + 1 if at_floor else 0
            if trust_stall_rounds > 0 and rounds_at_min_radius >= trust_stall_rounds:
                converged_at_round = round_index
                converged_reason = "trust region collapsed"
                if verbose:
                    print(
                        f"  stopping after {round_index + 1} rounds: trust radius has sat at "
                        f"its floor for {rounds_at_min_radius} rounds -- every meaningful "
                        "trajectory step hurts the gain block."
                    )
                break

        # Stagnation rule on the *scoring* loss. Two departures from the
        # standalone trajectory optimizer's rule, both required here:
        #
        #  * it runs on the frozen-set score, because the standalone rule
        #    assumes a stationary objective and `gain_loss_pre` is not one;
        #  * it tests |improvement|, not improvement. The one-sided form treats
        #    a *rising* loss as convergence, which on this loop is the common
        #    case -- the trajectory block makes the gain problem harder -- and
        #    it is precisely wrong: a run whose score is moving is a run that
        #    has not settled. Rises are handled by keeping the best iterate, not
        #    by quitting on them.
        if alternating:
            gain_steps_taken += 1
            if window_reference_score is None:
                window_reference_score = gain_score
            elif convergence_rel_tol >= 0.0 and gain_steps_taken % convergence_window == 0:
                improvement = float(relative_improvement(window_reference_score, gain_score))
                stagnated = has_stagnated(window_reference_score, gain_score, convergence_rel_tol)
                window_reference_score = gain_score
                if stagnated:
                    converged_at_round = round_index
                    converged_reason = "validation loss stagnated"
                    if verbose:
                        print(
                            f"  stopping after {round_index + 1} rounds: validation loss moved "
                            f"{improvement:+.2e} (relative) over the last "
                            f"{convergence_window} gain rounds."
                        )
                    break
    loop_seconds = time.time() - loop_start

    # The last round always gets a frame, whatever the stride and however the
    # loop ended: an animation that stops short of the state the run actually
    # shipped is misleading.
    if trajectory_trace_stride > 0 and history["gains"] and (
        not trajectory_trace or trajectory_trace[-1].round_index != round_index
    ):
        trajectory_trace.append(take_snapshot(round_index, state, gains, gain_score))

    # `gains` is the best iterate on the frozen scoring set; the last iterate is
    # kept alongside as `final_gains` for diagnostics only. On a loop whose two
    # blocks keep moving each other's objective the last iterate is a tail
    # sample, not a result.
    gains = controller_gains_from_optimizer_values(
        best_gain_values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    final_gains = controller_gains_from_optimizer_values(
        state.gain_values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    control_points = jax.vmap(trajectory_pipeline.control_points_from_decision_variables)(
        state.decision_variables
    )
    reference_states = jax.vmap(trajectory_pipeline.reference_states_from_control_points)(
        control_points
    )
    start_offsets = offsets_from_free(state.free_offsets)

    rounds_run = len(history["gain_loss_pre"])
    history = {name: np.asarray(values) for name, values in history.items()}
    history["uphill_fraction_gain"] = _uphill_fraction(
        history["gain_loss_pre"], history["gain_loss_post"]
    )
    history["uphill_fraction_trajectory"] = _uphill_fraction(
        history["trajectory_loss_pre"], history["trajectory_loss_post"]
    )
    # The only measure that sees the alternation itself: the gain loss at the
    # start of a round against the gain loss at the start of the previous one,
    # i.e. including the trajectory move in between.
    finite_gain_loss = history["gain_loss_pre"][np.isfinite(history["gain_loss_pre"])]
    history["uphill_fraction_joint"] = (
        float(np.mean(np.diff(finite_gain_loss) > 0.0)) if finite_gain_loss.size > 1 else float("nan")
    )
    history["converged_at_round"] = converged_at_round
    history["converged_reason"] = converged_reason
    history["best_gain_round"] = best_gain_round
    history["rejected_rounds"] = rejected_rounds
    history["rounds_at_min_trust_radius"] = rounds_at_min_radius
    history["best_gain_score"] = best_gain_score

    if verbose:
        print(f"Best gains (round {best_gain_round}, validation loss {best_gain_score:.6f}): " + ", ".join(
            f"{name}={float(value):.6g}" for name, value in zip(GAIN_NAMES, gains)
        ))
        print(f"  last iterate: " + ", ".join(
            f"{name}={float(value):.6g}" for name, value in zip(GAIN_NAMES, final_gains)
        ))
        print(
            f"  uphill fractions -- gain {history['uphill_fraction_gain']:.4f}, "
            f"trajectory {history['uphill_fraction_trajectory']:.4f}, "
            f"joint {history['uphill_fraction_joint']:.4f}"
        )
        if trust_region_enabled:
            print(
                f"  trust region: {rejected_rounds} of {rounds_run} trajectory steps rejected, "
                f"final radius {trust_radius:.3g} (initial {max_trust_radius:.3g})."
            )
        print(f"  {rounds_run} rounds in {loop_seconds:.1f} s "
              f"({loop_seconds / max(rounds_run, 1):.3f} s/round).")

    return JointTuningResult(
        gains=gains,
        final_gains=final_gains,
        best_gain_score=best_gain_score,
        control_points=control_points,
        reference_states=reference_states,
        start_offsets=start_offsets,
        realizations=realizations,
        initial_decision_variables=decision_variables,
        warm_start_decision_variables=warm_start_decision_variables,
        state=state,
        trajectory_pipeline=trajectory_pipeline,
        gain_pipeline=gain_pipeline,
        trajectory_trace=tuple(trajectory_trace),
        history=history,
        config={
            "problem_path": problem_path,
            "mode": mode,
            "num_rounds": num_rounds,
            "rounds_scheduled": len(schedule),
            "warm_start_rounds": warm_start_rounds,
            "num_trajectories": num_trajectories,
            "num_control_points": num_control_points,
            "num_realizations": num_realizations,
            "trajectory_learning_rate": trajectory_learning_rate,
            "trust_radius": trust_radius,
            "initial_trust_radius": max_trust_radius,
            "trust_gain_loss_increase": trust_gain_loss_increase,
            "trust_stall_rounds": trust_stall_rounds,
            "gain_steps_per_round": gain_steps_per_round,
            "validation_trajectories_dir": validation_trajectories_dir,
            "num_validation_trajectories": (
                0 if validation_trajectories is None else len(validation_trajectories)
            ),
            "convergence_rel_tol": convergence_rel_tol,
            "convergence_window": convergence_window,
            "trajectory_steps_per_round": trajectory_steps_per_round,
            "trajectory_trace_stride": trajectory_trace_stride,
            "warm_start_trajectories_dir": warm_start_trajectories_dir,
            "start_offset_mode": start_offset_mode,
            "init_offset_radius": init_offset_radius,
            "init_offset_angle": init_offset_angle,
            "constraint_violation_tolerance": constraint_violation_tolerance,
            "criterion": criterion,
            "time_scaling": trajectory_pipeline.time_scaling,
            "seed": seed,
            "k_min_stab": k_min_stab,
            "k_max_stab": k_max_stab,
            "k_max_rest": k_max_rest,
            # Both sides roll out the same plant; recorded per run as provenance
            # (the encoder low-pass is always on, read from the problem config).
            "wheel_lp_tau": float(trajectory_pipeline.wheel_lp_tau),
            "gain_wheel_lp_tau": float(gain_pipeline.estimator.wheel_lp_tau),
        },
        timing={
            "construct_s": construct_seconds,
            "loop_s": loop_seconds,
            "rounds_run": rounds_run,
            "seconds_per_round": loop_seconds / max(rounds_run, 1),
        },
    )


def _uphill_fraction(loss_pre: np.ndarray, loss_post: np.ndarray) -> float:
    """Fraction of a block's own steps that raised its own objective, with the
    other block held fixed."""
    finite = np.isfinite(loss_pre) & np.isfinite(loss_post)
    if not finite.any():
        return float("nan")
    return float(np.mean(loss_post[finite] > loss_pre[finite]))
