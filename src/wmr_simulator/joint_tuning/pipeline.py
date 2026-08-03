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
from wmr_simulator.trajectory_optimization.start_offsets import (
    START_OFFSET_MODE_STATIC,
    inverse_squash_start_offsets,
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

    ``gain_values`` are optimizer-space (log/sqrt) values, not gains, and
    ``free_offsets`` are the unconstrained pre-squash start-offset variables --
    both blocks' search spaces, not their physical readings.
    """

    gain_values: jax.Array              # (5,)
    gain_opt_state: optax.OptState
    decision_variables: jax.Array       # (T, 2K)
    free_offsets: jax.Array             # (R, 3)
    trajectory_opt_state: optax.OptState


class JointTuningResult(NamedTuple):
    gains: jax.Array                    # (5,)
    control_points: jax.Array           # (T, K, 2)
    reference_states: jax.Array         # (T, N, 8)
    start_offsets: jax.Array            # (R, 3)
    realizations: Realizations
    initial_decision_variables: jax.Array       # (T, 2K) before any step
    warm_start_decision_variables: jax.Array    # (T, 2K) when the gain block first ran
    state: JointState                   # both Adam states, as the loop left them
    trajectory_pipeline: TrajectoryOptimizationPipeline
    gain_pipeline: ControllerTuningPipeline
    history: dict
    config: dict
    timing: dict


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
    gain_learning_rate: float = 1e-4,
    start_offset_mode: str = "random",
    init_offset_radius: float = float(GAIN_TUNING_DEFAULTS["init_offset_radius"]),
    init_offset_angle: float = float(GAIN_TUNING_DEFAULTS["init_offset_angle"]),
    constraint_violation_tolerance: float = DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
    constraint_smooth_max_beta: float = 20.0,
    criterion: str = DEFAULT_CRITERION,
    wheel_lp_tau: float | None = None,
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
    start_offset_mode = normalize_start_offset_mode(start_offset_mode)
    mode = normalize_mode(mode)
    criterion = normalize_criterion(criterion)
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
        wheel_lp_tau=wheel_lp_tau,
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
    frozen_offsets = (
        static_start_offsets(num_realizations, init_offset_radius, init_offset_angle)
        if start_offset_mode == START_OFFSET_MODE_STATIC
        else realizations.start_offsets
    )
    # The optimizing modes start *at* the frozen offsets rather than at zero, so
    # every mode begins under the same conditions and only their evolution differs.
    initial_free_offsets = inverse_squash_start_offsets(
        frozen_offsets, init_offset_radius, init_offset_angle
    )

    def offsets_from_free(free_offsets):
        return resolve_start_offsets(
            free_offsets, frozen_offsets, mask, init_offset_radius, init_offset_angle
        )

    def realizations_from_free(free_offsets):
        return realizations._replace(start_offsets=offsets_from_free(free_offsets))

    def reference_states_from_decision_variables(decision_variables):
        return trajectory_pipeline.reference_states_from_control_points(
            trajectory_pipeline.control_points_from_decision_variables(decision_variables)
        )

    # ---------------------------------------------------------------- trajectory
    def trajectory_terms(decision_variables, gains, free_offsets):
        """One trajectory's objective plus the decomposition, in one rollout set.

        This is ``trajectory_objective`` written out so the FIM and constraint
        terms fall out as aux instead of costing a second rollout batch; the
        tests pin it against ``fim_loss_from_control_points``.
        """
        reference_states = reference_states_from_decision_variables(decision_variables)
        fim_factor = trajectory_pipeline.compute_fim_factor(
            reference_states=reference_states,
            gains=gains,
            realizations=realizations_from_free(free_offsets),
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
        total = (
            fim_objective_term(fim_factor, criterion)
            + constraint_term / constraint_violation_tolerance
        )
        return total, (fim_term, constraint_term, component_vector)

    def trajectory_loss(trajectory_params, gains):
        decision_variables, free_offsets = trajectory_params
        totals, aux = jax.vmap(trajectory_terms, in_axes=(0, None, None))(
            decision_variables, gains, free_offsets
        )
        # Mean over trajectories: they are independent problems sharing only the
        # offsets, and Adam's per-coordinate normalization makes the 1/T factor
        # irrelevant to the control points while giving the shared offsets the
        # average of the per-trajectory gradients.
        return jnp.mean(totals), (totals,) + aux

    trajectory_optimizer = optax.adam(trajectory_learning_rate)

    @jax.jit
    def trajectory_step(trajectory_params, optimizer_state, gains):
        (loss_pre, aux), grads = jax.value_and_grad(trajectory_loss, has_aux=True)(
            trajectory_params, gains
        )
        updates, next_optimizer_state = trajectory_optimizer.update(
            grads, optimizer_state, trajectory_params
        )
        decision_variables, free_offsets = optax.apply_updates(trajectory_params, updates)
        next_trajectory_params = (
            jax.vmap(trajectory_pipeline.clamp_decision_variables)(decision_variables),
            free_offsets,
        )
        loss_post, _ = trajectory_loss(next_trajectory_params, gains)
        return next_trajectory_params, next_optimizer_state, loss_pre, loss_post, aux

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

        def terms_for_reference(reference_states):
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
                initial_pose_offsets=offsets,
            )

        return jnp.sum(jnp.mean(jax.vmap(terms_for_reference)(reference_trajectories), axis=0))

    gain_optimizer = optax.adam(gain_learning_rate)

    @jax.jit
    def gain_step(gain_values, optimizer_state, decision_variables, free_offsets):
        loss_pre, grads = jax.value_and_grad(gain_loss)(
            gain_values, decision_variables, free_offsets
        )
        updates, next_optimizer_state = gain_optimizer.update(grads, optimizer_state, gain_values)
        next_gain_values = clip_optimizer_values(optax.apply_updates(gain_values, updates))
        loss_post = gain_loss(next_gain_values, decision_variables, free_offsets)
        return next_gain_values, next_optimizer_state, loss_pre, loss_post

    # ---------------------------------------------------------------------- init
    decision_variables = jnp.stack(
        trajectory_pipeline.initial_decision_variable_candidates(
            num_control_points=num_control_points,
            num_trajectories=num_trajectories,
            seed=seed,
        ),
        axis=0,
    )
    gain_values = controller_gains_to_optimizer_values(
        trajectory_pipeline.controller_gains,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
    )
    trajectory_params = (decision_variables, initial_free_offsets)
    state = JointState(
        gain_values=gain_values,
        gain_opt_state=gain_optimizer.init(gain_values),
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
            "constraint_components",
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
            f"  learning rates: trajectory {trajectory_learning_rate:.3g}, "
            f"gain {gain_learning_rate:.3g}; constraint tolerance {constraint_violation_tolerance:.3g}."
        )

    loop_start = time.time()
    converged_at_round = None
    window_reference_loss = None
    previous_gain_loss = None
    gain_steps_taken = 0
    warm_start_decision_variables = state.decision_variables
    for round_index, (alternating, run_trajectory) in enumerate(schedule):
        if alternating and gain_steps_taken == 0:
            # The trajectories as the gain block first sees them: everything
            # after this point is what the alternation itself did.
            warm_start_decision_variables = state.decision_variables
        if alternating:
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
            (
                trajectory_params,
                trajectory_opt_state,
                trajectory_loss_pre,
                trajectory_loss_post,
                (per_trajectory_loss, fim_terms, constraint_terms, component_vectors),
            ) = trajectory_step(
                (state.decision_variables, state.free_offsets), state.trajectory_opt_state, gains
            )
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

        # Same stopping rule as the standalone trajectory optimizer, applied to
        # the gain loss: progress slowed to a crawl over a whole window, rather
        # than a single flat round. Only meaningful once the gains actually move.
        if alternating:
            gain_steps_taken += 1
            previous_gain_loss = float(gain_loss_pre)
            if window_reference_loss is None:
                window_reference_loss = previous_gain_loss
            elif gain_steps_taken % CONVERGENCE_WINDOW == 0:
                improvement = float(
                    relative_improvement(window_reference_loss, previous_gain_loss)
                )
                window_reference_loss = previous_gain_loss
                if improvement <= CONVERGENCE_REL_TOL:
                    converged_at_round = round_index
                    if verbose:
                        print(
                            f"  converged after {round_index + 1} rounds "
                            f"({improvement:.2e} relative gain-loss improvement over the "
                            f"last {CONVERGENCE_WINDOW} rounds)."
                        )
                    break
    loop_seconds = time.time() - loop_start

    gains = controller_gains_from_optimizer_values(
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

    if verbose:
        print(f"Final gains: " + ", ".join(
            f"{name}={float(value):.6g}" for name, value in zip(GAIN_NAMES, gains)
        ))
        print(
            f"  uphill fractions -- gain {history['uphill_fraction_gain']:.4f}, "
            f"trajectory {history['uphill_fraction_trajectory']:.4f}, "
            f"joint {history['uphill_fraction_joint']:.4f}"
        )
        print(f"  {rounds_run} rounds in {loop_seconds:.1f} s "
              f"({loop_seconds / max(rounds_run, 1):.3f} s/round).")

    return JointTuningResult(
        gains=gains,
        control_points=control_points,
        reference_states=reference_states,
        start_offsets=start_offsets,
        realizations=realizations._replace(start_offsets=start_offsets),
        initial_decision_variables=decision_variables,
        warm_start_decision_variables=warm_start_decision_variables,
        state=state,
        trajectory_pipeline=trajectory_pipeline,
        gain_pipeline=gain_pipeline,
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
            "gain_learning_rate": gain_learning_rate,
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
            # Both sides now roll out the same plant. Recorded per run because
            # the LP-on/LP-off ablation is exactly a comparison over this field.
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
