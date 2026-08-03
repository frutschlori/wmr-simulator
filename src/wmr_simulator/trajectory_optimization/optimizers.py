"""Adam over the B-spline control points, with a relative-progress stopping rule."""

import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax_tqdm import scan_tqdm


# Stop when the loss has improved by less than CONVERGENCE_REL_TOL (relative)
# over the last CONVERGENCE_WINDOW steps -- progress slowed to a crawl, not
# merely a flat step. The comparison is against the loss a whole window ago
# rather than the previous step, because Adam oscillates and any single good
# step would otherwise reset the count. Measured tail rate on the tuning
# objective: ~2% per 50 steps at step 300, ~0.3% by step 1400, so 0.5% ends a
# run that has essentially stopped moving. This makes --opt-steps a safe upper
# bound rather than an exact setting.
#
# NB this is a threshold, not a switch: 0.0 does not disable the rule, it just
# demands *any* positive improvement, so a batch whose loss is rising (a
# diverging learning rate, say) still trips it. Stopping such a run is the right
# outcome, but do not read "converged" as "found an optimum". Use a negative
# value to disable it outright.
CONVERGENCE_REL_TOL = 5e-3
CONVERGENCE_WINDOW = 50


def relative_improvement(reference_loss, loss):
    return (reference_loss - loss) / jnp.maximum(jnp.abs(reference_loss), 1e-12)


def print_progress(step: int, total_steps: int, loss_value: float, bar_width: int = 30):
    if total_steps <= 0:
        return

    completed = int(bar_width * step / total_steps)
    bar = "=" * completed + "." * (bar_width - completed)
    sys.stdout.write(f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}")
    if step == total_steps:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _adam_with_step_scale(learning_rate: float, step_scale=None):
    """Adam, optionally with a per-coordinate multiplier on its update.

    Adam's update magnitude is ~learning_rate in every coordinate regardless of
    the gradient's scale, so multiplying a block of coordinates by ``f`` gives
    that block an effective learning rate of ``f * learning_rate`` and leaves
    the rest untouched. That is what lets the start offsets move at their own
    pace without disturbing a control-point learning rate that is known to work.
    """
    if step_scale is None:
        return optax.adam(learning_rate)
    return optax.chain(optax.adam(learning_rate), optax.scale_by_learning_rate(step_scale, flip_sign=False))


def optimize_control_points(
    pipeline,
    num_control_points: int,
    num_steps: int,
    learning_rate: float,
    initial_control_points: jax.Array | None = None,
    window_length: int | None = None,
    measurement_variances=None,
    save_trace: bool = False,
    trace_stride: int = 5,
    constraint_weight: float = 1.0,
    constraint_component_weights: dict | None = None,
    constraint_smooth_max_beta: float = 20.0,
    verbose: bool = True,
):
    from wmr_simulator.trajectory_optimization.pipeline import OptimizationSnapshot

    if initial_control_points is None:
        initial_control_points = pipeline.initial_control_points(num_control_points)
    else:
        initial_control_points = pipeline.clamp_control_points(initial_control_points)
    initial_decision_variables = pipeline.decision_variables_from_control_points(initial_control_points)
    optimizer = _adam_with_step_scale(
        learning_rate, pipeline.decision_variable_step_scale(initial_decision_variables.shape[0])
    )
    opt_state = optimizer.init(initial_decision_variables)

    # No round-0 loss rescaling: the objective normalizes itself (the FIM term
    # is in log units, the penalty in units of a tolerated fractional violation
    # -- see trajectory_objective). Stacking a 1/initial_loss factor on top
    # would double-normalize, and on a loss that can legitimately be negative it
    # is not even well defined.
    def loss_fn(decision_variables):
        return pipeline.loss_from_decision_variables(
            decision_variables,
            window_length=window_length,
            measurement_variances=measurement_variances,
            constraint_weight=constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
        )

    initial_loss = loss_fn(initial_decision_variables)

    @jax.jit
    def train_step(decision_variables, optimizer_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(decision_variables)
        updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
        next_decision_variables = pipeline.clamp_decision_variables(
            optax.apply_updates(decision_variables, updates)
        )
        return next_decision_variables, next_optimizer_state, loss_value

    decision_variables = initial_decision_variables
    best_decision_variables = initial_decision_variables
    best_loss = jnp.inf
    loss_history = []
    snapshots = [] if save_trace else None

    if save_trace:
        initial_closed_loop_log = pipeline.run_closed_loop_deployment(
            reference_states=pipeline.reference_states_from_control_points(initial_control_points)
        )
        snapshots.append(
            OptimizationSnapshot(
                step=0,
                loss_value=float(loss_fn(initial_decision_variables)),
                control_points=np.asarray(initial_control_points),
                reference_states=np.asarray(pipeline.reference_states_from_control_points(initial_control_points)),
                closed_loop_log=initial_closed_loop_log,
            )
        )

    if verbose:
        print(f"Initial loss: {float(initial_loss):.8f}")
    if num_steps <= 0:
        optimized_control_points = pipeline.control_points_from_decision_variables(decision_variables)
        pipeline.set_start_offsets(pipeline.start_offsets_from_decision_variables(decision_variables))
        pipeline.set_control_points(optimized_control_points)
        pipeline.loss_history = loss_history
        pipeline.optimization_snapshots = snapshots
        return optimized_control_points, loss_history

    window_reference_loss = float(initial_loss)
    for step in range(num_steps):
        # loss_value is measured at `decision_variables` *before* the update, so
        # that is the point it belongs to.
        scored_decision_variables = decision_variables
        decision_variables, opt_state, loss_value = train_step(decision_variables, opt_state)
        loss_history.append(float(loss_value))
        if float(loss_value) < float(best_loss):
            best_loss = float(loss_value)
            best_decision_variables = scored_decision_variables
        if verbose:
            print_progress(step + 1, num_steps, float(loss_value))
        if (step + 1) % CONVERGENCE_WINDOW == 0:
            improvement = float(relative_improvement(window_reference_loss, float(loss_value)))
            window_reference_loss = float(loss_value)
            if improvement <= CONVERGENCE_REL_TOL:
                if verbose:
                    print(f"\nConverged after {step + 1} steps ({improvement:.2e} relative "
                          f"improvement over the last {CONVERGENCE_WINDOW} steps).")
                break
        if save_trace and ((step + 1) % trace_stride == 0 or step + 1 == num_steps):
            control_points = pipeline.control_points_from_decision_variables(decision_variables)
            reference_states = pipeline.reference_states_from_control_points(control_points)
            closed_loop_log = pipeline.run_closed_loop_deployment(reference_states=reference_states)
            snapshots.append(
                OptimizationSnapshot(
                    step=step + 1,
                    loss_value=float(loss_value),
                    control_points=np.asarray(control_points),
                    reference_states=np.asarray(reference_states),
                    closed_loop_log=closed_loop_log,
                )
            )

    optimized_control_points = pipeline.control_points_from_decision_variables(best_decision_variables)
    # Offsets first: set_control_points rolls the deployment out, and it should
    # do so from the starts this run settled on.
    pipeline.set_start_offsets(pipeline.start_offsets_from_decision_variables(best_decision_variables))
    pipeline.set_control_points(optimized_control_points)
    pipeline.loss_history = loss_history
    pipeline.optimization_snapshots = snapshots
    return optimized_control_points, loss_history


def optimize_control_points_batch(
    pipeline,
    initial_decision_variables: jax.Array,
    num_steps: int,
    learning_rate: float,
    window_length: int | None = None,
    measurement_variances=None,
    constraint_weight: float = 1.0,
    constraint_component_weights: dict | None = None,
    constraint_smooth_max_beta: float = 20.0,
    verbose: bool = True,
):
    """Optimize a batch of trajectories, each with its own optimizer state.

    The trajectories are independent problems that only share a compiled step,
    so the vmap goes around a single trajectory's update. Each trajectory
    returns the best point it visited, not its last -- Adam spends a few percent
    of its steps going uphill, and there is no reason to ship one of them.
    """
    constraint_weights = jnp.broadcast_to(
        jnp.asarray(constraint_weight, dtype=jnp.float32),
        (initial_decision_variables.shape[0],),
    )
    num_control_points = pipeline.control_points_from_decision_variables(
        initial_decision_variables[0]
    ).shape[0]
    optimizer = _adam_with_step_scale(
        learning_rate, pipeline.decision_variable_step_scale(initial_decision_variables.shape[1])
    )

    # The objective normalizes itself (see trajectory_objective); there is no
    # round-0 loss_scale here either.
    def loss_fn(decision_variables, current_constraint_weight):
        return pipeline.loss_from_decision_variables(
            decision_variables,
            window_length=window_length,
            measurement_variances=measurement_variances,
            constraint_weight=current_constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
        )

    clamp_decision_variables_batch = jax.vmap(pipeline.clamp_decision_variables)
    control_points_from_decision_variables_batch = jax.vmap(pipeline.control_points_from_decision_variables)
    loss_values_fn = jax.vmap(loss_fn, in_axes=(0, 0))

    def step_one(decision_variables, optimizer_state, current_constraint_weight):
        loss_value, grads = jax.value_and_grad(loss_fn)(decision_variables, current_constraint_weight)
        updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
        return optax.apply_updates(decision_variables, updates), next_optimizer_state, loss_value

    step_batch = jax.vmap(step_one, in_axes=(0, 0, 0))

    def optimize_batch(initial_decision_variables, current_constraint_weights):
        initial_decision_variables = clamp_decision_variables_batch(initial_decision_variables)
        initial_loss = loss_values_fn(initial_decision_variables, current_constraint_weights)
        initial_opt_state = jax.vmap(optimizer.init)(initial_decision_variables)

        def train_step(carry, step_index):
            (decision_variables, optimizer_state, reference, improving, last_loss_values,
             best_decision_variables, best_loss_values) = carry
            # Scalar predicate, so this is a real branch rather than a masked
            # `select`: once every trajectory has stopped improving, the
            # remaining scan iterations skip the rollouts entirely.
            all_converged = jnp.all(improving <= 0.0)

            def run_step(_):
                next_decision_variables, next_optimizer_state, loss_values = step_batch(
                    decision_variables, optimizer_state, current_constraint_weights
                )
                next_decision_variables = clamp_decision_variables_batch(next_decision_variables)
                # At the end of each window, score the improvement against the
                # loss a full window ago and start a fresh window here.
                window_closed = (step_index + 1) % CONVERGENCE_WINDOW == 0
                improved = relative_improvement(reference, loss_values) > CONVERGENCE_REL_TOL
                # loss_values belongs to `decision_variables` (pre-update), so
                # that is the point recorded when it is the best seen so far.
                is_best = loss_values < best_loss_values
                return (next_decision_variables, next_optimizer_state,
                        jnp.where(window_closed, loss_values, reference),
                        jnp.where(window_closed, improved.astype(jnp.float32), improving),
                        loss_values,
                        jnp.where(is_best[:, None], decision_variables, best_decision_variables),
                        jnp.where(is_best, loss_values, best_loss_values))

            def hold(_):
                return (decision_variables, optimizer_state, reference, improving,
                        last_loss_values, best_decision_variables, best_loss_values)

            carry = jax.lax.cond(all_converged, hold, run_step, operand=None)
            return carry, (carry[4], all_converged)

        train_step = scan_tqdm(
            num_steps,
            desc=(
                f"B-spline optimization "
                f"({num_control_points} control points, {initial_decision_variables.shape[0]} trajectories)"
            ),
        )(train_step)

        if num_steps <= 0:
            return (initial_decision_variables,
                    jnp.empty((0, initial_decision_variables.shape[0]), dtype=jnp.float32),
                    jnp.zeros((0,), dtype=bool), initial_loss)

        (_, _, _, _, _, best_decision_variables, _), (loss_history, converged) = jax.lax.scan(
            train_step,
            (initial_decision_variables, initial_opt_state,
             initial_loss, jnp.ones_like(initial_loss),
             initial_loss,
             initial_decision_variables, jnp.full_like(initial_loss, jnp.inf)),
            jnp.arange(num_steps),
        )
        return best_decision_variables, loss_history, converged, initial_loss

    optimize_batch = jax.jit(optimize_batch)
    best_decision_variables, loss_history_by_trajectory, converged, initial_loss = optimize_batch(
        initial_decision_variables,
        constraint_weights,
    )
    optimized_control_points = control_points_from_decision_variables_batch(best_decision_variables)

    if verbose:
        print(f"Initial batch loss: {np.asarray(initial_loss, dtype=float)}")
    if num_steps <= 0:
        return optimized_control_points, [], best_decision_variables

    # Trim the flat tail the short-circuit leaves behind, so the history plot
    # shows only the steps that did something.
    converged = np.asarray(converged, dtype=bool)
    steps_run = int(np.argmax(converged)) if converged.any() else len(converged)
    if verbose and converged.any():
        print(f"All trajectories converged after {steps_run} of {num_steps} steps.")
    loss_history = np.asarray(loss_history_by_trajectory, dtype=float)[:max(steps_run, 1)].tolist()

    return optimized_control_points, loss_history, best_decision_variables
