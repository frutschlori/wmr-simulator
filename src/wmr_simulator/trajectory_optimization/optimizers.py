"""Adam over the B-spline control points, with a relative-progress stopping rule."""

import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax_tqdm import scan_tqdm


# Stop when the loss improved by less than CONVERGENCE_REL_TOL (relative) over
# the last CONVERGENCE_WINDOW steps (a negative tolerance disables the rule).
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
    """Adam, optionally with a per-coordinate multiplier on its update
    (an effective per-block learning rate)."""
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
    constraint_weight: float = 1.0,
    constraint_component_weights: dict | None = None,
    constraint_smooth_max_beta: float = 20.0,
    motion_floor=(0.0, 0.0),
    verbose: bool = True,
):
    """Optimize one trajectory with Adam; returns the best visited control points."""
    if initial_control_points is None:
        initial_control_points = pipeline.initial_control_points(num_control_points)
    else:
        initial_control_points = pipeline.clamp_control_points(initial_control_points)
    initial_decision_variables = pipeline.decision_variables_from_control_points(initial_control_points)
    optimizer = _adam_with_step_scale(
        learning_rate, pipeline.decision_variable_step_scale(initial_decision_variables.shape[0])
    )
    opt_state = optimizer.init(initial_decision_variables)

    def loss_fn(decision_variables):
        return pipeline.loss_from_decision_variables(
            decision_variables,
            window_length=window_length,
            measurement_variances=measurement_variances,
            constraint_weight=constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            motion_floor=motion_floor,
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

    if verbose:
        print(f"Initial loss: {float(initial_loss):.8f}")
    if num_steps <= 0:
        optimized_control_points = pipeline.control_points_from_decision_variables(decision_variables)
        pipeline.set_start_offsets(pipeline.start_offsets_from_decision_variables(decision_variables))
        pipeline.set_control_points(optimized_control_points)
        pipeline.loss_history = loss_history
        return optimized_control_points, loss_history

    window_reference_loss = float(initial_loss)
    for step in range(num_steps):
        # loss_value belongs to the point before the update.
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

    optimized_control_points = pipeline.control_points_from_decision_variables(best_decision_variables)
    # Offsets first: set_control_points rolls the deployment out from them.
    pipeline.set_start_offsets(pipeline.start_offsets_from_decision_variables(best_decision_variables))
    pipeline.set_control_points(optimized_control_points)
    pipeline.loss_history = loss_history
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
    motion_floor=(0.0, 0.0),
    realizations_batch=None,
    verbose: bool = True,
):
    """Optimize a batch of independent trajectories (vmapped, each with its
    own optimizer state); each returns the best point it visited."""
    constraint_weights = jnp.broadcast_to(
        jnp.asarray(constraint_weight, dtype=jnp.float32),
        (initial_decision_variables.shape[0],),
    )
    # Per-trajectory [v_min, a_lat_min] floors.
    motion_floors = jnp.broadcast_to(
        jnp.asarray(motion_floor, dtype=jnp.float32),
        (initial_decision_variables.shape[0], 2),
    )
    num_control_points = pipeline.control_points_from_decision_variables(
        initial_decision_variables[0]
    ).shape[0]
    optimizer = _adam_with_step_scale(
        learning_rate, pipeline.decision_variable_step_scale(initial_decision_variables.shape[1])
    )

    if realizations_batch is None:
        def loss_fn(decision_variables, current_constraint_weight, current_motion_floor):
            return pipeline.loss_from_decision_variables(
                decision_variables,
                window_length=window_length,
                measurement_variances=measurement_variances,
                constraint_weight=current_constraint_weight,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=constraint_smooth_max_beta,
                motion_floor=current_motion_floor,
            )
    else:
        def loss_fn(decision_variables, current_constraint_weight, current_motion_floor, realizations):
            return pipeline.loss_from_decision_variables(
                decision_variables,
                window_length=window_length,
                measurement_variances=measurement_variances,
                constraint_weight=current_constraint_weight,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=constraint_smooth_max_beta,
                motion_floor=current_motion_floor,
                realizations=realizations,
            )

    clamp_decision_variables_batch = jax.vmap(pipeline.clamp_decision_variables)
    control_points_from_decision_variables_batch = jax.vmap(pipeline.control_points_from_decision_variables)
    if realizations_batch is None:
        loss_values_fn = jax.vmap(loss_fn, in_axes=(0, 0, 0))

        def step_one(decision_variables, optimizer_state, current_constraint_weight, current_motion_floor):
            loss_value, grads = jax.value_and_grad(loss_fn)(
                decision_variables, current_constraint_weight, current_motion_floor
            )
            updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
            return optax.apply_updates(decision_variables, updates), next_optimizer_state, loss_value

        step_batch = jax.vmap(step_one, in_axes=(0, 0, 0, 0))
    else:
        loss_values_fn = jax.vmap(loss_fn, in_axes=(0, 0, 0, 0))

        def step_one(decision_variables, optimizer_state, current_constraint_weight,
                     current_motion_floor, realizations):
            loss_value, grads = jax.value_and_grad(loss_fn)(
                decision_variables, current_constraint_weight, current_motion_floor, realizations
            )
            updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
            return optax.apply_updates(decision_variables, updates), next_optimizer_state, loss_value

        step_batch = jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0))

    def optimize_batch(initial_decision_variables, current_constraint_weights, current_motion_floors,
                       realizations_batch):
        initial_decision_variables = clamp_decision_variables_batch(initial_decision_variables)
        initial_loss = (
            loss_values_fn(initial_decision_variables, current_constraint_weights, current_motion_floors)
            if realizations_batch is None
            else loss_values_fn(initial_decision_variables, current_constraint_weights,
                                current_motion_floors, realizations_batch)
        )
        initial_opt_state = jax.vmap(optimizer.init)(initial_decision_variables)

        def train_step(carry, step_index):
            (decision_variables, optimizer_state, reference, improving, last_loss_values,
             best_decision_variables, best_loss_values) = carry
            # Once every trajectory has converged, skip the remaining rollouts.
            all_converged = jnp.all(improving <= 0.0)

            def run_step(_):
                if realizations_batch is None:
                    next_decision_variables, next_optimizer_state, loss_values = step_batch(
                        decision_variables, optimizer_state, current_constraint_weights, current_motion_floors
                    )
                else:
                    next_decision_variables, next_optimizer_state, loss_values = step_batch(
                        decision_variables, optimizer_state, current_constraint_weights,
                        current_motion_floors, realizations_batch
                    )
                next_decision_variables = clamp_decision_variables_batch(next_decision_variables)
                # Score the improvement at the end of each window.
                window_closed = (step_index + 1) % CONVERGENCE_WINDOW == 0
                improved = relative_improvement(reference, loss_values) > CONVERGENCE_REL_TOL
                # loss_values belongs to the pre-update point.
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
        motion_floors,
        realizations_batch,
    )
    optimized_control_points = control_points_from_decision_variables_batch(best_decision_variables)

    if verbose:
        print(f"Initial batch loss: {np.asarray(initial_loss, dtype=float)}")
    if num_steps <= 0:
        return optimized_control_points, [], best_decision_variables

    # Trim the steps after convergence.
    converged = np.asarray(converged, dtype=bool)
    steps_run = int(np.argmax(converged)) if converged.any() else len(converged)
    if verbose and converged.any():
        print(f"All trajectories converged after {steps_run} of {num_steps} steps.")
    loss_history = np.asarray(loss_history_by_trajectory, dtype=float)[:max(steps_run, 1)].tolist()

    return optimized_control_points, loss_history, best_decision_variables
