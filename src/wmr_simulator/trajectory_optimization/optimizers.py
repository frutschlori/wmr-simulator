import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax_tqdm import scan_tqdm


def print_progress(step: int, total_steps: int, loss_value: float, bar_width: int = 30):
    if total_steps <= 0:
        return

    completed = int(bar_width * step / total_steps)
    bar = "=" * completed + "." * (bar_width - completed)
    sys.stdout.write(f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}")
    if step == total_steps:
        sys.stdout.write("\n")
    sys.stdout.flush()


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
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_decision_variables)

    def unnormalized_loss_fn(decision_variables):
        control_points = pipeline.control_points_from_decision_variables(decision_variables)
        return pipeline.fim_loss_from_control_points(
            control_points,
            window_length=window_length,
            measurement_variances=measurement_variances,
            constraint_weight=constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
        )

    initial_loss = unnormalized_loss_fn(initial_decision_variables)
    loss_scale = 1.0 / jnp.maximum(initial_loss, 1e-12)
    # loss_scale = 1.0

    def loss_fn(decision_variables):
        return loss_scale * unnormalized_loss_fn(decision_variables)

    @jax.jit
    def train_step(decision_variables, optimizer_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(decision_variables)
        updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
        next_decision_variables = optax.apply_updates(decision_variables, updates)
        next_control_points = pipeline.control_points_from_decision_variables(next_decision_variables)
        next_decision_variables = pipeline.decision_variables_from_control_points(next_control_points)
        return next_decision_variables, next_optimizer_state, loss_value

    decision_variables = initial_decision_variables
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
        print(f"Initial unnormalized loss: {float(initial_loss):.8f}")
        print(f"Loss normalization scale: {float(loss_scale):.8f}")
    if num_steps <= 0:
        optimized_control_points = pipeline.control_points_from_decision_variables(decision_variables)
        pipeline.set_control_points(optimized_control_points)
        pipeline.loss_history = loss_history
        pipeline.optimization_snapshots = snapshots
        return optimized_control_points, loss_history

    for step in range(num_steps):
        decision_variables, opt_state, loss_value = train_step(decision_variables, opt_state)
        loss_history.append(float(loss_value))
        if verbose:
            print_progress(step + 1, num_steps, float(loss_value))
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

    optimized_control_points = pipeline.control_points_from_decision_variables(decision_variables)
    pipeline.set_control_points(optimized_control_points)
    pipeline.loss_history = loss_history
    pipeline.optimization_snapshots = snapshots
    return optimized_control_points, loss_history


def optimize_control_points_batch(
    pipeline,
    initial_control_points: jax.Array,
    num_steps: int,
    learning_rate: float,
    window_length: int | None = None,
    measurement_variances=None,
    constraint_weight: float = 1.0,
    constraint_component_weights: dict | None = None,
    constraint_smooth_max_beta: float = 20.0,
    verbose: bool = True,
):
    constraint_weights = jnp.broadcast_to(
        jnp.asarray(constraint_weight, dtype=jnp.float32),
        (initial_control_points.shape[0],),
    )
    optimizer = optax.adam(learning_rate)

    def unnormalized_loss_fn(decision_variables, current_constraint_weight):
        control_points = pipeline.control_points_from_decision_variables(decision_variables)
        return pipeline.fim_loss_from_control_points(
            control_points,
            window_length=window_length,
            measurement_variances=measurement_variances,
            constraint_weight=current_constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
        )

    def scaled_loss_fn(decision_variables, scale, current_constraint_weight):
        return scale * unnormalized_loss_fn(decision_variables, current_constraint_weight)

    decision_variables_from_control_points_batch = jax.vmap(pipeline.decision_variables_from_control_points)
    control_points_from_decision_variables_batch = jax.vmap(pipeline.control_points_from_decision_variables)
    loss_values_fn = jax.vmap(scaled_loss_fn, in_axes=(0, 0, 0))
    unnormalized_loss_values_fn = jax.vmap(unnormalized_loss_fn, in_axes=(0, 0))

    def optimize_batch(control_points, current_constraint_weights):
        initial_decision_variables = decision_variables_from_control_points_batch(control_points)
        initial_loss = unnormalized_loss_values_fn(initial_decision_variables, current_constraint_weights)
        loss_scale = 1.0 / jnp.maximum(initial_loss, 1e-12)
        initial_opt_state = optimizer.init(initial_decision_variables)

        def summed_loss_fn(decision_variables):
            loss_values = loss_values_fn(decision_variables, loss_scale, current_constraint_weights)
            return jnp.sum(loss_values), loss_values

        def train_step(carry, _):
            decision_variables, optimizer_state = carry
            (_, loss_values), grads = jax.value_and_grad(summed_loss_fn, has_aux=True)(decision_variables)
            updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
            next_decision_variables = optax.apply_updates(decision_variables, updates)
            next_control_points = control_points_from_decision_variables_batch(next_decision_variables)
            next_decision_variables = decision_variables_from_control_points_batch(next_control_points)
            return (next_decision_variables, next_optimizer_state), loss_values

        train_step = scan_tqdm(
            num_steps,
            desc=(
                f"B-spline optimization "
                f"({initial_control_points.shape[1]} control points, {initial_control_points.shape[0]} trajectories)"
            ),
        )(train_step)

        if num_steps <= 0:
            return control_points, jnp.empty((0, control_points.shape[0]), dtype=jnp.float32), initial_loss, loss_scale

        (final_decision_variables, _), loss_history = jax.lax.scan(
            train_step,
            (initial_decision_variables, initial_opt_state),
            jnp.arange(num_steps),
        )
        final_control_points = control_points_from_decision_variables_batch(final_decision_variables)
        return final_control_points, loss_history, initial_loss, loss_scale

    optimize_batch = jax.jit(optimize_batch)
    optimized_control_points, loss_history_by_trajectory, initial_loss, loss_scale = optimize_batch(
        initial_control_points,
        constraint_weights,
    )

    if verbose:
        print(f"Initial unnormalized batch loss: {np.asarray(initial_loss, dtype=float)}")
        print(f"Batch loss normalization scale: {np.asarray(loss_scale, dtype=float)}")
    if num_steps <= 0:
        return optimized_control_points, []

    loss_history = np.asarray(loss_history_by_trajectory, dtype=float).tolist()

    return optimized_control_points, loss_history
