import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax


def print_progress(step: int, total_steps: int, loss_value: float, bar_width: int = 30):
    if total_steps <= 0:
        return

    completed = int(bar_width * step / total_steps)
    bar = "=" * completed + "." * (bar_width - completed)
    sys.stdout.write(f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}")
    if step == total_steps:
        sys.stdout.write("\n")
    sys.stdout.flush()


def optimize_bezier_control_points(
    pipeline,
    order: int,
    num_steps: int,
    learning_rate: float,
    window_length: int | None = None,
    measurement_variances=None,
    save_trace: bool = False,
    trace_stride: int = 5,
    constraint_weight: float = 1.0,
    constraint_component_weights: dict | None = None,
    constraint_smooth_max_beta: float = 20.0,
    constraint_smooth_violation_alpha: float = 20.0,
):
    from wmr_simulator.trajectory_optimization.pipeline import OptimizationSnapshot

    initial_control_points = pipeline.initial_bezier_control_points(order)
    initial_decision_variables = initial_control_points[1:]
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
            constraint_smooth_violation_alpha=constraint_smooth_violation_alpha,
        )

    initial_loss = unnormalized_loss_fn(initial_decision_variables)
    loss_scale = 1.0 / jnp.maximum(initial_loss, 1e-12)
    # loss_scale = -1.0 / jnp.maximum(initial_loss, 1e-12)
    # loss_scale = 1.0

    def loss_fn(decision_variables):
        return loss_scale * unnormalized_loss_fn(decision_variables)

    @jax.jit
    def train_step(decision_variables, optimizer_state):
        loss_value, grads = jax.value_and_grad(loss_fn)(decision_variables)
        updates, next_optimizer_state = optimizer.update(grads, optimizer_state, decision_variables)
        next_decision_variables = optax.apply_updates(decision_variables, updates)
        next_control_points = pipeline.control_points_from_decision_variables(next_decision_variables)
        next_decision_variables = next_control_points[1:]
        return next_decision_variables, next_optimizer_state, loss_value

    decision_variables = initial_decision_variables
    loss_history = []
    snapshots = [] if save_trace else None

    if save_trace:
        initial_closed_loop_log = pipeline.run_closed_loop_deployment(
            reference_states=pipeline.bezier_reference_sequence(initial_control_points)
        )
        snapshots.append(
            OptimizationSnapshot(
                step=0,
                loss_value=float(loss_fn(initial_decision_variables)),
                control_points=np.asarray(initial_control_points),
                reference_states=np.asarray(pipeline.bezier_reference_sequence(initial_control_points)),
                closed_loop_log=initial_closed_loop_log,
            )
        )

    print(f"Initial unnormalized loss: {float(initial_loss):.8f}")
    print(f"Loss normalization scale: {float(loss_scale):.8f}")
    if num_steps <= 0:
        optimized_control_points = pipeline.control_points_from_decision_variables(decision_variables)
        pipeline.set_bezier_control_points(optimized_control_points)
        pipeline.loss_history = loss_history
        pipeline.optimization_snapshots = snapshots
        return optimized_control_points, loss_history

    for step in range(num_steps):
        decision_variables, opt_state, loss_value = train_step(decision_variables, opt_state)
        loss_history.append(float(loss_value))
        print_progress(step + 1, num_steps, float(loss_value))
        if save_trace and ((step + 1) % trace_stride == 0 or step + 1 == num_steps):
            control_points = pipeline.control_points_from_decision_variables(decision_variables)
            reference_states = pipeline.bezier_reference_sequence(control_points)
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
    pipeline.set_bezier_control_points(optimized_control_points)
    pipeline.loss_history = loss_history
    pipeline.optimization_snapshots = snapshots
    return optimized_control_points, loss_history
