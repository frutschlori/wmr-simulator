import argparse
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.system_identification import PhysicalParams, SystemIdentificationPipeline
from wmr_simulator.SI_visualization import (
    plot_system_id_realization_sweep,
    plot_tracking_error_surface,
)


def parameter_mse(estimated_params: PhysicalParams, hidden_params: PhysicalParams) -> float:
    estimated = jnp.asarray(
        [estimated_params.wheel_radius, estimated_params.base_diameter], dtype=jnp.float32
    )
    hidden = jnp.asarray(
        [hidden_params.wheel_radius, hidden_params.base_diameter], dtype=jnp.float32
    )
    return float(jnp.mean((estimated - hidden) ** 2))


def evaluate_seed(problem_path, init_params, steps, learning_rate, num_realizations, seed):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=init_params,
        seed=seed,
    )

    estimated_params, loss_history = pipeline.optimize(
        init_params=init_params,
        num_steps=steps,
        learning_rate=learning_rate,
        num_realizations=num_realizations,
    )
    final_tracking_loss = float(loss_history[-1])
    final_param_mse = parameter_mse(estimated_params, pipeline.hidden_params)
    return final_tracking_loss, final_param_mse, estimated_params


def evaluate_seed_task(task):
    return evaluate_seed(*task)


def build_realization_sweep_values(max_num_realizations: int) -> list[int]:
    if max_num_realizations <= 0:
        raise ValueError("max_num_realizations must be positive.")

    sweep_values = [2 ** i for i in range(int(jnp.floor(jnp.log2(max_num_realizations))) + 1)]
    if sweep_values[-1] != max_num_realizations:
        sweep_values.append(max_num_realizations)
    return sweep_values


def run_realization_sweep(args):
    sweep_values = build_realization_sweep_values(args.max_num_realizations)
    seeds = [args.seed + offset for offset in range(args.num_seeds)]
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
    )

    tracking_loss_means = []
    param_mse_means = []

    for num_realizations in sweep_values:
        print(
            f"\nRunning system ID with num_realizations={num_realizations} "
            f"averaged over {args.num_seeds} seeds"
        )
        tasks = [
            (
                args.problem,
                init_params,
                args.steps,
                args.learning_rate,
                num_realizations,
                seed,
            )
            for seed in seeds
        ]

        if args.num_workers == 1:
            results = [evaluate_seed_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                results = list(executor.map(evaluate_seed_task, tasks))

        tracking_losses = []
        param_mse_values = []

        for seed, (final_tracking_loss, final_param_mse, estimated_params) in zip(seeds, results):
            tracking_losses.append(final_tracking_loss)
            param_mse_values.append(final_param_mse)

            print(
                f"  seed={seed:>3}  "
                f"final_tracking_loss={final_tracking_loss:.8f}  "
                f"parameter_mse={final_param_mse:.8f}  "
                f"estimated_radius={float(estimated_params.wheel_radius):.6f}  "
                f"estimated_base={float(estimated_params.base_diameter):.6f}"
            )

        tracking_loss_mean = float(np.mean(tracking_losses))
        param_mse_mean = float(np.mean(param_mse_values))
        tracking_loss_means.append(tracking_loss_mean)
        param_mse_means.append(param_mse_mean)

        print(
            f"  mean_tracking_loss={tracking_loss_mean:.8f}  "
            f"mean_parameter_mse={param_mse_mean:.8f}"
        )

    plot_system_id_realization_sweep(
        num_realizations=sweep_values,
        tracking_losses=tracking_loss_means,
        parameter_mse=param_mse_means,
        num_seeds=args.num_seeds,
        out_prefix=args.output,
    )

    print("\nSweep summary:")
    for num_realizations, tracking_loss, param_mse_value in zip(
        sweep_values, tracking_loss_means, param_mse_means
    ):
        print(
            f"  num_realizations={num_realizations:>3}  "
            f"mean_tracking_loss={tracking_loss:.8f}  "
            f"mean_parameter_mse={param_mse_value:.8f}"
        )


def build_parameter_grid(min_value: float, max_value: float, num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("Grid resolution must be at least 2.")
    if min_value >= max_value:
        raise ValueError("Grid minimum must be smaller than grid maximum.")
    return np.linspace(min_value, max_value, num_points)


def run_tracking_error_surface(args):
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
    )
    pipeline = SystemIdentificationPipeline(
        problem_path=args.problem,
        initial_params=init_params,
        seed=args.seed,
    )

    wheel_radius_values = build_parameter_grid(args.radius_min, args.radius_max, args.radius_points)
    base_diameter_values = build_parameter_grid(args.base_min, args.base_max, args.base_points)
    replay_robot_keys = jax.random.split(pipeline.replay_robot_key, args.num_realizations)
    replay_estimator_keys = jax.random.split(pipeline.replay_estimator_key, args.num_realizations)
    wheel_radius_values_jax = jnp.asarray(wheel_radius_values, dtype=jnp.float32)
    base_diameter_values_jax = jnp.asarray(base_diameter_values, dtype=jnp.float32)

    def loss_for_params(wheel_radius, base_diameter):
        params = PhysicalParams(
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        )
        return pipeline.loss(params, replay_robot_keys, replay_estimator_keys)

    batched_loss_for_row = jax.vmap(loss_for_params, in_axes=(0, None))

    def scan_row(_, base_diameter):
        row_losses = batched_loss_for_row(wheel_radius_values_jax, base_diameter)
        return None, row_losses

    evaluate_surface = jax.jit(lambda: jax.lax.scan(scan_row, None, base_diameter_values_jax)[1])

    print(
        f"Evaluating tracking-error surface for seed={args.seed} "
        f"with num_realizations={args.num_realizations}"
    )
    print(
        f"  compiling and evaluating {len(base_diameter_values)} rows x "
        f"{len(wheel_radius_values)} columns"
    )
    tracking_error_surface = np.asarray(evaluate_surface())

    plot_tracking_error_surface(
        wheel_radius_values=wheel_radius_values,
        base_diameter_values=base_diameter_values,
        tracking_error_surface=tracking_error_surface,
        hidden_params=pipeline.hidden_params,
        init_params=init_params,
        num_realizations=args.num_realizations,
        seed=args.seed,
        out_prefix=args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-type", type=str, choices=["realization_sensitivity", "tracking_error_surface"])
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")

    # SI optimiztation and realization plot arguments
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-num-realizations", type=int, default=64) # max number of vectorized sims in each opt iteration
    parser.add_argument("--num-seeds", type=int, default=1)  # number of complete SI runs to be averaged
    parser.add_argument("--num-workers", type=int, default=8) # parallel workers

    # Tracking error surface plot arguments
    parser.add_argument("--num-realizations", type=int, default=1) # replay realizations
    parser.add_argument("--init-wheel-radius", type=float, default=0.08)
    parser.add_argument("--init-base-diameter", type=float, default=0.4)
    parser.add_argument("--radius-min", type=float, default=0.0015)
    parser.add_argument("--radius-max", type=float, default=0.1)
    parser.add_argument("--radius-points", type=int, default=100)
    parser.add_argument("--base-min", type=float, default=0.009)
    parser.add_argument("--base-max", type=float, default=0.5)
    parser.add_argument("--base-points", type=int, default=100)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    args.plot_type = "realization_sensitivity"
    # args.plot_type = "tracking_error_surface"

    if args.output is None:
        args.output = args.plot_type

    if args.plot_type == "realization_sensitivity":
        run_realization_sweep(args)
    else:
        run_tracking_error_surface(args)
