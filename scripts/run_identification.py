import argparse

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.types import (
    PhysicalParams,
    physical_params_to_array,
    physical_params_from_array,
)
from wmr_simulator.visualization.identification import (
    plot_loss_history,
    plot_system_id,
    plot_windowed_replay_trajectory,
)


def print_param_block(label: str, params: PhysicalParams, signed: bool = False):
    values_mm = 1000.0 * np.asarray(physical_params_to_array(params), dtype=float)
    value_format = "+.2f" if signed else ".2f"
    print(label)
    print(f"  wheel_radius   = {values_mm[0]:{value_format}} mm")
    print(f"  base_diameter  = {values_mm[1]:{value_format}} mm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/figure_eight.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--stochastic-replay", action="store_true", default=False)
    parser.add_argument("--num-realizations", type=int, default=8)
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.2)
    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
    )
    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
        window_length=args.window_length,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        deterministic_replay=not args.stochastic_replay,
    )
    pipeline = result["pipeline"]
    print_param_block("Initial guess:", init_params)
    print()
    print_param_block("Hidden parameters:", pipeline.hidden_params)
    if result["bootstrap"] is not None:
        bootstrap = result["bootstrap"]
        mean_params = physical_params_from_array(bootstrap["parameter_mean"])
        bias = physical_params_from_array(
            bootstrap["parameter_mean"] - physical_params_to_array(pipeline.hidden_params)
        )
        covariance_mm2 = 1e6 * np.asarray(bootstrap["parameter_covariance"])
        std_params = PhysicalParams(
            wheel_radius=jnp.asarray(np.sqrt(covariance_mm2[0, 0]) / 1000.0),
            base_diameter=jnp.asarray(np.sqrt(covariance_mm2[1, 1]) / 1000.0),
        )
        print()
        print(f"Bootstrap samples: {args.bootstrap_samples}")
        print(f"Mean loss: {result['loss_history'][-1]:.8f}")
        print()
        print_param_block("Mean estimated parameters:", mean_params)
        print_param_block("Bias of mean estimate:", bias, signed=True)
        print_param_block("Standard deviation:", std_params)
        print("Parameter covariance [mm²]:")
        print(np.array2string(covariance_mm2, precision=4, suppress_small=False))
    else:
        print()
        print_param_block("Estimated parameters:", result["estimated_params"])
        print()
        print(f"Final loss: {result['loss_history'][-1]:.8f}")
        print(f"Final parameter RMSE: {np.sqrt(result['parameter_mse_history'][-1]):.2f} mm")

    plot_system_id(
        pipeline=pipeline,
        init_target_log=result["init_target_log"],
        init_log=result["init_replay_log"],
        predicted_log=result["final_replay_log"],
    )
    plot_windowed_replay_trajectory(
        pipeline=pipeline,
        closed_loop_log=pipeline.target_log,
        replay_log=result["final_replay_log"],
        window_length=args.window_length,
        out_prefix="system_identification_trajectory",
    )
    plot_loss_history(
        loss_history=result["loss_history"],
        parameter_error_history=result["parameter_mse_history"],
    )


if __name__ == "__main__":
    main()
