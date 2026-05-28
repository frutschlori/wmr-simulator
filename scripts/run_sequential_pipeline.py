import argparse

import jax.numpy as jnp
import numpy as np

from wmr_simulator.active_learning.sequential_pipeline import run_si_then_gain_tuning
from wmr_simulator.types import (
    PhysicalParams,
    print_controller_gains,
    print_physical_params,
)
from wmr_simulator.visualization.gain_tuning import plot_controller_tuning_errors
from wmr_simulator.visualization.identification import (
    plot_loss_history,
    plot_system_id,
    plot_trajectory,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--steps-si", type=int, default=500)
    parser.add_argument("--learning-rate-si", type=float, default=1e-3)
    parser.add_argument("--steps-gain-tuning", type=int, default=1000)
    parser.add_argument("--learning-rate-gain-tuning", type=float, default=5e-2)
    parser.add_argument("--num-realizations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.2)
    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
    )
    result = run_si_then_gain_tuning(
        problem_path=args.problem,
        initial_params=init_params,
        steps_si=args.steps_si,
        learning_rate_si=args.learning_rate_si,
        steps_gain_tuning=args.steps_gain_tuning,
        learning_rate_gain_tuning=args.learning_rate_gain_tuning,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
        window_length=args.window_length,
    )

    id_result = result["identification"]
    gain_result = result["gain_tuning"]
    id_pipeline = id_result["pipeline"]
    gain_pipeline = gain_result["pipeline"]

    print_physical_params("True robot parameters:", id_pipeline.hidden_params)
    print_physical_params("Initial robot parameters guess:", init_params)
    print_physical_params("Estimated robot parameters:", id_result["estimated_params"])
    print(f"Final system ID loss: {id_result['loss_history'][-1]:.8f}")
    print(f"Final parameter RMSE: {np.sqrt(id_result['parameter_mse_history'][-1]):.2f} mm")
    print_controller_gains("Initial gains:", gain_pipeline.gains)
    print_controller_gains("Optimized gains:", gain_result["optimized_gains"])
    print(f"Final gain tuning loss: {gain_result['loss_history'][-1]:.8f}")

    plot_system_id(
        pipeline=id_pipeline,
        init_target_log=id_result["init_target_log"],
        init_log=id_result["init_replay_log"],
        predicted_log=id_result["final_replay_log"],
        out_prefix="sequential_opt",
    )
    plot_trajectory(
        gain_pipeline,
        tuned_log=gain_result["final_hidden_log"],
        untuned_log=id_result["init_target_log"],
        out_prefix="sequential_opt_trajectory",
    )
    plot_controller_tuning_errors(
        pipeline=gain_pipeline,
        init_log=id_result["init_target_log"],
        tuned_log=gain_result["final_hidden_log"],
        out_prefix="sequential_opt",
    )
    plot_loss_history(
        loss_history=id_result["loss_history"],
        parameter_error_history=id_result["parameter_mse_history"],
        out_prefix="system_id",
    )
    plot_loss_history(loss_history=gain_result["loss_history"], out_prefix="ctrl_tuning")


if __name__ == "__main__":
    main()
