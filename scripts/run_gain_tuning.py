import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
from wmr_simulator.gain_tuning.pipeline import (resolve_gain_robot_params, run_gain_tuning_experiment)
from wmr_simulator.types import print_controller_gains, print_physical_params
from wmr_simulator.visualization.gain_tuning import plot_controller_tuning_errors
from wmr_simulator.visualization.identification import (plot_loss_history, plot_trajectory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports/turbo/")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-realizations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)
    args = parser.parse_args()

    robot_params = resolve_gain_robot_params(args.problem, args.fixed_wheel_radius, args.fixed_base_diameter)
    result = run_gain_tuning_experiment(
        problem_path=args.problem,
        robot_params=robot_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
    )
    pipeline = result["pipeline"]
    print_physical_params("Robot parameters used for gain tuning:", robot_params)
    print_controller_gains("Initial gains:", pipeline.gains)
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    print(f"Final loss: {result['loss_history'][-1]:.8f}")

    plot_trajectory(
        pipeline,
        tuned_log=result["final_hidden_log"],
        untuned_log=result["init_hidden_log"],
        out_prefix="trajectories_gain_tuning_hidden_params",
    )
    plot_trajectory(
        pipeline,
        tuned_log=result["final_model_log"],
        untuned_log=result["init_model_log"],
        out_prefix="trajectories_gain_tuning_used_params",
    )
    plot_controller_tuning_errors(
        pipeline=pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
    )
    plot_loss_history(loss_history=result["loss_history"], out_prefix="ctrl_tuning")


if __name__ == "__main__":
    main()
