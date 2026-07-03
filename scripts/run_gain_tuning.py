import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from wmr_simulator.gain_tuning.pipeline import (resolve_gain_robot_params, run_gain_tuning_experiment)
from wmr_simulator.types import print_controller_gains, print_physical_params
from wmr_simulator.visualization.gain_tuning import plot_controller_tuning_errors, plot_gain_tuning_summary
from wmr_simulator.visualization.identification import plot_loss_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports/2026_06_22")
    # Optimization hyper-parameters
    parser.add_argument("--num-lhs-points", type=int, default=200) # points on initial search grid, 0 to disable
    parser.add_argument("--num-adam-optimizations", type=int, default=10) # number of best candidates to refine
    parser.add_argument("--steps", type=int, default=500)                 # adam steps
    parser.add_argument("--learning-rate", type=float, default=1e-3)      # adam learning rate
    parser.add_argument("--num-realizations", type=int, default=8) # noise realizations over 1 trajectory
    parser.add_argument("--seed", type=int, default=2)
    # Loss weights
    parser.add_argument("--input-weight", type=float, default=0.1)
    parser.add_argument("--input-delta-weight", type=float, default=1)
    # Gain bounds
    parser.add_argument("--k-min-stab", type=float, default=1e-3)
    parser.add_argument("--k-max-stab", type=float, default=20.0)
    parser.add_argument("--k-max-rest", type=float, default=100.0)
    # Optional overwrite of robot model parameters
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
        input_weight=args.input_weight,
        input_delta_weight=args.input_delta_weight,
        k_min_stab=args.k_min_stab,
        k_max_stab=args.k_max_stab,
        k_max_rest=args.k_max_rest,
        num_lhs_points=args.num_lhs_points,
        num_adam_optimizations=args.num_adam_optimizations,
    )
    pipeline = result["pipeline"]
    print_physical_params("Robot parameters used for gain tuning:", robot_params)
    print_controller_gains("Initial gains:", pipeline.gains)
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    print(f"Input regularization weight: {args.input_weight:.8g}")
    print(f"Input delta regularization weight: {args.input_delta_weight:.8g}")
    print(f"Stable gain search range: [{args.k_min_stab:.8g}, {args.k_max_stab:.8g}]")
    print(f"I/D motor gain search max: {args.k_max_rest:.8g}")
    print(f"LHS points: {args.num_lhs_points}")
    print(f"Adam starts: {args.num_adam_optimizations}")
    print(f"Final loss: {result['loss_history'][-1]:.8f}")

    plot_gain_tuning_summary(
        pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
        out_prefix="summary_gain_tuning",
    )
    plot_controller_tuning_errors(
        pipeline=pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
    )
    plot_loss_history(loss_history=result["loss_history"], out_prefix="ctrl_tuning")


if __name__ == "__main__":
    main()
