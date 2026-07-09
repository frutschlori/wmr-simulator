import os
# os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from wmr_simulator.gain_tuning.pipeline import (resolve_gain_robot_params, run_gain_tuning_experiment)
from wmr_simulator.types import print_controller_gains, print_physical_params
from wmr_simulator.visualization.gain_tuning import (
    plot_controller_tuning_errors,
    plot_gain_tuning_summary,
    plot_training_trajectory_summary,
    plot_validation_trajectory_summary,
)
from wmr_simulator.visualization.identification import plot_loss_history


def print_loss_breakdown(label: str, component_history: dict[str, list[float]] | None) -> None:
    if component_history is None:
        return
    final_terms = {name: float(values[-1]) for name, values in component_history.items()}
    total = sum(final_terms.values())
    print(label)
    for name in ("tracking", "velocity_tracking", "input", "input_delta", "gain_delta"):
        if name not in final_terms:
            continue
        value = final_terms[name]
        share = 100.0 * value / total if total > 0.0 else 0.0
        print(f"  {name:<16}: {value:.8f} ({share:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports/tuning_optimized_5000it")
    parser.add_argument("--validation-split", type=float, default=0.2)
    # Optimization hyper-parameters
    parser.add_argument("--num-lhs-points", type=int, default=250) # points on initial search grid, 0 to disable
    parser.add_argument("--num-adam-optimizations", type=int, default=10) # number of best candidates to refine
    parser.add_argument("--steps", type=int, default=500)                 # adam steps
    parser.add_argument("--learning-rate", type=float, default=1e-3)      # adam learning rate
    parser.add_argument("--num-realizations", type=int, default=16) # noise realizations over 1 trajectory
    parser.add_argument("--seed", type=int, default=2)
    # Loss weights
    parser.add_argument("--velocity-tracking-weight", type=float, default=1)
    parser.add_argument("--input-weight", type=float, default=0.0)
    parser.add_argument("--input-delta-weight", type=float, default=1)
    # Gain bounds
    parser.add_argument("--k-min-stab", type=float, default=1e-3)
    parser.add_argument("--k-max-stab", type=float, default=100.0)
    parser.add_argument("--k-max-rest", type=float, default=100.0)
    # Optional overwrite of robot model parameters
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)
    parser.add_argument("--num-summary-training-trajectories", type=int, default=None)
    # Gain schedule: jointly tune base gains + outer-gain schedule (W), default follows problem yaml, --no-gain-schedule forces W=0 (static)
    parser.add_argument("--gain-schedule", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gain-delta-weight", type=float, default=0.0)
    # Learned residual dynamics checkpoint (scripts/train_residual_model.py); tuning
    # then rolls out the residual-augmented dynamics (model params stay fixed).
    # parser.add_argument("--residual-model", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--residual-model", type=str, default=None)
    args = parser.parse_args()

    residual_model = None
    if args.residual_model is not None:
        from wmr_simulator.residual_model import load_residual_model

        residual_model, checkpoint = load_residual_model(args.residual_model)
        print(f"Loaded residual dynamics model: {args.residual_model}")
        print(f"  config: {checkpoint['config']}")

    robot_params = resolve_gain_robot_params(args.problem, args.fixed_wheel_radius, args.fixed_base_diameter)
    result = run_gain_tuning_experiment(
        problem_path=args.problem,
        robot_params=robot_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
        validation_split=args.validation_split,
        velocity_tracking_weight=args.velocity_tracking_weight,
        input_weight=args.input_weight,
        input_delta_weight=args.input_delta_weight,
        k_min_stab=args.k_min_stab,
        k_max_stab=args.k_max_stab,
        k_max_rest=args.k_max_rest,
        num_lhs_points=args.num_lhs_points,
        num_adam_optimizations=args.num_adam_optimizations,
        schedule_enabled=args.gain_schedule,
        gain_delta_weight=args.gain_delta_weight,
        residual_model=residual_model,
    )
    pipeline = result["pipeline"]
    print_physical_params("Robot parameters used for gain tuning:", robot_params)
    print_controller_gains("Initial gains:", pipeline.gains)
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    print(f"Velocity tracking weight: {args.velocity_tracking_weight:.8g}")
    print(f"Input regularization weight: {args.input_weight:.8g}")
    print(f"Input delta regularization weight: {args.input_delta_weight:.8g}")
    print(f"Training trajectories: {pipeline.training_reference_trajectories.shape[0]}")
    print(f"Validation trajectories: {pipeline.validation_reference_trajectories.shape[0]}")
    print(f"Stable gain search range: [{args.k_min_stab:.8g}, {args.k_max_stab:.8g}]")
    print(f"I/D motor gain search max: {args.k_max_rest:.8g}")
    print(f"LHS points: {args.num_lhs_points}")
    print(f"Adam starts: {args.num_adam_optimizations}")
    print(f"Final loss: {result['loss_history'][-1]:.8f}")
    print_loss_breakdown("Final training loss components:", result["loss_component_history"])
    if result["validation_loss_history"] is not None:
        print(f"Final validation loss: {result['validation_loss_history'][-1]:.8f}")
        print_loss_breakdown("Final validation loss components:", result["validation_loss_component_history"])

    print(f"Gain schedule enabled: {result['schedule_enabled']}")
    if result.get("schedule_params") is not None:
        import numpy as _np
        schedule_params = result["schedule_params"]
        print(f"Gain delta weight: {args.gain_delta_weight:.8g}")
        print("Scheduled indices:", list(map(int, schedule_params.scheduled_indices)))
        print("Feature scale [v_max, omega_max]:", list(map(float, pipeline.gain_schedule_feature_scale)))
        print("rho:", _np.array2string(_np.asarray(schedule_params.rho), precision=5))
        print("W (rows = scheduled gains, cols = [v_d, |omega_d|]):")
        print(_np.array2string(_np.asarray(schedule_params.W), precision=5))

    plot_gain_tuning_summary(
        pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
        out_prefix="summary_gain_tuning",
    )
    plot_training_trajectory_summary(
        pipeline,
        robot_params=robot_params,
        tuned_gains=result["optimized_gains"],
        max_trajectories=args.num_summary_training_trajectories,
        out_prefix="summary_training",
    )
    plot_validation_trajectory_summary(
        pipeline,
        robot_params=robot_params,
        tuned_gains=result["optimized_gains"],
        out_prefix="summary_validation",
    )
    plot_controller_tuning_errors(
        pipeline=pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
    )
    plot_loss_history(
        loss_history=result["loss_history"],
        validation_loss_history=result["validation_loss_history"],
        loss_component_history=result["loss_component_history"],
        validation_loss_component_history=result["validation_loss_component_history"],
        out_prefix="ctrl_tuning",
    )


if __name__ == "__main__":
    main()
