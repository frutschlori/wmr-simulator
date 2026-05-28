import argparse

import numpy as np

from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline


def main():
    parser = argparse.ArgumentParser(description="Initialize trajectory optimization inputs.")
    parser.add_argument("problem", nargs="?", default="problems/problem_hidden.yaml")
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--trajectory-generator", choices=["planner", "bezier"], default="bezier")
    parser.add_argument("--time-scaling", choices=["s-curve", "linear"], default="s-curve")
    parser.add_argument("--bezier-order", type=int, default=20)
    parser.add_argument("--opt-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constraint-v-weight", type=float, default=1.0)
    parser.add_argument("--constraint-a-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lateral-weight", type=float, default=1.0)
    parser.add_argument("--constraint-omega-weight", type=float, default=1.0)
    parser.add_argument("--constraint-alpha-weight", type=float, default=1.0)
    parser.add_argument("--constraint-smooth-max-beta", type=float, default=20.0)
    parser.add_argument("--constraint-smooth-violation-alpha", type=float, default=20.0)
    parser.add_argument("--save-opt-GIF", action="store_true", default=False)
    parser.add_argument("--export-opt-reference-states", action="store_true", default=False)
    parser.add_argument("--opt-trace-stride", type=int, default=50)
    parser.add_argument("--save-trajectory", action="store_true", default=True)
    args = parser.parse_args()

    pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        trajectory_generator_type=args.trajectory_generator,
        time_scaling=args.time_scaling,
    )

    print(f"Loaded problem: {pipeline.problem.path}")
    print(f"Robot: {type(pipeline.robot).__name__}")
    print(f"Trajectory generator: {type(pipeline.trajectory_generator).__name__}")
    if args.trajectory_generator == "bezier":
        print(f"Time scaling: {pipeline.time_scaling}")
    print(f"Trajectory samples: {len(pipeline.trajectory.time)}")
    print(f"Start pose: {pipeline.trajectory.poses[0]}")
    print(f"Goal pose:  {pipeline.trajectory.poses[-1]}")
    print(f"Window length: {pipeline.resolve_window_length(args.window_length)}")
    print("Motion limits:")
    print({name: float(value) for name, value in pipeline.motion_limits().items()})
    print("Measurement vector shape:")
    print(pipeline.measurement_vector(pipeline.nominal_parameters(), window_length=args.window_length).shape)
    print("FIM:")
    print(pipeline.compute_fim_matrix(window_length=args.window_length))

    if args.trajectory_generator == "bezier":
        initial_control_points = pipeline.initial_bezier_control_points(args.bezier_order)
        pipeline.set_bezier_control_points(initial_control_points)

    pipeline.plot_trajectory(
        window_length=args.window_length,
        out_prefix=f"traj_initial_{args.trajectory_generator}",
    )

    if args.opt_steps and args.trajectory_generator == "bezier":
        constraint_component_weights = {
            "v": args.constraint_v_weight,
            "a": args.constraint_a_weight,
            "lateral": args.constraint_lateral_weight,
            "omega": args.constraint_omega_weight,
            "alpha": args.constraint_alpha_weight,
        }
        print("Initial control points:")
        print(np.asarray(initial_control_points))
        optimized_control_points, loss_history = pipeline.optimize_bezier_trajectory(
            order=args.bezier_order,
            num_steps=args.opt_steps,
            learning_rate=args.learning_rate,
            window_length=args.window_length,
            save_trace=args.save_opt_GIF,
            trace_stride=args.opt_trace_stride,
            constraint_weight=args.constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=args.constraint_smooth_max_beta,
            constraint_smooth_violation_alpha=args.constraint_smooth_violation_alpha,
        )
        objective_terms = pipeline.objective_terms_from_control_points(
            optimized_control_points,
            window_length=args.window_length,
            constraint_weight=args.constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=args.constraint_smooth_max_beta,
            constraint_smooth_violation_alpha=args.constraint_smooth_violation_alpha,
        )
        print("Optimized control points:")
        print(np.asarray(optimized_control_points))
        print("Final optimization loss:")
        print(loss_history[-1])
        print("Final objective terms:")
        print(f"  FIM:         {float(objective_terms['fim']):.8e}")
        print(f"  Constraints: {float(objective_terms['constraints']):.8e}")
        print(f"  Total:       {float(objective_terms['total']):.8e}")
        print(f"  Constraint share: {100.0 * float(objective_terms['constraint_share']):.2f}%")
        print("Optimized FIM:")
        print(pipeline.compute_fim_matrix(window_length=args.window_length))
        pipeline.plot_trajectory(
            window_length=args.window_length,
            out_prefix="traj_optimized_bezier",
        )
        pipeline.plot_loss_history(out_prefix="traj_opt_loss_history")
        if args.save_opt_GIF:
            pipeline.save_optimization_GIF(
                window_length=args.window_length,
                out_prefix="traj_opt",
            )
        if args.export_opt_reference_states:
            export_dir, saved_paths = pipeline.save_optimization_reference_states(
                filename_prefix="traj_opt_reference_states",
            )
            print("Exported optimization reference states:")
            print(export_dir)
            print(f"Saved {len(saved_paths)} snapshots")

    if args.save_trajectory:
        saved_path = pipeline.save_reference_states_pickle(
            filename_prefix=f"{args.trajectory_generator}_reference_states",
        )
        print("Saved trajectory pickle:")
        print(saved_path)


if __name__ == "__main__":
    main()
