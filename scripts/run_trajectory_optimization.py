import argparse
import os
import re
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.analysis import (
    create_stacked_tracking_surface_trace_gif,
    render_tracking_surface_frames_for_optimization_trace,
)
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline
from wmr_simulator.types import PhysicalParams


def filename_stem(title: str | None) -> str | None:
    if title is None:
        return None
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip()).strip("_")
    if not stem:
        raise ValueError("--title must contain at least one filename-safe character.")
    return stem


def main():
    parser = argparse.ArgumentParser(description="Initialize trajectory optimization inputs.")
    # Setup description
    parser.add_argument("--problem", default="problems/pololu.yaml")
    parser.add_argument("--title", type=str, default="medium_speed_new_scurve")
    # Optimization Settings
    parser.add_argument("--save-trajectory", action="store_true", default=True)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--opt-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    # Path settings
    parser.add_argument("--time-scaling", choices=["s-curve", "linear"], default="s-curve")
    parser.add_argument("--bezier-order", type=int, default=20)
    # Constraints
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constraint-v-weight", type=float, default=1.0)
    parser.add_argument("--constraint-a-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lateral-weight", type=float, default=1.0)
    parser.add_argument("--constraint-omega-weight", type=float, default=1.0)
    parser.add_argument("--constraint-alpha-weight", type=float, default=1.0)
    parser.add_argument("--tangent-floor-weight", type=float, default=1.0)
    parser.add_argument("--constraint-smooth-max-beta", type=float, default=20.0) # barrier constant
    # Visualization settings
    parser.add_argument("--save-opt-GIF", action="store_true", default=False)
    parser.add_argument("--opt-trace-stride", type=int, default=500)
    parser.add_argument("--stacked-tracking-surface", action="store_true", default=False)
    args = parser.parse_args()
    output_stem = filename_stem(args.title)

    pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        time_scaling=args.time_scaling,
    )

    print(f"Loaded problem: {pipeline.problem.path}")
    print(f"Robot: {type(pipeline.robot).__name__}")
    print("Trajectory generator: Bezier")
    print(f"Time scaling: {pipeline.time_scaling}")
    print(f"Reference samples: {len(pipeline.reference_states)} at dt={pipeline.problem.geometry_dt}")
    print(f"Closed-loop pose samples: {len(pipeline.closed_loop_log.pose.states)} at dt={pipeline.problem.wheel_dt}")
    print(f"Start pose: {pipeline.problem.start}")
    print(f"Goal pose:  {pipeline.problem.goal}")
    print(f"Window length: {pipeline.simulation.resolve_window_length(args.window_length)}")
    print("Motion limits:")
    print({name: float(value) for name, value in pipeline.motion_limits().items()})
    print("Measurement vector shape:")
    print(pipeline.measurement_vector(pipeline.nominal_parameters(), window_length=args.window_length).shape)
    print("FIM:")
    print(pipeline.compute_fim_matrix(window_length=args.window_length))

    initial_control_points = pipeline.initial_bezier_control_points(args.bezier_order)
    pipeline.set_bezier_control_points(initial_control_points)

    pipeline.plot_trajectory(
        window_length=args.window_length,
        out_prefix="traj_initial_bezier" if output_stem is None else f"{output_stem}_initial",
    )

    if args.opt_steps:
        constraint_component_weights = {
            "v": args.constraint_v_weight,
            "a": args.constraint_a_weight,
            "lateral": args.constraint_lateral_weight,
            "omega": args.constraint_omega_weight,
            "alpha": args.constraint_alpha_weight,
        }
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
            tangent_floor_weight=args.tangent_floor_weight,
        )
        objective_terms = pipeline.objective_terms_from_control_points(
            optimized_control_points,
            window_length=args.window_length,
            constraint_weight=args.constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=args.constraint_smooth_max_beta,
            tangent_floor_weight=args.tangent_floor_weight,
        )
        constraint_components = pipeline.constraint_components_from_control_points(
            optimized_control_points,
            constraint_weight=args.constraint_weight,
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=args.constraint_smooth_max_beta,
        )
        print("Final optimization loss:")
        print(loss_history[-1])
        print("Final objective terms:")
        print(f"  FIM:         {float(objective_terms['fim']):.8e}")
        print(f"  Constraints: {float(objective_terms['constraints']):.8e}")
        print(f"  Tangent:     {float(objective_terms['tangent_floor']):.8e}")
        print("  Constraint components:")
        for name in ("v", "a", "lateral", "omega", "alpha"):
            print(f"    {name:<7}: {float(constraint_components[name]):.8e}")
        print(f"  Total:       {float(objective_terms['total']):.8e}")
        print(f"  Constraint share: {100.0 * float(objective_terms['constraint_share']):.2f}%")
        print("Optimized FIM:")
        print(pipeline.compute_fim_matrix(window_length=args.window_length))
        pipeline.plot_trajectory(
            window_length=args.window_length,
            out_prefix="traj_optimized_bezier" if output_stem is None else f"{output_stem}_optimized",
        )
        pipeline.plot_loss_history(out_prefix="traj_opt_loss_history" if output_stem is None else f"{output_stem}_loss_history")
        if args.save_opt_GIF:
            frames_root = os.path.join("visualize", "Trajectory Optimization Frames")
            trace_stem = "traj_opt" if output_stem is None else output_stem
            trajectory_frames_dir = os.path.join(frames_root, f"{trace_stem}_trajectory_frames")
            trajectory_gif_path = os.path.join(frames_root, f"{trace_stem}.gif")
            pipeline.save_optimization_GIF(
                window_length=args.window_length,
                out_prefix=trace_stem,
                frames_dir=trajectory_frames_dir,
                gif_path=trajectory_gif_path,
            )
            if args.stacked_tracking_surface:
                robot_cfg = pipeline.problem.robot_cfg
                surface_initial_params = PhysicalParams(
                    wheel_radius=jnp.asarray(robot_cfg["wheel_radius"], dtype=jnp.float32),
                    base_diameter=jnp.asarray(robot_cfg["base_diameter"], dtype=jnp.float32),
                    max_wheel_speed=jnp.asarray(robot_cfg["max_wheel_speed"], dtype=jnp.float32),
                    time_constant=jnp.asarray(robot_cfg["time_constant"], dtype=jnp.float32),
                )
                surface_frames_dir = render_tracking_surface_frames_for_optimization_trace(
                    problem_path=args.problem,
                    snapshots=pipeline.optimization_snapshots,
                    initial_params=surface_initial_params,
                    output_dir_name=os.path.join(frames_root, f"{trace_stem}_tracking_surface_frames"),
                    radius_min=0.01,
                    radius_max=0.1,
                    radius_points=50,
                    base_min=0.01,
                    base_max=0.5,
                    base_points=50,
                    window_length=args.window_length,
                )
                stacked_gif_path = create_stacked_tracking_surface_trace_gif(
                    trajectory_frames_dir=trajectory_frames_dir,
                    surface_frames_dir=surface_frames_dir,
                    output_dir_name=os.path.join(frames_root, f"{trace_stem}_stacked_frames"),
                    gif_name=f"{trace_stem}_tracking_surface.gif",
                    gif_path=os.path.join(frames_root, f"{trace_stem}_tracking_surface.gif"),
                )
                print(f"Stacked tracking-surface GIF: {stacked_gif_path}")

    if args.save_trajectory:
        saved_path = pipeline.save_reference_states_pickle(
            filename_prefix="bezier_reference_states" if output_stem is None else output_stem,
        )
        print("Saved trajectory pickle:")
        print(saved_path)


if __name__ == "__main__":
    main()
