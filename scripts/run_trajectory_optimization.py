import argparse
from datetime import datetime
import os
import re

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np

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
    parser.add_argument("--problem", default="problems/pololu_gains.yaml")
    parser.add_argument("--title", type=str, default="id_optimized")
    # Optimization Settings
    parser.add_argument("--save-trajectory", action="store_true", default=True)
    parser.add_argument("--no-save-trajectory", dest="save_trajectory", action="store_false")
    parser.add_argument("--window-length", type=int, default=50) # replay window length, only for identification mode
    parser.add_argument("--opt-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--objective-mode", choices=["identification", "gain-tuning"],
                        default="identification")
    # Settings for multiple trajectory synthesis
    parser.add_argument("--num-trajectories", type=int, default=1)
    parser.add_argument("--constraint-weight-jitter", type=float, default=0.2) # factor for diverse constraints
    parser.add_argument("--vectorize-trajectories", action="store_true", default=True)
    # Path settings
    parser.add_argument("--time-scaling", choices=["s-curve", "linear"], default="s-curve")
    parser.add_argument("--bezier-order", type=int, default=15)
    parser.add_argument("--trajectory-seed", type=int, default=0)
    # Constraints
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constraint-v-weight", type=float, default=1.0)
    parser.add_argument("--constraint-a-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lateral-weight", type=float, default=1.0)
    parser.add_argument("--constraint-omega-weight", type=float, default=1.0)
    parser.add_argument("--constraint-alpha-weight", type=float, default=1.0)
    parser.add_argument("--tangent-floor-weight", type=float, default=1.0) # penalize 0 linear velocity to avoid num instability
    parser.add_argument("--constraint-smooth-max-beta", type=float, default=20.0) # barrier constant
    # Visualization settings
    parser.add_argument("--save-opt-GIF", action="store_true", default=False)
    parser.add_argument("--opt-trace-stride", type=int, default=500)
    parser.add_argument("--stacked-tracking-surface", action="store_true", default=False)
    args = parser.parse_args()
    output_stem = filename_stem(args.title)
    run_stem = "trajectory_optimization" if output_stem is None else output_stem
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.num_trajectories <= 0:
        raise ValueError("--num-trajectories must be positive.")
    if args.num_trajectories > 1 and args.save_opt_GIF:
        raise ValueError("--save-opt-GIF is only supported for single-trajectory optimization.")
    if args.bezier_order < 2:
        raise ValueError("--bezier-order must be >= 2.")

    pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        time_scaling=args.time_scaling,
        objective_mode=args.objective_mode,
    )

    print(f"Loaded problem: {pipeline.problem.path}")
    print(f"Robot: {type(pipeline.robot).__name__}")
    print("Trajectory generator: Bezier")
    print(f"Time scaling: {pipeline.time_scaling}")
    print(f"Objective mode: {pipeline.objective_mode}")
    print(f"Optimized trajectories: {args.num_trajectories}")
    if args.num_trajectories > 1:
        print(f"Bezier order: {args.bezier_order}")
        print(f"Constraint weight jitter: +/-{100.0 * args.constraint_weight_jitter:.1f}%")
    print(f"Reference samples: {len(pipeline.reference_states)} at dt={pipeline.problem.geometry_dt}")
    print(f"Closed-loop pose samples: {len(pipeline.closed_loop_log.pose.states)} at dt={pipeline.problem.wheel_dt}")
    if args.num_trajectories == 1:
        print(f"Start pose: {pipeline.problem.start}")
        print(f"Goal pose:  {pipeline.problem.goal}")
        print(f"Window length: {pipeline.simulation.resolve_window_length(args.window_length)}")
        print("Motion limits:")
        print({name: float(value) for name, value in pipeline.motion_limits().items()})
        print("Measurement vector shape:")
        print(pipeline.measurement_vector(pipeline.nominal_parameters(), window_length=args.window_length).shape)
        print("FIM parameter vector shape:")
        print(pipeline.nominal_parameters().shape)
        print("FIM:")
        print(pipeline.compute_fim_matrix(window_length=args.window_length))

    initial_control_points = pipeline.initial_bezier_control_points(args.bezier_order)
    pipeline.set_bezier_control_points(initial_control_points)

    pipeline.plot_trajectory(
        window_length=args.window_length,
        out_prefix="traj_initial_bezier" if output_stem is None else f"{output_stem}_initial",
    )

    optimized_control_point_batch = None

    if args.opt_steps:
        constraint_component_weights = {
            "v": args.constraint_v_weight,
            "a": args.constraint_a_weight,
            "lateral": args.constraint_lateral_weight,
            "omega": args.constraint_omega_weight,
            "alpha": args.constraint_alpha_weight,
        }
        if args.num_trajectories == 1:
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
        else:
            optimized_control_point_batch, loss_history = pipeline.optimize_bezier_trajectories(
                order=args.bezier_order,
                num_steps=args.opt_steps,
                learning_rate=args.learning_rate,
                num_trajectories=args.num_trajectories,
                vectorized=args.vectorize_trajectories,
                constraint_weight_jitter=args.constraint_weight_jitter,
                seed=args.trajectory_seed,
                window_length=args.window_length,
                constraint_weight=args.constraint_weight,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=args.constraint_smooth_max_beta,
                tangent_floor_weight=args.tangent_floor_weight,
                verbose=False,
            )
            final_losses = np.asarray(pipeline.batch_final_losses, dtype=float)
            best_index = int(np.argmin(np.where(np.isfinite(final_losses), final_losses, np.inf)))
            optimized_control_points = optimized_control_point_batch[best_index]
            selected_constraint_weight = float(pipeline.batch_constraint_weights[best_index])
            sampled_constraint_weights = np.asarray(pipeline.batch_constraint_weights, dtype=float)
            sampled_constraint_factors = (
                sampled_constraint_weights / args.constraint_weight
                if args.constraint_weight != 0.0
                else sampled_constraint_weights
            )
            print("Sampled constraint factors:")
            print(sampled_constraint_factors)
        if args.num_trajectories == 1:
            selected_constraint_weight = args.constraint_weight
            objective_terms = pipeline.objective_terms_from_control_points(
                optimized_control_points,
                window_length=args.window_length,
                constraint_weight=selected_constraint_weight,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=args.constraint_smooth_max_beta,
                tangent_floor_weight=args.tangent_floor_weight,
            )
            constraint_components = pipeline.constraint_components_from_control_points(
                optimized_control_points,
                constraint_weight=selected_constraint_weight,
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
            print(f"  Constraint weight used: {selected_constraint_weight:.8g}")
            print("Optimized FIM:")
            print(pipeline.compute_fim_matrix(window_length=args.window_length))
        if optimized_control_point_batch is None:
            pipeline.plot_trajectory(
                window_length=args.window_length,
                out_prefix="traj_optimized_bezier" if output_stem is None else f"{output_stem}_optimized",
            )
        else:
            plot_dir = os.path.join("visualize", f"{run_stem}_trajectories_{run_timestamp}")
            os.makedirs(plot_dir, exist_ok=True)
            for index, control_points in enumerate(optimized_control_point_batch):
                pipeline.set_bezier_control_points(control_points)
                pipeline.plot_trajectory(
                    window_length=args.window_length,
                    out_path=os.path.join(plot_dir, f"trajectory_{index:02d}.pdf"),
                )
            pipeline.set_bezier_control_points(optimized_control_points)
            print("Saved trajectory plots:")
            print(plot_dir)
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
        if optimized_control_point_batch is None:
            saved_path = pipeline.save_reference_states_pickle(
                filename_prefix="bezier_reference_states" if output_stem is None else output_stem,
            )
            print("Saved trajectory pickle:")
            print(saved_path)
        else:
            saved_paths = []
            filename_prefix = "bezier_reference_states" if output_stem is None else output_stem
            export_dir = os.path.join("trajectory_exports", f"{filename_prefix}_{run_timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            for index, control_points in enumerate(optimized_control_point_batch):
                pipeline.set_bezier_control_points(control_points)
                saved_paths.append(
                    pipeline.save_reference_states_pickle(
                        out_dir=export_dir,
                        filename_prefix=f"{filename_prefix}_{index:02d}",
                    )
                )
            pipeline.set_bezier_control_points(optimized_control_points)
            print("Saved trajectory pickles:")
            print(export_dir)
            for saved_path in saved_paths:
                print(saved_path)


if __name__ == "__main__":
    main()
