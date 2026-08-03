import argparse

import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.analysis import run_trajectory_optimization_trace
from wmr_simulator.types import PhysicalParams


def main():
    parser = argparse.ArgumentParser(description="Run Bezier trajectory optimization and save its trace GIF.")
    parser.add_argument("problem", nargs="?", default="problems/problem_hidden.yaml")
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--time-scaling", choices=["s-curve", "linear"], default="linear")
    parser.add_argument("--num-control-points", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--opt-steps", type=int, default=4000)
    parser.add_argument("--trace-stride", type=int, default=1000)
    parser.add_argument("--constraint-weight", type=float, default=0.0)
    parser.add_argument("--constraint-v-weight", type=float, default=1.0)
    parser.add_argument("--constraint-a-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lateral-weight", type=float, default=1.0)
    parser.add_argument("--constraint-omega-weight", type=float, default=1.0)
    parser.add_argument("--constraint-alpha-weight", type=float, default=1.0)
    parser.add_argument("--constraint-smooth-max-beta", type=float, default=20.0)
    parser.add_argument("--output", type=str, default="traj_opt")
    parser.add_argument("--frame-duration", type=float, default=0.2)
    parser.add_argument("--frames-root", type=str, default="visualize/Trajectory Optimization Frames")
    parser.add_argument("--export-reference-states", action="store_true")
    parser.add_argument("--save-final-reference-states", action="store_true")
    parser.add_argument("--stacked-tracking-surface", action="store_true", default=True)
    parser.add_argument("--surface-init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--surface-init-base-diameter", type=float, default=0.2)
    parser.add_argument("--surface-radius-min", type=float, default=0.01)
    parser.add_argument("--surface-radius-max", type=float, default=0.1)
    parser.add_argument("--surface-radius-points", type=int, default=50)
    parser.add_argument("--surface-base-min", type=float, default=0.01)
    parser.add_argument("--surface-base-max", type=float, default=0.5)
    parser.add_argument("--surface-base-points", type=int, default=50)
    parser.add_argument("--surface-num-realizations", type=int, default=1)
    parser.add_argument("--surface-seed", type=int, default=0)
    parser.add_argument("--surface-window-length", type=int, default=None)
    args = parser.parse_args()

    surface_initial_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.surface_init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.surface_init_base_diameter, dtype=jnp.float32),
    )
    result = run_trajectory_optimization_trace(
        problem_path=args.problem,
        window_length=args.window_length,
        time_scaling=args.time_scaling,
        num_control_points=args.num_control_points,
        num_steps=args.opt_steps,
        learning_rate=args.learning_rate,
        trace_stride=args.trace_stride,
        constraint_weight=args.constraint_weight,
        constraint_component_weights={
            "v": args.constraint_v_weight,
            "a": args.constraint_a_weight,
            "lateral": args.constraint_lateral_weight,
            "omega": args.constraint_omega_weight,
            "alpha": args.constraint_alpha_weight,
        },
        constraint_smooth_max_beta=args.constraint_smooth_max_beta,
        out_prefix=args.output,
        frame_duration=args.frame_duration,
        frames_root=args.frames_root,
        export_reference_states=args.export_reference_states,
        save_final_reference_states=args.save_final_reference_states,
        include_tracking_surface=args.stacked_tracking_surface,
        surface_initial_params=surface_initial_params,
        surface_radius_min=args.surface_radius_min,
        surface_radius_max=args.surface_radius_max,
        surface_radius_points=args.surface_radius_points,
        surface_base_min=args.surface_base_min,
        surface_base_max=args.surface_base_max,
        surface_base_points=args.surface_base_points,
        surface_num_realizations=args.surface_num_realizations,
        surface_seed=args.surface_seed,
        surface_window_length=args.surface_window_length,
    )
    print(f"Final loss: {result['loss_history'][-1]:.8f}")
    print("Final objective terms:")
    print(f"  FIM:         {result['objective_terms']['fim']:.8e}")
    print(f"  log(FIM):    {result['objective_terms']['log_fim']:.8f}")
    print(f"  Constraints: {result['objective_terms']['constraints']:.8e}")
    print(f"  Constraint term (weighted): {result['objective_terms']['constraint_term']:.8f}")
    print(f"  Total:       {result['objective_terms']['total']:.8f}")
    print(f"Trajectory-only GIF: {result['trajectory_gif_path']}")
    if result["stacked_gif_path"] is not None:
        print(f"Stacked tracking-surface GIF: {result['stacked_gif_path']}")
    if result["export_dir"] is not None:
        print(f"Exported reference-state snapshots: {result['export_dir']}")
    if result["final_reference_path"] is not None:
        print(f"Saved final reference states: {result['final_reference_path']}")


if __name__ == "__main__":
    main()
