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
from wmr_simulator.trajectory_optimization.bspline import (
    DEFAULT_MIN_TANGENT_FRACTION,
    GUARD_TANGENT_FRACTION,
)
from wmr_simulator.trajectory_optimization.start_offsets import START_OFFSET_MODE_RANDOM, START_OFFSET_MODES, START_OFFSET_MODE_OPTIMIZE
from wmr_simulator.trajectory_optimization.objectives import CRITERIA, DEFAULT_CRITERION
from wmr_simulator.trajectory_optimization.pipeline import (
    OBJECTIVE_MODE_GAIN_TUNING,
    TrajectoryOptimizationPipeline,
)
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
    parser.add_argument("--title", type=str, default="gain_optimized")
    # Optimization Settings
    parser.add_argument("--save-trajectory", action="store_true", default=True)
    parser.add_argument("--window-length", type=int, default=50) # replay window length, only for identification mode
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--opt-steps", type=int, default=500)
    parser.add_argument("--objective-mode", choices=["identification", "gain-tuning"],
                        default="gain-tuning")
    parser.add_argument("--fim-a-slip-max", action=argparse.BooleanOptionalAction, default=True,
                        help="Include a_slip_max in the FIM parameters (identification mode); "
                             "--no-fim-a-slip-max drops it when its low sensitivity makes the FIM stiff.")
    # Settings for multiple trajectory synthesis
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--constraint-weight-jitter", type=float, default=0.3) # factor for diverse constraints
    parser.add_argument("--vectorize-trajectories", action="store_true", default=True)
    # Path settings
    parser.add_argument("--time-scaling", choices=["s-curve", "linear"], default="s-curve")
    # B-spline control points. This is the parametrization's stiffness knob:
    # more of them means finer detail but a curve that reacts harder to each
    # one, so the motion constraints bind sooner (see bspline.py).
    parser.add_argument("--num-control-points", type=int, default=6)
    parser.add_argument("--trajectory-seed", type=int, default=0)
    parser.add_argument("--criterion", choices=list(CRITERIA), default=DEFAULT_CRITERION)
    # What happens to the rollout start offsets the FIM is averaged over:
    # 'random' keeps the frozen draw, 'static' a deterministic spread, and the
    # 'optimize*' modes hand them to the optimizer alongside the control points.
    # They ship in the trajectory pickles either way.
    parser.add_argument("--start-offset-mode", choices=sorted(START_OFFSET_MODES), default=START_OFFSET_MODE_OPTIMIZE)
    parser.add_argument("--offset-displacement-step-factor", type=float, default=1.0)
    parser.add_argument("--offset-heading-step-factor", type=float, default=1.0)
    # Constraints
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constraint-v-weight", type=float, default=1.0)
    parser.add_argument("--constraint-a-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lateral-weight", type=float, default=1.0)
    parser.add_argument("--constraint-omega-weight", type=float, default=1.0)
    parser.add_argument("--constraint-alpha-weight", type=float, default=0.4)
    parser.add_argument("--constraint-smooth-max-beta", type=float, default=10.0) # barrier constant
    # Floor on |dpos/ds| the curve is held above, as a fraction of the curve's
    # own rms tangent (see bspline.tangent_floor_loss). Bunched control points
    # otherwise stall the curve into a cusp, which the FIM rewards and the
    # motion limits cannot see; 0 disables the term. Relative so it does not
    # have to be retuned when the environment box or the path length changes.
    parser.add_argument("--min-tangent-fraction", type=float, default=DEFAULT_MIN_TANGENT_FRACTION)
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
    if args.num_control_points < 4:
        raise ValueError("--num-control-points must be >= 4 (cubic B-spline).")

    pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        time_scaling=args.time_scaling,
        objective_mode=args.objective_mode,
        fim_a_slip_max=args.fim_a_slip_max,
        criterion=args.criterion,
        start_offset_mode=args.start_offset_mode,
        offset_displacement_step_factor=args.offset_displacement_step_factor,
        offset_heading_step_factor=args.offset_heading_step_factor,
        min_tangent_fraction=args.min_tangent_fraction,
    )

    print(f"Loaded problem: {pipeline.problem.path}")
    print(f"Encoder low-pass: wheel_lp_tau = {pipeline.wheel_lp_tau:.4g} s")
    print(f"Design criterion: {pipeline.criterion}")
    print(
        f"Start offsets: {pipeline.start_offset_mode} "
        f"({'optimized with the control points' if pipeline.optimize_start_offsets else 'frozen'}, "
        f"{int(pipeline.realizations.start_offsets.shape[0])} realizations)"
    )
    if pipeline.optimize_start_offsets:
        print(
            f"  step factors: displacement {pipeline.offset_displacement_step_factor:.4g}, "
            f"heading {pipeline.offset_heading_step_factor:.4g} "
            f"(x learning rate {args.learning_rate:.4g})"
        )
    print(f"Robot: {type(pipeline.robot).__name__}")
    print(f"Time scaling: {pipeline.time_scaling}")
    print(
        "Tangent floor: "
        + (
            f"|dpos/ds| >= {pipeline.min_tangent_fraction:.4g} x rms|dpos/ds| "
            f"({pipeline.min_tangent_fraction / GUARD_TANGENT_FRACTION:.3g}x the "
            "fallback-tangent threshold)"
            if pipeline.min_tangent_fraction > 0.0
            else "off"
        )
    )
    print(f"Objective mode: {pipeline.objective_mode}")
    print(f"Optimized trajectories: {args.num_trajectories}")
    if args.num_trajectories > 1:
        print(f"B-spline control points: {args.num_control_points}")
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

    initial_control_points = pipeline.initial_control_points(args.num_control_points)
    pipeline.set_control_points(initial_control_points)

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
            optimized_control_points, loss_history = pipeline.optimize_trajectory(
                num_control_points=args.num_control_points,
                num_steps=args.opt_steps,
                learning_rate=args.learning_rate,
                window_length=args.window_length,
                save_trace=args.save_opt_GIF,
                trace_stride=args.opt_trace_stride,
                constraint_weight=args.constraint_weight,
                constraint_component_weights=constraint_component_weights,
                constraint_smooth_max_beta=args.constraint_smooth_max_beta,
            )
        else:
            optimized_control_point_batch, loss_history = pipeline.optimize_trajectories(
                num_control_points=args.num_control_points,
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
            print(f"  log(FIM):    {float(objective_terms['log_fim']):.8f}")
            print(f"  Constraints: {float(objective_terms['constraints']):.8e}")
            print("  Constraint components:")
            for name in ("v", "a", "lateral", "omega", "alpha"):
                print(f"    {name:<7}: {float(constraint_components[name]):.8e}")
            print(f"  Constraint term (weighted): {float(objective_terms['constraint_term']):.8f}")
            print(f"  Tangent floor (weighted):   {float(objective_terms['tangent_floor_term']):.8f}")
            print(f"  Total:       {float(objective_terms['total']):.8f}")
            print(f"  Constraint weight used: {selected_constraint_weight:.8g}")
            print("Optimized FIM:")
            print(pipeline.compute_fim_matrix(window_length=args.window_length))
        if optimized_control_point_batch is None:
            pipeline.plot_trajectory(
                window_length=args.window_length,
                out_prefix="traj_optimized_bezier" if output_stem is None else f"{output_stem}_optimized",
            )
        else:
            pipeline.plot_trajectory_batch(
                optimized_control_point_batch,
                out_prefix=f"{run_stem}_trajectories",
                start_offset_batch=pipeline.batch_start_offsets,
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
        if optimized_control_point_batch is None:
            saved_path = pipeline.save_reference_states_pickle(
                filename_prefix="bezier_reference_states" if output_stem is None else output_stem,
            )
            print("Saved trajectory pickle:")
            print(saved_path)
        else:
            saved_paths = []
            filename_prefix = "bezier_reference_states" if output_stem is None else output_stem
            # Check the whole batch before writing any of it: the per-export
            # check would abort partway and leave a half-written directory, and
            # one stalled design usually means the run's settings are wrong for
            # every design, which is easier to see listed together.
            stalled = {}
            for index, control_points in enumerate(optimized_control_point_batch):
                report = pipeline.tangent_diagnostics(pipeline.clamp_control_points(control_points))
                if report["guarded_samples"]:
                    stalled[index] = report
            if stalled:
                detail = "; ".join(
                    f"{index:02d}: {report['guarded_samples']}/{report['num_samples']} samples, "
                    f"min |dpos/ds| {report['min_tangent_norm']:.4g} vs threshold "
                    f"{report['guard_threshold']:.4g}"
                    for index, report in stalled.items()
                )
                raise ValueError(
                    f"{len(stalled)} of {len(optimized_control_point_batch)} designs stalled onto "
                    f"the fallback tangent and were not exported -- {detail}. Their heading is the "
                    f"constant [1, 0] over those samples, not the curve's. Raise "
                    f"--min-tangent-fraction (currently {args.min_tangent_fraction:.4g}) and rerun."
                )
            export_dir = os.path.join("trajectory_exports", f"{filename_prefix}_{run_timestamp}")
            os.makedirs(export_dir, exist_ok=True)
            for index, control_points in enumerate(optimized_control_point_batch):
                clamped_control_points = pipeline.clamp_control_points(control_points)
                saved_paths.append(
                    pipeline.save_reference_states_pickle(
                        out_dir=export_dir,
                        filename_prefix=f"{filename_prefix}_{index:02d}",
                        reference_states=pipeline.reference_states_from_control_points(
                            clamped_control_points
                        ),
                        # The curve itself, not just its samples: a warm start
                        # picks up the decision variables directly.
                        control_points=clamped_control_points,
                        # Each trajectory ships the offsets it was designed
                        # under, which the gain tuner then tunes on. In
                        # identification mode there are none: the start is where
                        # the robot is placed.
                        start_offsets=(
                            pipeline.batch_start_offsets[index]
                            if pipeline.objective_mode == OBJECTIVE_MODE_GAIN_TUNING
                            else None
                        ),
                    )
                )
            print("Saved trajectory pickles:")
            print(export_dir)
            for saved_path in saved_paths:
                print(saved_path)


if __name__ == "__main__":
    main()
