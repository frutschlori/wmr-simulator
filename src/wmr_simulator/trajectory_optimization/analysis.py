import os

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.analysis import (
    apply_reference_states_to_pipeline,
    build_physical_parameter_surface,
    make_surface_pipeline,
)
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline
from wmr_simulator.types import PhysicalParams
from wmr_simulator.visualization.animation import (
    create_gif_from_png_frames,
    stack_png_frame_directories_vertically,
)
from wmr_simulator.visualization.identification import plot_tracking_error_surface


def render_tracking_surface_frames_for_optimization_trace(
    problem_path: str,
    snapshots,
    initial_params: PhysicalParams,
    output_dir_name: str,
    radius_min: float = 0.01,
    radius_max: float = 0.1,
    radius_points: int = 100,
    base_min: float = 0.01,
    base_max: float = 0.5,
    base_points: int = 100,
    num_realizations: int = 1,
    seed: int = 0,
    window_length: int | None = None,
) -> str:
    if not snapshots:
        raise ValueError("No optimization snapshots available for surface rendering.")

    if os.path.dirname(output_dir_name):
        output_dir = output_dir_name
    else:
        output_dir = os.path.join("visualize", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    frame_data = []
    print(f"Evaluating {len(snapshots)} tracking-error surface frames")
    for frame_idx, snapshot in enumerate(snapshots):
        print(f"  evaluate {frame_idx + 1:>3}/{len(snapshots)}  step={snapshot.step}")
        surface_pipeline = make_surface_pipeline(
            problem_path=problem_path,
            initial_params=initial_params,
            seed=seed,
            window_length=window_length,
        )
        apply_reference_states_to_pipeline(surface_pipeline, np.asarray(snapshot.reference_states, dtype=float))
        wheel_radius_values, base_diameter_values, tracking_error_surface = build_physical_parameter_surface(
            pipeline=surface_pipeline,
            radius_min=radius_min,
            radius_max=radius_max,
            radius_points=radius_points,
            base_min=base_min,
            base_max=base_max,
            base_points=base_points,
        )
        frame_data.append(
            {
                "step": snapshot.step,
                "wheel_radius_values": wheel_radius_values,
                "base_diameter_values": base_diameter_values,
                "tracking_error_surface": tracking_error_surface,
                "loss_min": float(np.min(tracking_error_surface)),
                "loss_max": float(np.max(tracking_error_surface)),
            }
        )

    global_z_min = float(min(frame["loss_min"] for frame in frame_data))
    global_z_max = float(max(frame["loss_max"] for frame in frame_data))

    print(
        f"Rendering {len(frame_data)} surface frames into {output_dir} "
        f"with z-limits [{global_z_min:.8f}, {global_z_max:.8f}]"
    )
    for frame_idx, (snapshot, frame) in enumerate(zip(snapshots, frame_data)):
        surface_pipeline = make_surface_pipeline(
            problem_path=problem_path,
            initial_params=initial_params,
            seed=seed,
            window_length=window_length,
        )
        apply_reference_states_to_pipeline(surface_pipeline, np.asarray(snapshot.reference_states, dtype=float))
        out_path = os.path.join(output_dir, f"frame_{snapshot.step:05d}.png")
        plot_tracking_error_surface(
            wheel_radius_values=frame["wheel_radius_values"],
            base_diameter_values=frame["base_diameter_values"],
            tracking_error_surface=frame["tracking_error_surface"],
            hidden_params=surface_pipeline.hidden_params,
            init_params=initial_params,
            num_realizations=num_realizations,
            seed=seed,
            out_path=out_path,
            z_min=global_z_min,
            z_max=global_z_max,
            title=f"Tracking Error Surface - Iteration {snapshot.step}",
        )
        print(f"  render {frame_idx + 1:>3}/{len(frame_data)}  step={snapshot.step}")

    return output_dir


def create_stacked_tracking_surface_trace_gif(
    trajectory_frames_dir: str,
    surface_frames_dir: str,
    output_dir_name: str,
    gif_name: str,
    frame_time: float = 0.2,
    stop_time: float = 1.0,
    gif_path: str | None = None,
) -> str:
    if os.path.dirname(output_dir_name):
        output_dir = output_dir_name
    else:
        output_dir = os.path.join("visualize", output_dir_name)
    stack_png_frame_directories_vertically(
        top_frames_dir=trajectory_frames_dir,
        bottom_frames_dir=surface_frames_dir,
        output_dir=output_dir,
    )
    return create_gif_from_png_frames(
        frames_dir=output_dir,
        gif_name=gif_name,
        frame_time=frame_time,
        stop_time=stop_time,
        gif_path=gif_path,
    )


def run_trajectory_optimization_trace(
    problem_path: str,
    window_length: int | None = 50,
    time_scaling: str | None = None,
    num_segments: int = 7,
    num_steps: int = 2000,
    learning_rate: float = 1e-2,
    trace_stride: int = 50,
    constraint_weight: float = 1.0,
    constraint_component_weights: dict | None = None,
    constraint_smooth_max_beta: float = 20.0,
    out_prefix: str = "traj_opt",
    frame_duration: float = 0.2,
    export_reference_states: bool = False,
    save_final_reference_states: bool = False,
    include_tracking_surface: bool = True,
    surface_initial_params: PhysicalParams | None = None,
    surface_radius_min: float = 0.01,
    surface_radius_max: float = 0.1,
    surface_radius_points: int = 100,
    surface_base_min: float = 0.01,
    surface_base_max: float = 0.5,
    surface_base_points: int = 100,
    surface_num_realizations: int = 1,
    surface_seed: int = 0,
    surface_window_length: int | None = None,
    frames_root: str = os.path.join("visualize", "Trajectory Optimization Frames"),
):
    pipeline = TrajectoryOptimizationPipeline(
        problem_path,
        time_scaling=time_scaling,
    )

    initial_control_points = pipeline.initial_control_points(num_segments)
    pipeline.set_control_points(initial_control_points)
    optimized_control_points, loss_history = pipeline.optimize_trajectory(
        num_segments=num_segments,
        num_steps=num_steps,
        learning_rate=learning_rate,
        window_length=window_length,
        save_trace=True,
        trace_stride=trace_stride,
        constraint_weight=constraint_weight,
        constraint_component_weights=constraint_component_weights,
        constraint_smooth_max_beta=constraint_smooth_max_beta,
    )
    objective_terms = pipeline.objective_terms_from_control_points(
        optimized_control_points,
        window_length=window_length,
        constraint_weight=constraint_weight,
        constraint_component_weights=constraint_component_weights,
        constraint_smooth_max_beta=constraint_smooth_max_beta,
    )
    pipeline.plot_trajectory(
        window_length=window_length,
        out_prefix=f"{out_prefix}_final_trajectory",
    )
    pipeline.plot_loss_history(out_prefix=f"{out_prefix}_loss_history")
    os.makedirs(frames_root, exist_ok=True)
    trajectory_frames_dir = os.path.join(frames_root, f"{out_prefix}_trajectory_frames")
    trajectory_gif_path = os.path.join(frames_root, f"{out_prefix}.gif")
    tracking_surface_frame_dir_name = os.path.join(frames_root, f"{out_prefix}_tracking_surface_frames")
    stacked_frame_dir_name = os.path.join(frames_root, f"{out_prefix}_stacked_frames")
    stacked_gif_output_path = os.path.join(frames_root, f"{out_prefix}_tracking_surface.gif")

    trajectory_gif_path = pipeline.save_optimization_GIF(
        window_length=window_length,
        out_prefix=out_prefix,
        frame_duration=frame_duration,
        frames_dir=trajectory_frames_dir,
        gif_path=trajectory_gif_path,
    )

    tracking_surface_frames_dir = None
    stacked_gif_path = None
    if include_tracking_surface:
        if surface_initial_params is None:
            surface_initial_params = PhysicalParams(
                wheel_radius=jnp.asarray(0.02, dtype=jnp.float32),
                base_diameter=jnp.asarray(0.2, dtype=jnp.float32),
            )
        if surface_window_length is None:
            surface_window_length = window_length

        tracking_surface_frames_dir = render_tracking_surface_frames_for_optimization_trace(
            problem_path=problem_path,
            snapshots=pipeline.optimization_snapshots,
            initial_params=surface_initial_params,
            output_dir_name=tracking_surface_frame_dir_name,
            radius_min=surface_radius_min,
            radius_max=surface_radius_max,
            radius_points=surface_radius_points,
            base_min=surface_base_min,
            base_max=surface_base_max,
            base_points=surface_base_points,
            num_realizations=surface_num_realizations,
            seed=surface_seed,
            window_length=surface_window_length,
        )
        stacked_gif_path = create_stacked_tracking_surface_trace_gif(
            trajectory_frames_dir=trajectory_frames_dir,
            surface_frames_dir=tracking_surface_frames_dir,
            output_dir_name=stacked_frame_dir_name,
            gif_name=f"{out_prefix}_tracking_surface.gif",
            frame_time=frame_duration,
            gif_path=stacked_gif_output_path,
        )

    export_dir = None
    saved_reference_paths = []
    if export_reference_states:
        export_dir, saved_reference_paths = pipeline.save_optimization_reference_states(
            filename_prefix=f"{out_prefix}_reference_states",
        )

    final_reference_path = None
    if save_final_reference_states:
        final_reference_path = pipeline.save_reference_states_pickle(
            filename_prefix=f"{out_prefix}_final_reference_states",
        )

    return {
        "pipeline": pipeline,
        "initial_control_points": np.asarray(initial_control_points),
        "optimized_control_points": np.asarray(optimized_control_points),
        "loss_history": loss_history,
        "objective_terms": {name: float(value) for name, value in objective_terms.items()},
        "frames_root": frames_root,
        "trajectory_frames_dir": trajectory_frames_dir,
        "trajectory_gif_path": trajectory_gif_path,
        "tracking_surface_frames_dir": tracking_surface_frames_dir,
        "stacked_gif_path": stacked_gif_path,
        "export_dir": export_dir,
        "saved_reference_paths": saved_reference_paths,
        "final_reference_path": final_reference_path,
    }
