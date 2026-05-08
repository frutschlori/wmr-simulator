import argparse
import os
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy.optimize import linprog

from wmr_simulator.system_identification import PhysicalParams
from wmr_simulator.system_identification import ControllerTuningPipeline
from wmr_simulator.SI_visualization import (
    plot_tracking_error_frame,
    plot_tracking_error_surface,
    plot_trajectory
)


def build_parameter_grid(min_value: float, max_value: float, num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("Grid resolution must be at least 2.")
    if min_value >= max_value:
        raise ValueError("Grid minimum must be smaller than grid maximum.")
    return np.linspace(min_value, max_value, num_points)


def resolve_reference_window(num_steps: int, start_idx: int, end_idx: int | None) -> tuple[int, int]:
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative.")
    resolved_end_idx = num_steps if end_idx is None else end_idx
    if resolved_end_idx > num_steps:
        raise ValueError("end_idx must not exceed the reference trajectory length.")
    if start_idx >= resolved_end_idx:
        raise ValueError("start_idx must be smaller than end_idx.")
    return start_idx, resolved_end_idx


def apply_reference_window(pipeline: ControllerTuningPipeline, start_idx: int, end_idx: int | None):
    full_reference_states = pipeline.reference_states
    window_start, window_end = resolve_reference_window(len(full_reference_states), start_idx, end_idx)
    clipped_reference_states = full_reference_states[window_start:window_end]
    start_pose = clipped_reference_states[0, :3]

    def init_states_for_window(self, robot_key, estimator_key):
        robot_state0 = self.robot.get_init_state(key=robot_key, init_pose=start_pose)
        est_state0 = self.estimator.get_init_state(key=estimator_key, start_pose=start_pose)
        ctrl_state0 = jnp.zeros(2, dtype=jnp.float32)
        return robot_state0, est_state0, ctrl_state0

    pipeline.full_reference_states = full_reference_states
    pipeline.reference_states = clipped_reference_states
    pipeline.sim_time_grid = pipeline.sim_time_grid[window_start:window_end + 1]
    pipeline._init_states = MethodType(init_states_for_window, pipeline)

    return window_start, window_end


def create_pipeline(args):
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
    )
    pipeline = ControllerTuningPipeline(
        problem_path=args.problem,
        robot_params=init_params,
        seed=args.seed)
    return init_params, pipeline


def evaluate_tracking_error_surface(pipeline: ControllerTuningPipeline, args):
    kx_values = build_parameter_grid(args.kx_min, args.kx_max, args.kx_points)
    ky_values = build_parameter_grid(args.ky_min, args.ky_max, args.ky_points)
    robot_keys = jax.random.split(pipeline.robot_key, args.num_realizations)
    estimator_keys = jax.random.split(pipeline.estimator_key, args.num_realizations)
    kx_values_jax = jnp.asarray(kx_values, dtype=jnp.float32)
    ky_values_jax = jnp.asarray(ky_values, dtype=jnp.float32)

    def loss_for_gains(kx, ky):
        gains = pipeline.gains.at[0].set(kx).at[1].set(ky)
        return pipeline.loss(gains, robot_keys, estimator_keys)

    batched_loss_for_row = jax.vmap(loss_for_gains, in_axes=(0, None))

    def scan_row(_, ky):
        row_losses = batched_loss_for_row(kx_values_jax, ky)
        return None, row_losses

    evaluate_surface = jax.jit(lambda: jax.lax.scan(scan_row, None, ky_values_jax)[1])
    tracking_error_surface = np.asarray(evaluate_surface())
    return kx_values, ky_values, tracking_error_surface


def create_init_gains_marker(pipeline: ControllerTuningPipeline):
    return PhysicalParams(
        wheel_radius=jnp.asarray(pipeline.gains[0], dtype=jnp.float32),
        base_diameter=jnp.asarray(pipeline.gains[1], dtype=jnp.float32),
    )


def save_tracking_error_plots(
    pipeline: ControllerTuningPipeline,
    args,
    kx_values,
    ky_values,
    tracking_error_surface,
    z_min=None,
    z_max=None,
    combined_out_path=None,
    trajectory_out_path=None,
    surface_out_path=None,
):
    init_gains = create_init_gains_marker(pipeline)

    if combined_out_path is not None:
        plot_tracking_error_frame(
            pipeline=pipeline,
            wheel_radius_values=kx_values,
            base_diameter_values=ky_values,
            tracking_error_surface=tracking_error_surface,
            init_params=init_gains,
            num_realizations=args.num_realizations,
            seed=args.seed,
            out_path=combined_out_path,
            z_min=z_min,
            z_max=z_max,
            x_label="kx",
            y_label="ky",
            surface_label="Closed-loop loss",
            init_label="Initial gains",
        )
        return

    plot_trajectory(
        pipeline=pipeline,
        out_prefix="surface_reference",
        out_path=trajectory_out_path,
    )
    plot_tracking_error_surface(
        wheel_radius_values=kx_values,
        base_diameter_values=ky_values,
        tracking_error_surface=tracking_error_surface,
        hidden_params=None,
        init_params=init_gains,
        num_realizations=args.num_realizations,
        seed=args.seed,
        out_prefix=args.output,
        out_path=surface_out_path,
        x_label="kx",
        y_label="ky",
        surface_label="Closed-loop loss",
        title="Closed-Loop Loss Surface",
        init_label="Initial gains",
    )


def run_tracking_error_surface(args):
    _, pipeline = create_pipeline(args)
    start_idx, end_idx = apply_reference_window(pipeline, args.start_idx, args.end_idx)

    print(
        f"Evaluating closed-loop loss surface over controller gains for seed={args.seed} "
        f"with num_realizations={args.num_realizations}"
    )
    print(
        f"  reference window: start_idx={start_idx}, end_idx={end_idx} "
        f"(num_steps={pipeline.reference_states.shape[0]})"
    )
    print(
        f"  compiling and evaluating {args.ky_points} rows x "
        f"{args.kx_points} columns"
    )
    kx_values, ky_values, tracking_error_surface = evaluate_tracking_error_surface(pipeline, args)
    save_tracking_error_plots(pipeline, args, kx_values, ky_values, tracking_error_surface)


def build_frame_windows(num_steps: int, window_size: int, window_stride: int) -> list[tuple[int, int]]:
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if window_stride <= 0:
        raise ValueError("window_stride must be positive.")

    windows = []
    for start_idx in range(0, num_steps, window_stride):
        end_idx = min(start_idx + window_size, num_steps)
        if end_idx - start_idx < 2:
            break
        windows.append((start_idx, end_idx))
        if end_idx == num_steps:
            break
    return windows


def compute_polynomial_z_max(max_values: list[float], polynomial_order: int) -> list[float]:
    if polynomial_order < 0:
        raise ValueError("z_poly_order must be non-negative.")
    if not max_values:
        return []

    if len(max_values) == 1:
        return [float(max_values[0])]

    scaled_indices = np.linspace(-1.0, 1.0, len(max_values), dtype=float)
    vandermonde = np.vander(scaled_indices, N=polynomial_order + 1, increasing=True)
    objective = np.sum(vandermonde, axis=0)

    result = linprog(
        c=objective,
        A_ub=-vandermonde,
        b_ub=-np.asarray(max_values, dtype=float),
        bounds=[(None, None)] * vandermonde.shape[1],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Polynomial z-limit fit failed: {result.message}")

    fitted_values = vandermonde @ result.x
    fitted_values = np.maximum(fitted_values, np.asarray(max_values, dtype=float))
    return [float(value) for value in fitted_values]


def run_gif_frame_sweep(args):
    _, base_pipeline = create_pipeline(args)
    total_steps = len(base_pipeline.reference_states)
    frame_windows = build_frame_windows(total_steps, args.window_size, args.window_stride)
    output_dir = os.path.join("visualize", args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Generating {len(frame_windows)} frame windows into {output_dir} "
        f"(window_size={args.window_size}, window_stride={args.window_stride})"
    )

    frame_data = []
    for frame_idx, (start_idx, end_idx) in enumerate(frame_windows):
        print(
            f"  evaluating frame {frame_idx + 1:>3}/{len(frame_windows)}: "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )
        _, pipeline = create_pipeline(args)
        apply_reference_window(pipeline, start_idx, end_idx)
        kx_values, ky_values, tracking_error_surface = evaluate_tracking_error_surface(pipeline, args)
        frame_data.append(
            {
                "frame_idx": frame_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "kx_values": kx_values,
                "ky_values": ky_values,
                "tracking_error_surface": tracking_error_surface,
                "loss_min": float(np.min(tracking_error_surface)),
                "loss_max": float(np.max(tracking_error_surface)),
            }
        )

    global_loss_min = float(min(frame["loss_min"] for frame in frame_data))
    polynomial_loss_maxs = compute_polynomial_z_max(
        [frame["loss_max"] for frame in frame_data],
        args.z_poly_order,
    )

    print(
        f"Rendering {len(frame_data)} frames with polynomial z-axis upper envelope "
        f"(order={args.z_poly_order})"
    )

    for frame, z_max in zip(frame_data, polynomial_loss_maxs):
        frame_idx = frame["frame_idx"]
        start_idx = frame["start_idx"]
        end_idx = frame["end_idx"]
        print(
            f"  rendering frame {frame_idx + 1:>3}/{len(frame_data)}: "
            f"start_idx={start_idx}, end_idx={end_idx}  "
            f"z_limits=[{global_loss_min:.8f}, {z_max:.8f}]"
        )
        _, pipeline = create_pipeline(args)
        apply_reference_window(pipeline, start_idx, end_idx)
        frame_tag = f"{frame_idx:04d}"
        save_tracking_error_plots(
            pipeline,
            args,
            frame["kx_values"],
            frame["ky_values"],
            frame["tracking_error_surface"],
            z_min=global_loss_min,
            z_max=z_max,
            combined_out_path=os.path.join(output_dir, f"frame_{frame_tag}.png"),
        )


def create_gif_from_frames(args):
    frames_dir = os.path.join("visualize", args.frames_dir)
    if not os.path.isdir(frames_dir):
        raise ValueError(f"Frame directory does not exist: {frames_dir}")

    frame_names = sorted(
        file_name for file_name in os.listdir(frames_dir)
        if file_name.lower().endswith(".png")
    )
    if not frame_names:
        raise ValueError(f"No PNG frames found in: {frames_dir}")

    gif_path = os.path.join(frames_dir, args.gif_name)
    frame_duration_ms = int(1000 * args.frame_time)
    stop_duration_ms = int(1000 * args.stop_time)

    frames = []
    for frame_name in frame_names:
        frame_path = os.path.join(frames_dir, frame_name)
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGBA"))

    durations = [frame_duration_ms] * len(frames)
    durations[-1] = stop_duration_ms

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    print(f"Saved GIF to: {gif_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-type", type=str, choices=["tracking_error_surface", "gif_frames", "create_gif"],
                        default="tracking_error_surface")
    parser.add_argument("--problem", type=str, default="problems/figure_eight.yaml")

    # Tracking error surface plot arguments
    parser.add_argument("--num-realizations", type=int, default=2) # replay realizations
    parser.add_argument("--init-wheel-radius", type=float, default=0.015)
    parser.add_argument("--init-base-diameter", type=float, default=0.09)
    parser.add_argument("--kx-min", type=float, default=0.0)
    parser.add_argument("--kx-max", type=float, default=20.0)
    parser.add_argument("--kx-points", type=int, default=100)
    parser.add_argument("--ky-min", type=float, default=0.0)
    parser.add_argument("--ky-max", type=float, default=20.0)
    parser.add_argument("--ky-points", type=int, default=100)
    parser.add_argument("--start-idx", type=int, default=380)
    parser.add_argument("--end-idx", type=int, default=None)

    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--z-poly-order", type=int, default=0)

    # GIF mode
    parser.add_argument("--frames-dir", type=str, default="tracking_error_surface_frames")
    parser.add_argument("--gif-name", type=str, default="animation.gif")
    parser.add_argument("--frame-time", type=float, default=0.2)
    parser.add_argument("--stop-time", type=float, default=1.0)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # args.plot_type = "tracking_error_surface"
    args.plot_type = "gif_frames"

    if args.output is None:
        args.output = "tracking_error_surface_frames" if args.plot_type == "gif_frames" else "tracking_error_surface"

    if args.plot_type == "tracking_error_surface":
        run_tracking_error_surface(args)
    elif args.plot_type == "gif_frames":
        run_gif_frame_sweep(args)
        create_gif_from_frames(args)
