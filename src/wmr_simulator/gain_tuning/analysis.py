import os
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline
from wmr_simulator.types import PhysicalParams
from wmr_simulator.visualization.animation import create_gif_from_png_frames
from wmr_simulator.visualization.identification import (
    plot_tracking_error_frame,
    plot_tracking_error_surface,
    plot_trajectory,
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


def apply_reference_window(
    pipeline: ControllerTuningPipeline,
    start_idx: int,
    end_idx: int | None,
) -> tuple[int, int]:
    full_reference_states = pipeline.reference_states
    window_start, window_end = resolve_reference_window(len(full_reference_states), start_idx, end_idx)
    clipped_reference_states = full_reference_states[window_start:window_end]
    start_pose = clipped_reference_states[0, :3]

    def init_states_for_window(self, robot_key, estimator_key, reference_states=None, start_pose=None):
        robot_state0 = self.robot.get_init_state(key=robot_key, init_pose=start_pose)
        est_state0 = self.estimator.get_init_state(key=estimator_key, start_pose=start_pose)
        ctrl_state0 = jnp.zeros(2, dtype=jnp.float32)
        delayed_wheel_ref0 = jnp.zeros(2, dtype=jnp.float32)
        geometry_state0 = self.controller.initial_geometry_state()
        return robot_state0, est_state0, ctrl_state0, delayed_wheel_ref0, geometry_state0

    pipeline.full_reference_states = full_reference_states
    pipeline.reference_states = clipped_reference_states
    pipeline.sim_time_grid = pipeline.sim_time_grid[window_start:window_end + 1]
    pipeline._init_states = MethodType(init_states_for_window, pipeline)
    return window_start, window_end


def make_gain_surface_pipeline(
    problem_path: str,
    robot_params: PhysicalParams,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
) -> ControllerTuningPipeline:
    return ControllerTuningPipeline(
        problem_path=problem_path,
        robot_params=robot_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
    )


def evaluate_gain_tracking_error_surface(
    pipeline: ControllerTuningPipeline,
    kx_min: float,
    kx_max: float,
    kx_points: int,
    ky_min: float,
    ky_max: float,
    ky_points: int,
    num_realizations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx_values = build_parameter_grid(kx_min, kx_max, kx_points)
    ky_values = build_parameter_grid(ky_min, ky_max, ky_points)
    robot_keys = jax.random.split(pipeline.robot_key, num_realizations)
    estimator_keys = jax.random.split(pipeline.estimator_key, num_realizations)
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


def create_initial_gains_marker(pipeline: ControllerTuningPipeline) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=jnp.asarray(pipeline.gains[0], dtype=jnp.float32),
        base_diameter=jnp.asarray(pipeline.gains[1], dtype=jnp.float32),
    )


def save_gain_tracking_error_plots(
    pipeline: ControllerTuningPipeline,
    kx_values: np.ndarray,
    ky_values: np.ndarray,
    tracking_error_surface: np.ndarray,
    num_realizations: int,
    seed: int,
    out_prefix: str = "gain_tracking_error_surface",
    z_min: float | None = None,
    z_max: float | None = None,
    combined_out_path: str | None = None,
    trajectory_out_path: str | None = None,
    surface_out_path: str | None = None,
) -> None:
    init_gains = create_initial_gains_marker(pipeline)

    if combined_out_path is not None:
        plot_tracking_error_frame(
            pipeline=pipeline,
            wheel_radius_values=kx_values,
            base_diameter_values=ky_values,
            tracking_error_surface=tracking_error_surface,
            init_params=init_gains,
            num_realizations=num_realizations,
            seed=seed,
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
        num_realizations=num_realizations,
        seed=seed,
        out_prefix=out_prefix,
        out_path=surface_out_path,
        x_label="kx",
        y_label="ky",
        surface_label="Closed-loop loss",
        title="Closed-Loop Loss Surface",
        init_label="Initial gains",
    )


def run_gain_tracking_error_surface(
    problem_path: str,
    robot_params: PhysicalParams,
    kx_min: float = 0.0,
    kx_max: float = 20.0,
    kx_points: int = 100,
    ky_min: float = 0.0,
    ky_max: float = 20.0,
    ky_points: int = 100,
    num_realizations: int = 2,
    seed: int = 0,
    start_idx: int = 0,
    end_idx: int | None = None,
    reference_trajectories_dir: str | None = None,
    out_prefix: str = "gain_tracking_error_surface",
    save_plots: bool = True,
):
    pipeline = make_gain_surface_pipeline(
        problem_path=problem_path,
        robot_params=robot_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
    )
    window_start, window_end = apply_reference_window(pipeline, start_idx, end_idx)

    kx_values, ky_values, tracking_error_surface = evaluate_gain_tracking_error_surface(
        pipeline=pipeline,
        kx_min=kx_min,
        kx_max=kx_max,
        kx_points=kx_points,
        ky_min=ky_min,
        ky_max=ky_max,
        ky_points=ky_points,
        num_realizations=num_realizations,
    )

    if save_plots:
        save_gain_tracking_error_plots(
            pipeline=pipeline,
            kx_values=kx_values,
            ky_values=ky_values,
            tracking_error_surface=tracking_error_surface,
            num_realizations=num_realizations,
            seed=seed,
            out_prefix=out_prefix,
        )

    return {
        "pipeline": pipeline,
        "window_start": window_start,
        "window_end": window_end,
        "kx_values": kx_values,
        "ky_values": ky_values,
        "tracking_error_surface": tracking_error_surface,
    }


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

    try:
        from scipy.optimize import linprog
    except ImportError:
        global_max = float(max(max_values))
        return [global_max for _ in max_values]

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
        global_max = float(max(max_values))
        return [global_max for _ in max_values]

    fitted_values = vandermonde @ result.x
    fitted_values = np.maximum(fitted_values, np.asarray(max_values, dtype=float))
    return [float(value) for value in fitted_values]


def render_gain_surface_frames(
    problem_path: str,
    robot_params: PhysicalParams,
    output_dir_name: str = "gain_tracking_error_surface_frames",
    kx_min: float = 0.0,
    kx_max: float = 20.0,
    kx_points: int = 100,
    ky_min: float = 0.0,
    ky_max: float = 20.0,
    ky_points: int = 100,
    num_realizations: int = 2,
    seed: int = 0,
    window_size: int = 60,
    window_stride: int = 5,
    z_poly_order: int = 0,
    reference_trajectories_dir: str | None = None,
) -> str:
    base_pipeline = make_gain_surface_pipeline(
        problem_path=problem_path,
        robot_params=robot_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
    )
    frame_windows = build_frame_windows(len(base_pipeline.reference_states), window_size, window_stride)
    output_dir = os.path.join("visualize", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"Generating {len(frame_windows)} gain surface frames into {output_dir} "
        f"(window_size={window_size}, window_stride={window_stride})"
    )

    frame_data = []
    for frame_idx, (start_idx, end_idx) in enumerate(frame_windows):
        print(
            f"  evaluating frame {frame_idx + 1:>3}/{len(frame_windows)}: "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )
        pipeline = make_gain_surface_pipeline(
            problem_path=problem_path,
            robot_params=robot_params,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
        )
        apply_reference_window(pipeline, start_idx, end_idx)
        kx_values, ky_values, tracking_error_surface = evaluate_gain_tracking_error_surface(
            pipeline=pipeline,
            kx_min=kx_min,
            kx_max=kx_max,
            kx_points=kx_points,
            ky_min=ky_min,
            ky_max=ky_max,
            ky_points=ky_points,
            num_realizations=num_realizations,
        )
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
        z_poly_order,
    )

    print(
        f"Rendering {len(frame_data)} frames with polynomial z-axis upper envelope "
        f"(order={z_poly_order})"
    )
    for frame, z_max in zip(frame_data, polynomial_loss_maxs):
        print(
            f"  rendering frame {frame['frame_idx'] + 1:>3}/{len(frame_data)}: "
            f"start_idx={frame['start_idx']}, end_idx={frame['end_idx']}  "
            f"z_limits=[{global_loss_min:.8f}, {z_max:.8f}]"
        )
        pipeline = make_gain_surface_pipeline(
            problem_path=problem_path,
            robot_params=robot_params,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
        )
        apply_reference_window(pipeline, frame["start_idx"], frame["end_idx"])
        frame_tag = f"{frame['frame_idx']:04d}"
        save_gain_tracking_error_plots(
            pipeline=pipeline,
            kx_values=frame["kx_values"],
            ky_values=frame["ky_values"],
            tracking_error_surface=frame["tracking_error_surface"],
            num_realizations=num_realizations,
            seed=seed,
            z_min=global_loss_min,
            z_max=z_max,
            combined_out_path=os.path.join(output_dir, f"frame_{frame_tag}.png"),
        )

    return output_dir


def create_gain_surface_gif(
    problem_path: str,
    robot_params: PhysicalParams,
    output_dir_name: str = "gain_tracking_error_surface_frames",
    gif_name: str = "gain_surface_animation.gif",
    frame_time: float = 0.02,
    stop_time: float = 1.0,
    **surface_kwargs,
) -> str:
    frames_dir = render_gain_surface_frames(
        problem_path=problem_path,
        robot_params=robot_params,
        output_dir_name=output_dir_name,
        **surface_kwargs,
    )
    return create_gif_from_png_frames(
        frames_dir=frames_dir,
        gif_name=gif_name,
        frame_time=frame_time,
        stop_time=stop_time,
    )
