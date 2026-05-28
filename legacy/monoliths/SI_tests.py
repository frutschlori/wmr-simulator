import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from wmr_simulator.system_identification import PhysicalParams, SystemIdentificationPipeline
from wmr_simulator.simultaneous_optimization_online import IDandGainPipeline
from wmr_simulator.SI_visualization import (
    plot_system_id_realization_sweep,
    plot_tracking_error_surface,
    plot_trajectory,
)


def parameter_mse(estimated_params: PhysicalParams, hidden_params: PhysicalParams) -> float:
    estimated = jnp.asarray(
        [estimated_params.wheel_radius, estimated_params.base_diameter], dtype=jnp.float32
    )
    hidden = jnp.asarray(
        [hidden_params.wheel_radius, hidden_params.base_diameter], dtype=jnp.float32
    )
    return float(jnp.mean((estimated - hidden) ** 2))


def evaluate_seed(problem_path, init_params, steps, learning_rate, num_realizations, seed,
                  reference_trajectories_dir=None, window_length=None):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=init_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
    )

    estimated_params, loss_history, _ = pipeline.optimize(
        init_params=init_params,
        num_steps=steps,
        learning_rate=learning_rate,
        num_realizations=num_realizations,
    )
    final_tracking_loss = float(loss_history[-1])
    final_param_mse = parameter_mse(estimated_params, pipeline.hidden_params)
    return final_tracking_loss, final_param_mse, estimated_params


def evaluate_seed_task(task):
    return evaluate_seed(*task)


def build_realization_sweep_values(max_num_realizations: int) -> list[int]:
    if max_num_realizations <= 0:
        raise ValueError("max_num_realizations must be positive.")

    sweep_values = [2 ** i for i in range(int(jnp.floor(jnp.log2(max_num_realizations))) + 1)]
    if sweep_values[-1] != max_num_realizations:
        sweep_values.append(max_num_realizations)
    return sweep_values


def run_realization_sweep(args):
    sweep_values = build_realization_sweep_values(args.max_num_realizations)
    seeds = [args.seed + offset for offset in range(args.num_seeds)]
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
    )

    tracking_loss_means = []
    param_mse_means = []

    for num_realizations in sweep_values:
        print(
            f"\nRunning system ID with num_realizations={num_realizations} "
            f"averaged over {args.num_seeds} seeds"
        )
        tasks = [
            (
                args.problem,
                init_params,
                args.steps,
                args.learning_rate,
                num_realizations,
                seed,
                args.reference_trajectories_dir,
                args.window_length,
            )
            for seed in seeds
        ]

        if args.num_workers == 1:
            results = [evaluate_seed_task(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                results = list(executor.map(evaluate_seed_task, tasks))

        tracking_losses = []
        param_mse_values = []

        for seed, (final_tracking_loss, final_param_mse, estimated_params) in zip(seeds, results):
            tracking_losses.append(final_tracking_loss)
            param_mse_values.append(final_param_mse)

            print(
                f"  seed={seed:>3}  "
                f"final_tracking_loss={final_tracking_loss:.8f}  "
                f"parameter_mse={final_param_mse:.8f}  "
                f"estimated_radius={float(estimated_params.wheel_radius):.6f}  "
                f"estimated_base={float(estimated_params.base_diameter):.6f}"
            )

        tracking_loss_mean = float(np.mean(tracking_losses))
        param_mse_mean = float(np.mean(param_mse_values))
        tracking_loss_means.append(tracking_loss_mean)
        param_mse_means.append(param_mse_mean)

        print(
            f"  mean_tracking_loss={tracking_loss_mean:.8f}  "
            f"mean_parameter_mse={param_mse_mean:.8f}"
        )

    plot_system_id_realization_sweep(
        num_realizations=sweep_values,
        tracking_losses=tracking_loss_means,
        parameter_mse=param_mse_means,
        num_seeds=args.num_seeds,
        out_prefix=args.output,
    )

    print("\nSweep summary:")
    for num_realizations, tracking_loss, param_mse_value in zip(
        sweep_values, tracking_loss_means, param_mse_means
    ):
        print(
            f"  num_realizations={num_realizations:>3}  "
            f"mean_tracking_loss={tracking_loss:.8f}  "
            f"mean_parameter_mse={param_mse_value:.8f}"
        )


def build_parameter_grid(min_value: float, max_value: float, num_points: int) -> np.ndarray:
    if num_points < 2:
        raise ValueError("Grid resolution must be at least 2.")
    if min_value >= max_value:
        raise ValueError("Grid minimum must be smaller than grid maximum.")
    return np.linspace(min_value, max_value, num_points)


def list_reference_trajectory_pickles(reference_trajectories_dir: str) -> list[str]:
    if not os.path.isdir(reference_trajectories_dir):
        raise ValueError(f"Reference trajectory directory does not exist: {reference_trajectories_dir}")

    pickle_paths = sorted(
        os.path.join(reference_trajectories_dir, filename)
        for filename in os.listdir(reference_trajectories_dir)
        if filename.endswith(".pkl")
    )
    if not pickle_paths:
        raise ValueError(f"No pickle files found in: {reference_trajectories_dir}")
    return pickle_paths


def load_reference_states(reference_trajectory_path: str) -> np.ndarray:
    with open(reference_trajectory_path, "rb") as file:
        reference_states = pickle.load(file)

    reference_states = np.asarray(reference_states, dtype=float)
    if reference_states.ndim != 2 or reference_states.shape[1] != 8:
        raise ValueError(
            f"Loaded reference states from {reference_trajectory_path} must have shape (N, 8), "
            f"got {reference_states.shape}"
        )
    return reference_states


def apply_reference_states_to_pipeline(pipeline: SystemIdentificationPipeline, reference_states: np.ndarray):
    pipeline.reference_states = jnp.asarray(
        pipeline._extend_reference_states(reference_states),
        dtype=jnp.float32,
    )
    pipeline.loaded_reference_trajectory_path = None
    pipeline.target_log = pipeline.simulate(
        pipeline.initial_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
    )


def build_tracking_error_surface_data(pipeline, args):
    wheel_radius_values = build_parameter_grid(args.radius_min, args.radius_max, args.radius_points)
    base_diameter_values = build_parameter_grid(args.base_min, args.base_max, args.base_points)
    robot_keys = jax.random.split(pipeline.robot_key, args.num_realizations)
    estimator_keys = jax.random.split(pipeline.estimator_key, args.num_realizations)
    wheel_radius_values_jax = jnp.asarray(wheel_radius_values, dtype=jnp.float32)
    base_diameter_values_jax = jnp.asarray(base_diameter_values, dtype=jnp.float32)

    def loss_for_params(wheel_radius, base_diameter):
        dec_variables = PhysicalParams(
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        )
        if args.loss == "closed-loop":
            dec_variables = (dec_variables, pipeline.gains)
        return pipeline.loss(dec_variables, robot_keys, estimator_keys)

    batched_loss_for_row = jax.vmap(loss_for_params, in_axes=(0, None))

    def scan_row(_, base_diameter):
        row_losses = batched_loss_for_row(wheel_radius_values_jax, base_diameter)
        return None, row_losses

    evaluate_surface = jax.jit(lambda: jax.lax.scan(scan_row, None, base_diameter_values_jax)[1])

    print(
        f"Evaluating tracking-error surface for seed={args.seed} "
        f"with num_realizations={args.num_realizations}"
    )
    print(
        f"  compiling and evaluating {len(base_diameter_values)} rows x "
        f"{len(wheel_radius_values)} columns"
    )
    tracking_error_surface = np.asarray(evaluate_surface())
    return wheel_radius_values, base_diameter_values, tracking_error_surface


def build_surface_pipeline(args, init_params):
    if args.loss == "closed-loop":
        return IDandGainPipeline(
            problem_path=args.problem,
            initial_params=init_params,
            seed=args.seed,
        )

    return SystemIdentificationPipeline(
        problem_path=args.problem,
        initial_params=init_params,
        seed=args.seed,
        reference_trajectories_dir=None,
        window_length=args.window_length,
    )


def run_tracking_error_surface(args):
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
    )
    pipeline = build_surface_pipeline(args, init_params)
    if args.reference_trajectories_dir is not None and args.loss != "closed-loop":
        reference_trajectory_paths = list_reference_trajectory_pickles(args.reference_trajectories_dir)
        latest_reference_path = max(reference_trajectory_paths, key=os.path.getctime)
        apply_reference_states_to_pipeline(pipeline, load_reference_states(latest_reference_path))

    wheel_radius_values, base_diameter_values, tracking_error_surface = build_tracking_error_surface_data(
        pipeline,
        args,
    )

    plot_trajectory(pipeline=pipeline,out_prefix="surface_reference")
    plot_tracking_error_surface(
        wheel_radius_values=wheel_radius_values,
        base_diameter_values=base_diameter_values,
        tracking_error_surface=tracking_error_surface,
        hidden_params=pipeline.hidden_params,
        init_params=init_params,
        num_realizations=args.num_realizations,
        seed=args.seed,
        out_prefix=args.output,
    )


def render_surface_frames_from_reference_directory(args, output_dir_name: str | None = None):
    if args.loss == "closed-loop":
        raise ValueError("Surface-frame rendering from saved reference trajectories only supports replay loss.")

    if args.reference_trajectories_dir is None:
        raise ValueError("--reference-trajectories-dir is required for surface frame rendering.")

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
    )
    reference_paths = list_reference_trajectory_pickles(args.reference_trajectories_dir)
    if output_dir_name is None:
        output_dir_name = args.output
    output_dir = os.path.join("visualize", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating {len(reference_paths)} surface frames for shared z-axis limits")
    frame_data = []
    for frame_idx, reference_path in enumerate(reference_paths):
        print(f"  evaluate {frame_idx + 1:>3}/{len(reference_paths)}  {os.path.basename(reference_path)}")
        pipeline = build_surface_pipeline(args, init_params)
        apply_reference_states_to_pipeline(pipeline, load_reference_states(reference_path))
        wheel_radius_values, base_diameter_values, tracking_error_surface = build_tracking_error_surface_data(
            pipeline,
            args,
        )
        frame_data.append(
            {
                "frame_idx": frame_idx,
                "reference_path": reference_path,
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
        f"with fixed z-limits [{global_z_min:.8f}, {global_z_max:.8f}]"
    )
    for frame in frame_data:
        frame_idx = frame["frame_idx"]
        reference_path = frame["reference_path"]
        print(f"  render   {frame_idx + 1:>3}/{len(frame_data)}  {os.path.basename(reference_path)}")
        pipeline = build_surface_pipeline(args, init_params)
        apply_reference_states_to_pipeline(pipeline, load_reference_states(reference_path))
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        plot_tracking_error_surface(
            wheel_radius_values=frame["wheel_radius_values"],
            base_diameter_values=frame["base_diameter_values"],
            tracking_error_surface=frame["tracking_error_surface"],
            hidden_params=pipeline.hidden_params,
            init_params=init_params,
            num_realizations=args.num_realizations,
            seed=args.seed,
            out_path=out_path,
            z_min=global_z_min,
            z_max=global_z_max,
        )
    return output_dir


def create_subplot_frames(args):
    if args.trajectory_frames_dir is None:
        raise ValueError("--trajectory-frames-dir is required for subplot frame creation.")

    trajectory_dir = os.path.join("visualize", args.trajectory_frames_dir)
    if not os.path.isdir(trajectory_dir):
        raise ValueError(f"Trajectory frame directory does not exist: {trajectory_dir}")

    surface_dir = render_surface_frames_from_reference_directory(
        args,
        output_dir_name=args.surface_frames_output_dir,
    )

    trajectory_frame_names = sorted(
        file_name for file_name in os.listdir(trajectory_dir) if file_name.lower().endswith(".png")
    )
    surface_frame_names = sorted(
        file_name for file_name in os.listdir(surface_dir) if file_name.lower().endswith(".png")
    )
    if not trajectory_frame_names:
        raise ValueError(f"No PNG trajectory frames found in: {trajectory_dir}")
    if not surface_frame_names:
        raise ValueError(f"No PNG surface frames found in: {surface_dir}")
    if len(trajectory_frame_names) != len(surface_frame_names):
        raise ValueError(
            f"Frame count mismatch: {len(trajectory_frame_names)} trajectory frames vs "
            f"{len(surface_frame_names)} surface frames."
        )

    output_dir = os.path.join("visualize", args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating {len(trajectory_frame_names)} combined subplot frames into {output_dir}")
    for frame_idx, (trajectory_name, surface_name) in enumerate(zip(trajectory_frame_names, surface_frame_names)):
        trajectory_path = os.path.join(trajectory_dir, trajectory_name)
        surface_path = os.path.join(surface_dir, surface_name)
        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")

        with Image.open(trajectory_path) as trajectory_image, Image.open(surface_path) as surface_image:
            trajectory_image = trajectory_image.convert("RGBA")
            surface_image = surface_image.convert("RGBA")

            target_width = max(trajectory_image.width, surface_image.width)

            def resize_to_width(image):
                if image.width == target_width:
                    return image
                scale = target_width / image.width
                target_height = int(round(image.height * scale))
                return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            trajectory_image = resize_to_width(trajectory_image)
            surface_image = resize_to_width(surface_image)

            combined_height = trajectory_image.height + surface_image.height
            combined_image = Image.new("RGBA", (target_width, combined_height), (255, 255, 255, 255))
            combined_image.paste(trajectory_image, (0, 0))
            combined_image.paste(surface_image, (0, trajectory_image.height))
            combined_image.save(out_path)

        print(f"  saved {out_path}")

    create_gif_from_frames(
        frames_dir=output_dir,
        gif_name=args.gif_name,
        frame_time=args.frame_time,
        stop_time=args.stop_time,
    )


def create_gif_from_frames(frames_dir: str, gif_name: str, frame_time: float, stop_time: float):
    if not os.path.isdir(frames_dir):
        raise ValueError(f"Frame directory does not exist: {frames_dir}")

    frame_names = sorted(
        file_name for file_name in os.listdir(frames_dir)
        if file_name.lower().endswith(".png")
    )
    if not frame_names:
        raise ValueError(f"No PNG frames found in: {frames_dir}")

    gif_path = os.path.join(frames_dir, gif_name)
    frame_duration_ms = int(1000 * frame_time)
    stop_duration_ms = int(1000 * stop_time)

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
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=[
            "standard",
            "realization_sensitivity",
            "tracking_error_surface",
            "surface_frames",
            "subplot_frames",
        ], default="suplot_frames"
    )
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_opt_reference_exports")
    parser.add_argument("--window-length", type=int, default=50)

    # SI optimiztation and realization plot arguments
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-num-realizations", type=int, default=64) # max number of vectorized sims in each opt iteration
    parser.add_argument("--num-seeds", type=int, default=1)  # number of complete SI runs to be averaged
    parser.add_argument("--num-workers", type=int, default=1) # parallel workers

    # Tracking error surface plot arguments
    parser.add_argument("--num-realizations", type=int, default=1) # replay realizations
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.2)
    parser.add_argument("--radius-min", type=float, default=0.01)
    parser.add_argument("--radius-max", type=float, default=0.1)
    parser.add_argument("--radius-points", type=int, default=100)
    parser.add_argument("--base-min", type=float, default=0.01)
    parser.add_argument("--base-max", type=float, default=0.5)
    parser.add_argument("--base-points", type=int, default=100)
    parser.add_argument("--loss", type=str, default="replay")
    parser.add_argument("--trajectory-frames-dir", type=str, default="traj_opt_frames")
    parser.add_argument("--surface-frames-output-dir", type=str, default="si_surface_frames")
    parser.add_argument("--gif-name", type=str, default="traj_opt_animation.gif")
    parser.add_argument("--frame-time", type=float, default=0.2)
    parser.add_argument("--stop-time", type=float, default=1.0)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.plot_type

    if args.plot_type == "standard":
        run_tracking_error_surface(args)
    elif args.plot_type == "realization_sensitivity":
        run_realization_sweep(args)
    elif args.plot_type == "tracking_error_surface":
        run_tracking_error_surface(args)
    elif args.plot_type == "surface_frames":
        render_surface_frames_from_reference_directory(args)
    else:
        create_subplot_frames(args)
