import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.pipeline import SystemIdentificationPipeline
from wmr_simulator.types import PhysicalParams
from wmr_simulator.visualization.identification import (
    plot_tracking_error_surface,
    plot_trajectory,
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
        reference_payload = pickle.load(file)

    if isinstance(reference_payload, dict):
        if "reference_states" not in reference_payload:
            raise ValueError(
                f"Loaded reference trajectory payload from {reference_trajectory_path} must contain a "
                "'reference_states' field."
            )
        reference_states = reference_payload["reference_states"]
    else:
        reference_states = reference_payload
    reference_states = np.asarray(reference_states, dtype=float)
    if reference_states.ndim != 2 or reference_states.shape[1] != 8:
        raise ValueError(
            f"Loaded reference states from {reference_trajectory_path} must have shape (N, 8), "
            f"got {reference_states.shape}"
        )
    return reference_states


def apply_reference_states_to_pipeline(
    pipeline: SystemIdentificationPipeline,
    reference_states: np.ndarray,
) -> SystemIdentificationPipeline:
    pipeline.reference_states = jnp.asarray(
        pipeline._extend_reference_states(reference_states),
        dtype=jnp.float32,
    )
    pipeline.loaded_reference_trajectory_path = None
    pipeline.target_log = pipeline.run_closed_loop(
        pipeline.initial_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
    )
    return pipeline


def build_physical_parameter_surface(
    pipeline: SystemIdentificationPipeline,
    radius_min: float,
    radius_max: float,
    radius_points: int,
    base_min: float,
    base_max: float,
    base_points: int,
    num_realizations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_replay_realizations = pipeline.resolve_replay_realizations(num_realizations)
    wheel_radius_values = build_parameter_grid(radius_min, radius_max, radius_points)
    base_diameter_values = build_parameter_grid(base_min, base_max, base_points)
    robot_keys = jax.random.split(pipeline.robot_key, num_replay_realizations)
    estimator_keys = jax.random.split(pipeline.estimator_key, num_replay_realizations)
    wheel_radius_values_jax = jnp.asarray(wheel_radius_values, dtype=jnp.float32)
    base_diameter_values_jax = jnp.asarray(base_diameter_values, dtype=jnp.float32)

    def loss_for_params(wheel_radius, base_diameter):
        params = PhysicalParams(
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        )
        return pipeline.loss(params, robot_keys, estimator_keys)

    batched_loss_for_row = jax.vmap(loss_for_params, in_axes=(0, None))

    def scan_row(_, base_diameter):
        row_losses = batched_loss_for_row(wheel_radius_values_jax, base_diameter)
        return None, row_losses

    evaluate_surface = jax.jit(lambda: jax.lax.scan(scan_row, None, base_diameter_values_jax)[1])
    tracking_error_surface = np.asarray(evaluate_surface())
    return wheel_radius_values, base_diameter_values, tracking_error_surface


def make_surface_pipeline(
    problem_path: str,
    initial_params: PhysicalParams,
    seed: int = 0,
    window_length: int | None = None,
    deterministic_replay: bool = True,
) -> SystemIdentificationPipeline:
    return SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        reference_trajectories_dir=None,
        window_length=window_length,
        deterministic_replay=deterministic_replay,
    )


def run_physical_parameter_surface(
    problem_path: str,
    initial_params: PhysicalParams,
    radius_min: float = 0.01,
    radius_max: float = 0.1,
    radius_points: int = 100,
    base_min: float = 0.01,
    base_max: float = 0.5,
    base_points: int = 100,
    num_realizations: int = 1,
    seed: int = 0,
    window_length: int | None = None,
    reference_trajectory_path: str | None = None,
    reference_trajectories_dir: str | None = None,
    out_prefix: str = "si_tracking_error_surface",
    save_plots: bool = True,
):
    pipeline = make_surface_pipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        window_length=window_length,
    )
    selected_reference_path = reference_trajectory_path
    if selected_reference_path is None and reference_trajectories_dir is not None:
        selected_reference_path = max(
            list_reference_trajectory_pickles(reference_trajectories_dir),
            key=os.path.getctime,
        )
    if selected_reference_path is not None:
        apply_reference_states_to_pipeline(pipeline, load_reference_states(selected_reference_path))

    wheel_radius_values, base_diameter_values, tracking_error_surface = build_physical_parameter_surface(
        pipeline=pipeline,
        radius_min=radius_min,
        radius_max=radius_max,
        radius_points=radius_points,
        base_min=base_min,
        base_max=base_max,
        base_points=base_points,
        num_realizations=num_realizations,
    )

    if save_plots:
        plot_trajectory(pipeline=pipeline, out_prefix="surface_reference")
        plot_tracking_error_surface(
            wheel_radius_values=wheel_radius_values,
            base_diameter_values=base_diameter_values,
            tracking_error_surface=tracking_error_surface,
            hidden_params=pipeline.hidden_params,
            init_params=initial_params,
            num_realizations=num_realizations,
            seed=seed,
            out_prefix=out_prefix,
        )

    return {
        "pipeline": pipeline,
        "reference_trajectory_path": selected_reference_path,
        "wheel_radius_values": wheel_radius_values,
        "base_diameter_values": base_diameter_values,
        "tracking_error_surface": tracking_error_surface,
    }
