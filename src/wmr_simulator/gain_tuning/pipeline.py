import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_objective,
)
from wmr_simulator.gain_tuning.optimizers import optimize_controller_gains
from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.types import PhysicalParams, clip_physical_params


class ControllerTuningPipeline(SimulationPipeline):
    def __init__(
        self,
        problem_path,
        robot_params=None,
        seed=0,
        reference_trajectories_dir: str | None = None,
        validation_split: float = 0.0,
    ):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=None,
        )
        robot_params = self.hidden_params if robot_params is None else robot_params
        self.robot_params = clip_physical_params(robot_params)
        self.reference_trajectory_paths = []
        self.training_reference_trajectories = self.reference_states[None, ...]
        self.validation_reference_trajectories = jnp.empty((0,) + self.reference_states.shape, dtype=jnp.float32)
        if reference_trajectories_dir is not None:
            self._load_reference_trajectory_sets(reference_trajectories_dir, validation_split, seed)

    def _load_reference_trajectory_sets(
        self,
        reference_trajectories_dir: str,
        validation_split: float,
        seed: int,
    ) -> None:
        validation_split = self._normalize_validation_split(validation_split)
        pickle_paths = sorted(
            os.path.join(reference_trajectories_dir, name)
            for name in os.listdir(reference_trajectories_dir)
            if name.endswith(".pkl")
        )
        if not pickle_paths:
            raise ValueError(f"No pickle files found in {reference_trajectories_dir}")

        reference_trajectories = []
        for path in pickle_paths:
            with open(path, "rb") as file:
                payload = pickle.load(file)
            reference_states = np.asarray(payload["reference_states"] if isinstance(payload, dict) else payload, dtype=float)
            reference_trajectories.append(self._fit_reference_states(reference_states))

        indices = np.arange(len(reference_trajectories))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        num_validation = int(np.floor(len(indices) * validation_split))
        validation_indices = indices[:num_validation]
        training_indices = indices[num_validation:]
        if len(training_indices) == 0:
            raise ValueError("validation_split leaves no training trajectories.")

        self.reference_trajectory_paths = pickle_paths
        self.training_reference_trajectories = jnp.asarray(
            np.stack([reference_trajectories[index] for index in training_indices], axis=0),
            dtype=jnp.float32,
        )
        self.validation_reference_trajectories = jnp.asarray(
            np.stack([reference_trajectories[index] for index in validation_indices], axis=0)
            if len(validation_indices)
            else np.empty((0,) + reference_trajectories[0].shape, dtype=float),
            dtype=jnp.float32,
        )
        self.reference_states = self.training_reference_trajectories[0]
        self.loaded_reference_trajectory_path = pickle_paths[int(training_indices[0])]
        self.reference_trajectories_dir = reference_trajectories_dir
        print(
            "Loaded reference trajectories: "
            f"{len(training_indices)} training, {len(validation_indices)} validation "
            f"from {reference_trajectories_dir}"
        )

    @staticmethod
    def _normalize_validation_split(validation_split: float) -> float:
        validation_split = float(validation_split)
        if validation_split > 1.0:
            validation_split /= 100.0
        if validation_split < 0.0 or validation_split >= 1.0:
            raise ValueError("validation_split must satisfy 0 <= split < 1, or be a percentage in [0, 100).")
        return validation_split

    @staticmethod
    def _clip_controller_gains(gains: jax.Array):
        return clip_controller_gains(gains)

    def loss(
        self,
        gains: jax.Array,
        replay_robot_keys: jax.Array,
        replay_estimator_keys: jax.Array,
        input_weight: float = 0.0,
        input_delta_weight: float = 0.0,
    ):
        return closed_loop_objective(
            self,
            gains,
            replay_robot_keys,
            replay_estimator_keys,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
        )

    def optimize(
        self,
        init_gains: jax.Array,
        num_steps: int,
        learning_rate: float,
        num_realizations: int,
        velocity_tracking_weight: float = 0.0,
        input_weight: float = 0.0,
        input_delta_weight: float = 0.0,
        k_min_stab: float = 1e-3,
        k_max_stab: float = 20.0,
        k_max_rest: float = 20.0,
        num_lhs_points: int = 0,
        num_adam_optimizations: int = 1,
    ):
        return optimize_controller_gains(
            pipeline=self,
            init_gains=init_gains,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            num_lhs_points=num_lhs_points,
            num_adam_optimizations=num_adam_optimizations,
            training_reference_trajectories=self.training_reference_trajectories,
            validation_reference_trajectories=self.validation_reference_trajectories,
        )


def resolve_gain_robot_params(problem_path: str, fixed_wheel_radius, fixed_base_diameter) -> PhysicalParams:
    with open(problem_path, "r", encoding="utf-8") as file:
        problem_cfg = yaml.safe_load(file)

    wheel_radius = fixed_wheel_radius
    base_diameter = fixed_base_diameter
    if wheel_radius is None:
        wheel_radius = problem_cfg["robot"]["wheel_radius"]
    if base_diameter is None:
        base_diameter = problem_cfg["robot"]["base_diameter"]

    return PhysicalParams(
        wheel_radius=jnp.asarray(wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(base_diameter, dtype=jnp.float32),
        max_wheel_speed=jnp.asarray(problem_cfg["robot"]["max_wheel_speed"], dtype=jnp.float32),
        time_constant=jnp.asarray(problem_cfg["robot"]["time_constant"], dtype=jnp.float32),
    )


def run_gain_tuning_experiment(
    problem_path: str,
    robot_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    num_realizations: int,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
    validation_split: float = 0.0,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    k_min_stab: float = 1e-3,
    k_max_stab: float = 20.0,
    k_max_rest: float = 20.0,
    num_lhs_points: int = 0,
    num_adam_optimizations: int = 1,
):
    pipeline = ControllerTuningPipeline(
        problem_path=problem_path,
        robot_params=robot_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        validation_split=validation_split,
    )
    init_hidden_log = pipeline.run_closed_loop(robot_params, use_hidden_robot=True)
    init_model_log = pipeline.run_closed_loop(robot_params)
    (
        optimized_gains,
        loss_history,
        validation_loss_history,
        loss_component_history,
        validation_loss_component_history,
    ) = pipeline.optimize(
        init_gains=pipeline.gains,
        num_steps=num_steps,
        learning_rate=learning_rate,
        num_realizations=num_realizations,
        velocity_tracking_weight=velocity_tracking_weight,
        input_weight=input_weight,
        input_delta_weight=input_delta_weight,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
        num_lhs_points=num_lhs_points,
        num_adam_optimizations=num_adam_optimizations,
    )
    final_hidden_log = pipeline.run_closed_loop(
        robot_params,
        use_hidden_robot=True,
        controller_gains=optimized_gains,
    )
    final_model_log = pipeline.run_closed_loop(
        robot_params,
        controller_gains=optimized_gains,
    )
    return {
        "pipeline": pipeline,
        "init_hidden_log": init_hidden_log,
        "init_model_log": init_model_log,
        "optimized_gains": optimized_gains,
        "loss_history": loss_history,
        "validation_loss_history": validation_loss_history,
        "loss_component_history": loss_component_history,
        "validation_loss_component_history": validation_loss_component_history,
        "final_hidden_log": final_hidden_log,
        "final_model_log": final_model_log,
    }
