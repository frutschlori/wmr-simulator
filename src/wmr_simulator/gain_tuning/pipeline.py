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
from wmr_simulator.gain_tuning.optimizers import histories_for_start, optimize_controller_gains
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
        residual_model=None,
    ):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=None,
            residual_model=residual_model,
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
        schedule_enabled: bool = False,
        velocity_tracking_weight: float = 0.0,
        input_weight: float = 0.0,
        input_delta_weight: float = 0.0,
        gain_delta_weight: float = 0.0,
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
            schedule_template=self.gain_schedule_params,
            schedule_enabled=schedule_enabled,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            gain_delta_weight=gain_delta_weight,
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
        a_slip_max=jnp.asarray(problem_cfg["robot"].get("a_slip_max", 0.0), dtype=jnp.float32),
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
    schedule_enabled: bool | None = None,
    gain_delta_weight: float = 0.0,
    residual_model=None,
    static_pretune: bool = False,
    static_pretune_steps: int | None = None,
    static_pretune_learning_rate: float | None = None,
):
    """Tune controller gains (optionally jointly with a gain parametrization).

    With ``static_pretune`` and an enabled parametrization, the optimization is
    split in two stages: first the full static routine (LHS presearch +
    multistart Adam over the base gains only), then a parametrization stage
    that continues every Adam start from its own static result (no new LHS
    round; base gains keep refining jointly). The static stage uses
    ``static_pretune_steps``/``static_pretune_learning_rate`` when given and
    falls back to ``num_steps``/``learning_rate``; the parametrization stage
    always uses the latter. The returned loss histories concatenate the
    winning start's lineage across both stages (continuous up to the one
    pre-update Adam step between a stage's last recorded loss and its final
    parameters).
    """
    pipeline = ControllerTuningPipeline(
        problem_path=problem_path,
        robot_params=robot_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        validation_split=validation_split,
        residual_model=residual_model,
    )
    schedule_enabled = pipeline.gain_schedule_enabled if schedule_enabled is None else bool(schedule_enabled)
    init_hidden_log = pipeline.run_closed_loop(robot_params, use_hidden_robot=True)
    init_model_log = pipeline.run_closed_loop(robot_params)

    def optimize(init_gains, stage_schedule_enabled, stage_lhs_points, stage_adam_starts,
                 stage_num_steps, stage_learning_rate):
        return pipeline.optimize(
            init_gains=init_gains,
            num_steps=stage_num_steps,
            learning_rate=stage_learning_rate,
            num_realizations=num_realizations,
            schedule_enabled=stage_schedule_enabled,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            gain_delta_weight=gain_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            num_lhs_points=stage_lhs_points,
            num_adam_optimizations=stage_adam_starts,
        )

    static_pretune = static_pretune and schedule_enabled
    static_gains = None
    static_optimization = None
    init_gains = pipeline.gains
    lhs_points = num_lhs_points
    adam_starts = num_adam_optimizations
    if static_pretune:
        pretune_steps = num_steps if static_pretune_steps is None else int(static_pretune_steps)
        pretune_learning_rate = learning_rate if static_pretune_learning_rate is None else float(static_pretune_learning_rate)
        print(
            "Static pretune stage: optimizing base gains only "
            f"({pretune_steps} steps, learning rate {pretune_learning_rate:.8g})."
        )
        static_optimization = optimize(
            init_gains, stage_schedule_enabled=False, stage_lhs_points=num_lhs_points,
            stage_adam_starts=num_adam_optimizations,
            stage_num_steps=pretune_steps, stage_learning_rate=pretune_learning_rate,
        )
        static_gains = static_optimization["gains"]
        # No new LHS round: every Adam start continues from its own static result.
        init_gains = static_optimization["final_gains_per_start"]
        lhs_points = 0
        num_starts = int(init_gains.shape[0])
        print(
            "Parametrization stage: training the gain parametrization on top of the "
            f"{num_starts} static result{'s' if num_starts != 1 else ''}."
        )

    optimization = optimize(init_gains, stage_schedule_enabled=schedule_enabled, stage_lhs_points=lhs_points,
                            stage_adam_starts=adam_starts,
                            stage_num_steps=num_steps, stage_learning_rate=learning_rate)
    optimized_gains = optimization["gains"]
    schedule_params = optimization["schedule_params"]
    (
        loss_history,
        validation_loss_history,
        loss_component_history,
        validation_loss_component_history,
    ) = histories_for_start(optimization, optimization["best_start_index"])

    if static_optimization is not None:
        # For a continuous plotted curve, prepend the history of the *static
        # start the eventual winner descended from* (stage-2 candidates are the
        # per-start static results in stage-1 start order), not the static best.
        lineage_index = int(optimization["start_candidate_indices"][optimization["best_start_index"]])
        (
            static_loss_history,
            static_validation_loss_history,
            static_loss_component_history,
            static_validation_loss_component_history,
        ) = histories_for_start(static_optimization, lineage_index)
        static_final_loss = float(
            static_optimization["loss_history_per_start"][-1, static_optimization["best_start_index"]]
        )
        scheduled_final_loss = float(loss_history[-1])
        if lineage_index != static_optimization["best_start_index"]:
            print(
                f"Parametrization winner descended from static start {lineage_index} "
                f"(static best was start {static_optimization['best_start_index']})."
            )
        print(f"Static tune best final loss:    {static_final_loss:.8f}")
        print(f"Scheduled tune best final loss: {scheduled_final_loss:.8f}")
        if static_final_loss > 0.0:
            improvement = 100.0 * (1.0 - scheduled_final_loss / static_final_loss)
            print(f"Improvement from gain parametrization: {improvement:.2f}%")
        loss_history = static_loss_history + loss_history
        if static_validation_loss_history is not None and validation_loss_history is not None:
            validation_loss_history = static_validation_loss_history + validation_loss_history
        loss_component_history = {
            name: static_loss_component_history[name] + values
            for name, values in loss_component_history.items()
        }
        if static_validation_loss_component_history is not None and validation_loss_component_history is not None:
            validation_loss_component_history = {
                name: static_validation_loss_component_history[name] + values
                for name, values in validation_loss_component_history.items()
            }
    final_hidden_log = pipeline.run_closed_loop(
        robot_params,
        use_hidden_robot=True,
        controller_gains=optimized_gains,
        schedule_params=schedule_params,
    )
    final_model_log = pipeline.run_closed_loop(
        robot_params,
        controller_gains=optimized_gains,
        schedule_params=schedule_params,
    )
    return {
        "pipeline": pipeline,
        "init_hidden_log": init_hidden_log,
        "init_model_log": init_model_log,
        "optimized_gains": optimized_gains,
        "static_pretune": static_pretune,
        "static_gains": static_gains,
        "schedule_enabled": schedule_enabled,
        "schedule_params": schedule_params,
        "loss_history": loss_history,
        "validation_loss_history": validation_loss_history,
        "loss_component_history": loss_component_history,
        "validation_loss_component_history": validation_loss_component_history,
        "final_hidden_log": final_hidden_log,
        "final_model_log": final_model_log,
    }
