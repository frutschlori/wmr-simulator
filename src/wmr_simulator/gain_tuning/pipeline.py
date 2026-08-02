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
        omega_delta_weight: float = 0.0,
        gain_delta_weight: float = 0.0,
        k_min_stab: float = 1e-3,
        k_max_stab: float = 20.0,
        k_max_rest: float = 20.0,
        num_lhs_points: int = 0,
        num_adam_optimizations: int = 1,
        presearch_relative_range: float = 0.0,
        warm_start_schedule: bool = False,
        init_offset_radius: float = 0.0,
        init_offset_angle: float = 0.0,
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
            omega_delta_weight=omega_delta_weight,
            gain_delta_weight=gain_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            num_lhs_points=num_lhs_points,
            num_adam_optimizations=num_adam_optimizations,
            presearch_relative_range=presearch_relative_range,
            warm_start_schedule=warm_start_schedule,
            init_offset_radius=init_offset_radius,
            init_offset_angle=init_offset_angle,
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
    omega_delta_weight: float = 0.0,
    k_min_stab: float = 1e-3,
    k_max_stab: float = 20.0,
    k_max_rest: float = 20.0,
    num_lhs_points: int = 0,
    num_adam_optimizations: int = 1,
    schedule_enabled: bool | None = None,
    gain_delta_weight: float = 0.0,
    residual_model=None,
    static_tune: bool = False,
    static_tune_steps: int | None = None,
    static_tune_learning_rate: float | None = None,
    static_init_gains=None,
    seed_parametrization_from_static: bool = False,
    presearch_relative_range: float = 0.0,
    warm_start_schedule: bool = False,
    init_offset_radius: float = 0.0,
    init_offset_angle: float = 0.0,
):
    """Tune controller gains (optionally jointly with a gain parametrization).

    ``init_offset_radius`` / ``init_offset_angle`` randomize the rollout start
    pose around the reference start, one draw per noise realization, so the
    tracking gains see real error to act on (see
    :func:`gain_tuning.objectives.sample_initial_pose_offsets`). Both 0 starts
    every rollout exactly on the reference.

    ``presearch_relative_range`` (> 0) narrows an LHS presearch to a +/- band
    around its run's init gains instead of the full [k_min_stab, k_max_stab]
    range -- useful for refining across active-learning iterations.
    ``warm_start_schedule`` initializes the parametrization from the problem's
    gain_parametrization (e.g. the previous iteration's trained schedule)
    rather than the identity mapping.

    With ``static_tune`` and an enabled parametrization, two *independent*
    optimizations run, so the static controller and the parametrized one are
    two separate options rather than two stages of one search:

    * the static run: LHS presearch + multistart Adam over the base gains only,
      centered on ``static_init_gains`` (the previous iteration's static gains;
      the problem's gains when None) and using
      ``static_tune_steps``/``static_tune_learning_rate`` when given.
    * the parametrization run: LHS presearch with the (warm-started)
      parametrization already active, centered on the problem's gains, followed
      directly by joint multistart Adam over base gains + parametrization,
      using ``num_steps``/``learning_rate``.

    Their loss histories are returned separately (``loss_history`` is the
    parametrization run's, ``static_*`` the static run's).

    ``seed_parametrization_from_static`` adds the static run's per-start Adam
    results to the parametrization run's presearch as extra candidates (the LHS
    points still compete, and the band stays centered on the problem's gains).
    Meant for the first active-learning iteration, where there is no trained
    parametrization to warm-start from: the two presearches are then the same
    evaluation (identity parametrization), so the converged static gains are
    free information. From the second iteration on the runs should stay fully
    independent so each option refines its own lineage.
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

    def optimize(init_gains, run_schedule_enabled, run_num_steps, run_learning_rate):
        return pipeline.optimize(
            init_gains=init_gains,
            num_steps=run_num_steps,
            learning_rate=run_learning_rate,
            num_realizations=num_realizations,
            schedule_enabled=run_schedule_enabled,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            omega_delta_weight=omega_delta_weight,
            gain_delta_weight=gain_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            num_lhs_points=num_lhs_points,
            num_adam_optimizations=num_adam_optimizations,
            presearch_relative_range=presearch_relative_range,
            warm_start_schedule=warm_start_schedule,
            init_offset_radius=init_offset_radius,
            init_offset_angle=init_offset_angle,
        )

    static_tune = static_tune and schedule_enabled
    static_gains = None
    static_optimization = None
    if static_tune:
        static_steps = num_steps if static_tune_steps is None else int(static_tune_steps)
        static_learning_rate = (
            learning_rate if static_tune_learning_rate is None else float(static_tune_learning_rate)
        )
        static_init = pipeline.gains if static_init_gains is None else jnp.asarray(static_init_gains, dtype=jnp.float32)
        print(
            "Static run: optimizing base gains only "
            f"({static_steps} steps, learning rate {static_learning_rate:.8g})."
        )
        static_optimization = optimize(
            static_init, run_schedule_enabled=False,
            run_num_steps=static_steps, run_learning_rate=static_learning_rate,
        )
        static_gains = static_optimization["gains"]
        print("Parametrization run: independent presearch with the parametrization active.")

    # Row 0 stays the problem's gains: it is what a narrowed presearch centers on.
    parametrization_init_gains = jnp.atleast_2d(pipeline.gains)
    if static_optimization is not None and seed_parametrization_from_static:
        seeds = static_optimization["final_gains_per_start"]
        parametrization_init_gains = jnp.concatenate([parametrization_init_gains, seeds], axis=0)
        print(
            f"  seeding its presearch with the {int(seeds.shape[0])} static Adam "
            "result(s) as extra candidates (first iteration: nothing to warm-start from)."
        )

    optimization = optimize(parametrization_init_gains, run_schedule_enabled=schedule_enabled,
                            run_num_steps=num_steps, run_learning_rate=learning_rate)
    optimized_gains = optimization["gains"]
    schedule_params = optimization["schedule_params"]
    (
        loss_history,
        validation_loss_history,
        loss_component_history,
        validation_loss_component_history,
    ) = histories_for_start(optimization, optimization["best_start_index"])

    static_loss_history = None
    static_validation_loss_history = None
    static_loss_component_history = None
    static_validation_loss_component_history = None
    if static_optimization is not None:
        # The two runs are independent, so their histories stay separate curves.
        (
            static_loss_history,
            static_validation_loss_history,
            static_loss_component_history,
            static_validation_loss_component_history,
        ) = histories_for_start(static_optimization, static_optimization["best_start_index"])
        static_final_loss = float(static_optimization["best_training_loss"])
        scheduled_final_loss = float(optimization["best_training_loss"])
        print(f"Static tune best final loss:    {static_final_loss:.8f}")
        print(f"Scheduled tune best final loss: {scheduled_final_loss:.8f}")
        if static_final_loss > 0.0:
            improvement = 100.0 * (1.0 - scheduled_final_loss / static_final_loss)
            print(f"Improvement from gain parametrization: {improvement:.2f}%")
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
    # Rollout of the static run's gains (base gains only, no parametrization)
    # so the plots can overlay "tuned (static)" against "tuned (param)". None
    # when there was no static run.
    static_hidden_log = (
        None
        if static_gains is None
        else pipeline.run_closed_loop(
            robot_params, use_hidden_robot=True, controller_gains=static_gains
        )
    )
    return {
        "pipeline": pipeline,
        "init_hidden_log": init_hidden_log,
        "init_model_log": init_model_log,
        "optimized_gains": optimized_gains,
        "static_tune": static_tune,
        "static_gains": static_gains,
        "static_hidden_log": static_hidden_log,
        "schedule_enabled": schedule_enabled,
        "schedule_params": schedule_params,
        # Losses of the *returned* gains. The histories below are raw per-step
        # traces whose last entry belongs to the last iterate, not to the best
        # one that is actually returned -- use these for reporting/export.
        "final_loss": float(optimization["best_training_loss"]),
        "final_validation_loss": optimization["best_validation_loss"],
        "static_final_loss": (
            None if static_optimization is None else float(static_optimization["best_training_loss"])
        ),
        "static_final_validation_loss": (
            None if static_optimization is None else static_optimization["best_validation_loss"]
        ),
        "loss_history": loss_history,
        "validation_loss_history": validation_loss_history,
        "loss_component_history": loss_component_history,
        "validation_loss_component_history": validation_loss_component_history,
        "static_loss_history": static_loss_history,
        "static_validation_loss_history": static_validation_loss_history,
        "static_loss_component_history": static_loss_component_history,
        "static_validation_loss_component_history": static_validation_loss_component_history,
        "final_hidden_log": final_hidden_log,
        "final_model_log": final_model_log,
    }
