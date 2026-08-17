import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_objective,
    make_realizations,
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
        # Per-trajectory start-pose offsets (T, R, 3), read off the trajectory
        # pickles when the design shipped them, else None (the tuner then draws
        # its own from init_offset_radius/angle).
        self.training_start_offsets = None
        self.validation_start_offsets = None
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
        start_offsets = []
        for path in pickle_paths:
            with open(path, "rb") as file:
                payload = pickle.load(file)
            reference_states = np.asarray(payload["reference_states"] if isinstance(payload, dict) else payload, dtype=float)
            reference_trajectories.append(self._fit_reference_states(reference_states))
            start_offsets.append(
                None
                if not isinstance(payload, dict) or payload.get("start_offsets") is None
                else np.asarray(payload["start_offsets"], dtype=float)
            )
        start_offsets = self._stack_start_offsets(start_offsets, pickle_paths)

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
        if start_offsets is not None:
            self.training_start_offsets = jnp.asarray(start_offsets[training_indices], dtype=jnp.float32)
            self.validation_start_offsets = jnp.asarray(
                start_offsets[validation_indices], dtype=jnp.float32
            )
        self.reference_states = self.training_reference_trajectories[0]
        self.loaded_reference_trajectory_path = pickle_paths[int(training_indices[0])]
        self.reference_trajectories_dir = reference_trajectories_dir
        print(
            "Loaded reference trajectories: "
            f"{len(training_indices)} training, {len(validation_indices)} validation "
            f"from {reference_trajectories_dir}"
        )
        if start_offsets is not None:
            print(
                f"  with designed start-pose offsets: {start_offsets.shape[1]} per trajectory "
                "(the tuner rolls out on these instead of drawing its own)"
            )

    @staticmethod
    def _stack_start_offsets(start_offsets: list, pickle_paths: list) -> np.ndarray | None:
        """Stack the per-trajectory offsets into (T, R, 3), or None when the
        trajectories carry none.

        All-or-none, and one common R: a run whose trajectories were scored
        under different numbers of realizations -- or half of them under
        designed starts and half under drawn ones -- is not one experiment, and
        silently reconciling that would hide it.
        """
        present = [offsets is not None for offsets in start_offsets]
        if not any(present):
            return None
        if not all(present):
            missing = [path for path, has in zip(pickle_paths, present) if not has]
            raise ValueError(
                "Some reference trajectories carry designed start offsets and some do not: "
                f"{missing}. Export them all with offsets, or none."
            )
        shapes = {offsets.shape for offsets in start_offsets}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[1] != 3:
            raise ValueError(
                f"Designed start offsets must all have the same shape (R, 3); got {sorted(shapes)}."
            )
        return np.stack(start_offsets, axis=0)

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
        position_tracking_weight: float = 1.0,
        heading_tracking_weight: float = 1.0,
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
        optimizer: str = "adam",
        presearch_relative_range: float = 0.0,
        warm_start_schedule: bool = False,
        init_offset_radius: float = 0.0,
        init_offset_angle: float = 0.0,
        realizations=None,
        outlier_loss_factor: float = 0.0,
    ):
        return optimize_controller_gains(
            pipeline=self,
            init_gains=init_gains,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
            schedule_template=self.gain_schedule_params,
            schedule_enabled=schedule_enabled,
            position_tracking_weight=position_tracking_weight,
            heading_tracking_weight=heading_tracking_weight,
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
            optimizer=optimizer,
            presearch_relative_range=presearch_relative_range,
            warm_start_schedule=warm_start_schedule,
            init_offset_radius=init_offset_radius,
            init_offset_angle=init_offset_angle,
            realizations=realizations,
            outlier_loss_factor=outlier_loss_factor,
            training_reference_trajectories=self.training_reference_trajectories,
            validation_reference_trajectories=self.validation_reference_trajectories,
            training_start_offsets=self.training_start_offsets,
            validation_start_offsets=self.validation_start_offsets,
        )


def start_offsets_for_set(reference_trajectories, designed_offsets, realizations) -> jax.Array:
    """The (T, R, 3) start offsets for a set of reference trajectories.

    Designed offsets travel with their trajectory, one set each. Without them
    every trajectory shares the run's single drawn set, which is what the tuner
    scores them under -- broadcast so both cases are one array shape.
    """
    num_trajectories = int(jnp.asarray(reference_trajectories).shape[0])
    if designed_offsets is not None:
        return jnp.asarray(designed_offsets, dtype=jnp.float32)
    return jnp.broadcast_to(
        realizations.start_offsets, (num_trajectories,) + realizations.start_offsets.shape
    ).astype(jnp.float32)


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
    position_tracking_weight: float = 1.0,
    heading_tracking_weight: float = 1.0,
    velocity_tracking_weight: float = 0.0,
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
    omega_delta_weight: float = 0.0,
    k_min_stab: float = 1e-3,
    k_max_stab: float = 20.0,
    k_max_rest: float = 20.0,
    num_lhs_points: int = 0,
    num_adam_optimizations: int = 1,
    # "adam" or "bfgs"; see gain_tuning.optimizers. BFGS needs no learning rate
    # (its line search sets the step length) and reaches kimotor = 0 by
    # gradient, which is on the box boundary and out of Adam's reach.
    optimizer: str = "adam",
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
    outlier_loss_factor: float = 0.0,
):
    """Tune controller gains (optionally jointly with a gain parametrization).

    ``init_offset_radius`` / ``init_offset_angle`` randomize the rollout start
    pose around the reference start, one draw per noise realization, so the
    tracking gains see real error to act on (see
    :func:`trajectory_optimization.start_offsets.sample_initial_pose_offsets`). Both 0 starts
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

    * the static run: LHS presearch + multistart refinement over the base gains only,
      centered on ``static_init_gains`` (the previous iteration's static gains;
      the problem's gains when None) and using
      ``static_tune_steps``/``static_tune_learning_rate`` when given.
    * the parametrization run: LHS presearch with the (warm-started)
      parametrization already active, centered on the problem's gains, followed
      directly by joint multistart refinement over base gains + parametrization,
      using ``num_steps``/``learning_rate``.

    Their loss histories are returned separately (``loss_history`` is the
    parametrization run's, ``static_*`` the static run's).

    ``seed_parametrization_from_static`` skips the parametrization run's own LHS
    presearch and multistarts the refinement directly from the problem's gains plus
    the static run's per-start results. Meant for the first active-learning
    iteration, where there is no trained parametrization to warm-start from: the
    two presearches would then be the same evaluation (identity
    parametrization), so re-running one is wasted candidate evaluations and the
    converged static gains are free information instead. From the second
    iteration on the runs should stay fully independent (this flag off) so each
    option refines its own lineage.
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
    # One realization bundle for the whole experiment: the tuner scores its
    # objective on it and the summary figures roll out on it, so the plots show
    # the conditions the gains were actually chosen under. Designed offsets
    # (read off the trajectory pickles) win over the drawn ones and set R.
    if pipeline.training_start_offsets is not None:
        designed_realizations = int(pipeline.training_start_offsets.shape[1])
        if designed_realizations != num_realizations:
            print(
                f"Trajectories were designed under {designed_realizations} start offsets; "
                f"using that instead of the requested {num_realizations} realizations."
            )
        num_realizations = designed_realizations
    realizations = make_realizations(
        pipeline.robot_key,
        pipeline.estimator_key,
        num_realizations,
        init_offset_radius,
        init_offset_angle,
    )
    training_start_offsets = start_offsets_for_set(
        pipeline.training_reference_trajectories, pipeline.training_start_offsets, realizations
    )
    validation_start_offsets = start_offsets_for_set(
        pipeline.validation_reference_trajectories, pipeline.validation_start_offsets, realizations
    )
    # The single-run summary is trajectory 0, realization 0 -- rolled out from
    # its offset start, not from the reference, so the figure shows the same
    # transient the tuning loss was computed on.
    summary_offsets = training_start_offsets[0]
    summary_initial_pose = pipeline.initial_reference_pose() + summary_offsets[0]
    init_hidden_log = pipeline.run_closed_loop(
        robot_params, use_hidden_robot=True, initial_pose=summary_initial_pose
    )
    init_model_log = pipeline.run_closed_loop(robot_params, initial_pose=summary_initial_pose)

    def optimize(init_gains, run_schedule_enabled, run_num_steps, run_learning_rate, run_num_lhs_points=None):
        return pipeline.optimize(
            realizations=realizations,
            init_gains=init_gains,
            num_steps=run_num_steps,
            learning_rate=run_learning_rate,
            num_realizations=num_realizations,
            schedule_enabled=run_schedule_enabled,
            position_tracking_weight=position_tracking_weight,
            heading_tracking_weight=heading_tracking_weight,
            velocity_tracking_weight=velocity_tracking_weight,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
            omega_delta_weight=omega_delta_weight,
            gain_delta_weight=gain_delta_weight,
            k_min_stab=k_min_stab,
            k_max_stab=k_max_stab,
            k_max_rest=k_max_rest,
            num_lhs_points=num_lhs_points if run_num_lhs_points is None else run_num_lhs_points,
            num_adam_optimizations=num_adam_optimizations,
            optimizer=optimizer,
            presearch_relative_range=presearch_relative_range,
            warm_start_schedule=warm_start_schedule,
            init_offset_radius=init_offset_radius,
            init_offset_angle=init_offset_angle,
            outlier_loss_factor=outlier_loss_factor,
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
        # The budget itself is reported by the "Refining ..." line inside
        # optimize_controller_gains, which knows how each optimizer reads it.
        print("Static run: optimizing base gains only.")
        static_optimization = optimize(
            static_init, run_schedule_enabled=False,
            run_num_steps=static_steps, run_learning_rate=static_learning_rate,
        )
        static_gains = static_optimization["gains"]

    # Row 0 stays the problem's gains: it is what a narrowed presearch centers on.
    parametrization_init_gains = jnp.atleast_2d(pipeline.gains)
    # When seeding from the static run, its own LHS presearch already searched
    # this same evaluation (identity parametrization), so a second presearch
    # here would spend the full candidate budget only to be beaten by the
    # already-converged static seeds -- skip it and multistart the refinement directly
    # from the problem's gains + the static run's results.
    parametrization_num_lhs_points = num_lhs_points
    if static_optimization is not None and seed_parametrization_from_static:
        seeds = static_optimization["final_gains_per_start"]
        parametrization_init_gains = jnp.concatenate([parametrization_init_gains, seeds], axis=0)
        parametrization_num_lhs_points = 0
        print(
            "Parametrization run: skipping its own presearch, refining directly from the "
            f"problem's gains + the {int(seeds.shape[0])} static run result(s) "
            "(first iteration: nothing to warm-start from)."
        )
    else:
        print("Parametrization run: independent presearch with the parametrization active.")

    optimization = optimize(parametrization_init_gains, run_schedule_enabled=schedule_enabled,
                            run_num_steps=num_steps, run_learning_rate=learning_rate,
                            run_num_lhs_points=parametrization_num_lhs_points)
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
        initial_pose=summary_initial_pose,
    )
    final_model_log = pipeline.run_closed_loop(
        robot_params,
        controller_gains=optimized_gains,
        schedule_params=schedule_params,
        initial_pose=summary_initial_pose,
    )
    # Rollout of the static run's gains (base gains only, no parametrization)
    # so the plots can overlay "tuned (static)" against "tuned (param)". None
    # when there was no static run.
    static_hidden_log = (
        None
        if static_gains is None
        else pipeline.run_closed_loop(
            robot_params,
            use_hidden_robot=True,
            controller_gains=static_gains,
            initial_pose=summary_initial_pose,
        )
    )
    return {
        "pipeline": pipeline,
        # The start offsets everything above was scored and rolled out under:
        # (T, R, 3) per trajectory set, and the (R, 3) of the summary's
        # trajectory 0. The plots need them to draw the same conditions.
        "realizations": realizations,
        "training_start_offsets": training_start_offsets,
        "validation_start_offsets": validation_start_offsets,
        "summary_start_offsets": summary_offsets,
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
