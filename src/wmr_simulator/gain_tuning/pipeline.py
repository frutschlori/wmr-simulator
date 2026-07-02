import jax
import jax.numpy as jnp
import yaml

from wmr_simulator.gain_tuning.objectives import (
    clip_controller_gains,
    closed_loop_objective,
)
from wmr_simulator.gain_tuning.optimizers import optimize_controller_gains
from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.types import PhysicalParams, clip_physical_params


class ControllerTuningPipeline(SimulationPipeline):
    def __init__(self, problem_path, robot_params=None, seed=0, reference_trajectories_dir: str | None = None):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
        )
        robot_params = self.hidden_params if robot_params is None else robot_params
        self.robot_params = clip_physical_params(robot_params)

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
        input_weight: float = 0.0,
        input_delta_weight: float = 0.0,
    ):
        return optimize_controller_gains(
            pipeline=self,
            init_gains=init_gains,
            num_steps=num_steps,
            learning_rate=learning_rate,
            num_realizations=num_realizations,
            input_weight=input_weight,
            input_delta_weight=input_delta_weight,
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
    input_weight: float = 0.0,
    input_delta_weight: float = 0.0,
):
    pipeline = ControllerTuningPipeline(
        problem_path=problem_path,
        robot_params=robot_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
    )
    init_hidden_log = pipeline.run_closed_loop(robot_params, use_hidden_robot=True)
    init_model_log = pipeline.run_closed_loop(robot_params)
    optimized_gains, loss_history = pipeline.optimize(
        init_gains=pipeline.gains,
        num_steps=num_steps,
        learning_rate=learning_rate,
        num_realizations=num_realizations,
        input_weight=input_weight,
        input_delta_weight=input_delta_weight,
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
        "final_hidden_log": final_hidden_log,
        "final_model_log": final_model_log,
    }
