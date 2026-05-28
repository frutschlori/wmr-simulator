import argparse
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from wmr_simulator.controller_jax import Controller
from wmr_simulator.estimator_jax import DiffDriveEstimator, EstimatorState
from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.robot_jax import DiffDrive, DiffDriveState
from wmr_simulator.SI_visualization import (plot_controller_tuning_errors,
                                            plot_loss_history,
                                            plot_system_id, plot_trajectory)


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    base_diameter: jax.Array


class SimulationLog(NamedTuple):
    robot_states: DiffDriveState
    estimator_states: EstimatorState


class IDandGainPipeline:
    def __init__(self, problem_path: str, initial_params: PhysicalParams, seed: int = 0):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.problem = yaml.safe_load(file)

        self.problem_path = problem_path
        self.dt = float(self.problem["time_step"])
        self.sim_time = float(self.problem["sim_time"])
        self.sim_steps = int(self.sim_time / self.dt)
        self.sim_time_grid = np.linspace(0.0, self.sim_steps * self.dt, self.sim_steps + 1)

        planner_cfg = self.problem["planner"]
        planner_time = float(planner_cfg["time"])
        planner_steps = int(planner_time / self.dt)
        planner_time_grid = np.linspace(0.0, planner_steps * self.dt, planner_steps + 1)

        start = self.problem["start"]
        goal = self.problem["goal"]
        waypoints = planner_cfg["waypoints"]
        reference_states, _ = compute_reference_trajectory(start, goal, waypoints, planner_time_grid)
        self.reference_states = jnp.asarray(self._extend_reference_states(reference_states), dtype=jnp.float32)

        self.robot_cfg = self.problem["robot"]
        self.estimator_cfg = self.problem["estimator"]
        self.controller_cfg = self.problem["controller"]

        self.robot = DiffDrive(robot_cfg=self.robot_cfg, dt=self.dt)
        self.estimator = DiffDriveEstimator(estimator_cfg=self.estimator_cfg, dt=self.dt)
        self.controller = Controller(
            robot_param=self.estimator_cfg,
            gains=self.controller_cfg["gains"],
            cmd_limits=[-self.robot_cfg["max_wheel_speed"], self.robot_cfg["max_wheel_speed"]],
            dt=self.dt)

        self.hidden_params = PhysicalParams(
            wheel_radius=jnp.asarray(self.robot_cfg["wheel_radius"], dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot_cfg["base_diameter"], dtype=jnp.float32))

        self.gains = jnp.array([jnp.asarray(value) for value in self.controller_cfg["gains"]])

        master_key = jax.random.PRNGKey(seed)
        self.robot_key, self.estimator_key = jax.random.split(master_key, 2)

        self.target_log = self.simulate(initial_params, controller_gains=self.gains) # just for plots


    def _extend_reference_states(self, reference_states: np.ndarray) -> np.ndarray:
        if len(reference_states) >= len(self.sim_time_grid):
            return reference_states[: len(self.sim_time_grid)]

        num_extra_steps = len(self.sim_time_grid) - len(reference_states)
        last_ref_state = reference_states[-1]
        return np.vstack([reference_states, np.tile(last_ref_state, (num_extra_steps, 1))])

    def _init_states(self, robot_key, estimator_key):
        robot_state0 = self.robot.get_init_state(key=robot_key, init_pose=self.problem["start"])
        est_state0 = self.estimator.get_init_state(key=estimator_key, start_pose=self.estimator_cfg["start"])
        ctrl_state0 = jnp.zeros(2, dtype=jnp.float32)
        return robot_state0, est_state0, ctrl_state0

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @staticmethod
    def pose_mse(predicted_poses, target_poses):
        pos_error = predicted_poses[:, :2] - target_poses[:, :2]
        angle_error = IDandGainPipeline._wrap_to_pi(predicted_poses[:, 2] - target_poses[:, 2])
        squared_error = jnp.sum(pos_error ** 2, axis=1) + angle_error ** 2
        return jnp.mean(squared_error)

    @staticmethod
    def _print_progress(step: int, total_steps: int, loss_value: float, bar_width: int = 30):
        if total_steps <= 0:
            return

        completed = int(bar_width * step / total_steps)
        bar = "=" * completed + "." * (bar_width - completed)
        sys.stdout.write(f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}")
        if step == total_steps:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def simulate(self, robot_params, controller_gains=None, robot_key=None, estimator_key=None, ref=None):
        def sim_step(carry, ref_k):
            robot_state, est_state, ctrl_state = carry

            ur_true, ul_true = self.robot.get_wheel_speeds(robot_state)
            pose_true = self.robot.get_pose(robot_state)

            next_est_state = self.estimator.update(est_state, ur_true, ul_true, pose_true,
                                                   wheel_radius=robot_params.wheel_radius,
                                                   base_diameter=robot_params.base_diameter)
            pose_est = self.estimator.get_est_pose(next_est_state)
            wheel_est = self.estimator.get_est_wheel_speeds(next_est_state)

            next_ctrl_state, wheel_cmd = self.controller.compute(ctrl_state, ref_k, pose_est, wheel_est,
                                                                 gains=controller_gains,
                                                                 wheel_radius=robot_params.wheel_radius,
                                                                 base_diameter=robot_params.base_diameter)
            next_robot_state = self.robot.step(robot_state, wheel_cmd)

            return (next_robot_state, next_est_state, next_ctrl_state), (next_robot_state, next_est_state)

        robot_key = self.robot_key if robot_key is None else robot_key
        estimator_key = self.estimator_key if estimator_key is None else estimator_key
        carry0 = self._init_states(robot_key, estimator_key)

        ref = self.reference_states if ref is None else ref

        _, logs = jax.lax.scan(sim_step, carry0, ref)
        robot_states, estimator_states = logs
        return SimulationLog(robot_states=robot_states, estimator_states=estimator_states)


    @staticmethod
    def _clip_physical_params(params: PhysicalParams):
        min_radius = 1e-4
        min_base = 1e-4
        return PhysicalParams(wheel_radius=jnp.clip(params.wheel_radius, min=min_radius),
                              base_diameter=jnp.clip(params.base_diameter, min=min_base))

    @staticmethod
    def _clip_controller_gains(gains: jax.Array):
        min_gain = 0
        return jnp.clip(gains, min=min_gain)

    def loss(self, dec_variables: tuple, robot_keys: jax.Array, estimator_keys: jax.Array, ref_poses = None):
        params, gains = dec_variables
        reference_poses = self.reference_states[:, :3] if ref_poses is None else ref_poses

        def single_loss(robot_key, estimator_key):
            log = self.simulate(params, controller_gains=gains,
                                robot_key=robot_key, estimator_key=estimator_key)
            return self.pose_mse(log.robot_states.pose, reference_poses)

        # Stack multiple realizations via vmap
        losses = jax.vmap(single_loss)(robot_keys, estimator_keys)
        return jnp.mean(losses)

    def optimize(self, init_params: PhysicalParams, init_gains: jax.Array, num_steps: int, learning_rate: float, num_realizations: int):
        optimizer = optax.adam(learning_rate)
        current_params = self._clip_physical_params(init_params)
        current_gains = self._clip_controller_gains(init_gains)
        current_dec_variables = (current_params, current_gains)
        current_opt_state = optimizer.init(current_dec_variables)

        robot_keys = jax.random.split(self.robot_key, num_realizations)
        estimator_keys = jax.random.split(self.estimator_key, num_realizations)

        @jax.jit
        def train_step(dec_variables, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss)(dec_variables, robot_keys, estimator_keys)
            updates, next_opt_state = optimizer.update(grads, opt_state, dec_variables)
            next_dec_variables = optax.apply_updates(dec_variables, updates)
            next_params, next_gains = next_dec_variables
            next_params = self._clip_physical_params(next_params)
            next_gains = self._clip_controller_gains(next_gains)
            return (next_params, next_gains), next_opt_state, loss_value

        loss_history = []
        for step in range(num_steps):
            current_dec_variables, current_opt_state, loss_value = train_step(current_dec_variables, current_opt_state)
            loss_value_float = float(loss_value)
            loss_history.append(loss_value_float)
            self._print_progress(step + 1, num_steps, loss_value_float)

        return current_dec_variables, loss_history


def print_physical_params(label: str, params: PhysicalParams):
    print(label)
    print(f"  wheel_radius={float(params.wheel_radius):.7f} m")
    print(f"  base_diameter={float(params.base_diameter):.7f} m")


def print_controller_gains(label: str, gains: jax.Array):
    print(label)
    print(f"  kx={float(gains[0]):.7f}")
    print(f"  ky={float(gains[1]):.7f}")
    print(f"  kth={float(gains[2]):.7f}")
    print(f"  kprmotor={float(gains[3]):.7f}")
    print(f"  kplmotor={float(gains[4]):.7f}")
    print(f"  kirmotor={float(gains[5]):.7f}")
    print(f"  kilmotor={float(gains[6]):.7f}")


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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-realizations", type=int, default=1)
    # parser.add_argument("--seed", type=int, default=np.random.randint(low=0, high=1e16))
    parser.add_argument("--seed", type=int, default=0)

    # Robot parameters to start system ID with
    parser.add_argument("--init-wheel-radius", type=float, default=0.08)
    parser.add_argument("--init-base-diameter", type=float, default=0.4)

    # Robot parameters to use for controller gain tuning
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)

    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter))

    # Setup pipeline
    pipeline = IDandGainPipeline(problem_path=args.problem, initial_params=init_params, seed=args.seed)
    # Save initial run (hidden robot with guessed est and ctrl) for comparison in plots
    init_log = pipeline.simulate(init_params)

    # simultaneous optimization of robot parameters and gains
    print("Starting simultaneous optimization...")
    tuned_dec_variables, loss_history = pipeline.optimize(
        init_params=init_params,
        init_gains=pipeline.gains,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations)

    estimated_params, tuned_gains = tuned_dec_variables

    # Log simulation after tuning for plots
    final_log = pipeline.simulate(estimated_params, controller_gains=tuned_gains)

    # Print summary
    print_physical_params("True robot parameters:", pipeline.hidden_params)
    print_physical_params("Initial robot parameters guess:", init_params)
    print_physical_params("Estimated robot parameters:", estimated_params)
    print_controller_gains("Initial gains:", pipeline.gains)
    print_controller_gains("Optimized gains:", tuned_gains)
    print(f"Final tuning loss: {loss_history[-1]:.8f}")

    # Plots
    plot_system_id(pipeline=pipeline, init_target_log=init_log, init_log=init_log,
                   predicted_log=final_log, out_prefix="simultaneous_opt")
    plot_trajectory(pipeline, tuned_log=final_log, untuned_log=init_log, out_prefix="simultaneous_opt_trajectory")
    plot_controller_tuning_errors(pipeline=pipeline, init_log=init_log, tuned_log=final_log,
                                  out_prefix="simultaneous_opt")
    plot_loss_history(loss_history=loss_history, out_prefix="simultaneous_opt")
