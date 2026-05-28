import argparse
import os
import pickle
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


class BasePipeline:
    def __init__(
        self,
        problem_path: str,
        seed: int = 0,
        reference_trajectories_dir: str | None = None,
        window_length: int | None = None,
    ):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.problem = yaml.safe_load(file)

        self.problem_path = problem_path
        self.reference_trajectories_dir = reference_trajectories_dir
        self.loaded_reference_trajectory_path = None
        self.window_length = window_length
        self.dt = float(self.problem["time_step"])
        self.sim_time = float(self.problem["sim_time"])
        self.sim_steps = int(self.sim_time / self.dt)
        self.sim_time_grid = np.linspace(0.0, self.sim_steps * self.dt, self.sim_steps + 1)

        planner_cfg = self.problem["planner"]
        planner_time = float(planner_cfg["time"])
        planner_steps = int(planner_time / self.dt)
        planner_time_grid = np.linspace(0.0, planner_steps * self.dt, planner_steps + 1)

        if reference_trajectories_dir is None:
            start = self.problem["start"]
            goal = self.problem["goal"]
            waypoints = planner_cfg["waypoints"]
            reference_states, _ = compute_reference_trajectory(start, goal, waypoints, planner_time_grid)
        else:
            reference_states = self._load_latest_reference_states(reference_trajectories_dir)
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
        estimator_init_state = self.estimator.get_init_state(
            key=jax.random.PRNGKey(1),
            start_pose=self.estimator_cfg["start"],
        )
        self.initial_estimator_covariance = estimator_init_state.P

        self.hidden_params = PhysicalParams(
            wheel_radius=jnp.asarray(self.robot_cfg["wheel_radius"], dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot_cfg["base_diameter"], dtype=jnp.float32))
        self.gains = jnp.array([jnp.asarray(value) for value in self.controller_cfg["gains"]])

        master_key = jax.random.PRNGKey(seed)
        target_key, replay_key = jax.random.split(master_key, 2)
        self.target_robot_key, self.target_estimator_key = jax.random.split(target_key, 2)
        self.robot_key, self.estimator_key = jax.random.split(replay_key, 2)

    def _load_latest_reference_states(self, reference_trajectories_dir: str) -> np.ndarray:
        if not os.path.isdir(reference_trajectories_dir):
            raise ValueError(f"Reference trajectory directory does not exist: {reference_trajectories_dir}")

        pickle_files = [
            os.path.join(reference_trajectories_dir, filename)
            for filename in os.listdir(reference_trajectories_dir)
            if filename.endswith(".pkl")
        ]
        if not pickle_files:
            raise ValueError(f"No pickle files found in reference trajectory directory: {reference_trajectories_dir}")

        latest_file = max(pickle_files, key=os.path.getctime)
        with open(latest_file, "rb") as file:
            reference_states = pickle.load(file)

        reference_states = np.asarray(reference_states, dtype=float)
        if reference_states.ndim != 2 or reference_states.shape[1] != 8:
            raise ValueError(
                f"Loaded reference states from {latest_file} must have shape (N, 8), got {reference_states.shape}"
            )

        self.loaded_reference_trajectory_path = latest_file
        print(f"Loaded Reference Trajectory: {latest_file}")
        return reference_states

    def _extend_reference_states(self, reference_states: np.ndarray) -> np.ndarray:
        if len(reference_states) >= len(self.sim_time_grid):
            return reference_states[: len(self.sim_time_grid)]

        num_extra_steps = len(self.sim_time_grid) - len(reference_states)
        last_ref_state = reference_states[-1]
        return np.vstack([reference_states, np.tile(last_ref_state, (num_extra_steps, 1))])

    def resolve_window_length(self, window_length: int | None = None) -> int:
        if window_length is None:
            if self.window_length is None:
                return len(self.reference_states)
            return int(self.window_length)

        window_length = int(window_length)
        if window_length <= 0:
            raise ValueError("window_length must be a positive integer.")
        return window_length

    def _init_states(self, robot_key, estimator_key):
        robot_state0 = self.robot.get_init_state(key=robot_key, init_pose=self.problem["start"])
        est_state0 = self.estimator.get_init_state(key=estimator_key, start_pose=self.estimator_cfg["start"])
        ctrl_state0 = jnp.zeros(2, dtype=jnp.float32)
        return robot_state0, est_state0, ctrl_state0

    @staticmethod
    def pose_mse(predicted_poses, target_poses):
        pos_error = predicted_poses[:, :2] - target_poses[:, :2]
        angle_error = predicted_poses[:, 2] - target_poses[:, 2]
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

    def simulate(self, robot_params, est_params=None, use_hidden_robot=False, controller_gains=None, wheel_cmds=None, robot_key=None, estimator_key=None):
        closed_loop_flag = wheel_cmds is None

        def sim_step(carry, inputs):
            robot_state, est_state, ctrl_state = carry
            ref_k, wheel_cmd_experiment = inputs

            ur_true, ul_true = self.robot.get_wheel_speeds(robot_state)
            pose_true = self.robot.get_pose(robot_state)

            model_params = robot_params if est_params is None else est_params
            next_est_state = self.estimator.update(
                                    est_state, ur_true, ul_true, pose_true,
                                    wheel_radius=model_params.wheel_radius,
                                    base_diameter=model_params.base_diameter)
            pose_est = self.estimator.get_est_pose(next_est_state)
            wheel_est = self.estimator.get_est_wheel_speeds(next_est_state)

            if closed_loop_flag:
                next_ctrl_state, wheel_cmd = self.controller.compute(
                                                    ctrl_state, ref_k, pose_est, wheel_est,
                                                    gains=controller_gains,
                                                    wheel_radius=robot_params.wheel_radius,
                                                    base_diameter=robot_params.base_diameter)
                if use_hidden_robot:
                    next_robot_state = self.robot.step(robot_state, wheel_cmd) # for experiment simulation in SI
                else:
                    next_robot_state = self.robot.step(robot_state, wheel_cmd, # for controller gain tuning
                                                       wheel_radius=robot_params.wheel_radius,
                                                       base_diameter=robot_params.base_diameter)
            else:
                next_ctrl_state = ctrl_state
                next_robot_state = self.robot.step(
                                            robot_state, wheel_cmd_experiment,
                                            wheel_radius=robot_params.wheel_radius,
                                            base_diameter=robot_params.base_diameter)

            return (next_robot_state, next_est_state, next_ctrl_state), (next_robot_state, next_est_state)

        if wheel_cmds is None:
            robot_key = self.target_robot_key if robot_key is None else robot_key
            estimator_key = self.target_estimator_key if estimator_key is None else estimator_key
            carry0 = self._init_states(robot_key, estimator_key)
            # keep dummy commands for shape consistency so that JAX is happy
            dummy_wheel_cmds = jnp.zeros((self.reference_states.shape[0], 2), dtype=jnp.float32)
            inputs = (self.reference_states, dummy_wheel_cmds)
        else:
            robot_key = self.robot_key if robot_key is None else robot_key
            estimator_key = self.estimator_key if estimator_key is None else estimator_key
            carry0 = self._init_states(robot_key, estimator_key)
            inputs = (self.reference_states, wheel_cmds)

        _, logs = jax.lax.scan(sim_step, carry0, inputs)
        robot_states, estimator_states = logs
        return SimulationLog(robot_states=robot_states, estimator_states=estimator_states)

class SystemIdentificationPipeline(BasePipeline):
    def __init__(
        self,
        problem_path: str,
        initial_params: PhysicalParams,
        seed: int = 0,
        reference_trajectories_dir: str | None = None,
        window_length: int | None = None,
    ):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
            window_length=window_length,
        )
        self.initial_params = initial_params
        self.target_log = self.simulate(initial_params, use_hidden_robot=True, controller_gains=self.gains)

    @staticmethod
    def _clip_physical_params(params: PhysicalParams):
        min_radius = 1e-4
        min_base = 1e-4
        return PhysicalParams(
            wheel_radius=jnp.clip(params.wheel_radius, min=min_radius),
            base_diameter=jnp.clip(params.base_diameter, min=min_base),
        )

    @staticmethod
    def parameter_mse(params: PhysicalParams, target_params: PhysicalParams):
        squared_errors = jnp.array([
            (params.wheel_radius - target_params.wheel_radius) ** 2,
            (params.base_diameter - target_params.base_diameter) ** 2,
        ])
        return jnp.mean(squared_errors)

    def _initial_replay_states(
        self,
        init_pose,
        init_wheel_speeds,
        init_u_hat,
        init_u_true,
        init_covariance,
        robot_key,
        estimator_key,
    ):
        robot_state = DiffDriveState(
            pose=jnp.asarray(init_pose, dtype=jnp.float32),
            wheel_speeds=jnp.asarray(init_wheel_speeds, dtype=jnp.float32),
            key=robot_key,
            vel_omega=jnp.zeros(2, dtype=jnp.float32),
            wheel_cmd=jnp.zeros(2, dtype=jnp.float32),
        )
        estimator_state = EstimatorState(
            pose_hat=jnp.asarray(init_pose, dtype=jnp.float32),
            pose_meas=jnp.asarray(init_pose, dtype=jnp.float32),
            u_hat=jnp.asarray(init_u_hat, dtype=jnp.float32),
            u_true=jnp.asarray(init_u_true, dtype=jnp.float32),
            P=jnp.asarray(init_covariance, dtype=jnp.float32),
            key=estimator_key,
        )
        return robot_state, estimator_state

    def _extract_estimated_pose_series(self, sim_log: SimulationLog):
        if self.estimator.filter_type == "kf":
            return sim_log.estimator_states.pose_hat
        return sim_log.estimator_states.pose_meas

    def replay_rollout(
        self,
        robot_params,
        target_log: SimulationLog,
        est_params=None,
        robot_key=None,
        estimator_key=None,
        window_length: int | None = None,
    ):
        window_length = self.resolve_window_length(window_length)
        robot_key = self.robot_key if robot_key is None else robot_key
        estimator_key = self.estimator_key if estimator_key is None else estimator_key
        model_params = robot_params if est_params is None else est_params

        wheel_cmds = target_log.robot_states.wheel_cmd
        target_measurements = self._extract_estimated_pose_series(target_log)
        target_wheel_speeds = target_log.robot_states.wheel_speeds
        target_u_hat = target_log.estimator_states.u_hat
        target_u_true = target_log.estimator_states.u_true
        target_covariances = target_log.estimator_states.P

        num_steps = wheel_cmds.shape[0]
        num_windows = int(np.ceil(num_steps / window_length))
        padded_num_steps = num_windows * window_length
        pad_steps = padded_num_steps - num_steps

        padded_wheel_cmds = jnp.pad(wheel_cmds, ((0, pad_steps), (0, 0)))
        wheel_cmd_windows = padded_wheel_cmds.reshape(num_windows, window_length, 2)

        window_start_indices = jnp.arange(num_windows, dtype=jnp.int32) * window_length
        init_poses = target_measurements[window_start_indices]
        init_wheel_speeds = target_wheel_speeds[window_start_indices]
        init_u_hat = target_u_hat[window_start_indices]
        init_u_true = target_u_true[window_start_indices]
        init_covariances = target_covariances[window_start_indices]

        window_robot_keys = jax.random.split(robot_key, num_windows)
        window_estimator_keys = jax.random.split(estimator_key, num_windows)

        def replay_step(carry, wheel_cmd):
            robot_state, est_state = carry

            next_robot_state = self.robot.step(
                robot_state,
                wheel_cmd,
                wheel_radius=robot_params.wheel_radius,
                base_diameter=robot_params.base_diameter,
            )
            ur_true, ul_true = self.robot.get_wheel_speeds(next_robot_state)
            pose_true = self.robot.get_pose(next_robot_state)
            next_est_state = self.estimator.update(
                est_state,
                ur_true,
                ul_true,
                pose_true,
                wheel_radius=model_params.wheel_radius,
                base_diameter=model_params.base_diameter,
            )
            return (next_robot_state, next_est_state), (next_robot_state, next_est_state)

        def replay_single_window(
            init_pose,
            init_window_wheel_speeds,
            init_window_u_hat,
            init_window_u_true,
            init_window_covariance,
            window_robot_key,
            window_estimator_key,
            wheel_cmd_window,
        ):
            carry0 = self._initial_replay_states(
                init_pose,
                init_window_wheel_speeds,
                init_window_u_hat,
                init_window_u_true,
                init_window_covariance,
                window_robot_key,
                window_estimator_key,
            )
            _, outputs = jax.lax.scan(replay_step, carry0, wheel_cmd_window)
            return outputs

        robot_state_windows, estimator_state_windows = jax.vmap(replay_single_window)(
            init_poses,
            init_wheel_speeds,
            init_u_hat,
            init_u_true,
            init_covariances,
            window_robot_keys,
            window_estimator_keys,
            wheel_cmd_windows,
        )

        robot_states = DiffDriveState(
            pose=robot_state_windows.pose.reshape(-1, 3)[:num_steps],
            wheel_speeds=robot_state_windows.wheel_speeds.reshape(-1, 2)[:num_steps],
            key=robot_state_windows.key.reshape(-1, 2)[:num_steps],
            vel_omega=robot_state_windows.vel_omega.reshape(-1, 2)[:num_steps],
            wheel_cmd=robot_state_windows.wheel_cmd.reshape(-1, 2)[:num_steps],
        )
        estimator_states = EstimatorState(
            pose_hat=estimator_state_windows.pose_hat.reshape(-1, 3)[:num_steps],
            pose_meas=estimator_state_windows.pose_meas.reshape(-1, 3)[:num_steps],
            u_hat=estimator_state_windows.u_hat.reshape(-1, 2)[:num_steps],
            u_true=estimator_state_windows.u_true.reshape(-1, 2)[:num_steps],
            P=estimator_state_windows.P.reshape(-1, 3, 3)[:num_steps],
            key=estimator_state_windows.key.reshape(-1, 2)[:num_steps],
        )
        return SimulationLog(robot_states=robot_states, estimator_states=estimator_states)

    def loss(self, params: PhysicalParams, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        wheel_cmds = self.target_log.robot_states.wheel_cmd
        del wheel_cmds
        target_pose_hat = self._extract_estimated_pose_series(self.target_log)

        def replay_loss(robot_key, estimator_key):
            predicted_log = self.replay_rollout(
                params,
                target_log=self.target_log,
                est_params=self.initial_params,
                robot_key=robot_key,
                estimator_key=estimator_key,
                window_length=self.window_length,
            )
            predicted_pose_hat = self._extract_estimated_pose_series(predicted_log)
            return self.pose_mse(predicted_pose_hat, target_pose_hat)

        losses = jax.vmap(replay_loss)(replay_robot_keys, replay_estimator_keys)
        return jnp.mean(losses)

    def optimize(self, init_params: PhysicalParams, num_steps: int, learning_rate: float, num_realizations: int):
        optimizer = optax.adam(learning_rate)
        current_params = self._clip_physical_params(init_params)
        current_opt_state = optimizer.init(current_params)

        replay_robot_keys = jax.random.split(self.robot_key, num_realizations)
        replay_estimator_keys = jax.random.split(self.estimator_key, num_realizations)

        @jax.jit
        def train_step(params, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss)(params, replay_robot_keys, replay_estimator_keys)
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            next_params = self._clip_physical_params(next_params)
            return next_params, next_opt_state, loss_value

        loss_history = []
        parameter_mse_history = []
        for step in range(num_steps):
            current_params, current_opt_state, loss_value = train_step(current_params, current_opt_state)
            loss_value_float = float(loss_value)
            loss_history.append(loss_value_float)
            parameter_mse_history.append(float(self.parameter_mse(current_params, self.hidden_params)))
            self._print_progress(step + 1, num_steps, loss_value_float)

        return current_params, loss_history, parameter_mse_history


class ControllerTuningPipeline(BasePipeline):
    def __init__(self, problem_path, robot_params=None, seed=0, reference_trajectories_dir: str | None = None):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
        )
        robot_params = self.hidden_params if robot_params is None else robot_params
        self.robot_params = SystemIdentificationPipeline._clip_physical_params(robot_params)

    @staticmethod
    def _clip_controller_gains(gains: jax.Array):
        min_gain = 0
        return jnp.clip(gains, min=min_gain)

    def loss(self, gains: jax.Array, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        reference_poses = self.reference_states[:,:3]
        def realization_loss(robot_key, estimator_key):
            predicted_log = self.simulate(
                self.robot_params,
                controller_gains=gains,
                robot_key=robot_key,
                estimator_key=estimator_key)
            predicted_poses = predicted_log.robot_states.pose
            return self.pose_mse(predicted_poses, reference_poses)

        losses = jax.vmap(realization_loss)(replay_robot_keys, replay_estimator_keys)
        return jnp.mean(losses)

    def optimize(self, init_gains: jax.Array, num_steps: int, learning_rate: float, num_realizations: int):
        optimizer = optax.adam(learning_rate)
        current_gains = self._clip_controller_gains(init_gains)
        current_opt_state = optimizer.init(current_gains)

        replay_robot_keys = jax.random.split(self.robot_key, num_realizations)
        replay_estimator_keys = jax.random.split(self.estimator_key, num_realizations)

        @jax.jit
        def train_step(gains, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss)(gains, replay_robot_keys, replay_estimator_keys)
            updates, next_opt_state = optimizer.update(grads, opt_state, gains)
            next_gains = optax.apply_updates(gains, updates)
            next_gains = self._clip_controller_gains(next_gains)
            return next_gains, next_opt_state, loss_value

        loss_history = []
        for step in range(num_steps):
            current_gains, current_opt_state, loss_value = train_step(current_gains, current_opt_state)
            loss_value_float = float(loss_value)
            loss_history.append(loss_value_float)
            self._print_progress(step + 1, num_steps, loss_value_float)

        return current_gains, loss_history


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
        base_diameter=jnp.asarray(base_diameter, dtype=jnp.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports")
    # parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--steps-si", type=int, default=500)
    parser.add_argument("--learning-rate-si", type=float, default=1e-3)
    parser.add_argument("--steps-ctrl-tuning", type=int, default=1000)
    parser.add_argument("--learning-rate-ctrl-tuning", type=float, default=5e-2)
    parser.add_argument("--num-realizations", type=int, default=16)
    # parser.add_argument("--seed", type=int, default=np.random.randint(low=0, high=1e16))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("SI", "gains", "SI-then-gains"),
                                                default="SI")

    # Robot parameters to start system ID with
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.2)

    # Robot parameters to use for controller gain tuning
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)

    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter))

    if args.mode == "SI":
        # Setup pipeline
        pipeline = SystemIdentificationPipeline(
            problem_path=args.problem,
            initial_params=init_params,
            seed=args.seed,
            reference_trajectories_dir=args.reference_trajectories_dir,
            window_length=args.window_length,
        )

        # Save initial experiment run (hidden robot with guessed est and ctrl) for comparison in plots
        init_target_log = pipeline.target_log
        # Save initial replay run (logged target commands feedforwarded through system with guessed params) for plots
        init_log = pipeline.replay_rollout(
            init_params,
            target_log=pipeline.target_log,
            est_params=pipeline.initial_params,
            window_length=args.window_length,
        )

        # estimate robot parameters
        estimated_params, loss_history, parameter_mse_history = pipeline.optimize(
            init_params=init_params,
            num_steps=args.steps_si,
            learning_rate=args.learning_rate_si,
            num_realizations=args.num_realizations)

        # log simulations with estimated params for comparison in plots
        pipeline.target_log = pipeline.simulate(estimated_params,
                                                use_hidden_robot=True,
                                                controller_gains=pipeline.gains) # closed-loop
        final_log = pipeline.replay_rollout(
            estimated_params,
            target_log=pipeline.target_log,
            est_params=pipeline.initial_params,
            window_length=args.window_length,
        )
        # print summary
        print_physical_params("Initial guess:", init_params)
        print_physical_params("Hidden parameters:", pipeline.hidden_params)
        print_physical_params("Estimated parameters:", estimated_params)
        print(f"Final loss: {loss_history[-1]:.8f}")
        print(f"Final parameter MSE: {parameter_mse_history[-1]:.8f}")

        plot_system_id(pipeline=pipeline, init_target_log=init_target_log, init_log=init_log, predicted_log=final_log)
        plot_loss_history(loss_history=loss_history, parameter_error_history=parameter_mse_history)

    elif args.mode == "gains":
        # Get robot params
        fixed_robot_params = resolve_gain_robot_params(args.problem, args.fixed_wheel_radius, args.fixed_base_diameter)
        # Setup pipeline
        pipeline = ControllerTuningPipeline(
            problem_path=args.problem,
            robot_params=fixed_robot_params,
            seed=args.seed,
            reference_trajectories_dir=args.reference_trajectories_dir,
        )
        # Simulate with initial gains for comparison
        init_gain_log = pipeline.simulate(fixed_robot_params, use_hidden_robot=True) # simulate deployment with real robot params
        init_log_ctrl_perspective = pipeline.simulate(fixed_robot_params)

        # Tune controller gains
        estimated_gains, loss_history = pipeline.optimize(
            init_gains=pipeline.gains,
            num_steps=args.steps_ctrl_tuning,
            learning_rate=args.learning_rate_ctrl_tuning,
            num_realizations=args.num_realizations)

        # Log simulation with tuned gains for comparison
        # simulate deployment with real robot params
        final_log = pipeline.simulate(fixed_robot_params, use_hidden_robot=True, controller_gains=estimated_gains)
        # log what the controller sees for tuning
        final_log_ctrl_perspective = pipeline.simulate(fixed_robot_params, controller_gains=estimated_gains)
        # Print summary
        print_physical_params("Robot parameters used for gain tuning:", fixed_robot_params)
        print_controller_gains("Initial gains:", pipeline.gains)
        print_controller_gains("Optimized gains:", estimated_gains)
        print(f"Final loss: {loss_history[-1]:.8f}")

        plot_trajectory(pipeline, tuned_log=final_log, untuned_log=init_gain_log,
                        out_prefix="trajectories_gain_tuning_hidden_params")
        plot_trajectory(pipeline, tuned_log=final_log_ctrl_perspective, untuned_log=init_log_ctrl_perspective,
                        out_prefix="trajectories_gain_tuning_used_params")
        plot_controller_tuning_errors(pipeline=pipeline, init_log=init_gain_log, tuned_log=final_log)
        plot_loss_history(loss_history=loss_history, out_prefix="ctrl_tuning")

    else:
        # First run system ID pipeline and then tune controller gains

        # Setup system ID pipeline
        sys_id_pipeline = SystemIdentificationPipeline(
            problem_path=args.problem,
            initial_params=init_params,
            seed=args.seed,
            reference_trajectories_dir=args.reference_trajectories_dir,
            window_length=args.window_length,
        )

        # Save initial experiment run (hidden robot with guessed est and ctrl) for comparison in plots
        init_target_log = sys_id_pipeline.target_log
        # Save initial replay run (logged target commands feedforwarded through system with guessed params) for plots
        init_sys_id_log = sys_id_pipeline.replay_rollout(
            init_params,
            target_log=sys_id_pipeline.target_log,
            est_params=sys_id_pipeline.initial_params,
            window_length=args.window_length,
        )

        # estimate robot parameters
        print("Starting system identification...")
        estimated_params, id_loss_history, id_parameter_mse_history = sys_id_pipeline.optimize(
            init_params=init_params,
            num_steps=args.steps_si,
            learning_rate=args.learning_rate_si,
            num_realizations=args.num_realizations)

        # Setup controller gain tuning pipeline
        ctrl_pipeline = ControllerTuningPipeline(
            problem_path=args.problem,
            robot_params=estimated_params,
            seed=args.seed,
            reference_trajectories_dir=args.reference_trajectories_dir,
        )
        # Save initial simulation for pose error plot
        init_gain_log = ctrl_pipeline.simulate(estimated_params, use_hidden_robot=True)

        # Tune gains using identified robot parameters for estimator and ctrl, but hidden ones for robot step
        print("Starting gain tuning...")
        estimated_gains, gain_loss_history = ctrl_pipeline.optimize(
            init_gains=ctrl_pipeline.gains,
            num_steps=args.steps_ctrl_tuning,
            learning_rate=args.learning_rate_ctrl_tuning,
            num_realizations=args.num_realizations,
        )
        # Log simulation after gain tuning for plots
        final_log = ctrl_pipeline.simulate(estimated_params,
                                           use_hidden_robot=True,
                                           controller_gains=estimated_gains) # closed-loop simulation of "real experiment"
        sys_id_pipeline.target_log = final_log # copy to pipeline just for plot
        final_ff_log = sys_id_pipeline.replay_rollout(
            estimated_params,
            target_log=final_log,
            est_params=sys_id_pipeline.initial_params,
            window_length=args.window_length,
        )

        # Print summary
        print_physical_params("True robot parameters:", sys_id_pipeline.hidden_params)
        print_physical_params("Initial robot parameters guess:", init_params)
        print_physical_params("Estimated robot parameters:", estimated_params)
        print(f"Final system ID loss: {id_loss_history[-1]:.8f}")
        print(f"Final parameter MSE: {id_parameter_mse_history[-1]:.8f}")
        print_controller_gains("Initial gains:", ctrl_pipeline.gains)
        print_controller_gains("Optimized gains:", estimated_gains)
        print(f"Final gain tuning loss: {gain_loss_history[-1]:.8f}")

        # Plots
        plot_system_id(pipeline=sys_id_pipeline, init_target_log=init_target_log, init_log=init_sys_id_log,
                       predicted_log=final_ff_log, out_prefix="sequential_opt")
        plot_trajectory(ctrl_pipeline, tuned_log=final_log, untuned_log=init_target_log,
                        out_prefix="sequential_opt_trajectory")
        plot_controller_tuning_errors(pipeline=ctrl_pipeline, init_log=init_target_log, tuned_log=final_log,
                                      out_prefix="sequential_opt")
        plot_loss_history(
            loss_history=id_loss_history,
            parameter_error_history=id_parameter_mse_history,
            out_prefix="system_id",
        )
        plot_loss_history(loss_history=gain_loss_history, out_prefix="ctrl_tuning")
