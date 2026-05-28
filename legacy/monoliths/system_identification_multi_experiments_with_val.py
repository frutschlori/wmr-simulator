import argparse
import os
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
                                            plot_reference_trajectories,
                                            plot_system_id, plot_trajectory)


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    base_diameter: jax.Array


class SimulationLog(NamedTuple):
    robot_states: DiffDriveState
    estimator_states: EstimatorState


def load_trajectory_configs(experiments_path, section_name: str = "experiments", required: bool = True) -> list[dict]:
    with open(experiments_path, "r", encoding="utf-8") as file:
        experiments_cfg = yaml.safe_load(file)

    experiments = experiments_cfg.get(section_name, [])
    if required and not experiments:
        raise ValueError(f"No '{section_name}' entries found in '{experiments_path}'")

    return experiments


def build_reference_trajectory(problem_cfg: dict, dt: float, experiment_cfg: dict | None = None):
    # use trajectory info of problem config if no additional experiment is specified
    if experiment_cfg is None:
        planner_cfg = problem_cfg["planner"]
        start = problem_cfg["start"]
        goal = problem_cfg["goal"]
        waypoints = planner_cfg.get("waypoints", [])
        trajectory_time = float(planner_cfg["time"])
    # otherwise use specified experiment info
    else:
        start = experiment_cfg["start"]
        goal = experiment_cfg["goal"]
        waypoints = experiment_cfg.get("waypoints", [])
        trajectory_time = float(experiment_cfg["time"])

    planner_steps = int(trajectory_time / dt)
    planner_time_grid = np.linspace(0.0, planner_steps * dt, planner_steps + 1)
    reference_states, _ = compute_reference_trajectory(start, goal, waypoints, planner_time_grid)

    return reference_states


class BasePipeline:
    def __init__(self, problem_path: str, seed: int = 0, experiments_path: str | None = None):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.problem = yaml.safe_load(file)

        self.problem_path = problem_path
        self.experiments_path = experiments_path
        self.dt = float(self.problem["time_step"])
        self.sim_time = float(self.problem["sim_time"])
        self.sim_steps = int(self.sim_time / self.dt)
        self.sim_time_grid = np.linspace(0.0, self.sim_steps * self.dt, self.sim_steps + 1)
        self.experiment_configs = load_trajectory_configs(experiments_path, section_name="experiments")
        self.validation_configs = load_trajectory_configs(
            experiments_path,
            section_name="validation",
            required=False,
        )
        self.num_experiments = len(self.experiment_configs)
        self.experiment_names = [cfg.get("name", f"experiment_{idx}") for idx, cfg in enumerate(self.experiment_configs)]
        (
            self.reference_trajectories,
            self.experiment_start_poses,
            self.experiment_estimator_start_poses,
        ) = self._build_trajectory_batch(self.experiment_configs)
        self.reference_states = self.reference_trajectories[0]
        self.num_validation_experiments = len(self.validation_configs)
        self.validation_names = [
            cfg.get("name", f"validation_{idx}") for idx, cfg in enumerate(self.validation_configs)
        ]
        if self.validation_configs:
            (
                self.validation_reference_trajectories,
                self.validation_start_poses,
                self.validation_estimator_start_poses,
            ) = self._build_trajectory_batch(self.validation_configs)
        else:
            self.validation_reference_trajectories = None
            self.validation_start_poses = None
            self.validation_estimator_start_poses = None
        self.loss_history = []
        self.validation_loss_history = []
        self.hidden_loss_history = []

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
        target_key, replay_key = jax.random.split(master_key, 2)
        self.target_robot_key, self.target_estimator_key = jax.random.split(target_key, 2)
        self.replay_robot_key, self.replay_estimator_key = jax.random.split(replay_key, 2)

    def _build_trajectory_batch(self, trajectory_configs: list[dict]):
        start_poses = jnp.asarray([cfg["start"] for cfg in trajectory_configs], dtype=jnp.float32)
        estimator_start_poses = jnp.copy(start_poses)
        reference_trajectories = jnp.stack([
            jnp.asarray(
                self._extend_reference_states(
                    build_reference_trajectory(
                        problem_cfg=self.problem,
                        dt=self.dt,
                        experiment_cfg=trajectory_cfg,
                    )
                ),
                dtype=jnp.float32,
            )
            for trajectory_cfg in trajectory_configs
        ])
        return reference_trajectories, start_poses, estimator_start_poses

    def _extend_reference_states(self, reference_states: np.ndarray) -> np.ndarray:
        if len(reference_states) >= len(self.sim_time_grid):
            return reference_states[: len(self.sim_time_grid)]

        num_extra_steps = len(self.sim_time_grid) - len(reference_states)
        last_ref_state = reference_states[-1]
        return np.vstack([reference_states, np.tile(last_ref_state, (num_extra_steps, 1))])

    def _init_states(self, robot_key, estimator_key, start_pose, estimator_start_pose):
        robot_state0 = self.robot.get_init_state(key=robot_key, init_pose=start_pose)
        est_state0 = self.estimator.get_init_state(key=estimator_key, start_pose=estimator_start_pose)
        ctrl_state0 = jnp.zeros(2, dtype=jnp.float32)
        return robot_state0, est_state0, ctrl_state0

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @staticmethod
    def pose_mse(predicted_poses, target_poses):
        pos_error = predicted_poses[:, :2] - target_poses[:, :2]
        angle_error = BasePipeline._wrap_to_pi(predicted_poses[:, 2] - target_poses[:, 2])
        squared_error = jnp.sum(pos_error ** 2, axis=1) + angle_error ** 2
        return jnp.mean(squared_error)

    @staticmethod
    def _print_progress(
        step: int,
        total_steps: int,
        loss_value: float,
        validation_loss_value: float | None = None,
        bar_width: int = 30,
    ):
        if total_steps <= 0:
            return

        completed = int(bar_width * step / total_steps)
        bar = "=" * completed + "." * (bar_width - completed)
        progress_line = f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}"
        if validation_loss_value is not None:
            progress_line += f"  val_loss={validation_loss_value:.8f}"
        sys.stdout.write(progress_line)
        if step == total_steps:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def simulate(
        self,
        robot_params,
        estimator_params=None,
        use_hidden_robot=False,
        controller_gains=None,
        wheel_cmds=None,
        robot_key=None,
        estimator_key=None,
        reference_states=None,
        start_pose=None,
        estimator_start_pose=None,
    ):
        closed_loop_flag = wheel_cmds is None
        reference_states = self.reference_states if reference_states is None else reference_states
        start_pose = self.experiment_start_poses[0] if start_pose is None else start_pose
        estimator_start_pose = self.experiment_estimator_start_poses[0] if estimator_start_pose is None else estimator_start_pose

        def sim_step(carry, inputs):
            robot_state, est_state, ctrl_state = carry
            ref_k, wheel_cmd_experiment = inputs

            ur_true, ul_true = self.robot.get_wheel_speeds(robot_state)
            pose_true = self.robot.get_pose(robot_state)

            model_params = robot_params if estimator_params is None else estimator_params
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
                    # for experiment simulation, i.e. SI target run
                    next_robot_state = self.robot.step(robot_state, wheel_cmd)
                else:
                    # for controller gain tuning
                    next_robot_state = self.robot.step(robot_state, wheel_cmd,
                                                       wheel_radius=robot_params.wheel_radius,
                                                       base_diameter=robot_params.base_diameter)
            else:
                next_ctrl_state = ctrl_state
                next_robot_state = self.robot.step(
                                            robot_state, wheel_cmd_experiment,
                                            wheel_radius=robot_params.wheel_radius,
                                            base_diameter=robot_params.base_diameter)

            return (next_robot_state, next_est_state, next_ctrl_state), (next_robot_state, next_est_state)

        # Use different keys for experiment simulation and replay runs
        if wheel_cmds is None:
            robot_key = self.target_robot_key if robot_key is None else robot_key
            estimator_key = self.target_estimator_key if estimator_key is None else estimator_key
            carry0 = self._init_states(robot_key, estimator_key, start_pose, estimator_start_pose)
            # keep dummy commands for shape consistency so that JAX is happy
            dummy_wheel_cmds = jnp.zeros((reference_states.shape[0], 2), dtype=jnp.float32)
            inputs = (reference_states, dummy_wheel_cmds)
        else:
            robot_key = self.replay_robot_key if robot_key is None else robot_key
            estimator_key = self.replay_estimator_key if estimator_key is None else estimator_key
            carry0 = self._init_states(robot_key, estimator_key, start_pose, estimator_start_pose)
            inputs = (reference_states, wheel_cmds)

        _, logs = jax.lax.scan(sim_step, carry0, inputs)
        robot_states, estimator_states = logs
        return SimulationLog(robot_states=robot_states, estimator_states=estimator_states)

    def simulate_batch(
        self,
        robot_params,
        estimator_params=None,
        use_hidden_robot=False,
        controller_gains=None,
        wheel_cmds=None,
        robot_keys=None,
        estimator_keys=None,
        reference_trajectories=None,
        start_poses=None,
        estimator_start_poses=None,
    ):
        reference_trajectories = self.reference_trajectories if reference_trajectories is None else reference_trajectories
        start_poses = self.experiment_start_poses if start_poses is None else start_poses
        estimator_start_poses = self.experiment_estimator_start_poses if estimator_start_poses is None else estimator_start_poses
        num_trajectories = reference_trajectories.shape[0]

        if robot_keys is None:
            base_key = self.target_robot_key if wheel_cmds is None else self.replay_robot_key
            robot_keys = jax.random.split(base_key, num_trajectories)
        if estimator_keys is None:
            base_key = self.target_estimator_key if wheel_cmds is None else self.replay_estimator_key
            estimator_keys = jax.random.split(base_key, num_trajectories)

        def _simulate_single(reference_states, start_pose, estimator_start_pose, robot_key, estimator_key, wheel_cmd):
            return self.simulate(
                robot_params,
                estimator_params=estimator_params,
                use_hidden_robot=use_hidden_robot,
                controller_gains=controller_gains,
                wheel_cmds=wheel_cmd,
                robot_key=robot_key,
                estimator_key=estimator_key,
                reference_states=reference_states,
                start_pose=start_pose,
                estimator_start_pose=estimator_start_pose,
            )

        if wheel_cmds is None:
            return jax.vmap(
                lambda reference_states, start_pose, estimator_start_pose, robot_key, estimator_key:
                _simulate_single(reference_states, start_pose, estimator_start_pose, robot_key, estimator_key, None),
                in_axes=(0, 0, 0, 0, 0),
            )(reference_trajectories, start_poses, estimator_start_poses, robot_keys, estimator_keys)

        return jax.vmap(
            _simulate_single,
            in_axes=(0, 0, 0, 0, 0, 0),
        )(reference_trajectories, start_poses, estimator_start_poses, robot_keys, estimator_keys, wheel_cmds)


class SystemIdentificationPipeline(BasePipeline):
    def __init__(self, problem_path: str, initial_params: PhysicalParams, seed: int = 0, experiments_path: str | None = None):
        super().__init__(problem_path=problem_path, seed=seed, experiments_path=experiments_path)
        self.initial_params = initial_params
        self.target_logs = self.simulate_batch(initial_params, use_hidden_robot=True, controller_gains=self.gains)
        self.target_log = jax.tree_util.tree_map(lambda x: x[0], self.target_logs)
        if self.validation_reference_trajectories is not None:
            self.validation_target_logs = self.simulate_batch(
                initial_params,
                use_hidden_robot=True,
                controller_gains=self.gains,
                reference_trajectories=self.validation_reference_trajectories,
                start_poses=self.validation_start_poses,
                estimator_start_poses=self.validation_estimator_start_poses,
            )
        else:
            self.validation_target_logs = None

    @staticmethod
    def clip_physical_params(params: PhysicalParams):
        min_radius = 1e-4
        min_base = 1e-4
        return PhysicalParams(
            wheel_radius=jnp.clip(params.wheel_radius, min=min_radius),
            base_diameter=jnp.clip(params.base_diameter, min=min_base),
        )

    def hidden_param_loss(self, params: PhysicalParams):
        radius_error = params.wheel_radius - self.hidden_params.wheel_radius
        base_error = params.base_diameter - self.hidden_params.base_diameter
        return 0.5 * (radius_error ** 2 + base_error ** 2)

    def _loss_from_dataset(
        self,
        params: PhysicalParams,
        reference_trajectories: jax.Array,
        start_poses: jax.Array,
        estimator_start_poses: jax.Array,
        wheel_cmds: jax.Array,
        target_pose_hat: jax.Array,
        replay_robot_keys: jax.Array,
        replay_estimator_keys: jax.Array,
    ):
        def experiment_loss(reference_states, start_pose, estimator_start_pose, experiment_wheel_cmds, experiment_target_pose_hat,
                            experiment_robot_keys, experiment_estimator_keys):
            def replay_loss(robot_key, estimator_key):
                predicted_log = self.simulate(
                    params,
                    estimator_params=self.initial_params,
                    controller_gains=self.gains,
                    wheel_cmds=experiment_wheel_cmds,
                    robot_key=robot_key,
                    estimator_key=estimator_key,
                    reference_states=reference_states,
                    start_pose=start_pose,
                    estimator_start_pose=estimator_start_pose,
                )
                predicted_pose_hat = predicted_log.estimator_states.pose_hat
                return self.pose_mse(predicted_pose_hat, experiment_target_pose_hat)

            losses = jax.vmap(replay_loss)(experiment_robot_keys, experiment_estimator_keys)
            return jnp.mean(losses)

        experiment_losses = jax.vmap(experiment_loss)(
            reference_trajectories,
            start_poses,
            estimator_start_poses,
            wheel_cmds,
            target_pose_hat,
            replay_robot_keys,
            replay_estimator_keys,
        )
        return jnp.mean(experiment_losses)

    def loss(self, params: PhysicalParams, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        return self._loss_from_dataset(
            params,
            self.reference_trajectories,
            self.experiment_start_poses,
            self.experiment_estimator_start_poses,
            self.target_logs.robot_states.wheel_cmd,
            self.target_logs.estimator_states.pose_hat,
            replay_robot_keys,
            replay_estimator_keys,
        )

    def validation_loss(self, params: PhysicalParams, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        if self.validation_target_logs is None:
            raise ValueError("No validation trajectories configured.")
        return self._loss_from_dataset(
            params,
            self.validation_reference_trajectories,
            self.validation_start_poses,
            self.validation_estimator_start_poses,
            self.validation_target_logs.robot_states.wheel_cmd,
            self.validation_target_logs.estimator_states.pose_hat,
            replay_robot_keys,
            replay_estimator_keys,
        )

    def optimize(self, init_params: PhysicalParams, num_steps: int, learning_rate: float, num_realizations: int):
        optimizer = optax.adam(learning_rate)
        current_params = self.clip_physical_params(init_params)
        current_opt_state = optimizer.init(current_params)

        replay_robot_keys = jax.random.split(
            self.replay_robot_key, self.num_experiments * num_realizations).reshape(self.num_experiments, num_realizations, -1)
        replay_estimator_keys = jax.random.split(
            self.replay_estimator_key, self.num_experiments * num_realizations).reshape(self.num_experiments, num_realizations, -1)
        validation_replay_robot_keys = None
        validation_replay_estimator_keys = None
        if self.validation_target_logs is not None:
            validation_replay_robot_keys = jax.random.split(
                jax.random.fold_in(self.replay_robot_key, 1),
                self.num_validation_experiments * num_realizations,
            ).reshape(self.num_validation_experiments, num_realizations, -1)
            validation_replay_estimator_keys = jax.random.split(
                jax.random.fold_in(self.replay_estimator_key, 1),
                self.num_validation_experiments * num_realizations,
            ).reshape(self.num_validation_experiments, num_realizations, -1)

        @jax.jit
        def train_step(params, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss)(params, replay_robot_keys, replay_estimator_keys)
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            next_params = self.clip_physical_params(next_params)
            return next_params, next_opt_state, loss_value

        if self.validation_target_logs is not None:
            @jax.jit
            def validation_step(params):
                return self.validation_loss(params, validation_replay_robot_keys, validation_replay_estimator_keys)

        self.loss_history = []
        self.validation_loss_history = []
        self.hidden_loss_history = []
        for step in range(num_steps):
            current_params, current_opt_state, loss_value = train_step(current_params, current_opt_state)
            loss_value_float = float(loss_value)
            self.loss_history.append(loss_value_float)
            self.hidden_loss_history.append(float(self.hidden_param_loss(current_params)))
            validation_loss_value = None
            if self.validation_target_logs is not None:
                validation_loss_value = float(validation_step(current_params))
                self.validation_loss_history.append(validation_loss_value)
            self._print_progress(step + 1, num_steps, loss_value_float, validation_loss_value)

        return current_params, self.loss_history


class ControllerTuningPipeline(BasePipeline):
    def __init__(self, problem_path, robot_params=None, seed=0, experiments_path: str | None = None):
        super().__init__(problem_path=problem_path, seed=seed, experiments_path=experiments_path)
        robot_params = self.hidden_params if robot_params is None else robot_params
        self.robot_params = SystemIdentificationPipeline.clip_physical_params(robot_params)

    @staticmethod
    def _clip_controller_gains(gains: jax.Array):
        min_gain = 0
        return jnp.clip(gains, min=min_gain)

    def _loss_from_dataset(
        self,
        gains: jax.Array,
        reference_trajectories: jax.Array,
        start_poses: jax.Array,
        estimator_start_poses: jax.Array,
        replay_robot_keys: jax.Array,
        replay_estimator_keys: jax.Array,
    ):
        reference_poses = reference_trajectories[:, :, :3]
        def experiment_loss(reference_states, reference_pose, start_pose, estimator_start_pose,
                            experiment_robot_keys, experiment_estimator_keys):
            def realization_loss(robot_key, estimator_key):
                predicted_log = self.simulate(
                    self.robot_params,
                    controller_gains=gains,
                    robot_key=robot_key,
                    estimator_key=estimator_key,
                    reference_states=reference_states,
                    start_pose=start_pose,
                    estimator_start_pose=estimator_start_pose,
                )
                predicted_poses = predicted_log.robot_states.pose
                return self.pose_mse(predicted_poses, reference_pose)

            losses = jax.vmap(realization_loss)(experiment_robot_keys, experiment_estimator_keys)
            return jnp.mean(losses)

        experiment_losses = jax.vmap(experiment_loss)(
            reference_trajectories,
            reference_poses,
            start_poses,
            estimator_start_poses,
            replay_robot_keys,
            replay_estimator_keys,
        )
        return jnp.mean(experiment_losses)

    def loss(self, gains: jax.Array, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        return self._loss_from_dataset(
            gains,
            self.reference_trajectories,
            self.experiment_start_poses,
            self.experiment_estimator_start_poses,
            replay_robot_keys,
            replay_estimator_keys,
        )

    def validation_loss(self, gains: jax.Array, replay_robot_keys: jax.Array, replay_estimator_keys: jax.Array):
        if self.validation_reference_trajectories is None:
            raise ValueError("No validation trajectories configured.")
        return self._loss_from_dataset(
            gains,
            self.validation_reference_trajectories,
            self.validation_start_poses,
            self.validation_estimator_start_poses,
            replay_robot_keys,
            replay_estimator_keys,
        )

    def optimize(self, init_gains: jax.Array, num_steps: int, learning_rate: float, num_realizations: int):
        optimizer = optax.adam(learning_rate)
        current_gains = self._clip_controller_gains(init_gains)
        current_opt_state = optimizer.init(current_gains)

        replay_robot_keys = jax.random.split(
            self.replay_robot_key, self.num_experiments * num_realizations).reshape(self.num_experiments, num_realizations, -1)
        replay_estimator_keys = jax.random.split(
            self.replay_estimator_key, self.num_experiments * num_realizations).reshape(self.num_experiments, num_realizations, -1)
        validation_replay_robot_keys = None
        validation_replay_estimator_keys = None
        if self.validation_reference_trajectories is not None:
            validation_replay_robot_keys = jax.random.split(
                jax.random.fold_in(self.replay_robot_key, 1),
                self.num_validation_experiments * num_realizations,
            ).reshape(self.num_validation_experiments, num_realizations, -1)
            validation_replay_estimator_keys = jax.random.split(
                jax.random.fold_in(self.replay_estimator_key, 1),
                self.num_validation_experiments * num_realizations,
            ).reshape(self.num_validation_experiments, num_realizations, -1)

        @jax.jit
        def train_step(gains, opt_state):
            loss_value, grads = jax.value_and_grad(self.loss)(gains, replay_robot_keys, replay_estimator_keys)
            updates, next_opt_state = optimizer.update(grads, opt_state, gains)
            next_gains = optax.apply_updates(gains, updates)
            next_gains = self._clip_controller_gains(next_gains)
            return next_gains, next_opt_state, loss_value

        if self.validation_reference_trajectories is not None:
            @jax.jit
            def validation_step(gains):
                return self.validation_loss(gains, validation_replay_robot_keys, validation_replay_estimator_keys)

        self.loss_history = []
        self.validation_loss_history = []
        for step in range(num_steps):
            current_gains, current_opt_state, loss_value = train_step(current_gains, current_opt_state)
            loss_value_float = float(loss_value)
            self.loss_history.append(loss_value_float)
            validation_loss_value = None
            if self.validation_reference_trajectories is not None:
                validation_loss_value = float(validation_step(current_gains))
                self.validation_loss_history.append(validation_loss_value)
            self._print_progress(step + 1, num_steps, loss_value_float, validation_loss_value)

        return current_gains, self.loss_history


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
    default_experiments_path = "problems/experiments.yaml"
    if not os.path.exists(default_experiments_path):
        default_experiments_path = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--experiments", type=str, default=default_experiments_path)
    parser.add_argument("--steps-si", type=int, default=1000)
    parser.add_argument("--learning-rate-si", type=float, default=1e-3)
    parser.add_argument("--steps-ctrl-tuning", type=int, default=200)
    parser.add_argument("--learning-rate-ctrl-tuning", type=float, default=5e-2)
    parser.add_argument("--num-realizations", type=int, default=8)
    # parser.add_argument("--seed", type=int, default=np.random.randint(low=0, high=1e16))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("SI", "gains", "SI-then-gains"),
                        default="SI")

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

    if args.mode == "SI":
        # Setup pipeline
        pipeline = SystemIdentificationPipeline(
            problem_path=args.problem,
            initial_params=init_params,
            seed=args.seed,
            experiments_path=args.experiments,
        )
        plot_reference_trajectories(
            pipeline.reference_trajectories,
            pipeline.validation_reference_trajectories,
            out_prefix="reference_trajectories",
        )

        # Save initial experiment run (hidden robot with guessed est and ctrl) for comparison in plots
        init_target_log = pipeline.target_log
        # Save initial replay run (logged target commands feedforwarded through system with guessed params) for plots
        init_log = pipeline.simulate(init_params, wheel_cmds=pipeline.target_log.robot_states.wheel_cmd)

        # estimate robot parameters
        estimated_params, loss_history = pipeline.optimize(
            init_params=init_params,
            num_steps=args.steps_si,
            learning_rate=args.learning_rate_si,
            num_realizations=args.num_realizations)

        # log simulations with estimated params for comparison in plots
        pipeline.target_log = pipeline.simulate(estimated_params,
                                                use_hidden_robot=True,
                                                controller_gains=pipeline.gains) # closed-loop
        final_log = pipeline.simulate(estimated_params,
                                      use_hidden_robot=False,
                                      wheel_cmds=pipeline.target_log.robot_states.wheel_cmd,
                                      controller_gains=pipeline.gains) # feed-forward replay
        # print summary
        print_physical_params("Initial guess:", init_params)
        print_physical_params("Hidden parameters:", pipeline.hidden_params)
        print_physical_params("Estimated parameters:", estimated_params)
        print(f"Final loss: {loss_history[-1]:.8f}")
        if pipeline.validation_loss_history:
            print(f"Final validation loss: {pipeline.validation_loss_history[-1]:.8f}")
        if pipeline.hidden_loss_history:
            print(f"Final hidden parameter loss: {pipeline.hidden_loss_history[-1]:.8f}")

        plot_system_id(pipeline=pipeline, init_target_log=init_target_log, init_log=init_log, predicted_log=final_log)
        plot_trajectory(pipeline, tuned_log=final_log, untuned_log=init_log,
                        out_prefix="trajectory_sys_id")
        plot_loss_history(
            loss_history=loss_history,
            validation_loss_history=pipeline.validation_loss_history,
            hidden_loss_history=pipeline.hidden_loss_history,
        )

    elif args.mode == "gains":
        # Get robot params
        fixed_robot_params = resolve_gain_robot_params(args.problem, args.fixed_wheel_radius, args.fixed_base_diameter)
        # Setup pipeline
        pipeline = ControllerTuningPipeline(
            problem_path=args.problem,
            robot_params=fixed_robot_params,
            seed=args.seed,
            experiments_path=args.experiments)
        # Simulate with initial gains for comparison
        init_gain_log = pipeline.simulate(fixed_robot_params)
        plot_reference_trajectories(
            pipeline.reference_trajectories,
            pipeline.validation_reference_trajectories,
            out_prefix="reference_trajectories",
        )


        # Tune controller gains
        estimated_gains, loss_history = pipeline.optimize(
            init_gains=pipeline.gains,
            num_steps=args.steps_ctrl_tuning,
            learning_rate=args.learning_rate_ctrl_tuning,
            num_realizations=args.num_realizations)

        # Log simulation with tuned gains for comparison
        final_log = pipeline.simulate(fixed_robot_params, controller_gains=estimated_gains)
        # Print summary
        print_physical_params("Robot parameters used for gain tuning:", fixed_robot_params)
        print_controller_gains("Initial gains:", pipeline.gains)
        print_controller_gains("Optimized gains:", estimated_gains)
        print(f"Final loss: {loss_history[-1]:.8f}")
        if pipeline.validation_loss_history:
            print(f"Final validation loss: {pipeline.validation_loss_history[-1]:.8f}")

        plot_trajectory(pipeline, tuned_log=final_log, untuned_log=init_gain_log,
                        out_prefix="trajectory_gain_tuning")
        plot_controller_tuning_errors(pipeline=pipeline, init_log=init_gain_log, tuned_log=final_log)
        plot_loss_history(
            loss_history=loss_history,
            validation_loss_history=pipeline.validation_loss_history,
            out_prefix="ctrl_tuning",
        )

    else:
        # First run system ID pipeline and then tune controller gains

        # Setup system ID pipeline
        sys_id_pipeline = SystemIdentificationPipeline(
            problem_path=args.problem,
            initial_params=init_params,
            seed=args.seed,
            experiments_path=args.experiments,
        )
        plot_reference_trajectories(
            sys_id_pipeline.reference_trajectories,
            sys_id_pipeline.validation_reference_trajectories,
            out_prefix="reference_trajectories",
        )
        # Save initial experiment run (hidden robot with guessed est and ctrl) for comparison in plots
        init_target_log = sys_id_pipeline.target_log
        # Save initial replay run (logged target commands feedforwarded through system with guessed params) for plots
        init_sys_id_log = sys_id_pipeline.simulate(init_params,
            controller_gains=sys_id_pipeline.gains,
            wheel_cmds=sys_id_pipeline.target_log.robot_states.wheel_cmd)

        # estimate robot parameters
        print("Starting system identification...")
        estimated_params, id_loss_history = sys_id_pipeline.optimize(
            init_params=init_params,
            num_steps=args.steps_si,
            learning_rate=args.learning_rate_si,
            num_realizations=args.num_realizations)

        # Setup controller gain tuning pipeline
        ctrl_pipeline = ControllerTuningPipeline(
            problem_path=args.problem,
            robot_params=estimated_params,
            seed=args.seed,
            experiments_path=args.experiments)
        # Save initial simulation for pose error plot
        init_gain_log = ctrl_pipeline.simulate(estimated_params)

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
        final_ff_log = ctrl_pipeline.simulate(estimated_params,
                                              wheel_cmds=final_log.robot_states.wheel_cmd,
                                              controller_gains=estimated_gains) # feed-forward replay

        # Print summary
        print_physical_params("True robot parameters:", sys_id_pipeline.hidden_params)
        print_physical_params("Initial robot parameters guess:", init_params)
        print_physical_params("Estimated robot parameters:", estimated_params)
        print(f"Final system ID loss: {id_loss_history[-1]:.8f}")
        if sys_id_pipeline.validation_loss_history:
            print(f"Final system ID validation loss: {sys_id_pipeline.validation_loss_history[-1]:.8f}")
        if sys_id_pipeline.hidden_loss_history:
            print(f"Final system ID hidden parameter MSE: {sys_id_pipeline.hidden_loss_history[-1]:.8f}")
        print_controller_gains("Initial gains:", ctrl_pipeline.gains)
        print_controller_gains("Optimized gains:", estimated_gains)
        print(f"Final gain tuning loss: {gain_loss_history[-1]:.8f}")
        if ctrl_pipeline.validation_loss_history:
            print(f"Final gain tuning validation loss: {ctrl_pipeline.validation_loss_history[-1]:.8f}")

        # Plots
        plot_system_id(pipeline=sys_id_pipeline, init_target_log=init_target_log, init_log=init_sys_id_log,
                       predicted_log=final_log, out_prefix="sys_id_then_gain_tuning")
        plot_trajectory(ctrl_pipeline, tuned_log=final_log, untuned_log=init_gain_log,
                        out_prefix="trajectory_sequential_tuning")
        plot_controller_tuning_errors(pipeline=ctrl_pipeline, init_log=init_gain_log, tuned_log=final_log,
                                      out_prefix="ctrl_tuning_after_system_id")
        plot_loss_history(
            loss_history=id_loss_history,
            validation_loss_history=sys_id_pipeline.validation_loss_history,
            hidden_loss_history=sys_id_pipeline.hidden_loss_history,
            out_prefix="system_id",
        )
        plot_loss_history(
            loss_history=gain_loss_history,
            validation_loss_history=ctrl_pipeline.validation_loss_history,
            out_prefix="ctrl_tuning_after_system_id",
        )
