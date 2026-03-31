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
from wmr_simulator.visualize_jax import plot_loss_history, plot_system_id


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    base_diameter: jax.Array


class SimulationLog(NamedTuple):
    robot_states: DiffDriveState
    estimator_states: EstimatorState


class SystemIdentificationPipeline:
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
            dt=self.dt,
        )

        self.hidden_params = PhysicalParams(
            wheel_radius=jnp.asarray(self.robot_cfg["wheel_radius"], dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot_cfg["base_diameter"], dtype=jnp.float32),
        )
        master_key = jax.random.PRNGKey(seed)
        target_key, replay_key = jax.random.split(master_key, 2)
        self.target_robot_key, self.target_estimator_key = jax.random.split(target_key, 2)
        self.replay_robot_key, self.replay_estimator_key = jax.random.split(replay_key, 2)
        self.target_log = self.simulate(initial_params)

    def _extend_reference_states(self, reference_states: np.ndarray) -> np.ndarray:
        # Extend reference states if needed to match simulation horizon
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

    def simulate(self, robot_params: PhysicalParams, wheel_cmds=None):
        # Flag to initially simulate experiment run and later feed forward logged commands from that run
        simulate_experiment = wheel_cmds is None

        def sim_step(carry, inputs):
            robot_state, est_state, ctrl_state = carry
            ref_k, wheel_cmd_experiment = inputs

            # Get true wheel speeds and robot pose
            ur_true, ul_true = self.robot.get_wheel_speeds(robot_state)
            pose_true = self.robot.get_pose(robot_state)

            # Update estimator (with estimated robot params)
            next_est_state = self.estimator.update(est_state, ur_true, ul_true, pose_true,
                                                   wheel_radius=robot_params.wheel_radius,
                                                   base_diameter=robot_params.base_diameter)
            pose_est = self.estimator.get_est_pose(next_est_state)
            wheel_est = self.estimator.get_est_wheel_speeds(next_est_state)

            if simulate_experiment:
                # Compute control commands using reference at current time step (and guessed initial robot params)
                next_ctrl_state, wheel_cmd = self.controller.compute(ctrl_state, ref_k, pose_est, wheel_est,
                                                                     wheel_radius=robot_params.wheel_radius,
                                                                     base_diameter=robot_params.base_diameter)
                # step robot (with hidden params)
                next_robot_state = self.robot.step(robot_state, wheel_cmd)

            else:
                # propagate robot using logged commands from simulated experiment and estimated robot params
                next_ctrl_state = ctrl_state      # just a place holder for consistent carry shape
                next_robot_state = self.robot.step(robot_state, wheel_cmd_experiment,
                                                   wheel_radius=robot_params.wheel_radius,
                                                   base_diameter=robot_params.base_diameter)

            return (next_robot_state, next_est_state, next_ctrl_state), (next_robot_state, next_est_state)

        if wheel_cmds is None:
            carry0 = self._init_states(self.target_robot_key, self.target_estimator_key)
            # keep dummy commands for shape consistency so that JAX is happy
            dummy_wheel_cmds = jnp.zeros((self.reference_states.shape[0], 2), dtype=jnp.float32)
            inputs = (self.reference_states, dummy_wheel_cmds)
        else:
            carry0 = self._init_states(self.replay_robot_key, self.replay_estimator_key)
            # simulate robot with inputs from robot with hidden params (logged experiment inputs)
            inputs = (self.reference_states, wheel_cmds)

        # iterates over ref_states or with lax.scan()
        _, logs = jax.lax.scan(sim_step, carry0, inputs)

        robot_states, estimator_states = logs
        return SimulationLog(
            robot_states=robot_states,
            estimator_states=estimator_states,
        )

    @staticmethod
    def pose_mse(predicted_poses, target_poses):
        """ Mean Squared Error of WMR poses """
        pos_error = predicted_poses[:, :2] - target_poses[:, :2]
        angle_error = SystemIdentificationPipeline._wrap_to_pi(predicted_poses[:, 2] - target_poses[:, 2])
        squared_error = jnp.sum(pos_error ** 2, axis=1) + angle_error ** 2
        return jnp.mean(squared_error)

    def loss(self, params: PhysicalParams):
        """ Computes loss from estimated poses obtained with current params and est target poses """
        predicted_log = self.simulate(params, wheel_cmds=self.target_log.robot_states.wheel_cmd)
        predicted_pose_hat = predicted_log.estimator_states.pose_hat
        return self.pose_mse(predicted_pose_hat, self.target_log.estimator_states.pose_hat)

    @staticmethod
    def _clip_physical_params(params: PhysicalParams):
        """ Ensure non-negativity of physical params """
        min_radius = 1e-4
        min_base = 1e-4
        return PhysicalParams(
            wheel_radius=jnp.clip(params.wheel_radius, min_radius),
            base_diameter=jnp.clip(params.base_diameter, min_base),
        )

    @staticmethod
    def _print_progress(step: int, total_steps: int, loss_value: float, bar_width: int = 30):
        """ Print progress during optimization """
        if total_steps <= 0:
            return

        completed = int(bar_width * step / total_steps)
        bar = "=" * completed + "." * (bar_width - completed)
        sys.stdout.write(f"\rOptimization [{bar}] {step:>4}/{total_steps}  loss={loss_value:.8f}")
        if step == total_steps:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def optimize(self, init_params: PhysicalParams, num_steps: int, learning_rate: float):
        """ Optimize the system with given initial params """
        optimizer = optax.adam(learning_rate)
        current_params = self._clip_physical_params(init_params)
        opt_state = optimizer.init(current_params)

        @jax.jit
        def train_step(current_params, current_opt_state):
            loss_value, grads = jax.value_and_grad(self.loss)(current_params)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
            next_params = optax.apply_updates(current_params, updates)
            next_params = self._clip_physical_params(next_params)
            return next_params, next_opt_state, loss_value

        loss_history = []
        current_opt_state = opt_state

        for step in range(num_steps):
            current_params, current_opt_state, loss_value = train_step(current_params, current_opt_state)
            loss_value_float = float(loss_value)
            loss_history.append(loss_value_float)
            self._print_progress(step+1, num_steps, loss_value_float)

        return current_params, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--init-wheel-radius", type=float, default=0.08)  # hidden = 0.015
    parser.add_argument("--init-base-diameter", type=float, default=0.4)  # hidden = 0.09
    parser.add_argument("--output", type=str, default="system_identification")
    parser.add_argument("--seed", type=int, default=np.random.randint(low=0, high=1e16))
    args = parser.parse_args()
    # args.seed = 42

    # Set up initial params from arguments
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
    )

    # Initialize pipeline
    pipeline = SystemIdentificationPipeline(problem_path=args.problem, initial_params=init_params, seed=args.seed)

    # Simulate using initial params and log for plots
    init_log = pipeline.simulate(init_params, wheel_cmds=pipeline.target_log.robot_states.wheel_cmd)

    # Actual parameter identification
    estimated_params, loss_history = pipeline.optimize(
        init_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
    )

    # Simulate using optimized parameters and log for plots
    predicted_log = pipeline.simulate(estimated_params, wheel_cmds=pipeline.target_log.robot_states.wheel_cmd)

    print("Initial guess:")
    print(f"  wheel_radius={float(init_params.wheel_radius):.4f} m")
    print(f"  base_diameter={float(init_params.base_diameter):.4f} m")
    print("Hidden parameters:")
    print(f"  wheel_radius={float(pipeline.hidden_params.wheel_radius):.4f} m")
    print(f"  base_diameter={float(pipeline.hidden_params.base_diameter):.4f} m")
    print("Estimated parameters:")
    print(f"  wheel_radius={float(estimated_params.wheel_radius):.4f} m")
    print(f"  base_diameter={float(estimated_params.base_diameter):.4f} m")
    print(f"Final loss: {loss_history[-1]:.8f}")

    plot_system_id(
        hidden_log=pipeline.target_log,
        init_log=init_log,
        predicted_log=predicted_log,
        estimator_filter_type=pipeline.estimator.filter_type,
        time=pipeline.sim_time_grid,
        reference_states=pipeline.reference_states,
        out_prefix=args.output,
    )

    plot_loss_history(loss_history=loss_history, out_prefix=args.output)
