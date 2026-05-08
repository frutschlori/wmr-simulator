import argparse

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.robot_jax import DiffDrive


class ProblemDefinition:
    def __init__(self, problem_path: str):
        with open(problem_path, "r", encoding="utf-8") as file:
            self.raw = yaml.safe_load(file)

        self.path = problem_path
        self.dt = float(self.raw["time_step"])
        self.planner_time = float(self.raw["planner"]["time"])
        self.start = np.asarray(self.raw["start"], dtype=float)
        self.goal = np.asarray(self.raw["goal"], dtype=float)
        self.planner_waypoints = self.raw["planner"].get("waypoints", [])
        self.robot_cfg = self.raw["robot"]
        self.estimator_cfg = self.raw.get("estimator", {})

    def build_robot(self) -> DiffDrive:
        robot_type = self.robot_cfg.get("type")
        if robot_type != "differential_drive":
            raise ValueError(f"Unsupported robot type '{robot_type}'")
        return DiffDrive(robot_cfg=self.robot_cfg, dt=self.dt)

    def planner_time_grid(self) -> np.ndarray:
        num_steps = int(self.planner_time / self.dt)
        return np.linspace(0.0, num_steps * self.dt, num_steps + 1)


class Trajectory:
    def __init__(self, time: np.ndarray, poses: np.ndarray, speeds: np.ndarray):
        self.time = np.asarray(time, dtype=float)
        self.poses = np.asarray(poses, dtype=float)
        self.speeds = np.asarray(speeds, dtype=float)

    def linear_speed(self) -> np.ndarray:
        theta = self.poses[:, 2]
        vx = self.speeds[:, 0]
        vy = self.speeds[:, 1]
        return vx * np.cos(theta) + vy * np.sin(theta)


class TrajectoryGenerator:
    def generate(self, problem: ProblemDefinition) -> Trajectory:
        raise NotImplementedError("TrajectoryGenerator subclasses must implement generate().")


class PlannerTrajectoryGenerator(TrajectoryGenerator):
    def generate(self, problem: ProblemDefinition) -> Trajectory:
        time_grid = problem.planner_time_grid()
        reference_states, _ = compute_reference_trajectory(
            start=problem.start,
            goal=problem.goal,
            intermediate_waypoints=problem.planner_waypoints,
            time=time_grid,
        )

        poses = reference_states[:, :3]
        speeds = reference_states[:, 3:6]
        return Trajectory(time_grid, poses, speeds)


class TrajectoryOptimizationSetup:
    def __init__(self, problem_path: str, trajectory_generator: TrajectoryGenerator | None = None):
        self.problem = ProblemDefinition(problem_path)
        self.robot = self.problem.build_robot()

        if trajectory_generator is None:
            self.trajectory_generator = PlannerTrajectoryGenerator()
        else:
            self.trajectory_generator = trajectory_generator

        self.trajectory = self.trajectory_generator.generate(self.problem)

    def nominal_parameters(self) -> jnp.ndarray:
        return jnp.array([self.robot.r, self.robot.L], dtype=jnp.float32)

    def default_measurement_covariance(self) -> np.ndarray:
        noise_pos = float(self.problem.estimator_cfg.get("noise_pos", 1.0))
        noise_angle = float(self.problem.estimator_cfg.get("noise_angle", 1.0))

        pos_var = max(noise_pos ** 2, 1e-6)
        angle_var = max(noise_angle ** 2, 1e-6)
        return np.diag([pos_var, pos_var, angle_var])

    def build_wheel_command_sequence(self) -> np.ndarray:
        r = float(self.robot.r)
        L = float(self.robot.L)
        v = self.trajectory.linear_speed()
        omega = self.trajectory.speeds[:, 2]

        ur = (v + 0.5 * L * omega) / r
        ul = (v - 0.5 * L * omega) / r
        return np.column_stack([ur[:-1], ul[:-1]])

    def build_state_sequence(self) -> np.ndarray:
        r = float(self.robot.r)
        L = float(self.robot.L)
        v = self.trajectory.linear_speed()
        omega = self.trajectory.speeds[:, 2]

        ur = (v + 0.5 * L * omega) / r
        ul = (v - 0.5 * L * omega) / r
        return np.column_stack([self.trajectory.poses, ur, ul])

    def initial_state(self) -> jnp.ndarray:
        return jnp.array(
            [self.problem.start[0], self.problem.start[1], self.problem.start[2], 0.0, 0.0],
            dtype=jnp.float32,
        )

    def state_transition(self, state: jnp.ndarray, wheel_cmd: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        # same as in the robot model but without wrapping theta to avoid awkward derivatives at wrapping points
        x, y, theta, ur_prev, ul_prev = state
        wheel_radius = params[0]
        base_diameter = params[1]

        ur_cmd = jnp.clip(wheel_cmd[0], min=-self.robot.max_wheel_speed, max=self.robot.max_wheel_speed)
        ul_cmd = jnp.clip(wheel_cmd[1], min=-self.robot.max_wheel_speed, max=self.robot.max_wheel_speed)

        ur_next = self.robot.alpha * ur_prev + (1.0 - self.robot.alpha) * ur_cmd
        ul_next = self.robot.alpha * ul_prev + (1.0 - self.robot.alpha) * ul_cmd

        v = 0.5 * wheel_radius * (ur_next + ul_next)
        omega = (wheel_radius / base_diameter) * (ur_next - ul_next)

        x_next = x + v * jnp.cos(theta) * self.robot.dt
        y_next = y + v * jnp.sin(theta) * self.robot.dt
        theta_next = theta + omega * self.robot.dt

        return jnp.array([x_next, y_next, theta_next, ur_next, ul_next], dtype=jnp.float32)

    def measurement_model(self, state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        del params
        return state[:3]

    def compute_fim(self, measurement_covariance: np.ndarray | None = None) -> np.ndarray:
        if measurement_covariance is None:
            measurement_covariance = self.default_measurement_covariance()
        else:
            measurement_covariance = np.asarray(measurement_covariance, dtype=float)

        params = self.nominal_parameters()
        wheel_cmds = jnp.asarray(self.build_wheel_command_sequence(), dtype=jnp.float32)
        init_state = self.initial_state()
        state_sensitivity = jnp.zeros((5, 2), dtype=jnp.float32)

        covariance_inv = jnp.asarray(np.linalg.inv(measurement_covariance), dtype=jnp.float32)

        def fim_step(carry, wheel_cmd):
            state, state_sensitivity, fim = carry

            dfdx = jax.jacfwd(self.state_transition, argnums=0)(state, wheel_cmd, params)
            dfdtheta = jax.jacfwd(self.state_transition, argnums=2)(state, wheel_cmd, params)

            next_state = self.state_transition(state, wheel_cmd, params)
            # Discrete-time sensitivity recursion for x_{k+1} = f(x_k, u_k, theta).
            next_state_sensitivity = dfdx @ state_sensitivity + dfdtheta

            dhdx = jax.jacfwd(self.measurement_model, argnums=0)(next_state, params)
            dhdtheta_partial = jax.jacfwd(self.measurement_model, argnums=1)(next_state, params)
            dhdtheta = dhdx @ next_state_sensitivity + dhdtheta_partial

            fim_increment = dhdtheta.T @ covariance_inv @ dhdtheta
            next_fim = fim + fim_increment

            return (next_state, next_state_sensitivity, next_fim), fim_increment

        initial_carry = (init_state, state_sensitivity, jnp.zeros((2, 2), dtype=jnp.float32))
        final_carry, _ = jax.lax.scan(fim_step, initial_carry, wheel_cmds)
        fim = final_carry[2]

        return np.asarray(fim)

    def compute_stepwise_fim(self, measurement_covariance: np.ndarray | None = None) -> np.ndarray:
        if measurement_covariance is None:
            measurement_covariance = self.default_measurement_covariance()
        else:
            measurement_covariance = np.asarray(measurement_covariance, dtype=float)

        if measurement_covariance.shape != (3, 3):
            raise ValueError("measurement_covariance must have shape (3, 3) for pose measurements.")

        params = self.nominal_parameters()
        states = jnp.asarray(self.build_state_sequence()[:-1], dtype=jnp.float32)
        wheel_cmds = jnp.asarray(self.build_wheel_command_sequence(), dtype=jnp.float32)
        covariance_inv = jnp.asarray(np.linalg.inv(measurement_covariance), dtype=jnp.float32)

        def one_step_measurement(state: jnp.ndarray, wheel_cmd: jnp.ndarray, step_params: jnp.ndarray) -> jnp.ndarray:
            next_state = self.state_transition(state, wheel_cmd, step_params)
            return self.measurement_model(next_state, step_params)

        def fim_step(fim, inputs):
            state, wheel_cmd = inputs
            jacobian = jax.jacfwd(one_step_measurement, argnums=2)(state, wheel_cmd, params)
            fim_increment = jacobian.T @ covariance_inv @ jacobian
            return fim + fim_increment, fim_increment

        initial_fim = jnp.zeros((2, 2), dtype=jnp.float32)
        fim, _ = jax.lax.scan(fim_step, initial_fim, (states, wheel_cmds))
        return np.asarray(fim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize trajectory optimization inputs.")
    parser.add_argument("problem", nargs="?", default="problems/straight.yaml")
    args = parser.parse_args()

    setup = TrajectoryOptimizationSetup(args.problem)

    print(f"Loaded problem: {setup.problem.path}")
    print(f"Robot: {type(setup.robot).__name__}")
    print(f"Trajectory samples: {len(setup.trajectory.time)}")
    print(f"Start pose: {setup.trajectory.poses[0]}")
    print(f"Goal pose:  {setup.trajectory.poses[-1]}")
    print("Full-rollout FIM:")
    print(setup.compute_fim())
    print("Stepwise FIM:")
    print(setup.compute_stepwise_fim())
