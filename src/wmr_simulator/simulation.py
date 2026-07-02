import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.controller import Controller
from wmr_simulator.estimator import DiffDriveEstimator
from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.robot import DiffDrive, DiffDriveState
from wmr_simulator.types import PhysicalParams, PoseLog, ReferenceLog, SimulationLog, WheelLog


class SimulationPipeline:
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

        self.geometry_dt = float(self.problem["geometry_controller_dt"])
        self.wheel_dt = float(self.problem["wheel_controller_dt"])
        self.dt = self.geometry_dt
        self.sim_time = float(self.problem["sim_time"])
        self.inner_steps_per_geometry_step = self._inner_steps(self.geometry_dt, self.wheel_dt)
        self.reference_time_grid = self._time_grid(self.sim_time, self.geometry_dt)
        self.pose_time_grid = self._time_grid(self.sim_time, self.wheel_dt)
        self.sim_time_grid = self.pose_time_grid
        self.wheel_time_grid = self.pose_time_grid - self.wheel_dt
        self.command_time_grid = self.reference_time_grid[:-1] + 0.5 * self.wheel_dt
        self.reference_pose_indices = np.arange(
            0,
            len(self.reference_time_grid) * self.inner_steps_per_geometry_step,
            self.inner_steps_per_geometry_step,
            dtype=np.int32,
        )

        planner_cfg = self.problem["planner"]
        planner_time_grid = self._time_grid(float(planner_cfg["time"]), self.geometry_dt)
        if reference_trajectories_dir is None:
            reference_states, _ = compute_reference_trajectory(
                self.problem["start"],
                self.problem["goal"],
                planner_cfg["waypoints"],
                planner_time_grid,
            )
        else:
            reference_states = self._load_latest_reference_states(reference_trajectories_dir)
        self.reference_states = jnp.asarray(self._fit_reference_states(reference_states), dtype=jnp.float32)

        self.robot_cfg = self.problem["robot"]
        self.estimator_cfg = self.problem["estimator"]
        self.controller_cfg = self.problem["controller"]

        self.robot = DiffDrive(robot_cfg=self.robot_cfg, dt=self.wheel_dt)
        self.estimator = DiffDriveEstimator(estimator_cfg=self.estimator_cfg, dt=self.wheel_dt)
        self.controller = Controller(
            robot_param=self.robot_cfg,
            gains=self.controller_cfg["gains"],
            duty_limits=[-1.0, 1.0],
            dt=self.wheel_dt,
        )
        self.initial_estimator_covariance = self.estimator.get_init_state(
            key=jax.random.PRNGKey(1),
            start_pose=self.initial_reference_pose(),
        ).P

        self.hidden_params = PhysicalParams(
            wheel_radius=jnp.asarray(self.robot_cfg["wheel_radius"], dtype=jnp.float32),
            base_diameter=jnp.asarray(self.robot_cfg["base_diameter"], dtype=jnp.float32),
            max_wheel_speed=jnp.asarray(self.robot_cfg["max_wheel_speed"], dtype=jnp.float32),
            time_constant=jnp.asarray(self.robot_cfg["time_constant"], dtype=jnp.float32),
        )
        self.gains = jnp.asarray(self.controller_cfg["gains"], dtype=jnp.float32)

        master_key = jax.random.PRNGKey(seed)
        target_key, replay_key = jax.random.split(master_key, 2)
        self.target_robot_key, self.target_estimator_key = jax.random.split(target_key, 2)
        self.robot_key, self.estimator_key = jax.random.split(replay_key, 2)

    @staticmethod
    def _time_grid(duration: float, dt: float) -> np.ndarray:
        steps = int(np.round(duration / dt))
        return np.linspace(0.0, steps * dt, steps + 1)

    @staticmethod
    def _inner_steps(outer_dt: float, inner_dt: float) -> int:
        steps = int(np.round(outer_dt / inner_dt))
        if steps <= 0 or not np.isclose(steps * inner_dt, outer_dt):
            raise ValueError("geometry_controller_dt must be an integer multiple of wheel_controller_dt")
        return steps

    def _load_latest_reference_states(self, reference_trajectories_dir: str) -> np.ndarray:
        pickle_files = [
            os.path.join(reference_trajectories_dir, name)
            for name in os.listdir(reference_trajectories_dir)
            if name.endswith(".pkl")
        ]
        if not pickle_files:
            raise ValueError(f"No pickle files found in {reference_trajectories_dir}")
        latest_file = max(pickle_files, key=os.path.getctime)
        with open(latest_file, "rb") as file:
            payload = pickle.load(file)
        self.loaded_reference_trajectory_path = latest_file
        print(f"Loaded Reference Trajectory: {latest_file}")
        return np.asarray(payload["reference_states"] if isinstance(payload, dict) else payload, dtype=float)

    def _fit_reference_states(self, reference_states: np.ndarray) -> np.ndarray:
        reference_states = np.asarray(reference_states, dtype=float)
        target_len = len(self.reference_time_grid)
        if len(reference_states) >= target_len:
            return reference_states[:target_len]
        tail = np.tile(reference_states[-1], (target_len - len(reference_states), 1))
        return np.vstack([reference_states, tail])

    def resolve_window_length(self, window_length: int | None = None) -> int:
        if window_length is None:
            return len(self.reference_states) if self.window_length is None else int(self.window_length)
        window_length = int(window_length)
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        return window_length

    def resolve_replay_window_length(self, window_length: int | None, num_intervals: int) -> int:
        if window_length is None:
            if self.window_length is None:
                return max(int(num_intervals), 1)
            return self.resolve_window_length(self.window_length)
        return self.resolve_window_length(window_length)

    def initial_reference_pose(self, reference_states=None) -> jnp.ndarray:
        reference_states = self.reference_states if reference_states is None else reference_states
        return jnp.asarray(reference_states[0, :3], dtype=jnp.float32)

    def _init_states(self, robot_key, estimator_key, reference_states=None):
        pose0 = self.initial_reference_pose(reference_states)
        robot_state = self.robot.get_init_state(key=robot_key, init_pose=pose0)
        estimator_state = self.estimator.get_init_state(key=estimator_key, start_pose=pose0)
        controller_state = jnp.zeros(4, dtype=jnp.float32)
        delayed_wheel_ref = jnp.zeros(2, dtype=jnp.float32)
        return robot_state, estimator_state, controller_state, delayed_wheel_ref

    @staticmethod
    def pose_mse(predicted_poses, target_poses):
        pos_error = predicted_poses[:, :2] - target_poses[:, :2]
        angle_error = predicted_poses[:, 2] - target_poses[:, 2]
        return jnp.mean(jnp.sum(pos_error**2, axis=1) + 2.0 - 2.0 * jnp.cos(angle_error))

    def run_closed_loop(
        self,
        robot_params: PhysicalParams,
        est_params: PhysicalParams | None = None,
        use_hidden_robot: bool = False,
        controller_gains=None,
        robot_key=None,
        estimator_key=None,
        wheel_speed_log_source: str = "estimated",
        reference_states=None,
    ) -> SimulationLog:
        if wheel_speed_log_source not in {"estimated", "true"}:
            raise ValueError("wheel_speed_log_source must be 'estimated' or 'true'.")
        robot_key = self.target_robot_key if robot_key is None else robot_key
        estimator_key = self.target_estimator_key if estimator_key is None else estimator_key
        model_params = robot_params if est_params is None else est_params
        reference_states = self.reference_states if reference_states is None else reference_states
        carry0 = self._init_states(robot_key, estimator_key, reference_states)

        def geometry_step(carry, ref_state):
            robot_state, estimator_state, controller_state, delayed_wheel_ref = carry
            pose_est = self.estimator.get_est_pose(estimator_state)
            wheel_ref = self.controller.compute_wheel_reference(
                ref_state,
                pose_est,
                gains=controller_gains,
                wheel_radius=robot_params.wheel_radius,
                base_diameter=robot_params.base_diameter,
            )

            def wheel_step(inner_carry, inner_index):
                inner_robot_state, inner_estimator_state, inner_controller_state = inner_carry
                applied_wheel_ref = jnp.where(inner_index == 0, delayed_wheel_ref, wheel_ref)
                wheel_est = self.estimator.get_est_wheel_speeds(inner_estimator_state)
                next_controller_state, applied_duty_cycle = self.controller.compute_duty(
                    inner_controller_state,
                    applied_wheel_ref,
                    wheel_est,
                    gains=controller_gains,
                    max_wheel_speed=robot_params.max_wheel_speed,
                )
                if use_hidden_robot:
                    next_robot_state = self.robot.step(inner_robot_state, applied_duty_cycle, dt=self.wheel_dt)
                else:
                    next_robot_state = self.robot.step(
                        inner_robot_state,
                        applied_duty_cycle,
                        wheel_radius=robot_params.wheel_radius,
                        base_diameter=robot_params.base_diameter,
                        max_wheel_speed=robot_params.max_wheel_speed,
                        time_constant=robot_params.time_constant,
                        dt=self.wheel_dt,
                    )
                next_estimator_state = self.estimator.update(
                    inner_estimator_state,
                    next_robot_state.wheel_speeds[0],
                    next_robot_state.wheel_speeds[1],
                    self.robot.get_pose(next_robot_state),
                    wheel_radius=model_params.wheel_radius,
                    base_diameter=model_params.base_diameter,
                    dt=self.wheel_dt,
                )
                return (
                    next_robot_state,
                    next_estimator_state,
                    next_controller_state,
                ), (
                    self.robot.get_pose(next_robot_state),
                    self.estimator.get_est_pose(next_estimator_state),
                    next_robot_state.wheel_speeds,
                    self.estimator.get_est_wheel_speeds(next_estimator_state),
                    next_robot_state.vel_omega,
                    applied_duty_cycle,
                )

            (next_robot_state, next_estimator_state, next_controller_state), wheel_outputs = jax.lax.scan(
                wheel_step,
                (robot_state, estimator_state, controller_state),
                jnp.arange(self.inner_steps_per_geometry_step),
            )
            next_carry = (next_robot_state, next_estimator_state, next_controller_state, wheel_ref)
            return next_carry, (
                wheel_ref,
                wheel_outputs[0],
                wheel_outputs[1],
                wheel_outputs[2],
                wheel_outputs[3],
                wheel_outputs[4],
                wheel_outputs[5],
            )

        _, outputs = jax.lax.scan(geometry_step, carry0, reference_states[:-1])
        (
            wheel_cmds,
            true_pose_samples,
            pose_samples,
            true_wheel_speeds,
            estimated_wheel_speeds,
            wheel_vel_omega,
            duty_cycles,
        ) = outputs
        initial_pose = self.initial_reference_pose(reference_states)[None, :]
        pose_states = jnp.concatenate([initial_pose, pose_samples.reshape(-1, 3)], axis=0)
        true_pose_states = jnp.concatenate([initial_pose, true_pose_samples.reshape(-1, 3)], axis=0)
        duty_inputs = duty_cycles.reshape(-1, 2)
        selected_wheel_speeds = true_wheel_speeds if wheel_speed_log_source == "true" else estimated_wheel_speeds
        initial_wheel_speeds = carry0[0].wheel_speeds if wheel_speed_log_source == "true" else carry0[1].u_hat
        wheel_speed_log = jnp.vstack([initial_wheel_speeds, selected_wheel_speeds.reshape(-1, 2)])
        vel_omega_log = jnp.vstack([carry0[0].vel_omega, wheel_vel_omega.reshape(-1, 2)])
        duty_log = jnp.vstack([duty_inputs, duty_inputs[-1]])
        num_reference_samples = reference_states.shape[0]
        num_pose_samples = (num_reference_samples - 1) * self.inner_steps_per_geometry_step + 1

        return SimulationLog(
            reference=ReferenceLog(
                time_s=jnp.asarray(self.reference_time_grid[:num_reference_samples], dtype=jnp.float32),
                states=reference_states,
            ),
            wheel=WheelLog(
                time_s=jnp.asarray(self.wheel_time_grid[:num_pose_samples], dtype=jnp.float32),
                speeds=wheel_speed_log,
                vel_omega=vel_omega_log,
                duty_cycle=duty_log,
            ),
            pose=PoseLog(
                time_s=jnp.asarray(self.pose_time_grid[:num_pose_samples], dtype=jnp.float32),
                states=pose_states,
                true_states=true_pose_states,
                command_time_s=jnp.asarray(self.command_time_grid[: num_reference_samples - 1], dtype=jnp.float32),
                wheel_cmd=wheel_cmds,
            ),
        )

    def simulate(self, *args, **kwargs):
        return self.run_closed_loop(*args, **kwargs)


def make_replay_segment_plan(
    pose_times: np.ndarray,
    wheel_times: np.ndarray,
    window_length: int,
):
    pose_times = np.asarray(pose_times, dtype=float)
    wheel_times = np.asarray(wheel_times, dtype=float)
    segment_dt = []
    wheel_indices = []
    reset_mask = []
    reset_pose_indices = []
    record_indices = []

    for pose_index in range(len(pose_times) - 1):
        current_time = pose_times[pose_index]
        end_time = pose_times[pose_index + 1]
        wheel_index = max(0, int(np.searchsorted(wheel_times, current_time, side="right") - 1))
        first_segment = True

        event_indices = np.flatnonzero((wheel_times > current_time) & (wheel_times <= end_time))
        for event_index in event_indices:
            event_time = float(wheel_times[event_index])
            if event_time > current_time:
                _append_replay_segment(
                    segment_dt,
                    wheel_indices,
                    reset_mask,
                    reset_pose_indices,
                    event_time - current_time,
                    wheel_index,
                    pose_index if first_segment and pose_index % window_length == 0 else None,
                )
                first_segment = False
            current_time = event_time
            wheel_index = int(event_index)

        if end_time > current_time:
            _append_replay_segment(
                segment_dt,
                wheel_indices,
                reset_mask,
                reset_pose_indices,
                end_time - current_time,
                wheel_index,
                pose_index if first_segment and pose_index % window_length == 0 else None,
            )
        record_indices.append(len(segment_dt) - 1)

    return (
        np.asarray(segment_dt, dtype=np.float32),
        np.asarray(wheel_indices, dtype=np.int32),
        np.asarray(reset_mask, dtype=bool),
        np.asarray(reset_pose_indices, dtype=np.int32),
        np.asarray(record_indices, dtype=np.int32),
    )


def replay_simulation_log(
    robot: DiffDrive,
    robot_key: jax.Array,
    target_log: SimulationLog,
    robot_params: PhysicalParams,
    replay_segment_plan,
) -> SimulationLog:
    predicted_poses = replay_pose_states(
        robot=robot,
        robot_key=robot_key,
        initial_pose=target_log.pose.states[0],
        initial_wheel_speeds=target_log.wheel.speeds[0],
        pose_states=target_log.pose.states,
        wheel_speeds=target_log.wheel.speeds,
        duty_cycles=target_log.wheel.duty_cycle,
        robot_params=robot_params,
        replay_segment_plan=replay_segment_plan,
    )
    return SimulationLog(
        reference=target_log.reference,
        wheel=target_log.wheel,
        pose=PoseLog(
            time_s=target_log.pose.time_s[1:],
            states=predicted_poses,
            true_states=predicted_poses,
            command_time_s=target_log.pose.command_time_s,
            wheel_cmd=target_log.pose.wheel_cmd,
        ),
    )


def replay_pose_states(
    robot: DiffDrive,
    robot_key: jax.Array,
    initial_pose: jax.Array,
    initial_wheel_speeds: jax.Array,
    pose_states: jax.Array,
    wheel_speeds: jax.Array,
    duty_cycles: jax.Array,
    robot_params: PhysicalParams,
    replay_segment_plan,
) -> jax.Array:
    segment_dt, wheel_indices, reset_mask, reset_pose_indices, record_indices = replay_segment_plan
    segment_dt = jnp.asarray(segment_dt, dtype=jnp.float32)
    wheel_indices = jnp.asarray(wheel_indices, dtype=jnp.int32)
    reset_mask = jnp.asarray(reset_mask)
    reset_pose_indices = jnp.asarray(reset_pose_indices, dtype=jnp.int32)
    record_indices = jnp.asarray(record_indices, dtype=jnp.int32)

    state = DiffDriveState(
        pose=initial_pose,
        wheel_speeds=initial_wheel_speeds,
        key=robot_key,
        vel_omega=jnp.zeros(2, dtype=jnp.float32),
        duty_cycle=jnp.zeros(2, dtype=jnp.float32),
        wheel_speed_cmd=jnp.zeros(2, dtype=jnp.float32),
    )
    segment_speeds = wheel_speeds[wheel_indices]
    segment_duty = duty_cycles[wheel_indices]
    reset_poses = pose_states[reset_pose_indices]

    def replay_step(carry, inputs):
        dt, speed, duty, do_reset, reset_pose = inputs

        def reset_state(state):
            return state._replace(pose=reset_pose, wheel_speeds=speed)

        carry = jax.lax.cond(do_reset, reset_state, lambda state: state, carry)
        next_state = robot.step_kinematic(
            carry,
            speed,
            duty,
            wheel_radius=robot_params.wheel_radius,
            base_diameter=robot_params.base_diameter,
            dt=dt,
        )
        return next_state, next_state.pose

    _, segment_poses = jax.lax.scan(
        replay_step,
        state,
        (segment_dt, segment_speeds, segment_duty, reset_mask, reset_poses),
    )
    return segment_poses[record_indices]


def _append_replay_segment(
    segment_dt,
    wheel_indices,
    reset_mask,
    reset_pose_indices,
    dt: float,
    wheel_index: int,
    reset_pose_index: int | None,
):
    segment_dt.append(float(dt))
    wheel_indices.append(int(wheel_index))
    reset_mask.append(reset_pose_index is not None)
    reset_pose_indices.append(0 if reset_pose_index is None else int(reset_pose_index))
