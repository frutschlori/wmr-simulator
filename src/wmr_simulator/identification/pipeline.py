import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.losses import window_replay_mse
from wmr_simulator.identification.optimizers import bootstrap_identification_adam, optimize_physical_params_adam
from wmr_simulator.robot import DiffDriveState
from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.types import (
    PhysicalParams,
    PoseLog,
    SimulationLog,
    clip_physical_params,
    physical_params_from_array,
    physical_params_mse,
)


class SystemIdentificationPipeline(SimulationPipeline):
    def __init__(
        self,
        problem_path: str,
        initial_params: PhysicalParams,
        seed: int = 0,
        reference_trajectories_dir: str | None = None,
        window_length: int | None = None,
        target_log: SimulationLog | None = None,
        replay_wheel_speed_source: str = "estimated",
    ):
        super().__init__(
            problem_path=problem_path,
            seed=seed,
            reference_trajectories_dir=reference_trajectories_dir,
            window_length=window_length,
        )
        self.initial_params = initial_params
        self.replay_wheel_speed_source = replay_wheel_speed_source
        self.uses_external_target_log = target_log is not None
        self.target_log = target_log if target_log is not None else self.run_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=self.gains,
            wheel_speed_log_source=replay_wheel_speed_source,
        )
        if target_log is not None:
            self.reference_states = self.target_log.reference.states
            self.pose_time_grid = np.asarray(self.target_log.pose.time_s, dtype=float)
            self.wheel_time_grid = np.asarray(self.target_log.wheel.time_s, dtype=float)
            self.sim_time_grid = self.pose_time_grid
        self.replay_segment_plan = self.make_replay_segment_plan(self.target_log, self.window_length)

    @staticmethod
    def _clip_physical_params(params: PhysicalParams):
        return clip_physical_params(params)

    @staticmethod
    def parameter_mse(params: PhysicalParams, target_params: PhysicalParams):
        return physical_params_mse(params, target_params)

    def resolve_window_length(self, window_length: int | None = None) -> int:
        if window_length is None:
            return len(self.target_log.pose.states) if self.window_length is None else int(self.window_length)
        window_length = int(window_length)
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        return window_length

    def resolve_motor_window_length(self, window_length: int | None, num_wheel_samples: int) -> int:
        if window_length is None:
            window_length = self.window_length
        if window_length is None:
            return max(int(num_wheel_samples) - 1, 1)
        window_length = int(window_length)
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        return window_length

    def run_target_closed_loop(self, *args, **kwargs) -> SimulationLog:
        return self.run_closed_loop(*args, **kwargs)

    def motor_wheel_speed_rollout(
        self,
        params: PhysicalParams,
        target_log: SimulationLog,
        window_length: int | None = None,
    ):
        speeds = target_log.wheel.speeds
        window_length = self.resolve_motor_window_length(window_length, speeds.shape[0])
        duty = jnp.clip(target_log.wheel.duty_cycle, -1.0, 1.0)
        dts = jnp.diff(target_log.wheel.time_s)
        if speeds.shape[0] < 2:
            return speeds[:0]
        step_indices = jnp.arange(speeds.shape[0] - 1, dtype=jnp.int32)
        reset_mask = (step_indices % window_length) == 0

        def motor_step(wheel_speed, inputs):
            measured_speed, duty_k, dt, do_reset = inputs
            wheel_speed = jnp.where(do_reset, measured_speed, wheel_speed)
            safe_tau = jnp.maximum(params.time_constant, 1e-3)
            alpha = jnp.where(params.time_constant >= 1e-3, jnp.exp(-dt / safe_tau), 0.0)
            next_speed = alpha * wheel_speed + (1.0 - alpha) * params.max_wheel_speed * duty_k
            return next_speed, next_speed

        _, predicted = jax.lax.scan(motor_step, speeds[0], (speeds[:-1], duty[:-1], dts, reset_mask))
        return predicted

    def motor_wheel_speed_mse(
        self,
        params: PhysicalParams,
        target_log: SimulationLog,
        window_length: int | None = None,
    ):
        target = target_log.wheel.speeds[1:]
        predicted = self.motor_wheel_speed_rollout(params, target_log, window_length=window_length)
        error = predicted - target
        target_power = jnp.mean(jnp.sum(target**2, axis=1))
        return jnp.mean(jnp.sum(error**2, axis=1)) / jnp.maximum(target_power, 1.0)

    def replay_rollout(
        self,
        robot_params: PhysicalParams,
        target_log: SimulationLog,
        est_params=None,
        window_length: int | None = None,
        replay_segment_plan=None,
    ) -> SimulationLog:
        if replay_segment_plan is None:
            replay_segment_plan = self.make_replay_segment_plan(target_log, window_length)
        segment_dt, wheel_indices, reset_mask, reset_pose_indices, record_indices = replay_segment_plan
        segment_dt = jnp.asarray(segment_dt, dtype=jnp.float32)
        wheel_indices = jnp.asarray(wheel_indices, dtype=jnp.int32)
        reset_mask = jnp.asarray(reset_mask)
        reset_pose_indices = jnp.asarray(reset_pose_indices, dtype=jnp.int32)
        record_indices = jnp.asarray(record_indices, dtype=jnp.int32)

        state = DiffDriveState(
            pose=target_log.pose.states[0],
            wheel_speeds=target_log.wheel.speeds[0],
            key=self.robot_key,
            vel_omega=jnp.zeros(2, dtype=jnp.float32),
            duty_cycle=jnp.zeros(2, dtype=jnp.float32),
            wheel_speed_cmd=jnp.zeros(2, dtype=jnp.float32),
        )
        segment_speeds = target_log.wheel.speeds[wheel_indices]
        segment_duty = target_log.wheel.duty_cycle[wheel_indices]
        reset_poses = target_log.pose.states[reset_pose_indices]

        def replay_step(carry, inputs):
            dt, speed, duty, do_reset, reset_pose = inputs

            def reset_state(state):
                return state._replace(pose=reset_pose, wheel_speeds=speed)

            carry = jax.lax.cond(do_reset, reset_state, lambda state: state, carry)
            next_state = self.robot.step_kinematic(
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
        predicted_poses = segment_poses[record_indices]

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

    def make_replay_segment_plan(self, target_log: SimulationLog, window_length: int | None = None):
        return _replay_segments(
            np.asarray(target_log.pose.time_s, dtype=float),
            np.asarray(target_log.wheel.time_s, dtype=float),
            self.resolve_window_length(window_length),
        )

    def loss(self, params: PhysicalParams):
        return window_replay_mse(
            pipeline=self,
            params=params,
            target_log=self.target_log,
            est_params=self.initial_params,
            window_length=self.window_length,
        )

    def optimize(
        self,
        init_params: PhysicalParams,
        num_steps: int,
        learning_rate: float,
    ):
        return optimize_physical_params_adam(
            pipeline=self,
            init_params=init_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
        )


def _replay_segments(
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
                _append_segment(
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
            _append_segment(
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


def _append_segment(
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


def run_single_experiment_identification(
    problem_path: str,
    initial_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
    window_length: int | None = None,
    target_log: SimulationLog | None = None,
    bootstrap_samples: int | None = None,
    replay_wheel_speed_source: str = "estimated",
):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
        target_log=target_log,
        replay_wheel_speed_source=replay_wheel_speed_source,
    )
    init_target_log = pipeline.target_log
    init_replay_log = pipeline.replay_rollout(initial_params, target_log=pipeline.target_log, window_length=window_length)
    bootstrap = None
    if bootstrap_samples is not None and bootstrap_samples > 0:
        if pipeline.uses_external_target_log:
            raise ValueError("Bootstrap identification is only supported for simulated target logs.")
        bootstrap = bootstrap_identification_adam(
            pipeline=pipeline,
            initial_params=initial_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        estimated_params = physical_params_from_array(bootstrap["parameter_mean"])
        loss_history = jnp.mean(bootstrap["loss_history"], axis=0)
        motor_loss_history = jnp.mean(bootstrap["motor_loss_history"], axis=0)
        parameter_mse_history = jnp.mean(bootstrap["parameter_mse_history"], axis=0)
    else:
        estimated_params, loss_history, motor_loss_history, parameter_mse_history = pipeline.optimize(
            init_params=initial_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
        )
    if not pipeline.uses_external_target_log:
        pipeline.target_log = pipeline.run_closed_loop(
            estimated_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
            wheel_speed_log_source=pipeline.replay_wheel_speed_source,
        )
    pipeline.estimated_params = estimated_params
    final_replay_log = pipeline.replay_rollout(estimated_params, target_log=pipeline.target_log, window_length=window_length)
    return {
        "pipeline": pipeline,
        "init_target_log": init_target_log,
        "init_replay_log": init_replay_log,
        "estimated_params": estimated_params,
        "loss_history": loss_history,
        "motor_loss_history": motor_loss_history,
        "parameter_mse_history": parameter_mse_history,
        "bootstrap": bootstrap,
        "final_replay_log": final_replay_log,
    }
