from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.losses import window_replay_mse
from wmr_simulator.identification.optimizers import bootstrap_identification_adam, optimize_physical_params_adam
from wmr_simulator.simulation import (
    SimulationPipeline,
    make_replay_segment_plan,
    replay_simulation_log,
)
from wmr_simulator.types import (
    PhysicalParams,
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

    def run_target_closed_loop(self, *args, **kwargs) -> SimulationLog:
        return self.run_closed_loop(*args, **kwargs)

    def motor_wheel_speed_rollout(
        self,
        params: PhysicalParams,
        target_log: SimulationLog,
        window_length: int | None = None,
    ):
        speeds = target_log.wheel.speeds
        window_length = self.resolve_replay_window_length(window_length, speeds.shape[0] - 1)
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
        return replay_simulation_log(
            robot=self.robot,
            robot_key=self.robot_key,
            target_log=target_log,
            robot_params=robot_params,
            replay_segment_plan=replay_segment_plan,
        )

    def make_replay_segment_plan(self, target_log: SimulationLog, window_length: int | None = None):
        num_intervals = max(len(target_log.pose.time_s) - 1, 1)
        return make_replay_segment_plan(
            np.asarray(target_log.pose.time_s, dtype=float),
            np.asarray(target_log.wheel.time_s, dtype=float),
            self.resolve_replay_window_length(window_length, num_intervals),
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
            pipelines=[self],
            init_params=init_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
        )


def run_multi_log_identification(
    problem_path: str,
    initial_params: PhysicalParams,
    target_logs: Sequence[SimulationLog],
    num_steps: int,
    learning_rate: float,
    seed: int = 0,
    window_length: int | None = None,
    replay_wheel_speed_source: str = "estimated",
    identify_a_slip_max: bool = True,
):
    """Identify one parameter set jointly from several recorded logs.

    One replay pipeline per log (they differ in length, so they cannot be
    batched); the logs enter the fit through the mean of their normalized
    losses, see optimizers.optimize_physical_params_adam. Passing a single log
    reproduces a plain single-log identification. ``identify_a_slip_max=False``
    keeps the traction limit at ``initial_params``.
    """
    pipelines = [
        SystemIdentificationPipeline(
            problem_path=problem_path,
            initial_params=initial_params,
            seed=seed,
            window_length=window_length,
            target_log=target_log,
            replay_wheel_speed_source=replay_wheel_speed_source,
        )
        for target_log in target_logs
    ]
    init_replay_logs = [
        pipeline.replay_rollout(initial_params, target_log=pipeline.target_log, window_length=window_length)
        for pipeline in pipelines
    ]
    estimated_params, loss_history, motor_loss_history, parameter_mse_history = optimize_physical_params_adam(
        pipelines=pipelines,
        init_params=initial_params,
        num_steps=num_steps,
        learning_rate=learning_rate,
        identify_a_slip_max=identify_a_slip_max,
    )
    final_replay_logs = []
    for pipeline in pipelines:
        pipeline.estimated_params = estimated_params
        final_replay_logs.append(
            pipeline.replay_rollout(
                estimated_params,
                target_log=pipeline.target_log,
                window_length=window_length,
            )
        )
    return {
        "pipelines": pipelines,
        "estimated_params": estimated_params,
        "loss_history": loss_history,
        "motor_loss_history": motor_loss_history,
        "parameter_mse_history": parameter_mse_history,
        "init_replay_logs": init_replay_logs,
        "final_replay_logs": final_replay_logs,
    }


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
        pipeline.target_log = pipeline.run_target_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
            robot_key=bootstrap["target_robot_key"],
            estimator_key=bootstrap["target_estimator_key"],
            wheel_speed_log_source=pipeline.replay_wheel_speed_source,
        )
        plot_params = physical_params_from_array(bootstrap["parameter_samples"][0])
    else:
        estimated_params, loss_history, motor_loss_history, parameter_mse_history = pipeline.optimize(
            init_params=initial_params,
            num_steps=num_steps,
            learning_rate=learning_rate,
        )
        plot_params = estimated_params
    if bootstrap is None and not pipeline.uses_external_target_log:
        pipeline.target_log = pipeline.run_closed_loop(
            estimated_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
            wheel_speed_log_source=pipeline.replay_wheel_speed_source,
        )
    pipeline.estimated_params = estimated_params
    if bootstrap is not None:
        pipeline.estimated_params = plot_params
    final_replay_log = pipeline.replay_rollout(
        plot_params,
        target_log=pipeline.target_log,
        window_length=window_length,
    )
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
