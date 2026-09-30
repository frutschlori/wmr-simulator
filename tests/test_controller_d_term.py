"""The wheel-speed D-term (kdmotor): firmware parity, legacy gain vectors, design FIM."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.controller import (
    GAIN_NAMES,
    KDMOTOR_INDEX,
    NUM_GAINS,
    Controller,
    controller_gains_array,
    controller_gains_list,
    initial_controller_state,
)

MOTOR_GAIN = 223.0
GAINS = [4.5, 6.0, 12.0, 2.5, 5.0, 0.03]


def test_kdmotor_is_the_last_gain_so_old_indices_keep_their_meaning():
    assert GAIN_NAMES[:5] == ("kx", "ky", "kth", "kpmotor", "kimotor")
    assert KDMOTOR_INDEX == NUM_GAINS - 1 == 5


def test_a_five_gain_vector_is_read_with_kdmotor_zero():
    legacy = [4.5, 6.0, 12.0, 2.5, 5.0]
    np.testing.assert_allclose(np.asarray(controller_gains_array(legacy)), [*legacy, 0.0])
    assert controller_gains_list(legacy) == [*legacy, 0.0]
    batch = controller_gains_array(np.ones((3, 5)))
    assert batch.shape == (3, NUM_GAINS)
    np.testing.assert_allclose(np.asarray(batch[:, KDMOTOR_INDEX]), 0.0)
    with pytest.raises(ValueError, match="Expected 6 controller gains"):
        controller_gains_list([1.0, 2.0, 3.0])


def test_a_five_gain_problem_yaml_simulates_without_the_d_term():
    from wmr_simulator.simulation import SimulationPipeline

    pipeline = SimulationPipeline(problem_path="problems/pololu_stock.yaml", seed=0)
    assert pipeline.gains.shape == (NUM_GAINS,)
    assert float(pipeline.gains[KDMOTOR_INDEX]) == 0.0


def test_the_d_term_matches_the_firmware_inner_loop():
    """Same commands, wheels blocked (zero speed): the simulator's wheel-speed
    PID and the firmware port's ``InnerLoop`` put out the same duty tick by
    tick, including the derivative kick of a command step and the first tick,
    where the firmware's previous error starts at 0."""
    from wmr_simulator.mujoco_sim.firmware import FirmwareConfig, InnerLoop
    from wmr_simulator.pololu.robot_config import robot_config_values

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=MOTOR_GAIN)
    firmware_config = FirmwareConfig.from_mapping(
        robot_config_values(physical_params=physical, controller_gains=GAINS)
    )
    assert firmware_config.kd_inner == pytest.approx(GAINS[KDMOTOR_INDEX] / MOTOR_GAIN)
    inner = InnerLoop(firmware_config)
    inner.reset((0, 0))

    controller = Controller(
        robot_param={"wheel_radius": 0.016, "base_diameter": 0.0825, "max_wheel_speed": MOTOR_GAIN},
        gains=GAINS,
        duty_limits=[-1.0, 1.0],
        dt=inner.dt,
    )
    state = initial_controller_state()
    for command in (20.0, 20.0, 35.0, 35.0, 10.0, 10.0):
        inner.set_command(command, command)
        firmware_duty = inner.update((0, 0))
        state, sim_duty = controller.compute_duty(
            state, jnp.asarray([command, command]), jnp.asarray([0.0, 0.0])
        )
        np.testing.assert_allclose(np.asarray(sim_duty), firmware_duty, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.asarray(state[2:]), [command, command])


def test_kdmotor_zero_leaves_the_pi_controller_unchanged():
    robot = {"wheel_radius": 0.016, "base_diameter": 0.0825, "max_wheel_speed": MOTOR_GAIN}
    with_d = Controller(robot_param=robot, gains=GAINS, duty_limits=[-1.0, 1.0], dt=0.01)
    without_d = Controller(robot_param=robot, gains=GAINS[:5], duty_limits=[-1.0, 1.0], dt=0.01)
    state = jnp.asarray([0.1, -0.1, 3.0, 3.0], dtype=jnp.float32)
    ref, meas = jnp.asarray([30.0, 28.0]), jnp.asarray([25.0, 29.0])
    _, duty_pi = without_d.compute_duty(state, ref, meas)
    _, duty_pid = with_d.compute_duty(state, ref, meas)
    derivative = (np.asarray(ref - meas) - 3.0) / 0.01
    np.testing.assert_allclose(
        np.asarray(duty_pid - duty_pi), GAINS[KDMOTOR_INDEX] * derivative / MOTOR_GAIN, rtol=1e-4
    )


def test_the_kdmotor_fim_column_is_pinned_not_relative():
    """At a design point of kdmotor = 0 a relative scale would blank the column;
    it falls back to the stock value, and an explicit scale pins it."""
    from wmr_simulator.trajectory_optimization.pipeline import (
        KDMOTOR_FIM_SCALE_FALLBACK,
        TrajectoryOptimizationPipeline,
    )

    design_gains = [2.4, 5.1, 6.3, 0.9, 0.0, 0.0]
    tied = TrajectoryOptimizationPipeline(
        problem_path="problems/pololu_gains.yaml", objective_mode="gain-tuning", controller_gains=design_gains
    )
    assert tied.kdmotor_fim_scale == pytest.approx(KDMOTOR_FIM_SCALE_FALLBACK)
    scaling = np.asarray(tied.fim_parameter_scaling(tied.nominal_parameters()))
    assert scaling[KDMOTOR_INDEX] == pytest.approx(KDMOTOR_FIM_SCALE_FALLBACK)

    pinned = TrajectoryOptimizationPipeline(
        problem_path="problems/pololu_gains.yaml",
        objective_mode="gain-tuning",
        controller_gains=design_gains,
        kdmotor_fim_scale=0.05,
    )
    scaling = np.asarray(pinned.fim_parameter_scaling(pinned.nominal_parameters()))
    assert scaling[KDMOTOR_INDEX] == pytest.approx(0.05)
    # The other columns are untouched by the new one.
    np.testing.assert_allclose(scaling[:3], design_gains[:3], rtol=1e-6)
