"""The wheel-speed PI(D) controller against the firmware's inner loop."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.controller import (
    KDMOTOR_INDEX,
    Controller,
    initial_controller_state,
)

MOTOR_GAIN = 223.0
GAINS = [4.5, 6.0, 12.0, 2.5, 5.0, 0.03]


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
