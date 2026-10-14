"""Measure the hidden plant's effective parameters (rolling radius, wheelbase,
motor gain and time constant) by driving it open loop.

The effective wheelbase grows with yaw rate (tire scrub), and the motor time
constant seen in the logs includes the firmware's 3 Hz low-pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wmr_simulator.mujoco_sim.plant import MujocoPlant

# Duties that stay inside the plant's traction envelope.
STRAIGHT_DUTIES = (0.2, 0.3, 0.4, 0.5, 0.6)
SPIN_DUTIES = (0.10, 0.15)
ARC_DUTIES = ((0.18, 0.22), (0.12, 0.18))

# Open-loop driving is directionally unstable (CoM behind the axle), so every
# measurement window ends before the yaw drift grows.
SETTLE_TIME = 1.5
RISE_WINDOW_S = 1.5


@dataclass(frozen=True)
class PlantTruth:
    """The plant's effective parameters, in the units identification reports."""

    wheel_radius: float
    wheel_base_arc: float
    wheel_base_spin: float
    max_wheel_speed: float
    time_constant_mechanical: float
    time_constant_logged: float

    def __str__(self) -> str:
        return "\n".join(
            (
                f"{'wheel_radius':<27}{1000 * self.wheel_radius:8.3f} mm",
                f"{'base_diameter (arcs)':<27}{1000 * self.wheel_base_arc:8.2f} mm",
                f"{'base_diameter (spin)':<27}{1000 * self.wheel_base_spin:8.2f} mm",
                f"{'max_wheel_speed':<27}{self.max_wheel_speed:8.2f} rad/s",
                f"{'time_constant (mechanical)':<27}{self.time_constant_mechanical:8.4f} s",
                f"{'time_constant (logged)':<27}{self.time_constant_logged:8.4f} s",
            )
        )


def measure_plant_truth(plant: MujocoPlant | None = None) -> PlantTruth:
    """Drive the plant through steady manoeuvres and read its effective parameters."""
    plant = MujocoPlant(seed=0) if plant is None else plant

    radii = [_steady_radius(plant, duty) for duty in STRAIGHT_DUTIES]
    wheel_radius = float(np.mean(radii))
    return PlantTruth(
        wheel_radius=wheel_radius,
        wheel_base_arc=float(np.mean([_steady_wheel_base(plant, left, right, wheel_radius) for left, right in ARC_DUTIES])),
        wheel_base_spin=float(np.mean([_steady_wheel_base(plant, -duty, duty, wheel_radius) for duty in SPIN_DUTIES])),
        max_wheel_speed=_motor_gain(plant),
        time_constant_mechanical=_rise_time_constant(plant, logged=False),
        time_constant_logged=_rise_time_constant(plant, logged=True),
    )


def _drive(plant: MujocoPlant, duty_left: float, duty_right: float, seconds: float = SETTLE_TIME) -> None:
    plant.reset()
    plant.set_duty(duty_left, duty_right)
    plant.step(plant.steps_for(seconds))


def _steady_radius(plant: MujocoPlant, duty: float) -> float:
    """``v_true / omega_wheel`` in steady straight driving."""
    _drive(plant, duty, duty)
    speed, _ = plant.twist()
    return speed / float(np.mean(plant.wheel_speeds()))


def _steady_wheel_base(plant: MujocoPlant, duty_left: float, duty_right: float, wheel_radius: float) -> float:
    """``r_eff * (omega_r - omega_l) / w_true`` in a steady turn."""
    _drive(plant, duty_left, duty_right)
    _, yaw_rate = plant.twist()
    left, right = plant.wheel_speeds()
    return wheel_radius * (right - left) / yaw_rate


def _motor_gain(plant: MujocoPlant) -> float:
    """Loaded steady-state wheel speed per unit duty, over the usable duty range."""
    ratios = []
    for duty in (0.2, 0.4, 0.6):
        _drive(plant, duty, duty)
        ratios.append(float(np.mean(plant.wheel_speeds())) / duty)
    return float(np.mean(ratios))


def _rise_time_constant(plant: MujocoPlant, logged: bool, duty: float = 0.6) -> float:
    """Time to 63% of the steady wheel speed after a duty step from rest.

    ``logged=True`` reads the value through the firmware's inner-loop low-pass,
    i.e. the ``omega_*_meas`` an identification log actually carries.
    """
    from types import SimpleNamespace

    from wmr_simulator.mujoco_sim.firmware import INNER_PERIOD_S, FirmwareConfig, InnerLoop
    from wmr_simulator.pololu.robot_config import robot_config_values

    plant.reset()
    inner = None
    if logged:
        config = FirmwareConfig.from_mapping(
            robot_config_values(
                physical_params=SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0),
                controller_gains=[1.0, 1.0, 1.0, 1.0, 0.0],
            )
        )
        inner = InnerLoop(config)
        inner.reset(plant.encoder_counts())

    steps_per_tick = plant.steps_for(INNER_PERIOD_S)
    speeds = []
    for _ in range(int(round(RISE_WINDOW_S / INNER_PERIOD_S))):
        for _ in range(steps_per_tick):
            plant.set_duty(duty, duty)
            plant.step()
        if inner is not None:
            inner.update(plant.encoder_counts())
            speeds.append(float(np.mean(inner.omega_lp)))
        else:
            speeds.append(float(np.mean(plant.wheel_speeds())))

    speeds = np.asarray(speeds)
    # Last 0.5 s of the window.
    final = speeds[-50:].mean()
    return float(INNER_PERIOD_S * int(np.argmax(speeds >= (1.0 - np.exp(-1.0)) * final)))
