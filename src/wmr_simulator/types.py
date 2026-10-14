from typing import NamedTuple

import jax
import jax.numpy as jnp


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    # base_diameter is the effective wheelbase (it absorbs tire scrub in turns).
    base_diameter: jax.Array
    max_wheel_speed: jax.Array = jnp.asarray(150.0, dtype=jnp.float32)
    time_constant: jax.Array = jnp.asarray(0.0, dtype=jnp.float32)


class ReferenceLog(NamedTuple):
    time_s: jax.Array
    states: jax.Array  # [x, y, theta, vx, vy, omega, ax, ay]


class WheelLog(NamedTuple):
    time_s: jax.Array
    speeds: jax.Array      # [omega_right, omega_left]
    vel_omega: jax.Array   # [v, omega]
    duty_cycle: jax.Array  # [duty_right, duty_left]


class PoseLog(NamedTuple):
    time_s: jax.Array         # pose measurement times
    states: jax.Array         # [x, y, theta]
    true_states: jax.Array    # [x, y, theta]
    command_time_s: jax.Array # geometry-controller command times
    wheel_cmd: jax.Array      # [omega_right_cmd, omega_left_cmd]
    # [v_x, v_y, omega] body twist at the pose times from the smoothing filter
    # (Pololu logs); None for simulated logs.
    twists: jax.Array | None = None
    # Raw mocap poses that survived outlier rejection, and their times (Pololu
    # logs); None for simulated logs.
    clean_time_s: jax.Array | None = None
    clean_states: jax.Array | None = None
    # Controller gains applied at each geometry step (command_time_s); None for
    # replayed and recorded logs.
    gains: jax.Array | None = None


class SimulationLog(NamedTuple):
    reference: ReferenceLog
    wheel: WheelLog
    pose: PoseLog


def print_physical_params(label: str, params: PhysicalParams):
    print(label)
    print(f"  wheel_radius={1000.0 * float(params.wheel_radius):.2f} mm")
    print(f"  base_diameter={1000.0 * float(params.base_diameter):.2f} mm")
    print(f"  max_wheel_speed={float(params.max_wheel_speed):.2f} rad/s")
    print(f"  time_constant={float(params.time_constant):.4f} s")


def physical_params_to_array(params: PhysicalParams) -> jax.Array:
    return jnp.stack(
        [
            params.wheel_radius,
            params.base_diameter,
            params.max_wheel_speed,
            params.time_constant,
        ],
        axis=-1,
    )


def print_controller_gains(label: str, gains: jax.Array):
    from wmr_simulator.controller import GAIN_NAMES

    print(label)
    for name, value in zip(GAIN_NAMES, gains):
        print(f"  {name}={float(value):.7f}")


def clip_physical_params(params: PhysicalParams) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=jnp.clip(params.wheel_radius, min=1e-4),
        base_diameter=jnp.clip(params.base_diameter, min=1e-4),
        max_wheel_speed=jnp.clip(params.max_wheel_speed, min=1e-4),
        time_constant=jnp.clip(params.time_constant, min=1e-4),
    )


def physical_params_mse(params: PhysicalParams, target_params: PhysicalParams):
    squared_errors = jnp.stack(
        [
            (1000.0 * (params.wheel_radius - target_params.wheel_radius)) ** 2,
            (1000.0 * (params.base_diameter - target_params.base_diameter)) ** 2,
            (params.max_wheel_speed - target_params.max_wheel_speed) ** 2,
            (params.time_constant - target_params.time_constant) ** 2,
        ],
        axis=-1,
    )
    return jnp.mean(squared_errors, axis=-1)
