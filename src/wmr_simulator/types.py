from typing import NamedTuple

import jax
import jax.numpy as jnp

from wmr_simulator.estimator import EstimatorState
from wmr_simulator.robot import DiffDriveState


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    base_diameter: jax.Array
    max_wheel_speed: jax.Array = jnp.asarray(150.0, dtype=jnp.float32)
    time_constant: jax.Array = jnp.asarray(0.0, dtype=jnp.float32)


class SimulationLog(NamedTuple):
    robot_states: DiffDriveState
    estimator_states: EstimatorState


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


def physical_params_from_array(values: jax.Array) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=values[..., 0],
        base_diameter=values[..., 1],
        max_wheel_speed=values[..., 2],
        time_constant=values[..., 3],
    )


def print_controller_gains(label: str, gains: jax.Array):
    print(label)
    print(f"  kx={float(gains[0]):.7f}")
    print(f"  ky={float(gains[1]):.7f}")
    print(f"  kth={float(gains[2]):.7f}")
    print(f"  kprmotor={float(gains[3]):.7f}")
    print(f"  kplmotor={float(gains[4]):.7f}")
    print(f"  kirmotor={float(gains[5]):.7f}")
    print(f"  kilmotor={float(gains[6]):.7f}")


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
