from typing import NamedTuple

import jax
import jax.numpy as jnp

from wmr_simulator.estimator import EstimatorState
from wmr_simulator.robot import DiffDriveState


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    base_diameter: jax.Array


class SimulationLog(NamedTuple):
    robot_states: DiffDriveState
    estimator_states: EstimatorState


def print_physical_params(label: str, params: PhysicalParams):
    print(label)
    print(f"  wheel_radius={1000.0 * float(params.wheel_radius):.2f} mm")
    print(f"  base_diameter={1000.0 * float(params.base_diameter):.2f} mm")


def physical_params_to_array(params: PhysicalParams) -> jax.Array:
    return jnp.stack([params.wheel_radius, params.base_diameter], axis=-1)


def physical_params_from_array(values: jax.Array) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=values[..., 0],
        base_diameter=values[..., 1],
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
    )


def physical_params_mse(params: PhysicalParams, target_params: PhysicalParams):
    squared_errors = jnp.array([
        (1000.0 * (params.wheel_radius - target_params.wheel_radius)) ** 2,
        (1000.0 * (params.base_diameter - target_params.base_diameter)) ** 2,
    ])
    return jnp.mean(squared_errors)
