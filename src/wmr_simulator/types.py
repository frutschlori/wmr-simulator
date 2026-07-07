import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PhysicalParams(NamedTuple):
    wheel_radius: jax.Array
    # NB: base_diameter is the *effective* wheelbase, not the geometric one -- tire
    # scrub in turns acts as a constant multiplicative wheelbase correction that is
    # not separately identifiable (Borenstein & Feng 1996, E_b). Identification fits
    # the effective value directly; see wmr_simulator.slip module docstring.
    base_diameter: jax.Array
    max_wheel_speed: jax.Array = jnp.asarray(150.0, dtype=jnp.float32)
    time_constant: jax.Array = jnp.asarray(0.0, dtype=jnp.float32)
    # Slip model (see wmr_simulator.slip for formulas and references):
    a_slip_max: jax.Array = jnp.asarray(0.0, dtype=jnp.float32) # traction limit ~ mu*g (m/s^2); 0 disables
    b_backlash: jax.Array = jnp.asarray(0.0, dtype=jnp.float32) # gearbox play half-width at wheel output (rad); 0 disables
    slip_sigma: jax.Array = jnp.asarray(0.0, dtype=jnp.float32) # stationary std of AR(1) wheel slip fraction
    slip_tau: jax.Array = jnp.asarray(0.0, dtype=jnp.float32)   # correlation time of AR(1) wheel slip (s)


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
    # [v_x, v_y, omega] body twist at the pose times. Pololu logs fill this with
    # the analytic spline derivative (pololu.pose_smoothing); consumers should
    # prefer it over finite-differencing `states`. None for simulated logs.
    twists: jax.Array | None = None


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
    print(f"  a_slip_max={float(params.a_slip_max):.3f} m/s^2 (traction limit, 0 = ideal)")
    print(f"  b_backlash={math.degrees(float(params.b_backlash)):.3f} deg (gear play half-width)")
    print(f"  slip_sigma={100.0 * float(params.slip_sigma):.3f} % (AR(1) slip std)")
    print(f"  slip_tau={float(params.slip_tau):.4f} s (AR(1) slip correlation time)")


def physical_params_to_array(params: PhysicalParams) -> jax.Array:
    return jnp.stack(
        [
            params.wheel_radius,
            params.base_diameter,
            params.max_wheel_speed,
            params.time_constant,
            params.a_slip_max,
            params.b_backlash,
            params.slip_sigma,
            params.slip_tau,
        ],
        axis=-1,
    )


def physical_params_from_array(values: jax.Array) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=values[..., 0],
        base_diameter=values[..., 1],
        max_wheel_speed=values[..., 2],
        time_constant=values[..., 3],
        a_slip_max=values[..., 4],
        b_backlash=values[..., 5],
        slip_sigma=values[..., 6],
        slip_tau=values[..., 7],
    )


def print_controller_gains(label: str, gains: jax.Array):
    print(label)
    print(f"  kx={float(gains[0]):.7f}")
    print(f"  ky={float(gains[1]):.7f}")
    print(f"  kth={float(gains[2]):.7f}")
    print(f"  kpmotor={float(gains[3]):.7f}")
    print(f"  kimotor={float(gains[4]):.7f}")
    print(f"  kdmotor={float(gains[5]):.7f}")


def clip_physical_params(params: PhysicalParams) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=jnp.clip(params.wheel_radius, min=1e-4),
        base_diameter=jnp.clip(params.base_diameter, min=1e-4),
        max_wheel_speed=jnp.clip(params.max_wheel_speed, min=1e-4),
        time_constant=jnp.clip(params.time_constant, min=1e-4),
        a_slip_max=jnp.clip(params.a_slip_max, min=0.0),
        b_backlash=jnp.clip(params.b_backlash, min=0.0),
        slip_sigma=jnp.clip(params.slip_sigma, min=0.0),
        slip_tau=jnp.clip(params.slip_tau, min=0.0),
    )


def physical_params_mse(params: PhysicalParams, target_params: PhysicalParams):
    squared_errors = jnp.stack(
        [
            (1000.0 * (params.wheel_radius - target_params.wheel_radius)) ** 2,
            (1000.0 * (params.base_diameter - target_params.base_diameter)) ** 2,
            (params.max_wheel_speed - target_params.max_wheel_speed) ** 2,
            (params.time_constant - target_params.time_constant) ** 2,
            (params.a_slip_max - target_params.a_slip_max) ** 2,          # m/s^2 scale
            (100.0 * (params.b_backlash - target_params.b_backlash)) ** 2,  # centirad scale
            (100.0 * (params.slip_sigma - target_params.slip_sigma)) ** 2,
            (params.slip_tau - target_params.slip_tau) ** 2,
        ],
        axis=-1,
    )
    return jnp.mean(squared_errors, axis=-1)
