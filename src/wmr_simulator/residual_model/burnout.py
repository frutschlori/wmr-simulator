"""Traction-limited longitudinal slip ("burnout"), parameter ``a_slip_max``.

The tire can transmit at most F_max ~ mu*m*g to the ground, so the
ground-contact speed of each wheel is a *rate-limited* follower of the
motor-side wheel speed::

    delta_max = (a_slip_max / r) * dt
    u_ground' = u_ground + clip(u_motor - u_ground, -delta_max, +delta_max)

Full throttle from standstill makes the motor-side speed jump while the
ground speed ramps at a_slip_max ~ mu*g -- exactly the observed burnout.
This is the kinematic abstraction of the bounded tire longitudinal force
(J. Y. Wong, "Theory of Ground Vehicles", 4th ed., Wiley 2008, Ch. 1,
longitudinal slip vs. tractive effort). ``a_slip_max = 0`` disables the
limit (ideal traction).

This is the only structured slip element kept alongside the learned residual
model: without it, open-loop integration over a full trajectory drifts
noticeably, and the burnout transient is too sharp for the MLP residual to
absorb. a_slip_max is deterministic and gradient-identified alongside r, L,
u_max, tau (log-space optimizer: a zero init keeps it disabled since
0 * exp(theta) = 0 with zero gradient); it needs aggressive acceleration
segments (full-throttle steps) for excitation.
"""

import jax.numpy as jnp
import numpy as np


def traction_limited_ground_speeds(ground_speeds, motor_speeds, a_slip_max, wheel_radius, dt):
    """Rate-limit the ground-contact wheel speeds toward the motor-side speeds.

    The maximum ground-side wheel acceleration is a_slip_max / r (bounded tire
    force, Wong 2008 Ch. 1). ``a_slip_max = 0`` disables the limit: the ground
    speeds follow the motor speeds exactly (ideal traction).
    """
    delta = motor_speeds - ground_speeds
    max_step = a_slip_max / wheel_radius * dt
    limited = ground_speeds + jnp.clip(delta, -max_step, max_step)
    return jnp.where(a_slip_max > 0.0, limited, motor_speeds)


def rate_limited_series(speeds: np.ndarray, dt: np.ndarray, max_rate: float) -> np.ndarray:
    """Sequential rate limiter (traction limit) over an unevenly sampled series.

    Numpy counterpart of :func:`traction_limited_ground_speeds` for dataset
    construction from logs. ``max_rate`` is the maximum wheel angular
    acceleration a_slip_max / r; 0 disables.
    """
    if max_rate <= 0.0 or len(speeds) == 0:
        return speeds
    limited = np.empty_like(speeds)
    limited[0] = speeds[0]
    for index in range(1, len(speeds)):
        step = max_rate * max(float(dt[index - 1]), 0.0)
        delta = np.clip(speeds[index] - limited[index - 1], -step, step)
        limited[index] = limited[index - 1] + delta
    return limited
