"""Traction-limited longitudinal slip ("burnout"), parameter ``a_slip_max``.

The tire can transmit at most F_max ~ mu*m*g to the ground, so the
ground-contact speed of each wheel is a *rate-limited* follower of the
motor-side wheel speed::

    delta_max = (a_slip_max / r) * dt
    u_ground' = u_ground + soft_clip(u_motor - u_ground, delta_max)

The saturation is a smooth p-norm soft clip rather than a hard ``clip`` (see
:func:`soft_clip`): a hard clip's derivative with respect to ``a_slip_max`` is
exactly zero while unsaturated and jumps the instant it engages, which makes the
a_slip_max column of the trajectory-optimization FIM a sum over a *discrete* set
of saturated samples -- the FIM objective then becomes a staircase in the
control points (flat stretches separated by ~20% cliffs) and the optimizer
random-walks. It is also the more faithful tire model: the longitudinal
slip/tractive-effort curve saturates smoothly.

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


# Exponent of the p-norm soft clip. Higher is closer to a hard clip but stiffer;
# 4 keeps the unsaturated regime within ~1.5% of the identity (|x| = limit/2 maps
# to 0.985*x, where a tanh saturation would give 0.92*x) while staying smooth
# through the corner, which is the whole point. Must be an even *int*: see
# soft_clip.
SOFT_CLIP_SHARPNESS = 4


def soft_clip(x, limit, sharpness: int = SOFT_CLIP_SHARPNESS):
    """Smooth, symmetric saturation of ``x`` to +-``limit``.

    ``x / (1 + (x/limit)^p)^(1/p)`` for even ``p``: exact at ``x = 0``,
    asymptotic to ``+-limit``, and -- unlike ``clip`` -- differentiable in
    *both* arguments everywhere, with a ``d/d limit`` that is nonzero (not
    merely defined) below the limit. That derivative is what the FIM needs in
    order to see ``a_slip_max`` at all.

    ``sharpness`` must be an even integer, for two reasons that happen to
    coincide: an even power makes the ``abs`` unnecessary, and a Python int
    exponent routes through ``lax.integer_pow`` instead of ``lax.pow``. The
    latter matters -- ``lax.pow``'s JVP carries a ``log(base) * d exponent``
    term, and at zero slip that is ``-inf * 0 = NaN``, so a float exponent
    silently NaNs the whole Jacobian the moment a wheel is not slipping.
    The outer ``1/p`` exponent is safe as a float: its base is >= 1.

    The two algebraically identical branches exist to keep ``(.)^p`` away from
    overflow. ``|x| >> limit`` is not exotic here: the replay segment plan
    carries float-artifact steps as short as ``dt = 3.5e-18``, whose rate limit
    is ~1e-15 against a slip of order 1, and ``(1e15)^4`` is ``inf`` in float32.
    The primal survives that (``x / inf = 0``) but the tangent is ``inf / inf``,
    so a single such step NaNs the entire Jacobian. Each branch below raises
    only a quantity bounded by 1.
    """
    ratio = x / limit
    within_limit = jnp.abs(ratio) <= 1.0
    # Both operands are made safe unconditionally: an inf/NaN in the *untaken*
    # branch of a `where` still poisons the taken branch's derivative.
    bounded_ratio = jnp.where(within_limit, ratio, 1.0)
    bounded_inverse_ratio = jnp.where(within_limit, 1.0, ratio)
    return jnp.where(
        within_limit,
        x / (1.0 + bounded_ratio**sharpness) ** (1.0 / sharpness),
        limit * jnp.sign(ratio) / (1.0 + (1.0 / bounded_inverse_ratio) ** sharpness) ** (1.0 / sharpness),
    )


def traction_limited_ground_speeds(ground_speeds, motor_speeds, a_slip_max, wheel_radius, dt):
    """Rate-limit the ground-contact wheel speeds toward the motor-side speeds.

    The maximum ground-side wheel acceleration is a_slip_max / r (bounded tire
    force, Wong 2008 Ch. 1). ``a_slip_max = 0`` disables the limit: the ground
    speeds follow the motor speeds exactly (ideal traction).
    """
    delta = motor_speeds - ground_speeds
    max_step = a_slip_max / wheel_radius * dt
    # Two separate degenerate cases, and they want opposite answers: a_slip_max
    # of 0 disables the limit (follow the motor exactly), whereas a dt of 0 is a
    # zero-length step and must freeze the ground speeds. Both would divide by
    # zero inside soft_clip, and a division by zero in an *untaken* `where`
    # branch still poisons the taken branch's derivative, so the limit is made
    # safe before the call and the real answer selected afterwards.
    safe_max_step = jnp.where(max_step > 0.0, max_step, 1.0)
    limited = ground_speeds + jnp.where(
        max_step > 0.0, soft_clip(delta, safe_max_step), 0.0
    )
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
        delta = speeds[index] - limited[index - 1]
        if step > 0.0:
            # Same soft clip as :func:`soft_clip`, in its two overflow-safe
            # branches; scalar `if`s here rather than `where`, since this loop
            # is plain Python and carries no derivative.
            ratio = delta / step
            if abs(ratio) <= 1.0:
                delta = delta / (1.0 + ratio**SOFT_CLIP_SHARPNESS) ** (1.0 / SOFT_CLIP_SHARPNESS)
            else:
                delta = (
                    step
                    * np.sign(ratio)
                    / (1.0 + (1.0 / ratio) ** SOFT_CLIP_SHARPNESS) ** (1.0 / SOFT_CLIP_SHARPNESS)
                )
        limited[index] = limited[index - 1] + delta
    return limited
