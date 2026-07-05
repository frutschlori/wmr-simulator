"""Physically-inspired slip model for differential-drive robots.

Replaces the previous white multiplicative wheel-speed noise with components
that have literature backing and identifiable parameters:

1. Effective wheelbase (no separate parameter). The configured/identified
   ``base_diameter`` is the *effective* wheelbase, not the geometric one: in
   turns both tires scrub azimuthally, which acts as a constant multiplicative
   wheelbase correction (the E_b error of the UMBmark method; J. Borenstein and
   L. Feng, "Measurement and Correction of Systematic Odometry Errors in Mobile
   Robots", IEEE Trans. Robotics and Automation, 12(6), 1996). Because the
   correction and the geometric wheelbase only ever appear as a product, only
   the effective wheelbase is identifiable -- so that is what ``base_diameter``
   stores, and system identification fits it directly from data.

2. Traction-limited longitudinal slip ("burnout"), parameter ``a_slip_max``
   (m/s^2). The tire can transmit at most F_max ~ mu*m*g to the ground, so
   the ground-contact speed of each wheel is a *rate-limited* follower of the
   motor-side wheel speed::

       delta_max = (a_slip_max / r) * dt
       u_ground' = u_ground + clip(u_motor - u_ground, -delta_max, +delta_max)

   Full throttle from standstill makes the motor-side speed jump while the
   ground speed ramps at a_slip_max ~ mu*g -- exactly the observed burnout.
   This is the kinematic abstraction of the bounded tire longitudinal force
   (see J. Y. Wong, "Theory of Ground Vehicles", 4th ed., Wiley 2008, Ch. 1,
   longitudinal slip vs. tractive effort). ``a_slip_max = 0`` disables the
   limit (ideal traction).

3. Gearbox backlash, parameter ``b_backlash`` (rad, half-width of the play at
   the wheel output). The encoders sit on the motor shaft *before* the
   gearbox, so on a direction reversal the encoder sign flips immediately
   while the motor tooth first traverses the gap (2*b) before the wheel
   moves. Classic kinematic backlash element (M. Nordin and P.-O. Gutman,
   "Controlling mechanical systems with backlash -- a survey", Automatica
   38(10), 2002). State delta in [-b, +b] is the tooth position in the gap::

       delta'  = clip(delta + u_motor * dt, -b, +b)
       u_wheel = u_motor - (delta' - delta) / dt

   Engaged and pushing: u_wheel = u_motor (no effect). During reversals and
   low-speed dithering the wheel holds still while the encoder oscillates --
   which is why the measured wheel-speed difference can flip sign while the
   body keeps turning the same way. ``b_backlash = 0`` disables.

4. Correlated stochastic wheel slip, first-order Gauss-Markov (AR(1)/OU)
   per wheel, applied multiplicatively::

       rho    = exp(-dt / tau_slip)
       eta'   = rho * eta + sigma * sqrt(1 - rho^2) * eps,   eps ~ N(0, 1)
       u_eff  = u * (1 - eta)

   The exponentially-correlated first-order Gauss-Markov process is the
   standard model for non-white disturbance in navigation error modeling:
   P. S. Maybeck, "Stochastic Models, Estimation, and Control", Vol. 1,
   Academic Press, 1979 (Sec. 4.11). The stationary standard deviation is
   ``sigma`` and the autocorrelation time is ``tau_slip``.

Identification notes
--------------------
a_slip_max and b_backlash are deterministic parameters: they enter the
replay rollout and are identified by gradient descent alongside r, L
(effective), tau, u_max. a_slip_max needs aggressive acceleration segments
(full-throttle steps); b_backlash needs direction reversals (end-of-run
dithering is ideal excitation).

sigma and tau_slip are *noise* parameters: a deterministic replay loss has
zero sensitivity to them, so they must not be gradient-identified or included
in the FIM. Fit them from residual statistics with :func:`fit_ar1_moments`
(see wmr_simulator.identification.slip_noise).
"""

import jax
import jax.numpy as jnp


def slip_body_velocities(wheel_speeds, wheel_radius, base_diameter):
    """Body velocities (v, omega) from ground-contact wheel speeds.

    ``base_diameter`` is the *effective* wheelbase (see module docstring).
    This is ideal differential-drive kinematics (no lateral body velocity: a
    true two-wheel differential drive has no chassis side-slip).
    """
    ur, ul = wheel_speeds[0], wheel_speeds[1]
    v = 0.5 * wheel_radius * (ur + ul)
    omega = wheel_radius * (ur - ul) / base_diameter
    return v, omega


def integrate_planar_pose(pose, v, omega, dt):
    """Euler-integrate a planar pose from forward and angular body velocity."""
    x, y, theta = pose
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    return jnp.array(
        [
            x + v * cos_t * dt,
            y + v * sin_t * dt,
            _wrap_to_pi(theta + omega * dt),
        ]
    )


def integrate_planar_pose_lateral(pose, v_x, v_y, omega, dt):
    """Euler-integrate a planar pose from a full body twist (v_x, v_y, omega).

    A learned residual can introduce a lateral body velocity v_y (chassis
    side-slip) that the ideal differential-drive kinematics exclude; with
    v_y = 0 this reduces exactly to :func:`integrate_planar_pose`.
    """
    x, y, theta = pose
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    return jnp.array(
        [
            x + (v_x * cos_t - v_y * sin_t) * dt,
            y + (v_x * sin_t + v_y * cos_t) * dt,
            _wrap_to_pi(theta + omega * dt),
        ]
    )


def backlash_transmission(gap_offset, motor_speeds, b_backlash, dt):
    """Kinematic gearbox backlash element (Nordin & Gutman 2002).

    ``gap_offset`` (delta, per wheel) is the motor-tooth position inside the play
    band [-b, +b]. Motor motion absorbed by the gap does not reach the wheel:

        delta'  = clip(delta + u_motor * dt, -b, +b)
        u_wheel = u_motor - (delta' - delta) / dt

    Returns (new_gap_offset, wheel_side_speeds). ``b_backlash = 0`` disables.
    """
    next_offset = jnp.clip(gap_offset + motor_speeds * dt, -b_backlash, b_backlash)
    wheel_speeds = motor_speeds - (next_offset - gap_offset) / dt
    enabled = b_backlash > 0.0
    return (
        jnp.where(enabled, next_offset, jnp.zeros_like(gap_offset)),
        jnp.where(enabled, wheel_speeds, motor_speeds),
    )


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


def ar1_slip_update(slip_noise, key, sigma, tau_slip, dt):
    """One step of the stationary first-order Gauss-Markov slip process.

    ``slip_noise`` holds the per-wheel slip fractions eta = (eta_r, eta_l).
    Returns the updated slip fractions. With sigma = 0 the state decays to
    zero and the model is deterministic.
    """
    safe_tau = jnp.maximum(tau_slip, 1e-6)
    rho = jnp.where(tau_slip > 1e-6, jnp.exp(-dt / safe_tau), 0.0)
    innovation_std = sigma * jnp.sqrt(jnp.maximum(1.0 - rho**2, 0.0))
    eps = jax.random.normal(key, shape=slip_noise.shape)
    return rho * slip_noise + innovation_std * eps


def apply_wheel_slip(wheel_speeds, slip_noise):
    """Multiplicative slip: effective ground speed u_eff = u * (1 - eta)."""
    return wheel_speeds * (1.0 - slip_noise)


def fit_ar1_moments(residuals, dt):
    """Moment-match AR(1) parameters (sigma, tau) from a residual time series.

    ``residuals`` is a 1-D array sampled at fixed interval ``dt`` (e.g. the
    fractional wheel-speed residual after the deterministic identification
    fit). Uses the stationary std and the lag-1 autocorrelation:

        rho_1 = acf(1)  ->  tau = -dt / ln(rho_1),   sigma = std(residuals)

    This is the standard way to identify noise parameters that a
    deterministic replay loss cannot see (Maybeck 1979, Sec. 4.11).
    Returns (sigma, tau); tau is 0 when the lag-1 autocorrelation is
    non-positive (white residual).
    """
    residuals = jnp.asarray(residuals, dtype=jnp.float32).reshape(-1)
    centered = residuals - jnp.mean(residuals)
    variance = jnp.mean(centered**2)
    sigma = jnp.sqrt(variance)
    lag1 = jnp.mean(centered[:-1] * centered[1:]) / jnp.maximum(variance, 1e-12)
    lag1 = jnp.clip(lag1, -0.999, 0.999)
    tau = jnp.where(lag1 > 0.0, -dt / jnp.log(jnp.maximum(lag1, 1e-6)), 0.0)
    return sigma, tau


def _wrap_to_pi(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
