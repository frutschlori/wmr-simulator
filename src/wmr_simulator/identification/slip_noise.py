"""Estimate the AR(1) wheel-slip noise parameters (sigma, tau) from experiment logs.

The deterministic replay loss has zero sensitivity to noise parameters, so
slip_sigma / slip_tau cannot be gradient-identified. Instead they are fitted from
the *fractional wheel slip residual*, which is exactly the quantity the model
defines as slip (u_eff = u_enc * (1 - eta)):

1. Differentiate the (mocap) poses and project the displacement into the body
   frame -> measured body velocities v_meas, omega_meas.
2. Invert the identified kinematics (r, effective wheelbase L) to get the
   ground-side wheel speeds implied by the measured body motion:

       u_r_eff = (2*v + omega*L) / (2*r)
       u_l_eff = (2*v - omega*L) / (2*r)

3. Compare with the *deterministic model prediction* of the ground speeds: the
   encoder-side speeds passed through the identified traction limit
   (a_slip_max), so the burnout transient does not contaminate the noise fit.
   eta_i = 1 - u_eff_i / u_pred_i (masked where the predicted speed is too
   small for the ratio to be meaningful).
4. Moment-match the AR(1) parameters on the eta series
   (wmr_simulator.slip.fit_ar1_moments; first-order Gauss-Markov, Maybeck 1979).

Mocap differentiation amplifies measurement noise, which inflates sigma_hat and
biases tau_hat low; apply the zero-phase mocap filter in the log loader first
(pololu.log_loader, mocap_filter_window_s) to mitigate this.
"""

import numpy as np

from wmr_simulator.slip import fit_ar1_moments
from wmr_simulator.types import PhysicalParams, SimulationLog


def slip_residual_series(
    pose_time_s: np.ndarray,
    pose_states: np.ndarray,
    wheel_time_s: np.ndarray,
    wheel_speeds: np.ndarray,
    params: PhysicalParams,
    min_wheel_speed: float = 5.0,
) -> dict:
    """Fractional wheel slip residuals eta_r, eta_l from pose and encoder streams.

    ``pose_states`` are [x, y, theta] (mocap), ``wheel_speeds`` are the encoder
    wheel speeds [u_r, u_l]. Returns the per-wheel residual series, the validity
    mask (encoder speed above ``min_wheel_speed`` rad/s) and the median sample dt.
    """
    pose_time_s = np.asarray(pose_time_s, dtype=float)
    pose_states = np.asarray(pose_states, dtype=float)
    wheel_time_s = np.asarray(wheel_time_s, dtype=float)
    wheel_speeds = np.asarray(wheel_speeds, dtype=float)

    dt = np.diff(pose_time_s)
    valid_dt = dt > 1e-6
    dx = np.diff(pose_states[:, 0])
    dy = np.diff(pose_states[:, 1])
    dyaw = _wrap_to_pi(np.diff(pose_states[:, 2]))
    theta = pose_states[:-1, 2]  # interval start, consistent with Euler integration

    safe_dt = np.maximum(dt, 1e-9)
    # Body-frame projection: forward displacement -> v, heading change -> omega.
    v_meas = (dx * np.cos(theta) + dy * np.sin(theta)) / safe_dt
    omega_meas = dyaw / safe_dt

    r = float(params.wheel_radius)
    effective_wheelbase = float(params.base_diameter)
    u_r_eff = (2.0 * v_meas + omega_meas * effective_wheelbase) / (2.0 * r)
    u_l_eff = (2.0 * v_meas - omega_meas * effective_wheelbase) / (2.0 * r)

    interval_time = pose_time_s[:-1]
    u_r_enc = np.interp(interval_time, wheel_time_s, wheel_speeds[:, 0])
    u_l_enc = np.interp(interval_time, wheel_time_s, wheel_speeds[:, 1])
    # Deterministic model prediction: pass the encoder speeds through the identified
    # backlash and traction limit so those transients do not enter the noise residual.
    b_backlash = float(params.b_backlash)
    u_r_pred = _rate_limited(_backlash(u_r_enc, dt, b_backlash), dt, float(params.a_slip_max) / r)
    u_l_pred = _rate_limited(_backlash(u_l_enc, dt, b_backlash), dt, float(params.a_slip_max) / r)

    mask_r = valid_dt & (np.abs(u_r_pred) > min_wheel_speed)
    mask_l = valid_dt & (np.abs(u_l_pred) > min_wheel_speed)
    eta_r = np.where(mask_r, 1.0 - u_r_eff / np.where(mask_r, u_r_pred, 1.0), 0.0)
    eta_l = np.where(mask_l, 1.0 - u_l_eff / np.where(mask_l, u_l_pred, 1.0), 0.0)

    return {
        "time_s": interval_time,
        "eta_r": eta_r,
        "eta_l": eta_l,
        "mask_r": mask_r,
        "mask_l": mask_l,
        "median_dt": float(np.median(dt[valid_dt])) if np.any(valid_dt) else 0.0,
    }


def estimate_slip_noise(
    pose_time_s: np.ndarray,
    pose_states: np.ndarray,
    wheel_time_s: np.ndarray,
    wheel_speeds: np.ndarray,
    params: PhysicalParams,
    min_wheel_speed: float = 5.0,
) -> dict:
    """Fit AR(1) slip noise (sigma, tau) per wheel and averaged.

    Note: the AR(1) moments are computed on the masked series, which treats the
    samples as contiguous; long standstill gaps slightly bias tau_hat. Use logs
    with sustained motion for the fit.
    """
    residuals = slip_residual_series(
        pose_time_s, pose_states, wheel_time_s, wheel_speeds, params, min_wheel_speed
    )
    results = {}
    sigmas, taus = [], []
    for wheel in ("r", "l"):
        series = residuals[f"eta_{wheel}"][residuals[f"mask_{wheel}"]]
        if len(series) < 10:
            results[f"sigma_{wheel}"] = float("nan")
            results[f"tau_{wheel}"] = float("nan")
            continue
        sigma, tau = fit_ar1_moments(series, residuals["median_dt"])
        results[f"sigma_{wheel}"] = float(sigma)
        results[f"tau_{wheel}"] = float(tau)
        sigmas.append(float(sigma))
        taus.append(float(tau))
    results["sigma"] = float(np.mean(sigmas)) if sigmas else float("nan")
    results["tau"] = float(np.mean(taus)) if taus else float("nan")
    results["num_samples_r"] = int(np.sum(residuals["mask_r"]))
    results["num_samples_l"] = int(np.sum(residuals["mask_l"]))
    results["residuals"] = residuals
    return results


def estimate_slip_noise_from_log(
    target_log: SimulationLog,
    params: PhysicalParams,
    min_wheel_speed: float = 5.0,
) -> dict:
    """Convenience wrapper: fit AR(1) slip noise from a SimulationLog (e.g. a Pololu log)."""
    return estimate_slip_noise(
        np.asarray(target_log.pose.time_s),
        np.asarray(target_log.pose.states),
        np.asarray(target_log.wheel.time_s),
        np.asarray(target_log.wheel.speeds),
        params,
        min_wheel_speed=min_wheel_speed,
    )


def _backlash(speeds: np.ndarray, dt: np.ndarray, b_backlash: float) -> np.ndarray:
    """Sequential kinematic backlash element (Nordin & Gutman 2002) over the series.

    Assumes the gear starts engaged in the direction of the first sample; motion
    absorbed by the play band never reaches the output. ``b_backlash = 0`` disables.
    """
    if b_backlash <= 0.0 or len(speeds) == 0:
        return speeds
    output = np.empty_like(speeds)
    offset = b_backlash * np.sign(speeds[0])
    output[0] = speeds[0]
    for index in range(1, len(speeds)):
        step_dt = max(float(dt[index - 1]), 1e-9)
        next_offset = np.clip(offset + speeds[index] * step_dt, -b_backlash, b_backlash)
        output[index] = speeds[index] - (next_offset - offset) / step_dt
        offset = next_offset
    return output


def _rate_limited(speeds: np.ndarray, dt: np.ndarray, max_rate: float) -> np.ndarray:
    """Sequential rate limiter (traction limit) over an unevenly sampled series.

    ``max_rate`` is the maximum wheel angular acceleration a_slip_max / r; 0 disables.
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


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
