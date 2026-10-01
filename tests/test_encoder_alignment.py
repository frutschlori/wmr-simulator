import numpy as np

from wmr_simulator.pololu.measurement_smoothing import (
    estimate_encoder_lag,
    invert_encoder_lowpass,
    shift_series,
)


def firmware_lowpass(time_s, speeds, cutoff_hz=3.0):
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    filtered = speeds.copy()
    for k in range(1, len(time_s)):
        alpha = (time_s[k] - time_s[k - 1]) / (tau + time_s[k] - time_s[k - 1])
        filtered[k] = filtered[k - 1] + alpha * (speeds[k] - filtered[k - 1])
    return filtered


def test_inversion_undoes_the_firmware_lowpass_with_jittered_timestamps():
    rng = np.random.default_rng(0)
    time_s = np.cumsum(0.01 + 0.002 * rng.random(400))
    speeds = np.column_stack([np.sin(7 * time_s), np.cos(3 * time_s)]) * 50.0
    recovered = invert_encoder_lowpass(time_s, firmware_lowpass(time_s, speeds))
    np.testing.assert_allclose(recovered, speeds, atol=1e-9)


def test_lag_estimate_recovers_a_known_shift_and_skips_slipping_samples():
    time_s = np.arange(0.0, 8.0, 0.01)
    omega = 2.0 * np.sin(1.3 * time_s) + np.sin(3.1 * time_s)
    v = 0.6 + 0.3 * np.sin(0.7 * time_s)
    twists = np.column_stack([v, np.zeros_like(v), omega])
    # Encoders lag the mocap by 12 ms: the wheel speeds at t describe t - 0.012.
    lagged_omega = np.interp(time_s - 0.012, time_s, omega)
    lagged_v = np.interp(time_s - 0.012, time_s, v)
    right = (lagged_v + 0.04 * lagged_omega) / 0.016
    left = (lagged_v - 0.04 * lagged_omega) / 0.016
    speeds = np.column_stack([right, left])
    lag = estimate_encoder_lag(time_s, speeds, time_s, twists)
    assert abs(lag.lag_s - 0.012) < 1e-3

    # A slipping stretch (wheels spin, body does not follow) at high lateral
    # acceleration must not pull the estimate.
    slipping = (time_s > 5.0) & (time_s < 6.5)
    twists_slip = twists.copy()
    twists_slip[slipping, 0] = 2.5
    twists_slip[slipping, 2] = 3.0
    lag_slip = estimate_encoder_lag(time_s, speeds, time_s, twists_slip)
    assert abs(lag_slip.lag_s - 0.012) < 1e-3
    np.testing.assert_allclose(shift_series(time_s, speeds, lag.lag_s)[200:600], np.column_stack([
        (v + 0.04 * omega) / 0.016, (v - 0.04 * omega) / 0.016])[200:600], atol=0.2)
