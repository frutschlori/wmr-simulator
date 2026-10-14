"""Smoothing of the Pololu measurement streams (mocap poses, encoder speeds).

``smooth_pose_stream`` cleans the irregular mocap stream (repeated frames,
samples closer than ``min_dt``, residual outliers), resamples x, y and the
unwrapped yaw onto a uniform median-dt grid and runs a centered
Savitzky-Golay filter over each channel. Velocities come from the filter's
closed-form derivative; the smoothed grids are interpolated back to any time.

``smooth_and_align_encoder_speeds`` inverts the firmware's wheel-speed
low-pass, smooths the result (zero phase) and removes the remaining lag
against the mocap twist (``estimate_encoder_lag``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.signal import savgol_filter

__all__ = [
    "SmoothedPoseStream",
    "drop_repeated_poses",
    "reject_short_intervals",
    "reject_residual_outliers",
    "smooth_pose_stream",
    "savgol_smooth_series",
    "smooth_and_align_encoder_speeds",
    "DEFAULT_SAVGOL_WINDOW",
    "DEFAULT_SAVGOL_POLYORDER",
    "DEFAULT_MIN_DT",
    "DEFAULT_OUTLIER_SIGMA",
    "DEFAULT_ENCODER_LP_CUTOFF_HZ",
    "DEFAULT_ENCODER_LAG_SEARCH_S",
    "DEFAULT_ENCODER_LAG_MAX_LATERAL_ACCELERATION",
    "EncoderLag",
    "estimate_encoder_lag",
    "invert_encoder_lowpass",
    "shift_series",
    "DEFAULT_ENCODER_SAVGOL_WINDOW",
    "DEFAULT_ENCODER_SAVGOL_POLYORDER",
]

# Window length (in samples) and polynomial order of the Savitzky-Golay filter.
DEFAULT_SAVGOL_WINDOW = 35
DEFAULT_SAVGOL_POLYORDER = 3

# Outlier rejection (0 disables each): minimum sample spacing [s], and the
# robust sigmas of the residual from a preliminary fit.
DEFAULT_MIN_DT = 5e-3
DEFAULT_OUTLIER_SIGMA = 15.0

# Cutoff of the firmware's one-pole wheel-speed low-pass, which the loader
# inverts (y_k = y_{k-1} + alpha (x_k - y_{k-1}), alpha = dt / (tau + dt)).
# 0 disables the inversion.
DEFAULT_ENCODER_LP_CUTOFF_HZ = 3.0
# Search range of the per-log encoder lag against the mocap twist [s].
DEFAULT_ENCODER_LAG_SEARCH_S = (-0.08, 0.12)
# The lag is measured only where |v * omega| stays below this [m/s^2] (no slip).
DEFAULT_ENCODER_LAG_MAX_LATERAL_ACCELERATION = 3.0
# Zero-phase smoothing of the inverted speeds (against count quantization).
DEFAULT_ENCODER_SAVGOL_WINDOW = 21
DEFAULT_ENCODER_SAVGOL_POLYORDER = 3


def drop_repeated_poses(
    time_s: np.ndarray,
    pose_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples equal to the previous kept one (the same mocap frame
    logged twice) and samples whose timestamp does not strictly increase."""
    time_s = np.asarray(time_s, dtype=float)
    pose_states = np.asarray(pose_states, dtype=float)
    if len(time_s) == 0:
        return time_s, pose_states

    keep = np.ones(len(time_s), dtype=bool)
    last_kept = 0
    for index in range(1, len(time_s)):
        repeated = np.all(pose_states[index] == pose_states[last_kept])
        if repeated or time_s[index] <= time_s[last_kept]:
            keep[index] = False
            continue
        last_kept = index
    return time_s[keep], pose_states[keep]


def reject_short_intervals(
    time_s: np.ndarray,
    pose_states: np.ndarray,
    min_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop samples logged closer than ``min_dt`` to the previous kept one
    (``min_dt <= 0`` disables the filter)."""
    time_s = np.asarray(time_s, dtype=float)
    pose_states = np.asarray(pose_states, dtype=float)
    if min_dt <= 0.0 or len(time_s) == 0:
        return time_s, pose_states

    keep = np.ones(len(time_s), dtype=bool)
    last_kept = 0
    for index in range(1, len(time_s)):
        if time_s[index] - time_s[last_kept] < min_dt:
            keep[index] = False
            continue
        last_kept = index
    return time_s[keep], pose_states[keep]


def reject_residual_outliers(
    time_s: np.ndarray,
    pose_states: np.ndarray,
    sigma: float,
    *,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop poses whose residual from a preliminary Savitzky-Golay fit exceeds
    ``sigma`` robust sigmas (modified z-score ``|r - median| / (1.4826 MAD)``,
    per channel, combined as a max). ``sigma <= 0`` disables the filter.
    """
    time_s = np.asarray(time_s, dtype=float)
    pose_states = np.asarray(pose_states, dtype=float)
    # Need a few samples beyond the polynomial order to fit a meaningful residual.
    if sigma <= 0.0 or len(time_s) <= polyorder + 1:
        return time_s, pose_states

    grid_time, grid_pose, grid_velocity = _fit_grid(time_s, pose_states, window_length, polyorder)
    smoothed = SmoothedPoseStream(grid_time, grid_pose, grid_velocity, time_s, pose_states).pose(time_s)
    residual = np.column_stack(
        [
            pose_states[:, 0] - smoothed[:, 0],
            pose_states[:, 1] - smoothed[:, 1],
            (pose_states[:, 2] - smoothed[:, 2] + np.pi) % (2.0 * np.pi) - np.pi,
        ]
    )
    median = np.median(residual, axis=0)
    # Robust scale (MAD -> std), floored at the measurement resolution.
    scale = np.maximum(1.4826 * np.median(np.abs(residual - median), axis=0), 1e-3)
    score = np.max(np.abs(residual - median) / scale, axis=1)

    keep = score <= sigma
    return time_s[keep], pose_states[keep]


@dataclass(frozen=True)
class SmoothedPoseStream:
    """Savitzky-Golay smoothed pose stream on a uniform time grid; the query
    methods interpolate (clamped at the ends) onto arbitrary times."""

    grid_time: np.ndarray
    grid_pose: np.ndarray  # (N, 3): x, y, unwrapped yaw
    grid_velocity: np.ndarray  # (N, 3): x_dot, y_dot, yaw_dot
    clean_time: np.ndarray  # raw sample times surviving the rejection steps
    clean_pose: np.ndarray  # (M, 3) raw (unsmoothed) poses at clean_time

    @property
    def t_start(self) -> float:
        return float(self.grid_time[0])

    @property
    def t_end(self) -> float:
        return float(self.grid_time[-1])

    def _interp(self, time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
        # np.interp clamps to the endpoint values outside the grid range.
        t = np.asarray(time_s, dtype=float)
        return np.stack(
            [np.interp(t, self.grid_time, values[:, i]) for i in range(values.shape[1])],
            axis=-1,
        )

    def pose(self, time_s: np.ndarray) -> np.ndarray:
        """Smoothed poses (..., 3) with yaw wrapped to [-pi, pi)."""
        pose = self._interp(time_s, self.grid_pose)
        pose[..., 2] = (pose[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
        return pose

    def world_velocity(self, time_s: np.ndarray) -> np.ndarray:
        """Filter derivative (..., 3): x_dot, y_dot, yaw_dot."""
        return self._interp(time_s, self.grid_velocity)

    def body_twist(self, time_s: np.ndarray) -> np.ndarray:
        """World velocity rotated into the body frame (..., 3): v_x, v_y, omega."""
        velocity = self.world_velocity(time_s)
        yaw = self._interp(time_s, self.grid_pose[:, 2:3])[..., 0]
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        v_x = velocity[..., 0] * cos_t + velocity[..., 1] * sin_t
        v_y = -velocity[..., 0] * sin_t + velocity[..., 1] * cos_t
        return np.stack([v_x, v_y, velocity[..., 2]], axis=-1)


def smooth_pose_stream(
    time_s: np.ndarray,
    pose_states: np.ndarray,
    *,
    window_length: int = DEFAULT_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    min_dt: float = DEFAULT_MIN_DT,
    outlier_sigma: float = DEFAULT_OUTLIER_SIGMA,
    drop_duplicates: bool = True,
) -> SmoothedPoseStream:
    """Smooth a mocap pose stream with a centered Savitzky-Golay filter.

    The stream is cleaned first (``drop_duplicates``, ``min_dt``,
    ``outlier_sigma``; each disabled when 0), then resampled onto a uniform
    median-dt grid. ``window_length`` (odd, in grid samples) shrinks when the
    stream is too short.
    """
    if polyorder < 1:
        raise ValueError(f"polyorder must be >= 1 for velocities, got {polyorder}.")
    if drop_duplicates:
        time_s, pose_states = drop_repeated_poses(time_s, pose_states)
    else:
        time_s = np.asarray(time_s, dtype=float)
        pose_states = np.asarray(pose_states, dtype=float)
    time_s, pose_states = reject_short_intervals(time_s, pose_states, min_dt)
    if len(time_s) <= polyorder:
        raise ValueError(f"Need more than {polyorder} samples to fit polyorder {polyorder}, got {len(time_s)}.")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("Pose timestamps must be strictly increasing (use drop_duplicates=True).")
    time_s, pose_states = reject_residual_outliers(
        time_s, pose_states, outlier_sigma, window_length=window_length, polyorder=polyorder
    )
    if len(time_s) <= polyorder:
        raise ValueError(f"Outlier rejection left too few samples ({len(time_s)}) for polyorder {polyorder}.")

    grid_time, grid_pose, grid_velocity = _fit_grid(time_s, pose_states, window_length, polyorder)
    return SmoothedPoseStream(
        grid_time=grid_time,
        grid_pose=grid_pose,
        grid_velocity=grid_velocity,
        clean_time=time_s,
        clean_pose=pose_states,
    )


def _fit_grid(
    time_s: np.ndarray,
    pose_states: np.ndarray,
    window_length: int,
    polyorder: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample poses onto a uniform grid and Savitzky-Golay filter them
    (edge mode ``nearest``); returns (grid_time, grid_pose, grid_velocity)."""
    median_dt = float(np.median(np.diff(time_s)))
    num_intervals = max(int(np.round((time_s[-1] - time_s[0]) / median_dt)), polyorder + 1)
    grid_time = np.linspace(time_s[0], time_s[-1], num_intervals + 1)
    grid_dt = grid_time[1] - grid_time[0]

    yaw_unwrapped = np.unwrap(pose_states[:, 2])
    grid_signals = np.column_stack(
        [
            np.interp(grid_time, time_s, pose_states[:, 0]),
            np.interp(grid_time, time_s, pose_states[:, 1]),
            np.interp(grid_time, time_s, yaw_unwrapped),
        ]
    )

    window = _valid_window(window_length, polyorder, len(grid_time))
    grid_pose = savgol_filter(grid_signals, window, polyorder, deriv=0, axis=0, mode="nearest")
    grid_velocity = savgol_filter(grid_signals, window, polyorder, deriv=1, delta=grid_dt, axis=0, mode="nearest")
    return grid_time, grid_pose, grid_velocity


def savgol_smooth_series(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    """Savitzky-Golay smooth a jittered series ``(N,)`` or ``(N, K)`` on a
    uniform median-dt grid and interpolate back to ``time_s`` (unchanged when
    there are too few samples)."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    single_column = values.ndim == 1
    matrix = values[:, None] if single_column else values
    if len(time_s) <= polyorder + 1:
        return values

    median_dt = float(np.median(np.diff(time_s)))
    num_intervals = max(int(np.round((time_s[-1] - time_s[0]) / median_dt)), polyorder + 1)
    grid_time = np.linspace(time_s[0], time_s[-1], num_intervals + 1)
    grid_values = np.column_stack([np.interp(grid_time, time_s, matrix[:, i]) for i in range(matrix.shape[1])])

    window = _valid_window(window_length, polyorder, len(grid_time))
    grid_smoothed = savgol_filter(grid_values, window, polyorder, deriv=0, axis=0, mode="nearest")
    smoothed = np.column_stack(
        [np.interp(time_s, grid_time, grid_smoothed[:, i]) for i in range(grid_smoothed.shape[1])]
    )
    return smoothed[:, 0] if single_column else smoothed


def invert_encoder_lowpass(
    time_s: np.ndarray, speeds: np.ndarray, cutoff_hz: float = DEFAULT_ENCODER_LP_CUTOFF_HZ
) -> np.ndarray:
    """Undo the firmware's one-pole wheel-speed low-pass:
    ``x_k = y_{k-1} + (y_k - y_{k-1}) / alpha_k`` with ``alpha_k = dt_k / (tau + dt_k)``.
    ``cutoff_hz <= 0`` returns the series unchanged.
    """
    time_s = np.asarray(time_s, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    if cutoff_hz <= 0.0 or len(time_s) < 2:
        return speeds
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = np.maximum(np.diff(time_s), 1e-6)
    alpha = dt / (tau + dt)
    if speeds.ndim > 1:
        alpha = alpha[:, None]
    unfiltered = speeds.copy()
    unfiltered[1:] = speeds[:-1] + (speeds[1:] - speeds[:-1]) / alpha
    return unfiltered


def shift_series(time_s: np.ndarray, values: np.ndarray, lag_s: float) -> np.ndarray:
    """``values`` read ``lag_s`` later: the result at t is the input at t + lag_s
    (linear interpolation, ends held). A positive lag advances a delayed series."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if lag_s == 0.0 or len(time_s) <= 1:
        return values
    single_column = values.ndim == 1
    matrix = values[:, None] if single_column else values
    shifted = np.column_stack([np.interp(time_s + lag_s, time_s, matrix[:, i]) for i in range(matrix.shape[1])])
    return shifted[:, 0] if single_column else shifted


class EncoderLag(NamedTuple):
    """Lag of the encoder twist behind the mocap twist [s], and the peak
    correlations of the yaw-rate and forward-speed channels at it."""

    lag_s: float
    yaw_correlation: float
    speed_correlation: float


def estimate_encoder_lag(
    wheel_time_s: np.ndarray,
    wheel_speeds: np.ndarray,
    pose_time_s: np.ndarray,
    pose_twists: np.ndarray,
    search_s: tuple[float, float] = DEFAULT_ENCODER_LAG_SEARCH_S,
    resolution_s: float = 1e-3,
    edge_s: float = 0.3,
    max_lateral_acceleration: float = DEFAULT_ENCODER_LAG_MAX_LATERAL_ACCELERATION,
) -> EncoderLag | None:
    """The shift that best aligns the encoder twist with the mocap twist.

    Maximizes the summed Pearson correlations of ``omega_r - omega_l`` with the
    mocap yaw rate and ``omega_r + omega_l`` with the forward speed (no
    dependence on wheel radius or wheelbase), using only samples below
    ``max_lateral_acceleration``. None when too few samples remain.
    """
    wheel_time_s = np.asarray(wheel_time_s, dtype=float)
    wheel_speeds = np.asarray(wheel_speeds, dtype=float)
    pose_time_s = np.asarray(pose_time_s, dtype=float)
    pose_twists = np.asarray(pose_twists, dtype=float)
    start = max(wheel_time_s[0], pose_time_s[0]) + edge_s
    stop = min(wheel_time_s[-1], pose_time_s[-1]) - edge_s
    grid = np.arange(start, stop, 0.005)
    if len(grid) < 50:
        return None
    lateral = np.abs(np.interp(grid, pose_time_s, pose_twists[:, 0]) * np.interp(grid, pose_time_s, pose_twists[:, -1]))
    grid = grid[lateral <= max_lateral_acceleration]
    if len(grid) < 50:
        return None
    mocap = (np.interp(grid, pose_time_s, pose_twists[:, -1]), np.interp(grid, pose_time_s, pose_twists[:, 0]))
    encoder = (wheel_speeds[:, 0] - wheel_speeds[:, 1], wheel_speeds[:, 0] + wheel_speeds[:, 1])
    # A constant channel scores 0 instead of NaN.
    usable = [
        index for index in range(2) if np.std(mocap[index]) > 1e-9 and np.std(encoder[index]) > 1e-9
    ]
    if not usable:
        return None
    lags = np.arange(search_s[0], search_s[1] + 0.5 * resolution_s, resolution_s)

    def correlations(lag):
        values = []
        for index in range(2):
            if index not in usable:
                values.append(0.0)
                continue
            shifted = np.interp(grid + lag, wheel_time_s, encoder[index])
            value = np.corrcoef(mocap[index], shifted)[0, 1] if np.std(shifted) > 1e-9 else 0.0
            values.append(float(np.nan_to_num(value)))
        return values

    scores = np.asarray([sum(correlations(lag)) for lag in lags])
    best = int(np.argmax(scores))
    lag = float(lags[best])
    if 0 < best < len(lags) - 1:
        a, b, c = scores[best - 1], scores[best], scores[best + 1]
        curvature = a - 2.0 * b + c
        if curvature < 0.0:
            lag += 0.5 * (a - c) / curvature * resolution_s
    yaw_correlation, speed_correlation = correlations(lag)
    return EncoderLag(lag, float(yaw_correlation), float(speed_correlation))


def smooth_and_align_encoder_speeds(
    time_s: np.ndarray,
    speeds: np.ndarray,
    *,
    cutoff_hz: float = DEFAULT_ENCODER_LP_CUTOFF_HZ,
    lag_s: float = 0.0,
    window_length: int = DEFAULT_ENCODER_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_ENCODER_SAVGOL_POLYORDER,
) -> np.ndarray:
    """Invert the firmware low-pass, smooth (zero phase) and shift the
    encoder wheel speeds by ``lag_s``."""
    time_s = np.asarray(time_s, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    if len(time_s) <= 1:
        return speeds
    unfiltered = invert_encoder_lowpass(time_s, speeds, cutoff_hz)
    smoothed = savgol_smooth_series(time_s, unfiltered, window_length=window_length, polyorder=polyorder)
    return shift_series(time_s, smoothed, lag_s)


def _valid_window(window_length: int, polyorder: int, num_samples: int) -> int:
    """Largest odd window <= min(window_length, num_samples) and > polyorder."""
    window = min(int(window_length), num_samples)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        raise ValueError(
            f"Too few samples ({num_samples}) for polyorder {polyorder}; "
            "cannot form a valid Savitzky-Golay window."
        )
    return window
