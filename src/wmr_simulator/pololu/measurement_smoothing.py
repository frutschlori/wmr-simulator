"""Savitzky-Golay smoothing of the Pololu measurement streams.

Two consumers live here: mocap poses and encoder wheel speeds.

Mocap poses arrive as an irregular event stream. Finite differencing them is
fragile in two ways: an exactly repeated frame (the same pose logged twice a
few ms apart) produces a hard zero-velocity outlier, and small timestamp
jitter turns a real ~10 ms displacement divided by a jittered 3-5 ms dt into
multi-m/s spikes. ``smooth_pose_stream`` first cleans the stream (drop repeated
frames, samples closer than ``min_dt``, and residual outliers), then resamples
x, y and the *unwrapped* yaw onto a uniform median-dt grid and runs a centered
Savitzky-Golay filter over each channel. The smoothed pose is the filter with
``deriv=0``; velocities come from the filter's closed-form derivative
(``deriv=1``, scaled by the grid spacing) instead of finite differences, so
they are smooth by construction and have no phase delay. Smoothed grids are
linearly interpolated back to any query time.

Encoder wheel speeds arrive already firmware low-pass filtered, so they carry
no outliers; ``smooth_and_align_encoder_speeds`` just applies a light extra
(zero-phase) Savitzky-Golay smoothing and advances them by the firmware LP
group delay to realign them in time with the mocap motion.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "DEFAULT_ENCODER_LP_TAU_S",
    "DEFAULT_ENCODER_SAVGOL_WINDOW",
    "DEFAULT_ENCODER_SAVGOL_POLYORDER",
]

# Window length (in samples) and polynomial order of the Savitzky-Golay filter.
# At ~10 ms mocap sampling a window of 35 smooths over ~350 ms; a cubic keeps
# the fit flexible enough for the arcs while rejecting per-sample noise (a wider
# window is needed for the derivative, whose noise shrinks with window length).
DEFAULT_SAVGOL_WINDOW = 35
DEFAULT_SAVGOL_POLYORDER = 3

# Outlier-rejection defaults (0 disables each). ``min_dt`` drops samples logged
# closer than 5 ms to the previous kept one (double-logged mocap frames).
# ``outlier_sigma`` drops samples whose residual from a preliminary Savitzky-
# Golay fit exceeds that many robust sigmas (MAD-based modified z-score): an
# adaptive absolute gate that rejects nothing on clean data but catches genuine
# position glitches
DEFAULT_MIN_DT = 5e-3
DEFAULT_OUTLIER_SIGMA = 15.0

# Encoder wheel speeds are firmware low-pass filtered (one-pole IIR, f_c = 3 Hz).
# The filter's DC group delay equals its time constant tau = 1/(2*pi*f_c) ~ 53 ms,
# so the logged speed at time t reflects the true speed at t - tau; we advance the
# series by this constant to realign it with the mocap motion (0 disables). On top
# we apply a light Savitzky-Golay smoothing (window/order below); the encoders
# carry no outliers so no rejection is needed. NB the constant-tau shift only
# exactly compensates low-frequency content (the group delay falls off above f_c),
# but that is where the wheel-speed signal lives.
# DEFAULT_ENCODER_LP_TAU_S = 1.0 / (2.0 * np.pi * 3.0)
DEFAULT_ENCODER_LP_TAU_S = 0.03
DEFAULT_ENCODER_SAVGOL_WINDOW = 5
DEFAULT_ENCODER_SAVGOL_POLYORDER = 3


def drop_repeated_poses(
    time_s: np.ndarray,
    pose_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate mocap frames from a pose stream.

    Removes every sample whose (x, y, yaw) exactly equals the previous kept
    sample (the same mocap frame logged twice; keeping it would fabricate a
    zero-velocity interval) and any sample whose timestamp does not strictly
    increase. The first occurrence is kept as it carries the original receive
    time.
    """
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
    """Drop samples logged closer than ``min_dt`` to the previous kept one.

    Mocap frames double-logged a few ms apart (whether or not the pose is
    identical) turn a real ~10 ms displacement into a spuriously short interval
    and a large velocity. Greedily keeps the first sample of each such cluster.
    ``min_dt <= 0`` disables the filter.
    """
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
    """Drop poses whose residual from a preliminary SG fit is anomalously large.

    Fits the Savitzky-Golay smoother once, takes each sample's residual (raw
    minus smoothed, yaw wrapped), and scores it by a MAD-based modified z-score
    (``|residual - median| / (1.4826 * MAD)``, per channel, combined as a max).
    Samples above ``sigma`` robust sigmas are dropped and the smoother is refit
    on the survivors. Because the SG fit follows the real motion, genuine fast
    maneuvers have small residuals; only isolated glitches (mocap teleports)
    stand out. Unlike a reject-``N%`` rule this is an absolute, adaptive gate:
    it rejects nothing on clean data and however many samples are truly bad.
    ``sigma <= 0`` disables the filter.
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
    # Robust scale (MAD -> std) with a floor at measurement resolution (~1 mm /
    # 1 mrad): on near-noiseless data the MAD collapses to ~0 and would turn
    # negligible SG bias into huge z-scores; the floor keeps the gate from
    # firing there (it only ever makes the gate more lenient).
    scale = np.maximum(1.4826 * np.median(np.abs(residual - median), axis=0), 1e-3)
    score = np.max(np.abs(residual - median) / scale, axis=1)

    keep = score <= sigma
    return time_s[keep], pose_states[keep]


@dataclass(frozen=True)
class SmoothedPoseStream:
    """Savitzky-Golay smoothed pose stream sampled on a uniform time grid.

    Holds the smoothed x, y and *unwrapped* yaw on ``grid_time`` together with
    their filter derivatives; the query methods linearly interpolate these onto
    arbitrary times. Interpolation clamps to the grid endpoints, so evaluating
    outside the fitted domain returns the boundary pose/velocity.
    """

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

    ``polyorder`` is the local polynomial degree (velocities need >= 1, >= 2
    keeps the derivative smooth); ``window_length`` is the (odd) number of grid
    samples the filter fits over. The stream is cleaned first: repeated frames
    (``drop_duplicates``), samples closer than ``min_dt`` seconds to the
    previous kept one, and poses whose residual from a preliminary fit exceeds
    ``outlier_sigma`` robust sigmas are dropped (each disabled when 0). The
    survivors are resampled onto a uniform median-dt grid, so the filter's
    centering gives velocities without phase delay and its ``deriv=1`` mode
    gives the closed-form velocity. ``window_length`` is shrunk to the largest
    valid odd value when the stream has too few samples for the requested window.
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
    """Resample poses onto a uniform grid and Savitzky-Golay filter them.

    Returns (grid_time, grid_pose, grid_velocity). Gridding makes the
    (uniform-spacing) SG filter valid and its derivative scaling by a single dt
    correct. Assumes strictly increasing ``time_s`` with more than ``polyorder``
    samples. Boundary ``mode='nearest'`` holds the edge pose rather than
    extrapolating a polynomial (``mode='interp'``): a robot at rest at the start
    or end then reads ~zero velocity there instead of a spurious ramp.
    """
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
    """Savitzky-Golay smooth an evenly-meant-but-jittered scalar/vector series.

    Resamples ``values`` (shape (N,) or (N, K)) onto a uniform median-dt grid,
    applies a centered SG filter (``deriv=0``, ``mode='nearest'`` at the edges),
    and interpolates back to the original ``time_s``. For signals that carry no
    outliers (e.g. firmware-filtered encoder speeds) this is all the pose
    pipeline's smoothing without the pose-specific yaw unwrapping or velocity
    derivative. Returns the same shape as ``values``. Falls back to the input
    unchanged when there are too few samples to form a valid window.
    """
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


def smooth_and_align_encoder_speeds(
    time_s: np.ndarray,
    speeds: np.ndarray,
    *,
    lp_tau_s: float = DEFAULT_ENCODER_LP_TAU_S,
    window_length: int = DEFAULT_ENCODER_SAVGOL_WINDOW,
    polyorder: int = DEFAULT_ENCODER_SAVGOL_POLYORDER,
) -> np.ndarray:
    """Light-smooth encoder wheel speeds and advance them by the firmware LP delay.

    The firmware one-pole low-pass (f_c ~ 3 Hz) delays the logged speeds by its
    DC group delay ``lp_tau_s``: the value logged at t reflects the true speed at
    t - lp_tau_s. Applies a centered (zero-phase) Savitzky-Golay smoothing, then
    advances the series by ``lp_tau_s`` via ``np.interp`` at ``time_s + lp_tau_s``
    so it lines up in time with the mocap-derived motion. The encoders carry no
    outliers, so no rejection is done. ``lp_tau_s <= 0`` skips the shift. Returns
    the same shape as ``speeds`` (``(N,)`` or ``(N, K)``).
    """
    time_s = np.asarray(time_s, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    if len(time_s) <= 1:
        return speeds

    smoothed = savgol_smooth_series(time_s, speeds, window_length=window_length, polyorder=polyorder)
    if lp_tau_s <= 0.0:
        return smoothed

    single_column = smoothed.ndim == 1
    matrix = smoothed[:, None] if single_column else smoothed
    advanced = np.column_stack(
        [np.interp(time_s + lp_tau_s, time_s, matrix[:, i]) for i in range(matrix.shape[1])]
    )
    return advanced[:, 0] if single_column else advanced


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
