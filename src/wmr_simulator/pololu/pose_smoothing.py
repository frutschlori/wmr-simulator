"""Spline smoothing of mocap pose streams for artefact-free velocities.

Mocap poses arrive as an irregular event stream. Finite differencing them is
fragile in two ways: an exactly repeated frame (the same pose logged twice a
few ms apart) produces a hard zero-velocity outlier, and small timestamp
jitter turns a real ~10 ms displacement divided by a jittered 3-5 ms dt into
multi-m/s spikes.

This module first drops repeated frames, then fits penalized smoothing
B-splines of choosable order to x, y and the *unwrapped* yaw as functions of
time. Velocities come from the analytic spline derivative instead of finite
differences, so they are smooth by construction and insensitive to the exact
sample times. A smoothing B-spline is the closed-form solution of "optimize
control points for a good fit": the fit is linear least squares in the
control points with a residual budget set from the measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, splrep

__all__ = [
    "PoseSplines",
    "drop_repeated_poses",
    "fit_pose_splines",
]


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


@dataclass(frozen=True)
class PoseSplines:
    """Smoothing splines for one pose stream: x(t), y(t), unwrapped yaw(t)."""

    spline_x: BSpline
    spline_y: BSpline
    spline_yaw: BSpline  # fitted on the unwrapped yaw
    t_start: float
    t_end: float

    def _clip(self, time_s: np.ndarray) -> np.ndarray:
        # B-spline extrapolation is a high-order polynomial and diverges fast;
        # clamping to the fitted domain gives constant-pose behaviour instead.
        return np.clip(np.asarray(time_s, dtype=float), self.t_start, self.t_end)

    def pose(self, time_s: np.ndarray) -> np.ndarray:
        """Smoothed poses (..., 3) with yaw wrapped to [-pi, pi)."""
        t = self._clip(time_s)
        yaw = (self.spline_yaw(t) + np.pi) % (2.0 * np.pi) - np.pi
        return np.stack([self.spline_x(t), self.spline_y(t), yaw], axis=-1)

    def world_velocity(self, time_s: np.ndarray) -> np.ndarray:
        """Analytic spline derivative (..., 3): x_dot, y_dot, yaw_dot."""
        t = self._clip(time_s)
        return np.stack(
            [
                self.spline_x.derivative()(t),
                self.spline_y.derivative()(t),
                self.spline_yaw.derivative()(t),
            ],
            axis=-1,
        )

    def body_twist(self, time_s: np.ndarray) -> np.ndarray:
        """World velocity rotated into the body frame (..., 3): v_x, v_y, omega."""
        t = self._clip(time_s)
        velocity = self.world_velocity(t)
        yaw = self.spline_yaw(t)
        cos_t, sin_t = np.cos(yaw), np.sin(yaw)
        v_x = velocity[..., 0] * cos_t + velocity[..., 1] * sin_t
        v_y = -velocity[..., 0] * sin_t + velocity[..., 1] * cos_t
        return np.stack([v_x, v_y, velocity[..., 2]], axis=-1)


def fit_pose_splines(
    time_s: np.ndarray,
    pose_states: np.ndarray,
    *,
    order: int = 3,
    noise_std_xy: float = 1e-3,
    noise_std_yaw: float = 5e-3,
    smoothing_factor: float = 1.0,
    drop_duplicates: bool = True,
) -> PoseSplines:
    """Fit penalized smoothing B-splines to a mocap pose stream.

    ``order`` is the spline degree (1..5); velocities need >= 2, and 3 (cubic)
    additionally keeps accelerations continuous. ``noise_std_xy`` /
    ``noise_std_yaw`` are the assumed per-sample measurement stds; FITPACK is
    given weights 1/std and the recommended residual budget s = m samples, so
    the fit uses as few knots as possible while staying within roughly one
    noise std of the data on average. ``smoothing_factor`` scales that budget
    (> 1 smooths harder, < 1 follows the data more closely, 0 interpolates).
    """
    if not 1 <= order <= 5:
        raise ValueError(f"Spline order must be in 1..5, got {order}.")
    if drop_duplicates:
        time_s, pose_states = drop_repeated_poses(time_s, pose_states)
    else:
        time_s = np.asarray(time_s, dtype=float)
        pose_states = np.asarray(pose_states, dtype=float)
    m = len(time_s)
    if m <= order:
        raise ValueError(f"Need more than {order} unique samples to fit order {order}, got {m}.")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("Pose timestamps must be strictly increasing (use drop_duplicates=True).")

    yaw_unwrapped = np.unwrap(pose_states[:, 2])
    residual_budget = smoothing_factor * m

    def fit(values: np.ndarray, noise_std: float) -> BSpline:
        weights = np.full(m, 1.0 / noise_std)
        tck = splrep(time_s, values, w=weights, k=order, s=residual_budget)
        return BSpline(*tck)

    return PoseSplines(
        spline_x=fit(pose_states[:, 0], noise_std_xy),
        spline_y=fit(pose_states[:, 1], noise_std_xy),
        spline_yaw=fit(yaw_unwrapped, noise_std_yaw),
        t_start=float(time_s[0]),
        t_end=float(time_s[-1]),
    )
