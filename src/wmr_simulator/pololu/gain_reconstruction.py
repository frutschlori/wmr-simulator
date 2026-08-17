"""Reconstruct the controller gains applied along a recorded log.

The firmware does not log the per-step controller gains (saving MCU compute and
SD-card space). When a run used a gain parametrization, we recover the applied
gains offline by replaying the parametrization on the log's reference, measured
pose and encoder body twist -- the same inputs the on-robot controller fed the
parametrization every geometry step. The result aligns with the log's command
timestamps so it can overlay the log summary plot like a simulated rollout's
recorded gains.
"""

from __future__ import annotations

import numpy as np


def _interp_states(src_time, src_states, target_time: np.ndarray) -> np.ndarray:
    src_time = np.asarray(src_time, dtype=float)
    src_states = np.asarray(src_states, dtype=float)
    return np.column_stack(
        [np.interp(target_time, src_time, src_states[:, column]) for column in range(src_states.shape[1])]
    )


def applied_gains_over_log(log, base_gains, params) -> np.ndarray:
    """Controller gains applied per command step, shape ``(num_commands, num_gains)``.

    ``base_gains`` are the run's nominal gains and ``params`` the gain
    parametrization (``wmr_simulator.gain_parametrization`` params). The
    reference, pose and body-twist streams are interpolated onto the command
    timestamps before the parametrization is replayed.
    """
    from wmr_simulator.gain_parametrization import gains_over_samples

    command_time = np.asarray(log.pose.command_time_s, dtype=float)

    # Unwrap headings before interpolation so linear interp never crosses a
    # +/-pi jump; the parametrization re-wraps the heading error itself.
    reference = np.array(log.reference.states, dtype=float, copy=True)
    reference[:, 2] = np.unwrap(reference[:, 2])
    pose = np.array(log.pose.states, dtype=float, copy=True)
    pose[:, 2] = np.unwrap(pose[:, 2])

    ref_samples = _interp_states(log.reference.time_s, reference, command_time)
    pose_samples = _interp_states(log.pose.time_s, pose, command_time)
    # Spline-smoothed body twist columns [v, omega] (savgol; see pose_smoothing).
    twists = np.asarray(log.pose.twists, dtype=float)[:, [0, 2]]
    twist_samples = _interp_states(log.pose.time_s, twists, command_time)

    gains = gains_over_samples(base_gains, params, ref_samples, pose_samples, twist_samples)
    return np.asarray(gains, dtype=float)
