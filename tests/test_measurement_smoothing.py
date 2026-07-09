import numpy as np
import pytest

from wmr_simulator.pololu.measurement_smoothing import (
    drop_repeated_poses,
    reject_residual_outliers,
    reject_short_intervals,
    smooth_pose_stream,
)


def _circle_trajectory(rng, *, radius=0.5, omega=1.2, duration=6.0, dt=0.01, jitter=0.002):
    """Noisy mocap-like samples of a constant-twist circular arc.

    Returns (time_s, noisy_poses) plus the analytic body twist, which is
    constant: v_x = radius * omega, v_y = 0, omega.
    """
    time_s = np.arange(0.0, duration, dt)
    time_s = time_s + rng.uniform(-jitter, jitter, size=len(time_s))
    time_s = np.sort(time_s)
    yaw = omega * time_s
    x = radius * np.sin(yaw)
    y = radius * (1.0 - np.cos(yaw))
    poses = np.column_stack([x, y, (yaw + np.pi) % (2.0 * np.pi) - np.pi])
    noisy = poses + rng.normal(0.0, [5e-4, 5e-4, 2e-3], size=poses.shape)
    return time_s, noisy


def test_drop_repeated_poses_removes_duplicate_frames():
    time_s = np.array([0.0, 0.01, 0.013, 0.02, 0.03])
    poses = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1],  # exact repeat of the previous frame
            [0.2, 0.0, 0.1],
            [0.3, 0.0, 0.1],
        ]
    )
    kept_time, kept_poses = drop_repeated_poses(time_s, poses)
    np.testing.assert_allclose(kept_time, [0.0, 0.01, 0.02, 0.03])
    assert not np.any(np.all(np.diff(kept_poses, axis=0) == 0.0, axis=1))


def test_drop_repeated_poses_removes_non_increasing_timestamps():
    time_s = np.array([0.0, 0.01, 0.01, 0.02])
    poses = np.column_stack([np.arange(4.0), np.zeros(4), np.zeros(4)])
    kept_time, kept_poses = drop_repeated_poses(time_s, poses)
    assert np.all(np.diff(kept_time) > 0.0)
    assert len(kept_poses) == 3


def test_reject_short_intervals_drops_close_samples():
    time_s = np.array([0.0, 0.003, 0.01, 0.012, 0.02])  # 2nd and 4th arrive <5 ms later
    poses = np.column_stack([np.arange(5.0), np.zeros(5), np.zeros(5)])
    kept_time, kept_poses = reject_short_intervals(time_s, poses, min_dt=0.005)
    np.testing.assert_allclose(kept_time, [0.0, 0.01, 0.02])
    assert np.all(np.diff(kept_time) >= 0.005)
    # Disabled when min_dt <= 0.
    same_t, same_p = reject_short_intervals(time_s, poses, min_dt=0.0)
    np.testing.assert_array_equal(same_t, time_s)


def test_reject_residual_outliers_drops_corrupt_pose():
    rng = np.random.default_rng(7)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    corrupt = len(time_s) // 2
    noisy[corrupt, 0] += 0.5  # a teleport: huge residual from the smoothed track
    kept_time, kept_poses = reject_residual_outliers(time_s, noisy, sigma=15.0)
    assert time_s[corrupt] not in kept_time


def test_reject_residual_outliers_keeps_clean_data():
    # No glitch: an adaptive absolute gate should reject (almost) nothing,
    # unlike a fixed reject-N% rule.
    rng = np.random.default_rng(8)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    kept_time, _ = reject_residual_outliers(time_s, noisy, sigma=15.0)
    assert len(kept_time) >= len(time_s) - 1


def test_savgol_twist_matches_analytic_circle():
    rng = np.random.default_rng(0)
    radius, omega = 0.5, 1.2
    time_s, noisy = _circle_trajectory(rng, radius=radius, omega=omega)
    smoothed = smooth_pose_stream(time_s, noisy)

    eval_t = np.linspace(time_s[0] + 0.2, time_s[-1] - 0.2, 200)
    twist = smoothed.body_twist(eval_t)
    np.testing.assert_allclose(twist[:, 0], radius * omega, atol=0.02)
    np.testing.assert_allclose(twist[:, 1], 0.0, atol=0.02)
    np.testing.assert_allclose(twist[:, 2], omega, atol=0.06)


def test_savgol_smooths_duplicate_frame_outliers():
    rng = np.random.default_rng(1)
    radius, omega = 0.5, 1.2
    time_s, noisy = _circle_trajectory(rng, radius=radius, omega=omega)
    # Inject duplicate frames 3 ms after random samples, as seen in the logs.
    dup_indices = rng.choice(np.arange(50, len(time_s) - 50), size=5, replace=False)
    time_dup = np.concatenate([time_s, time_s[dup_indices] + 0.003])
    poses_dup = np.concatenate([noisy, noisy[dup_indices]])
    order = np.argsort(time_dup)
    smoothed = smooth_pose_stream(time_dup[order], poses_dup[order])

    twist = smoothed.body_twist(time_s[dup_indices])
    # Finite differences would report ~0 velocity on the duplicate intervals.
    np.testing.assert_allclose(twist[:, 0], radius * omega, atol=0.03)


def test_pose_yaw_wraps_and_matches_input():
    rng = np.random.default_rng(2)
    time_s, noisy = _circle_trajectory(rng, omega=1.5, duration=6.0)
    smoothed = smooth_pose_stream(time_s, noisy)
    pose = smoothed.pose(time_s)
    assert np.all(pose[:, 2] >= -np.pi) and np.all(pose[:, 2] < np.pi)
    # Smoothed poses stay close to the noisy measurements (compare wrapped) in
    # the interior; the 'nearest' boundary mode deliberately biases the edges
    # (it holds the edge pose, which suits an at-rest start/end but not this
    # constant-twist circle that moves through the boundary).
    interior = slice(20, -20)
    yaw_err = (pose[interior, 2] - noisy[interior, 2] + np.pi) % (2.0 * np.pi) - np.pi
    assert np.max(np.abs(pose[interior, :2] - noisy[interior, :2])) < 5e-3
    assert np.max(np.abs(yaw_err)) < 2e-2


@pytest.mark.parametrize("polyorder", [2, 3, 5])
def test_savgol_polyorder_is_choosable(polyorder):
    rng = np.random.default_rng(3)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    smoothed = smooth_pose_stream(time_s, noisy, polyorder=polyorder)
    twist = smoothed.body_twist(np.linspace(0.5, 2.5, 50))
    np.testing.assert_allclose(twist[:, 0], 0.6, atol=0.05)


def test_evaluation_is_clamped_to_fitted_domain():
    rng = np.random.default_rng(4)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    smoothed = smooth_pose_stream(time_s, noisy)
    inside = smoothed.pose(np.array([smoothed.t_end]))
    outside = smoothed.pose(np.array([smoothed.t_end + 10.0]))
    np.testing.assert_allclose(outside, inside)


def test_rejects_too_few_samples_and_bad_polyorder():
    time_s = np.array([0.0, 0.01, 0.02])
    poses = np.column_stack([np.arange(3.0), np.zeros(3), np.zeros(3)])
    with pytest.raises(ValueError):
        smooth_pose_stream(time_s, poses, polyorder=3)
    with pytest.raises(ValueError):
        smooth_pose_stream(time_s, poses, polyorder=0)
