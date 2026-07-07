import numpy as np
import pytest

from wmr_simulator.pololu.pose_smoothing import (
    drop_repeated_poses,
    fit_pose_splines,
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


def test_spline_twist_matches_analytic_circle():
    rng = np.random.default_rng(0)
    radius, omega = 0.5, 1.2
    time_s, noisy = _circle_trajectory(rng, radius=radius, omega=omega)
    splines = fit_pose_splines(time_s, noisy)

    eval_t = np.linspace(time_s[0] + 0.2, time_s[-1] - 0.2, 200)
    twist = splines.body_twist(eval_t)
    np.testing.assert_allclose(twist[:, 0], radius * omega, atol=0.02)
    np.testing.assert_allclose(twist[:, 1], 0.0, atol=0.02)
    np.testing.assert_allclose(twist[:, 2], omega, atol=0.05)


def test_spline_smooths_duplicate_frame_outliers():
    rng = np.random.default_rng(1)
    radius, omega = 0.5, 1.2
    time_s, noisy = _circle_trajectory(rng, radius=radius, omega=omega)
    # Inject duplicate frames 3 ms after random samples, as seen in the logs.
    dup_indices = rng.choice(np.arange(50, len(time_s) - 50), size=5, replace=False)
    time_dup = np.concatenate([time_s, time_s[dup_indices] + 0.003])
    poses_dup = np.concatenate([noisy, noisy[dup_indices]])
    order = np.argsort(time_dup)
    splines = fit_pose_splines(time_dup[order], poses_dup[order])

    twist = splines.body_twist(time_s[dup_indices])
    # Finite differences would report ~0 velocity on the duplicate intervals.
    np.testing.assert_allclose(twist[:, 0], radius * omega, atol=0.03)


def test_pose_yaw_wraps_and_matches_input():
    rng = np.random.default_rng(2)
    time_s, noisy = _circle_trajectory(rng, omega=1.5, duration=6.0)
    splines = fit_pose_splines(time_s, noisy)
    pose = splines.pose(time_s)
    assert np.all(pose[:, 2] >= -np.pi) and np.all(pose[:, 2] < np.pi)
    # Smoothed poses stay close to the noisy measurements (compare wrapped).
    yaw_err = (pose[:, 2] - noisy[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    assert np.max(np.abs(pose[:, :2] - noisy[:, :2])) < 5e-3
    assert np.max(np.abs(yaw_err)) < 2e-2


@pytest.mark.parametrize("order", [2, 3, 5])
def test_spline_order_is_choosable(order):
    rng = np.random.default_rng(3)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    splines = fit_pose_splines(time_s, noisy, order=order)
    twist = splines.body_twist(np.linspace(0.5, 2.5, 50))
    np.testing.assert_allclose(twist[:, 0], 0.6, atol=0.05)


def test_evaluation_is_clamped_to_fitted_domain():
    rng = np.random.default_rng(4)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    splines = fit_pose_splines(time_s, noisy)
    inside = splines.pose(np.array([splines.t_end]))
    outside = splines.pose(np.array([splines.t_end + 10.0]))
    np.testing.assert_allclose(outside, inside)


def test_rejects_too_few_samples_and_bad_order():
    time_s = np.array([0.0, 0.01, 0.02])
    poses = np.column_stack([np.arange(3.0), np.zeros(3), np.zeros(3)])
    with pytest.raises(ValueError):
        fit_pose_splines(time_s, poses, order=3)
    with pytest.raises(ValueError):
        fit_pose_splines(time_s, poses, order=0)
