import numpy as np

from wmr_simulator.pololu.measurement_smoothing import (
    drop_repeated_poses,
    reject_residual_outliers,
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


def test_reject_residual_outliers_drops_corrupt_pose():
    rng = np.random.default_rng(7)
    time_s, noisy = _circle_trajectory(rng, duration=3.0)
    corrupt = len(time_s) // 2
    noisy[corrupt, 0] += 0.5  # a teleport: huge residual from the smoothed track
    kept_time, kept_poses = reject_residual_outliers(time_s, noisy, sigma=15.0)
    assert time_s[corrupt] not in kept_time


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


def test_pose_yaw_wraps_and_matches_input():
    rng = np.random.default_rng(2)
    time_s, noisy = _circle_trajectory(rng, omega=1.5, duration=6.0)
    smoothed = smooth_pose_stream(time_s, noisy)
    pose = smoothed.pose(time_s)
    assert np.all(pose[:, 2] >= -np.pi) and np.all(pose[:, 2] < np.pi)
    # Smoothed poses stay close to the measurements in the interior (the edge
    # mode holds the edge pose).
    interior = slice(20, -20)
    yaw_err = (pose[interior, 2] - noisy[interior, 2] + np.pi) % (2.0 * np.pi) - np.pi
    assert np.max(np.abs(pose[interior, :2] - noisy[interior, :2])) < 5e-3
    assert np.max(np.abs(yaw_err)) < 2e-2
