import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.estimator import DiffDriveEstimator
from wmr_simulator.simulation import apply_noise_configuration


def _estimator(mocap_delay: float, dt: float = 0.01, filter_type: str = "kf") -> DiffDriveEstimator:
    return DiffDriveEstimator(
        estimator_cfg={
            "type": filter_type,
            "wheel_radius": 0.016,
            "base_diameter": 0.085,
            "noise_pos": 0.0,
            "noise_angle": 0.0,
            "enc_angle_noise": 0.0,
            "proc_pos_std": 0.0,
            "proc_theta_std": 0.0,
            "mocap_delay": mocap_delay,
        },
        dt=dt,
    )


def test_measurement_is_served_delayed():
    dt = 0.01
    delay_steps = 3
    estimator = _estimator(mocap_delay=delay_steps * dt, dt=dt)
    assert estimator.delay_steps == delay_steps

    state = estimator.get_init_state(jax.random.PRNGKey(0), start_pose=jnp.zeros(3))
    true_poses = [jnp.array([0.1 * step, 0.0, 0.0]) for step in range(1, 9)]
    measured = []
    for pose_true in true_poses:
        state = estimator.update(state, 0.0, 0.0, pose_true)
        measured.append(np.asarray(state.pose_meas))

    # With zero noise the measurement equals the true pose from delay_steps ago
    # (the buffer starts filled with the start pose).
    for step, meas in enumerate(measured):
        expected = np.zeros(3) if step < delay_steps else np.asarray(true_poses[step - delay_steps])
        assert meas == pytest.approx(expected, abs=1e-6)


def test_zero_delay_passes_pose_through():
    estimator = _estimator(mocap_delay=0.0)
    assert estimator.delay_steps == 0
    state = estimator.get_init_state(jax.random.PRNGKey(0), start_pose=jnp.zeros(3))
    pose_true = jnp.array([0.3, -0.1, 0.2])
    state = estimator.update(state, 0.0, 0.0, pose_true)
    assert np.asarray(state.pose_meas) == pytest.approx(np.asarray(pose_true), abs=1e-6)


def test_delay_buffer_is_scan_compatible():
    dt = 0.01
    estimator = _estimator(mocap_delay=2 * dt, dt=dt)
    state0 = estimator.get_init_state(jax.random.PRNGKey(0), start_pose=jnp.zeros(3))
    poses = jnp.stack([jnp.array([0.1 * i, 0.0, 0.0]) for i in range(1, 6)])

    def step(state, pose_true):
        next_state = estimator.update(state, 1.0, 1.0, pose_true)
        return next_state, next_state.pose_meas

    _, measurements = jax.lax.scan(step, state0, poses)
    assert np.asarray(measurements[-1]) == pytest.approx(np.asarray(poses[-3]), abs=1e-6)


def test_noise_configuration_toggle():
    problem_cfg = {
        "noise_enabled": False,
        "robot": {"slip_sigma": 0.02, "slip_tau": 0.3, "a_slip_max": 3.0},
        "estimator": {"noise_pos": 0.002, "noise_angle": 0.017, "enc_angle_noise": 0.01,
                      "proc_pos_std": 0.02},
    }
    apply_noise_configuration(problem_cfg)
    assert problem_cfg["robot"]["slip_sigma"] == 0.0
    assert problem_cfg["robot"]["slip_tau"] == 0.0
    assert problem_cfg["robot"]["a_slip_max"] == 3.0  # deterministic slip untouched
    assert problem_cfg["estimator"]["noise_pos"] == 0.0
    assert problem_cfg["estimator"]["noise_angle"] == 0.0
    assert problem_cfg["estimator"]["enc_angle_noise"] == 0.0
    assert problem_cfg["estimator"]["proc_pos_std"] == 0.02  # KF tuning untouched

    enabled_cfg = {"noise_enabled": True, "robot": {"slip_sigma": 0.02}, "estimator": {"noise_pos": 0.002}}
    apply_noise_configuration(enabled_cfg)
    assert enabled_cfg["robot"]["slip_sigma"] == 0.02
    assert enabled_cfg["estimator"]["noise_pos"] == 0.002
