import numpy as np
import pytest

from wmr_simulator.trajectory_optimization.baselines import (
    circle_reference,
    generate_baseline_reference,
    lemniscate_ramp_reference,
    lemniscate_reference,
    spin_reference,
    summarize_motion,
)

DT = 0.05
TWO_PI = 2.0 * np.pi


def speeds(reference_states: np.ndarray) -> np.ndarray:
    return np.linalg.norm(reference_states[:, 3:5], axis=1)


def test_circle_reference_shape_and_sample_count():
    states = circle_reference(radius=0.5, total_time=10.0, dt=DT)
    assert states.shape == (int(10.0 / DT) + 1, 8)


def test_circle_reference_stays_on_circle():
    radius = 0.4
    center = (0.3, -0.2)
    states = circle_reference(radius=radius, total_time=8.0, dt=DT, center=center)
    distances = np.linalg.norm(states[:, :2] - np.asarray(center), axis=1)
    assert np.allclose(distances, radius, atol=1e-6)


def test_circle_reference_zero_endpoint_velocity_and_acceleration():
    states = circle_reference(radius=0.5, total_time=10.0, dt=DT)
    for index in (0, -1):
        assert np.allclose(states[index, 3:], 0.0, atol=1e-5)


def test_circle_reference_closes_loop():
    states = circle_reference(radius=0.5, total_time=10.0, dt=DT)
    assert np.allclose(states[0, :3], states[-1, :3], atol=1e-4)


def test_circle_clockwise_flips_omega():
    ccw = circle_reference(radius=0.5, total_time=10.0, dt=DT)
    cw = circle_reference(radius=0.5, total_time=10.0, dt=DT, clockwise=True)
    assert np.max(ccw[:, 5]) > 0.0
    assert np.min(cw[:, 5]) < 0.0


def test_lemniscate_reference_hits_max_speed():
    max_speed = 0.4
    states = lemniscate_reference(total_time=12.0, max_speed=max_speed, dt=DT)
    assert np.isclose(np.max(speeds(states)), max_speed, rtol=1e-6)
    for index in (0, -1):
        assert np.allclose(states[index, 3:5], 0.0, atol=1e-6)


def test_lemniscate_reference_starts_and_ends_at_center():
    center = (0.1, 0.2)
    states = lemniscate_reference(total_time=12.0, max_speed=0.4, dt=DT, center=center)
    assert np.allclose(states[0, :2], center, atol=1e-6)
    assert np.allclose(states[-1, :2], center, atol=1e-4)


def test_lemniscate_ramp_completes_cycles_with_monotone_phase():
    cycles = 3
    states = lemniscate_ramp_reference(
        total_time=24.0, cycles=cycles, speed_rate=1.4, dt=DT, amplitude=0.5
    )
    # Both endpoints sit at the lemniscate origin after full cycles.
    assert np.allclose(states[0, :2], 0.0, atol=1e-6)
    assert np.allclose(states[-1, :2], 0.0, atol=1e-3)
    # Graceful start and stop.
    assert np.allclose(states[0, 3:5], 0.0, atol=1e-6)
    assert np.allclose(states[-1, 3:5], 0.0, atol=1e-4)


def test_lemniscate_ramp_speed_grows_between_cycles():
    states = lemniscate_ramp_reference(
        total_time=24.0, cycles=4, speed_rate=1.5, dt=DT, amplitude=0.5
    )
    speed = speeds(states)
    quarter = len(speed) // 4
    early_peak = np.max(speed[:quarter])
    late_peak = np.max(speed[2 * quarter : 3 * quarter])
    assert late_peak > early_peak


def test_lemniscate_ramp_max_speed_override():
    max_speed = 0.3
    states = lemniscate_ramp_reference(
        total_time=24.0, cycles=3, speed_rate=1.4, dt=DT, max_speed=max_speed
    )
    assert np.isclose(np.max(speeds(states)), max_speed, rtol=1e-6)


def test_spin_reference_stays_in_place_and_hits_max_omega():
    max_omega = 3.0
    center = (0.2, -0.1)
    states = spin_reference(total_time=8.0, max_omega=max_omega, dt=DT, center=center)
    assert np.allclose(states[:, :2], center, atol=1e-12)
    assert np.allclose(states[:, 3:5], 0.0, atol=1e-12)
    assert np.allclose(states[:, 6:8], 0.0, atol=1e-12)
    assert np.isclose(np.max(states[:, 5]), max_omega, rtol=1e-3)
    # Graceful spin-up and spin-down.
    assert np.isclose(states[0, 5], 0.0, atol=1e-9)
    assert np.isclose(states[-1, 5], 0.0, atol=1e-9)


def test_spin_reference_heading_consistent_with_omega():
    states = spin_reference(total_time=8.0, max_omega=3.0, dt=DT, start_angle=0.5)
    assert np.isclose(states[0, 2], 0.5)
    theta_rate = np.gradient(states[:, 2], DT)
    assert np.allclose(theta_rate[1:-1], states[1:-1, 5], atol=0.05)


def test_spin_reference_clockwise_flips_rotation():
    ccw = spin_reference(total_time=8.0, max_omega=3.0, dt=DT)
    cw = spin_reference(total_time=8.0, max_omega=3.0, dt=DT, clockwise=True)
    assert ccw[-1, 2] > ccw[0, 2]
    assert cw[-1, 2] < cw[0, 2]
    assert np.isclose(np.max(np.abs(cw[:, 5])), 3.0, rtol=1e-3)


def test_generate_baseline_reference_dispatch_and_unknown_type():
    states = generate_baseline_reference("circle", dt=DT, radius=0.5, total_time=10.0)
    assert states.shape[1] == 8
    with pytest.raises(ValueError, match="Unsupported baseline type"):
        generate_baseline_reference("spiral", dt=DT)


def test_check_firmware_limits_flags_oversized_references():
    from wmr_simulator.pololu.reference_exporter import (
        FIRMWARE_MAX_FILE_BYTES,
        FIRMWARE_MAX_TRAJECTORY_POINTS,
        check_firmware_limits,
    )

    assert check_firmware_limits(num_states=500, file_bytes=1000) == []
    warnings = check_firmware_limits(num_states=601, file_bytes=85044)
    assert len(warnings) == 2
    assert str(FIRMWARE_MAX_TRAJECTORY_POINTS) in warnings[0]
    assert str(FIRMWARE_MAX_FILE_BYTES) in warnings[1]


def test_summarize_motion_reports_peaks():
    states = circle_reference(radius=0.5, total_time=10.0, dt=DT)
    summary = summarize_motion(states, DT)
    assert summary["num_samples"] == states.shape[0]
    assert np.isclose(summary["v_peak"], np.max(speeds(states)))
    assert summary["v_peak"] > summary["v_mean"] > 0.0
