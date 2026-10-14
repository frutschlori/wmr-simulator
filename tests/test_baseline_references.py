import numpy as np

from wmr_simulator.trajectory_optimization.baselines import (
    circle_reference,
    lemniscate_reference,
)

DT = 0.05
TWO_PI = 2.0 * np.pi


def speeds(reference_states: np.ndarray) -> np.ndarray:
    return np.linalg.norm(reference_states[:, 3:5], axis=1)


def test_circle_reference_stays_on_circle():
    radius = 0.4
    center = (0.3, -0.2)
    states = circle_reference(radius=radius, total_time=8.0, dt=DT, center=center)
    distances = np.linalg.norm(states[:, :2] - np.asarray(center), axis=1)
    assert np.allclose(distances, radius, atol=1e-6)


def test_circle_reference_closes_loop():
    states = circle_reference(radius=0.5, total_time=10.0, dt=DT)
    assert np.allclose(states[0, :3], states[-1, :3], atol=1e-4)


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


def test_check_firmware_limits_flags_oversized_references():
    from wmr_simulator.pololu.reference_exporter import (
        FIRMWARE_MAX_FILE_BYTES,
        FIRMWARE_MAX_TRAJECTORY_POINTS,
        check_firmware_limits,
    )

    # MAX_POINTS in pololu-rs firmware/src/trajectory_reading.rs.
    assert FIRMWARE_MAX_TRAJECTORY_POINTS == 400
    assert check_firmware_limits(num_states=400, file_bytes=1000) == []
    assert len(check_firmware_limits(num_states=401, file_bytes=1000)) == 1
    warnings = check_firmware_limits(num_states=601, file_bytes=85044)
    assert len(warnings) == 2
    assert str(FIRMWARE_MAX_TRAJECTORY_POINTS) in warnings[0]
    assert str(FIRMWARE_MAX_FILE_BYTES) in warnings[1]
