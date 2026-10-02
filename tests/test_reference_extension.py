import pickle

import numpy as np
import pytest

from wmr_simulator.trajectory_optimization.reference_extension import (
    attach,
    extend_identification_reference,
    extended_payload,
    mirror,
    representative_trajectories,
    turn_in_place,
)

DT = 0.05


def arc(speed, omega, steps=40, start=(0.0, 0.0, 0.0)):
    """[x, y, theta, vx, vy, omega, ax, ay] of a constant-twist arc with world-frame
    velocity and acceleration, resting at both ends."""
    x, y, theta = start
    rows = []
    for k in range(steps + 1):
        v = speed if 0 < k < steps else 0.0
        w = omega if 0 < k < steps else 0.0
        rows.append([x, y, theta, v * np.cos(theta), v * np.sin(theta), w,
                     -v * w * np.sin(theta), v * w * np.cos(theta)])
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        theta += w * DT
    return np.asarray(rows)


def save(path, states, **extra):
    with open(path, "wb") as file:
        pickle.dump({"reference_states": states, "dt": DT, **extra}, file)
    return path


def test_attach_is_a_rigid_motion():
    segment = arc(1.0, 1.5, start=(0.3, -0.2, 0.4))
    end_pose = np.array([2.0, 1.0, -2.0])
    moved = attach(end_pose, segment)
    np.testing.assert_allclose(moved[0, :3], end_pose, atol=1e-12)
    # Speed, yaw rate and the path's shape are unchanged ...
    np.testing.assert_allclose(np.hypot(moved[:, 3], moved[:, 4]), np.hypot(segment[:, 3], segment[:, 4]), atol=1e-12)
    np.testing.assert_allclose(moved[:, 5], segment[:, 5], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(moved[:, :2], axis=0), axis=1),
                               np.linalg.norm(np.diff(segment[:, :2], axis=0), axis=1), atol=1e-12)
    # ... and the world-frame velocity still points along the moved heading.
    heading_of_velocity = np.arctan2(moved[1:-1, 4], moved[1:-1, 3])
    np.testing.assert_allclose(np.cos(heading_of_velocity - moved[1:-1, 2]), 1.0, atol=1e-9)


def test_extension_keeps_the_identified_part_and_joins_continuously():
    ident = np.vstack([arc(0.8, 0.5, steps=60), arc(2.0, 2.0, steps=80, start=(9, 9, 0))[1:]])
    segments = [arc(1.0, -1.0), arc(0.5, 2.0)]
    extended, _ = extend_identification_reference(ident, DT, 3.0, segments)
    np.testing.assert_array_equal(extended[:61], ident[:61])
    assert len(extended) == 61 + 40 + 40
    jumps = np.linalg.norm(np.diff(extended[:, :2], axis=0), axis=1)
    assert jumps.max() < 2.0 * DT + 1e-9  # no teleport at the joints (speeds <= 2 m/s)


def test_representative_picks_the_demanding_then_the_medoid(tmp_path):
    slow = [save(tmp_path / f"t{k}.pkl", arc(0.3 + 0.05 * k, 0.5)) for k in range(4)]
    fast = save(tmp_path / "t9.pkl", arc(1.5, 2.5))
    chosen = representative_trajectories(slow + [fast], 2)
    assert chosen[0] == fast
    assert chosen[1] in slow and chosen[1] != fast
    assert representative_trajectories(slow, 0) == []


def test_extended_payload_records_what_was_appended(tmp_path):
    ident = save(tmp_path / "identification.pkl", arc(0.8, 0.5, steps=140), identified_duration=3.0)
    tuning = [save(tmp_path / "a.pkl", arc(1.0, 1.0)), save(tmp_path / "b.pkl", arc(0.6, -1.0))]
    payload = extended_payload(ident, tuning)
    assert payload["identified_duration"] == 3.0
    assert payload["appended_tuning_trajectories"] == ["a.pkl", "b.pkl"]
    assert len(payload["reference_states"]) == 61 + 80


def test_extended_payload_needs_an_identified_duration(tmp_path):
    ident = save(tmp_path / "identification.pkl", arc(0.8, 0.5))
    with pytest.raises(ValueError, match="identified_duration"):
        extended_payload(ident, [save(tmp_path / "a.pkl", arc(1.0, 1.0))])


def test_mirror_turns_the_other_way():
    segment = arc(1.0, 1.5, start=(0.3, -0.2, 0.4))
    flipped = mirror(segment)
    np.testing.assert_allclose(flipped[0, :3], segment[0, :3], atol=1e-12)
    np.testing.assert_allclose(flipped[:, 5], -segment[:, 5], atol=1e-12)
    np.testing.assert_allclose(np.hypot(flipped[:, 3], flipped[:, 4]), np.hypot(segment[:, 3], segment[:, 4]), atol=1e-12)
    heading_of_velocity = np.arctan2(flipped[1:-1, 4], flipped[1:-1, 3])
    np.testing.assert_allclose(np.cos(heading_of_velocity - flipped[1:-1, 2]), 1.0, atol=1e-9)


def test_turn_in_place_rests_and_turns_by_the_angle():
    turn = turn_in_place(np.array([1.0, 2.0, 0.5, 0, 0, 0, 0, 0]), np.pi / 2, 0.75, DT, 8)
    np.testing.assert_allclose(turn[:, :2], [[1.0, 2.0]] * len(turn))
    np.testing.assert_allclose(turn[-1, 2], 0.5 + np.pi / 2, atol=1e-12)
    np.testing.assert_allclose(np.sum(turn[:, 5]) * DT, np.pi / 2, rtol=0.05)


def test_placement_keeps_the_reference_inside_the_box():
    ident = arc(0.8, 0.0, steps=60, start=(-0.5, -1.0, np.pi / 2))
    straight = [arc(1.0, 0.1), arc(1.0, -0.1)]  # 2 x ~1.95 m, would leave the box head on
    unplaced, _ = extend_identification_reference(ident, DT, 3.0, straight)
    assert unplaced[:, 1].max() > 2.3
    placed, chosen = extend_identification_reference(ident, DT, 3.0, straight, box_min=[-1.3, -2.3],
                                                     box_max=[1.3, 2.3], margin=0.15)
    assert chosen["box_violation"] == 0.0
    assert placed[:, 0].min() >= -1.15 - 1e-9 and placed[:, 0].max() <= 1.15 + 1e-9
    assert placed[:, 1].min() >= -2.15 - 1e-9 and placed[:, 1].max() <= 2.15 + 1e-9
