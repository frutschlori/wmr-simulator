import pickle

import numpy as np
import pytest

from wmr_simulator.trajectory_optimization.reference_extension import (
    attach,
    extend_identification_reference,
    arc_turn,
    representative_trajectories,
    traction_usage,
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


def test_arc_turn_drives_forward_on_the_arc_and_rests_at_both_ends():
    pose = np.array([1.0, 2.0, 0.5, 0, 0, 0, 0, 0])
    for angle in (np.pi / 2, -3 * np.pi / 4):
        turn = arc_turn(pose, angle, 0.25, DT, 8, lateral_acceleration=3.0, min_duration=0.75)
        np.testing.assert_allclose(turn[-1, 2], 0.5 + angle, atol=1e-12)
        # On the circle through the start, tangent to the start heading.
        sigma = np.sign(angle)
        center = pose[:2] + sigma * 0.25 * np.array([-np.sin(0.5), np.cos(0.5)])
        np.testing.assert_allclose(np.linalg.norm(turn[:, :2] - center, axis=1), 0.25, atol=1e-12)
        speed = np.hypot(turn[:, 3], turn[:, 4])
        # Forward along the heading, never on the spot inside, at rest at the end.
        np.testing.assert_allclose(turn[:-1, 3] * np.cos(turn[:-1, 2]) + turn[:-1, 4] * np.sin(turn[:-1, 2]),
                                   speed[:-1], atol=1e-12)
        assert speed[:-1].min() > 0.0 and speed[-1] < 1e-12
        np.testing.assert_allclose(turn[:, 5], sigma * speed / 0.25, atol=1e-12)
        assert (speed**2 / 0.25).max() <= 3.0 + 1e-9
        # The samples follow from one another at the listed speed.
        steps = np.linalg.norm(np.diff(np.vstack([pose[None, :2], turn[:, :2]]), axis=0), axis=1)
        assert steps.max() <= speed.max() * DT + 1e-9
    assert len(arc_turn(pose, 0.0, 0.25, DT, 8, lateral_acceleration=3.0, min_duration=0.75)) == 0


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
    # The pieces tile the reference in driving order: identified part, then a
    # turn before every turned segment, every segment once.
    pieces = chosen["pieces"]
    ends = [piece["end"] for piece in pieces]
    assert pieces[0] == {"kind": "identified", "end": 60}
    assert ends == sorted(ends) and len(set(ends)) == len(ends) and ends[-1] == len(placed) - 1
    assert [piece["segment"] for piece in pieces if piece["kind"] == "appended"] == chosen["order"]
    assert sum(piece["kind"] == "turn" for piece in pieces) == sum(angle != 0.0 for angle in chosen["turns_deg"])
    assert sum(angle != 0.0 for angle in chosen["turns_deg"]) > 0  # the box forces at least one turn here


def twist_states(speed, omega):
    """Reference states of given speed and yaw-rate profiles (heading integrated)."""
    theta = np.concatenate([[0.0], np.cumsum(omega[:-1]) * DT])
    states = np.zeros((len(speed), 8))
    states[:, 2] = theta
    states[:, 3] = speed * np.cos(theta)
    states[:, 4] = speed * np.sin(theta)
    states[:, 5] = omega
    return states


def test_representatives_skip_trajectories_over_the_traction_limit(tmp_path):
    steps = 41
    gentle = [save(tmp_path / f"g{k}.pkl", twist_states(np.full(steps, 1.0 + 0.1 * k), np.full(steps, 1.0)))
              for k in range(4)]
    # Braking hard while turning: most demanding by |v omega|, but over the limit.
    harsh = save(tmp_path / "h.pkl", twist_states(np.linspace(2.5, 0.5, steps) ** 2, np.full(steps, 2.0)))
    assert traction_usage(pickle.load(open(harsh, "rb"))["reference_states"], DT, 0.09) > 1.1
    assert representative_trajectories(gentle + [harsh], 2)[0] == harsh
    chosen = representative_trajectories(gentle + [harsh], 2, max_traction_usage=1.1, base_diameter=0.09)
    assert harsh not in chosen and len(chosen) == 2
    # Too few below the limit: the least-used ones over it fill up the count.
    assert representative_trajectories([gentle[0], harsh], 2, max_traction_usage=1.1, base_diameter=0.09) \
        == representative_trajectories([gentle[0], harsh], 2)
    with pytest.raises(ValueError):
        representative_trajectories(gentle, 2, max_traction_usage=1.1)
