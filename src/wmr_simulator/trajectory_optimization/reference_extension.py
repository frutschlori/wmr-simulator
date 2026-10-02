"""Tuning trajectories appended to the identification reference.

The residual is trained on the identification logs, and a residual is only
accurate where its data lies; what it has to be accurate for is the gain
tuner, which rolls out the tuning set. Driving the tuning set on the robot
costs an SD-card swap per trajectory, so instead the identification reference
carries some of it: its identified phase, then trajectories of the previous
iteration's tuning set, moved rigidly to start where the identified phase ends
(both rest at the joint). Measured in MuJoCo (2026-10-02, 5 seeds), two
appended trajectories -- the most demanding and the medoid of the set -- made
a closed-loop-trained residual predict held-out runs 2.4 % better and the gains
retuned on it track 1.1 mm better (5/5 seeds), as much as driving all fifteen;
one trajectory was not consistent, and the identification logs as they were
(fast phase far outside the tuning regime) made the residual 16 % worse.

Reference states are ``[x, y, theta, vx, vy, omega, ax, ay]`` with *world-frame*
velocity and acceleration, so a rigid motion rotates columns 3-4 and 6-7 along
with the path; omega is invariant.

Attached end to end, the segments carry on in whatever direction they happen to
end, and two tuning trajectories easily take the reference 5-8 m from its start
-- far outside the arena the robot is tracked in. Placement therefore has two
dynamically clean degrees of freedom, both at a point where the robot rests: a
segment may be *mirrored* about its start heading (the same motion with the yaw
rate negated), and a short smooth *turn in place* may re-aim it. A small search
over the segments' order, mirrors and turn angles keeps the whole reference
inside the environment box, preferring no turns and an end near the start (a
short return).
"""

from __future__ import annotations

import itertools
import math
import pickle
from pathlib import Path

import numpy as np

# Turn angles tried at a joint (multiples of 45 degrees; 0 means no turn).
TURN_ANGLES = tuple(math.radians(degrees) for degrees in (0, 45, -45, 90, -90, 135, -135, 180))


def _states(path: Path) -> np.ndarray:
    with Path(path).open("rb") as file:
        payload = pickle.load(file)
    states = payload["reference_states"] if isinstance(payload, dict) else payload
    return np.asarray(states, dtype=float)


def _motion_statistics(states: np.ndarray) -> np.ndarray:
    """(mean speed, mean |omega|, mean |v * omega|) of a reference."""
    speed = np.hypot(states[:, 3], states[:, 4])
    omega = np.abs(states[:, 5])
    return np.asarray([speed.mean(), omega.mean(), (speed * omega).mean()])


def representative_trajectories(pickles: list[Path], count: int) -> list[Path]:
    """``count`` trajectories of a tuning set: the most demanding one (largest
    mean lateral acceleration |v * omega|, the fast-turning regime), then the
    medoid (closest to the set's mean motion in units of its spread), then the
    remaining ones in decreasing demand."""
    pickles = sorted(Path(p) for p in pickles)
    if count <= 0 or not pickles:
        return []
    stats = np.stack([_motion_statistics(_states(path)) for path in pickles])
    by_demand = list(np.argsort(-stats[:, 2]))
    z = (stats - stats.mean(axis=0)) / (stats.std(axis=0) + 1e-9)
    medoid = int(np.argmin(np.sum(z**2, axis=1)))
    order = [by_demand[0]] + ([medoid] if medoid != by_demand[0] else [])
    order += [index for index in by_demand if index not in order]
    return [pickles[index] for index in order[:count]]


def attach(end_pose: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """``segment`` moved rigidly so it starts at ``end_pose`` [x, y, theta]."""
    moved = np.array(segment, dtype=float, copy=True)
    rotation = float(end_pose[2] - moved[0, 2])
    c, s = math.cos(rotation), math.sin(rotation)
    offset = moved[:, :2] - moved[0, :2]
    moved[:, 0] = end_pose[0] + c * offset[:, 0] - s * offset[:, 1]
    moved[:, 1] = end_pose[1] + s * offset[:, 0] + c * offset[:, 1]
    for a, b in ((3, 4), (6, 7)):
        if moved.shape[1] > b:
            u, w = moved[:, a].copy(), moved[:, b].copy()
            moved[:, a], moved[:, b] = c * u - s * w, s * u + c * w
    heading = np.unwrap(moved[:, 2])
    moved[:, 2] = heading - heading[0] + end_pose[2]
    return moved


def mirror(segment: np.ndarray) -> np.ndarray:
    """``segment`` reflected about its start heading line: the same rest-to-rest
    motion turning the other way (omega negated)."""
    segment = np.asarray(segment, dtype=float)
    origin, heading = segment[0, :2], segment[0, 2]
    c, s = math.cos(heading), math.sin(heading)
    out = segment.copy()

    def reflect(x, y):
        # into the start frame, flip its lateral axis, back out
        u = c * x + s * y
        w = -s * x + c * y
        return c * u + s * w, s * u - c * w

    out[:, 0], out[:, 1] = reflect(segment[:, 0] - origin[0], segment[:, 1] - origin[1])
    out[:, 0] += origin[0]
    out[:, 1] += origin[1]
    out[:, 2] = 2.0 * heading - np.unwrap(segment[:, 2])
    for a, b in ((3, 4), (6, 7)):
        if out.shape[1] > b:
            out[:, a], out[:, b] = reflect(segment[:, a], segment[:, b])
    out[:, 5] = -segment[:, 5]
    return out


def turn_in_place(pose: np.ndarray, angle: float, duration: float, dt: float, width: int) -> np.ndarray:
    """Samples of a smooth (minimum-jerk) turn by ``angle`` on the spot, after
    ``pose`` (the first sample is not repeated)."""
    steps = max(int(round(duration / dt)), 1)
    tau = np.arange(1, steps + 1) / steps
    shape = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    rate = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / (steps * dt)
    out = np.zeros((steps, width))
    out[:, 0], out[:, 1] = pose[0], pose[1]
    out[:, 2] = pose[2] + angle * shape
    out[:, 5] = angle * rate
    return out


def _box_violation(states: np.ndarray, box_min, box_max, margin: float) -> float:
    low = np.asarray(box_min, dtype=float) + margin
    high = np.asarray(box_max, dtype=float) - margin
    xy = states[:, :2]
    return float(max(np.max(low - xy), np.max(xy - high), 0.0))


def extend_identification_reference(
    states: np.ndarray,
    dt: float,
    identified_duration: float,
    segments: list[np.ndarray],
    *,
    box_min=None,
    box_max=None,
    margin: float = 0.15,
    turn_duration: float = 0.75,
) -> tuple[np.ndarray, dict]:
    """The identified part of ``states`` followed by ``segments``. Without a box
    the segments are attached end to end as they come; with one, the order,
    mirrors and turns in place that keep the reference inside it (by
    ``margin``) are searched, preferring fewer and smaller turns, then an end
    near the start. Returns the states and the placement chosen."""
    keep = int(round(float(identified_duration) / float(dt))) + 1
    if keep > len(states):
        raise ValueError(f"identified_duration {identified_duration} s exceeds the reference ({len(states)} samples).")
    base = np.asarray(states[:keep], dtype=float)
    width = base.shape[1]
    if box_min is None or box_max is None:
        extended = base
        for segment in segments:
            extended = np.vstack([extended, attach(extended[-1, :3], segment)[1:]])
        return extended, {"order": list(range(len(segments))), "mirrored": [False] * len(segments),
                          "turns_deg": [0.0] * len(segments), "box_violation": None}

    variants = [(segment, mirror(segment)) for segment in segments]
    best = None
    for order in itertools.permutations(range(len(segments))):
        for flips in itertools.product((False, True), repeat=len(segments)):
            for turns in itertools.product(TURN_ANGLES, repeat=len(segments)):
                extended = base
                for index, flip, turn in zip(order, flips, turns):
                    if turn != 0.0:
                        extended = np.vstack([extended, turn_in_place(extended[-1], turn, turn_duration, dt, width)])
                    extended = np.vstack([extended, attach(extended[-1, :3], variants[index][flip])[1:]])
                violation = _box_violation(extended, box_min, box_max, margin)
                score = (
                    round(violation, 3),
                    sum(turn != 0.0 for turn in turns),
                    round(sum(abs(turn) for turn in turns), 3),
                    float(np.linalg.norm(extended[-1, :2] - extended[0, :2])),
                )
                if best is None or score < best[0]:
                    best = (score, extended, order, flips, turns)
    score, extended, order, flips, turns = best
    return extended, {
        "order": list(order),
        "mirrored": [bool(flip) for flip in flips],
        "turns_deg": [round(math.degrees(turn), 1) for turn in turns],
        "box_violation": score[0],
        "end_to_start": round(score[3], 3),
    }


def extended_payload(
    identification_pickle: Path, tuning_pickles: list[Path], *, box_min=None, box_max=None, **placement
) -> dict:
    """The identification pickle's payload with the tuning trajectories appended
    after its identified phase (``identified_duration`` unchanged, so the fit
    still uses the identified part only), placed inside the box if given."""
    with Path(identification_pickle).open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or payload.get("identified_duration") is None:
        raise ValueError(f"{identification_pickle} records no identified_duration to append after.")
    dt = float(payload["dt"])
    segments = []
    for path in tuning_pickles:
        with Path(path).open("rb") as file:
            tuning = pickle.load(file)
        if abs(float(tuning["dt"]) - dt) > 1e-9:
            raise ValueError(f"{path} is sampled at {tuning['dt']} s, the identification reference at {dt} s.")
        segments.append(np.asarray(tuning["reference_states"], dtype=float))
    states, chosen = extend_identification_reference(
        payload["reference_states"], dt, payload["identified_duration"], segments,
        box_min=box_min, box_max=box_max, **placement,
    )
    return {
        "reference_states": states,
        "dt": dt,
        "identified_duration": float(payload["identified_duration"]),
        "appended_tuning_trajectories": [Path(path).name for path in tuning_pickles],
        "placement": chosen,
        "extends": Path(identification_pickle).name,
    }
