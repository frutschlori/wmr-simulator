"""The held-out trajectory set the joint loop's gains are scored on.

The loop's own trajectories are its *training* set: they are decision variables,
they move every round, and a gain vector scored on them is scored on a problem
that no longer exists a round later. Nothing in that answers "are these gains
better" -- only "are these gains better on the curve we happen to be holding".
This module supplies the other half: a fixed, neutral set of trajectories the
loop never optimizes, which is what makes the best-iterate selection and the
stopping rule mean something.

Deliberately *not* ``trajectory_optimization.load_reference_states_exports``.
That reader stacks its trajectories into one array, so it requires them to share
a sample count -- correct there, where the set is one batched design. A
validation set is the opposite: it wants a short, diverse collection, and
diversity includes length (the curated set mixes 101- and 161-sample curves).
So these stay a list and are scored one at a time. There are a handful of them
and they are scored without gradients, so the loop costs nothing that matters;
what it buys is that a validation set never has to be padded or truncated to
fit, which would change the very trajectories it exists to hold fixed.
"""

import glob
import os
import pickle
from typing import NamedTuple

import numpy as np


class ValidationTrajectory(NamedTuple):
    name: str
    reference_states: np.ndarray    # (N, 8)
    start_offsets: np.ndarray       # (R, 3)
    dt: float


def load_validation_trajectories(directory: str) -> list[ValidationTrajectory]:
    """Read a directory of exported trajectory pickles as a validation set.

    Every trajectory must carry the start offsets it is to be scored under: the
    tracking error the gain objective measures is dominated by driving the start
    offset out, so a set without them scores a different problem than the one
    the tuner solves. Sample counts may differ between trajectories; ``dt`` may
    not, since it is the rollout's timestep and not a property of the curve.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.pkl")))
    if not paths:
        raise ValueError(f"No trajectory pickles (*.pkl) in {directory}.")

    trajectories = []
    for path in paths:
        with open(path, "rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a reference-states export payload.")
        if payload.get("start_offsets") is None:
            raise ValueError(
                f"{path} carries no start offsets. A validation trajectory is scored under "
                "fixed start poses; without them the score is not comparable across rounds."
            )
        offsets = np.asarray(payload["start_offsets"], dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise ValueError(f"{path} start offsets must have shape (R, 3); got {offsets.shape}.")
        trajectories.append(
            ValidationTrajectory(
                name=os.path.splitext(os.path.basename(path))[0],
                reference_states=np.asarray(payload["reference_states"], dtype=float),
                start_offsets=offsets,
                dt=float(payload["dt"]),
            )
        )

    timesteps = {trajectory.dt for trajectory in trajectories}
    if len(timesteps) != 1:
        raise ValueError(f"Validation trajectories in {directory} differ in dt: {sorted(timesteps)}.")
    realization_counts = {trajectory.start_offsets.shape[0] for trajectory in trajectories}
    if len(realization_counts) != 1:
        raise ValueError(
            f"Validation trajectories in {directory} carry differing realization counts: "
            f"{sorted(realization_counts)}. They are averaged into one score, so the average "
            "would silently weight them unequally."
        )
    return trajectories
