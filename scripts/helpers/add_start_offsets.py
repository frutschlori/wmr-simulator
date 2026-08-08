"""Attach reproducible randomized start-pose offsets to trajectory pickles.

The gain-tuning loader reads ``start_offsets`` directly from each pickle.  This
helper gives every trajectory realization its own seed, so a directory can be
turned into a fixed validation set without changing the reference states.

Examples:
  uv run python scripts/helpers/add_start_offsets.py trajectory_exports/validation_trajectories
  uv run python scripts/helpers/add_start_offsets.py other_trajectories --seed 17 --dry-run
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_NUM_REALIZATIONS = 4
DEFAULT_OFFSET_RADIUS = 0.1
DEFAULT_OFFSET_ANGLE = 0.3
DEFAULT_SEED = 20_260_805


def sample_offset(seed: int, radius: float, angle: float) -> np.ndarray:
    """Sample ``[dx, dy, dtheta]`` uniformly in the configured pose bounds."""
    rng = np.random.default_rng(seed)
    displacement_radius = radius * np.sqrt(rng.uniform())
    bearing = rng.uniform(-np.pi, np.pi)
    heading = rng.uniform(-angle, angle)
    return np.asarray(
        [
            displacement_radius * np.cos(bearing),
            displacement_radius * np.sin(bearing),
            heading,
        ],
        dtype=float,
    )


def offsets_for_trajectory(
    trajectory_index: int,
    num_realizations: int,
    radius: float,
    angle: float,
    base_seed: int,
) -> tuple[np.ndarray, list[int]]:
    """Return one independently seeded offset per realization."""
    first_seed = base_seed + trajectory_index * num_realizations
    seeds = list(range(first_seed, first_seed + num_realizations))
    return np.stack([sample_offset(seed, radius, angle) for seed in seeds]), seeds


def update_pickle(
    path: Path,
    offsets: np.ndarray,
    seeds: list[int],
    radius: float,
    angle: float,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or "reference_states" not in payload:
        raise ValueError(f"{path} is not a reference-state export payload.")
    if "start_offsets" in payload and not overwrite:
        print(f"Skipped {path}: it already has start_offsets (pass --overwrite to replace them).")
        return False

    payload["start_offsets"] = offsets
    payload["start_offset_realization_seeds"] = seeds
    payload["start_offset_radius"] = radius
    payload["start_offset_angle"] = angle

    if dry_run:
        print(f"Would update {path}: {len(seeds)} offsets, seeds {seeds}.")
        return True

    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as file:
        pickle.dump(payload, file)
        temp_path = Path(file.name)
    os.replace(temp_path, path)
    print(f"Updated {path}: {len(seeds)} offsets, seeds {seeds}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing reference trajectory .pkl files.")
    parser.add_argument("--num-realizations", type=int, default=DEFAULT_NUM_REALIZATIONS)
    parser.add_argument("--offset-radius", type=float, default=DEFAULT_OFFSET_RADIUS)
    parser.add_argument("--offset-angle", type=float, default=DEFAULT_OFFSET_ANGLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="First per-realization seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing start offsets.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without rewriting any files.")
    args = parser.parse_args()

    if not args.directory.is_dir():
        raise SystemExit(f"Not a directory: {args.directory}")
    if args.num_realizations < 1:
        raise SystemExit("--num-realizations must be at least 1.")
    if args.offset_radius < 0.0 or args.offset_angle < 0.0:
        raise SystemExit("--offset-radius and --offset-angle must be non-negative.")

    paths = sorted(args.directory.glob("*.pkl"))
    if not paths:
        raise SystemExit(f"No .pkl files found in {args.directory}")

    updated = 0
    for trajectory_index, path in enumerate(paths):
        offsets, seeds = offsets_for_trajectory(
            trajectory_index,
            args.num_realizations,
            args.offset_radius,
            args.offset_angle,
            args.seed,
        )
        updated += update_pickle(
            path,
            offsets,
            seeds,
            args.offset_radius,
            args.offset_angle,
            args.overwrite,
            args.dry_run,
        )
    print(f"{updated}/{len(paths)} trajectory pickles {'would be ' if args.dry_run else ''}updated.")


if __name__ == "__main__":
    main()
