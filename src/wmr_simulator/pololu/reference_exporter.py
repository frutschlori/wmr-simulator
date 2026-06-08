from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferenceTrajectory:
    path: Path
    reference_states: np.ndarray
    dt: float


def load_latest_reference_trajectory(
    input_dir: str | Path,
    *,
    recursive: bool = False,
) -> ReferenceTrajectory:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")

    pattern = "**/*.pkl" if recursive else "*.pkl"
    candidates = sorted(input_dir.glob(pattern), key=lambda path: (_latest_reference_activity_time(path), path.name))
    if not candidates:
        raise ValueError(f"No pickle trajectories found in {input_dir}")

    return load_reference_trajectory(candidates[-1])


def load_reference_trajectory(path: str | Path) -> ReferenceTrajectory:
    path = Path(path)
    with path.open("rb") as file:
        payload = pickle.load(file)

    reference_states, dt = _unpack_reference_payload(payload, path)

    reference_states = np.asarray(reference_states, dtype=float)
    if reference_states.ndim != 2 or reference_states.shape[1] < 6:
        raise ValueError(
            "Reference states must be a 2D array with at least six columns "
            "[x, y, theta, vx, vy, omega]."
        )
    if reference_states.shape[0] < 2:
        raise ValueError("Reference trajectory must contain at least two states.")

    return ReferenceTrajectory(path=path, reference_states=reference_states, dt=float(dt))


def format_pololu_reference(
    trajectory: ReferenceTrajectory,
    *,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
) -> dict[str, Any]:
    states = trajectory.reference_states[:, :3]
    actions = _actions_from_reference_states(trajectory.reference_states[:-1])

    return {
        "result": [
            {
                "time_stamp": _round_float(time_stamp, decimals),
                "cost": _round_float(cost, decimals),
                "num_states": int(states.shape[0]),
                "states": _round_nested(states, decimals),
                "num_actions": int(actions.shape[0]),
                "actions": _round_nested(actions, decimals),
                "dt": _round_float(trajectory.dt, decimals),
                "start": _round_nested(states[0], decimals),
                "goal": _round_nested(states[-1], decimals),
            }
        ]
    }


def export_latest_reference(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    output_name: str | None = None,
    recursive: bool = False,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
) -> Path:
    trajectory = load_latest_reference_trajectory(input_dir, recursive=recursive)
    formatted = format_pololu_reference(
        trajectory,
        cost=cost,
        time_stamp=time_stamp,
        decimals=decimals,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"{trajectory.path.stem}.JSN"
    output_path = output_dir / output_name
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(formatted, file, indent=2)
        file.write("\n")
    return output_path


def _unpack_reference_payload(payload, path: Path) -> tuple[np.ndarray, float]:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a reference pickle dictionary with 'reference_states' and 'dt'.")
    if "reference_states" not in payload:
        raise ValueError(f"Reference pickle dictionary in {path} must contain a 'reference_states' key.")
    if "dt" not in payload:
        raise ValueError(f"Reference pickle dictionary in {path} must contain a 'dt' key.")
    dt = float(payload["dt"])
    if dt <= 0.0:
        raise ValueError(f"Reference dt in {path} must be positive, got {dt}.")
    return payload["reference_states"], dt


def _latest_reference_activity_time(path: Path) -> float:
    stat = path.stat()
    return max(stat.st_mtime, stat.st_ctime)


def _actions_from_reference_states(reference_states: np.ndarray) -> np.ndarray:
    theta = reference_states[:, 2]
    vx = reference_states[:, 3]
    vy = reference_states[:, 4]
    linear_speed = vx * np.cos(theta) + vy * np.sin(theta)
    omega = reference_states[:, 5]
    return np.column_stack([linear_speed, omega])


def _round_float(value: float, decimals: int) -> float:
    return float(np.round(float(value), decimals))


def _round_nested(values: np.ndarray, decimals: int):
    return np.round(np.asarray(values, dtype=float), decimals).tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Format the latest synthesized trajectory pickle as a Pololu reference JSN."
    )
    parser.add_argument("--input_dir", type=str, default="trajectory_exports")
    parser.add_argument("--output_dir", type=str, default="Pololu Data/References")
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--recursive", action="store_true", help="Search for pickle files recursively.")
    parser.add_argument("--cost", type=float, default=100.0)
    parser.add_argument("--time-stamp", type=float, default=0.0)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args(argv)

    output_path = export_latest_reference(
        args.input_dir,
        args.output_dir,
        output_name=args.output_name,
        recursive=args.recursive,
        cost=args.cost,
        time_stamp=args.time_stamp,
        decimals=args.decimals,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
