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
    input_dt: float | None = None,
    recursive: bool = False,
) -> ReferenceTrajectory:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")

    pattern = "**/*.pkl" if recursive else "*.pkl"
    candidates = sorted(input_dir.glob(pattern), key=lambda path: (_latest_reference_activity_time(path), path.name))
    if not candidates:
        raise ValueError(f"No pickle trajectories found in {input_dir}")

    return load_reference_trajectory(candidates[-1], input_dt=input_dt)


def load_reference_trajectory(path: str | Path, *, input_dt: float | None = None) -> ReferenceTrajectory:
    path = Path(path)
    with path.open("rb") as file:
        payload = pickle.load(file)

    reference_states, dt = _unpack_reference_payload(payload)
    if dt is None:
        if input_dt is None:
            raise ValueError(
                f"{path} does not contain a dt field. Pass --input-dt for legacy array-only pickles."
            )
        dt = input_dt

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
    output_dt: float,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
) -> dict[str, Any]:
    step = _downsample_step(trajectory.dt, output_dt)
    sampled = trajectory.reference_states[::step]
    if sampled.shape[0] < 2:
        raise ValueError(
            f"Downsampling from dt={trajectory.dt} to dt={output_dt} leaves fewer than two states."
        )

    states = sampled[:, :3]
    actions = _actions_from_reference_states(sampled[:-1])

    return {
        "result": [
            {
                "time_stamp": _round_float(time_stamp, decimals),
                "cost": _round_float(cost, decimals),
                "num_states": int(states.shape[0]),
                "states": _round_nested(states, decimals),
                "num_actions": int(actions.shape[0]),
                "actions": _round_nested(actions, decimals),
                "dt": _round_float(output_dt, decimals),
                "start": _round_nested(states[0], decimals),
                "goal": _round_nested(states[-1], decimals),
            }
        ]
    }


def export_latest_reference(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    output_dt: float,
    input_dt: float | None = None,
    output_name: str | None = None,
    recursive: bool = False,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
) -> Path:
    trajectory = load_latest_reference_trajectory(input_dir, input_dt=input_dt, recursive=recursive)
    formatted = format_pololu_reference(
        trajectory,
        output_dt=output_dt,
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


def _unpack_reference_payload(payload) -> tuple[np.ndarray, float | None]:
    if isinstance(payload, dict):
        if "reference_states" not in payload:
            raise ValueError("Reference pickle dictionary must contain a 'reference_states' key.")
        return payload["reference_states"], payload.get("dt")
    return payload, None


def _latest_reference_activity_time(path: Path) -> float:
    stat = path.stat()
    return max(stat.st_mtime, stat.st_ctime)


def _downsample_step(input_dt: float, output_dt: float) -> int:
    if input_dt <= 0.0:
        raise ValueError(f"Input dt must be positive, got {input_dt}.")
    if output_dt <= 0.0:
        raise ValueError(f"Output dt must be positive, got {output_dt}.")
    ratio = output_dt / input_dt
    step = int(round(ratio))
    if step < 1:
        raise ValueError(
            f"Output dt ({output_dt}) must be greater than or equal to input dt ({input_dt})."
        )
    if not np.isclose(ratio, step, rtol=1e-7, atol=1e-9):
        raise ValueError(
            f"Output dt ({output_dt}) must be an integer multiple of input dt ({input_dt}); "
            f"got ratio {ratio}."
        )
    return step


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
        description="Format the latest synthesized trajectory pickle as a Pololu reference JSON."
    )
    parser.add_argument("--input_dir", type=str, default="trajectory_exports")
    parser.add_argument("--output_dir", type=str, default="Pololu Data/References")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--input-dt", type=float, default=0.01,
        help="Input time step for legacy array-only pickles that do not contain dt.")
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--recursive", action="store_true", help="Search for pickle files recursively.")
    parser.add_argument("--cost", type=float, default=100.0)
    parser.add_argument("--time-stamp", type=float, default=0.0)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args(argv)

    output_path = export_latest_reference(
        args.input_dir,
        args.output_dir,
        output_dt=args.dt,
        input_dt=args.input_dt,
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
