from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


FinalActionPolicy = Literal["repeat", "zero"]
AccelerationPolicy = Literal["finite-difference", "zero"]


@dataclass(frozen=True)
class PololuReference:
    path: Path
    states: np.ndarray
    actions: np.ndarray
    dt: float
    metadata: dict[str, Any]


def load_pololu_reference(path: str | Path, *, result_index: int = 0) -> PololuReference:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError(f"{path} must contain a top-level 'result' field.")
    results = payload["result"]
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path} must contain a non-empty 'result' list.")
    if result_index < 0 or result_index >= len(results):
        raise ValueError(f"result_index {result_index} is out of range for {len(results)} result entries.")

    result = results[result_index]
    if not isinstance(result, dict):
        raise ValueError(f"Result entry {result_index} in {path} must be an object.")
    for key in ("states", "actions", "dt"):
        if key not in result:
            raise ValueError(f"Result entry {result_index} in {path} must contain '{key}'.")

    states = np.asarray(result["states"], dtype=float)
    actions = np.asarray(result["actions"], dtype=float)
    dt = float(result["dt"])
    _validate_pololu_arrays(path, states, actions, dt, result)

    metadata = {key: value for key, value in result.items() if key not in {"states", "actions"}}
    return PololuReference(
        path=path,
        states=states,
        actions=actions,
        dt=dt,
        metadata=metadata,
    )


def reference_states_from_pololu_reference(
    reference: PololuReference,
    *,
    final_action: FinalActionPolicy = "repeat",
    acceleration: AccelerationPolicy = "finite-difference",
    dtype=np.float32,
) -> np.ndarray:
    actions = _state_aligned_actions(reference.actions, reference.states.shape[0], final_action)
    theta = reference.states[:, 2]
    linear_speed = actions[:, 0]
    omega = actions[:, 1]

    vx = linear_speed * np.cos(theta)
    vy = linear_speed * np.sin(theta)
    ax, ay = _accelerations_from_velocity(vx, vy, reference.dt, acceleration)

    reference_states = np.column_stack(
        [
            reference.states[:, 0],
            reference.states[:, 1],
            theta,
            vx,
            vy,
            omega,
            ax,
            ay,
        ]
    )
    return reference_states.astype(dtype, copy=False)


def export_pololu_reference_pickle(
    input_path: str | Path,
    output_dir: str | Path = "trajectory_exports",
    *,
    output_name: str | None = None,
    result_index: int = 0,
    final_action: FinalActionPolicy = "repeat",
    acceleration: AccelerationPolicy = "finite-difference",
) -> Path:
    reference = load_pololu_reference(input_path, result_index=result_index)
    reference_states = reference_states_from_pololu_reference(
        reference,
        final_action=final_action,
        acceleration=acceleration,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (output_name or f"{reference.path.stem}.pkl")

    payload = {
        "reference_states": reference_states,
        "dt": reference.dt,
        "source": "pololu_reference",
        "source_path": str(reference.path),
        "final_action_policy": final_action,
        "acceleration_policy": acceleration,
        "pololu_metadata": reference.metadata,
    }
    with output_path.open("wb") as file:
        pickle.dump(payload, file)
    return output_path


def _validate_pololu_arrays(
    path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    dt: float,
    result: dict[str, Any],
) -> None:
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError(f"Pololu states in {path} must have shape (N, 3), got {states.shape}.")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Pololu actions in {path} must have shape (M, 2), got {actions.shape}.")
    if states.shape[0] < 2:
        raise ValueError(f"Pololu reference in {path} must contain at least two states.")
    if actions.shape[0] not in {states.shape[0] - 1, states.shape[0]}:
        raise ValueError(
            f"Pololu actions in {path} must contain either N-1 or N rows for N states; "
            f"got {actions.shape[0]} actions for {states.shape[0]} states."
        )
    if dt <= 0.0:
        raise ValueError(f"Pololu dt in {path} must be positive, got {dt}.")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"Pololu states in {path} contain non-finite values.")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"Pololu actions in {path} contain non-finite values.")

    expected_num_states = result.get("num_states")
    if expected_num_states is not None and int(expected_num_states) != states.shape[0]:
        raise ValueError(
            f"num_states in {path} is {expected_num_states}, but states contains {states.shape[0]} rows."
        )
    expected_num_actions = result.get("num_actions")
    if expected_num_actions is not None and int(expected_num_actions) != actions.shape[0]:
        raise ValueError(
            f"num_actions in {path} is {expected_num_actions}, but actions contains {actions.shape[0]} rows."
        )


def _state_aligned_actions(
    actions: np.ndarray,
    num_states: int,
    final_action: FinalActionPolicy,
) -> np.ndarray:
    if actions.shape[0] == num_states:
        return actions
    if final_action == "repeat":
        return np.vstack([actions, actions[-1]])
    if final_action == "zero":
        return np.vstack([actions, np.zeros((1, actions.shape[1]), dtype=actions.dtype)])
    raise ValueError("final_action must be 'repeat' or 'zero'.")


def _accelerations_from_velocity(
    vx: np.ndarray,
    vy: np.ndarray,
    dt: float,
    acceleration: AccelerationPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    if acceleration == "zero":
        return np.zeros_like(vx), np.zeros_like(vy)
    if acceleration != "finite-difference":
        raise ValueError("acceleration must be 'finite-difference' or 'zero'.")
    edge_order = 2 if vx.shape[0] >= 3 else 1
    return np.gradient(vx, dt, edge_order=edge_order), np.gradient(vy, dt, edge_order=edge_order)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Pololu reference JSON/JSN file into a simulator reference pickle."
    )
    parser.add_argument("--input", type=str, default="Pololu Data/References/20cp_bridge.JSN")
    parser.add_argument("--output-dir", type=str, default="trajectory_exports")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--result-index", type=int, default=0)
    parser.add_argument("--final-action", choices=("repeat", "zero"), default="repeat")
    parser.add_argument("--acceleration", choices=("finite-difference", "zero"), default="finite-difference")
    args = parser.parse_args(argv)

    output_path = export_pololu_reference_pickle(
        args.input,
        args.output_dir,
        output_name=args.output_name,
        result_index=args.result_index,
        final_action=args.final_action,
        acceleration=args.acceleration,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
