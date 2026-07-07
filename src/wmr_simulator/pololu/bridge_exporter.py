"""Append a wait + bridge-back segment to a Pololu reference JSN.

The bridged reference replays the original trajectory, waits at its goal, and
then drives a planned bridge path back to the start pose, so an experiment can
be repeated without repositioning the robot by hand. The bridged JSN is a
regular Pololu reference and is exported next to the unbridged one by default
(``<name>_bridge.JSN``).

The CLI wrapper lives in scripts/pololu_append_bridge_reference.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.pololu.reference_exporter import ReferenceTrajectory, format_pololu_reference
from wmr_simulator.pololu.reference_importer import (
    load_pololu_reference,
    reference_states_from_pololu_reference,
)


def time_grid(duration: float, dt: float) -> np.ndarray:
    steps = int(np.round(duration / dt))
    if steps <= 0:
        raise ValueError("duration must contain at least one dt interval.")
    return np.linspace(0.0, steps * dt, steps + 1)


def wait_reference_states(pose: np.ndarray, wait_time: float, dt: float) -> np.ndarray:
    intervals = int(np.round(wait_time / dt))
    if intervals <= 0:
        return np.empty((0, 8), dtype=float)
    states = np.zeros((intervals, 8), dtype=float)
    states[:, :3] = pose[:3]
    return states


def bridge_reference_states(start_pose: np.ndarray, goal_pose: np.ndarray, bridge_time: float, dt: float) -> np.ndarray:
    return compute_reference_trajectory(
        start=start_pose[:3],
        goal=goal_pose[:3],
        intermediate_waypoints=[],
        time=time_grid(bridge_time, dt),
    )[0]


def bridged_output_path(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}_bridge.JSN")


def append_bridge_reference(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    wait_time: float,
    bridge_time: float,
    result_index: int = 0,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
    plot_path: str | Path | None = None,
) -> Path:
    """Write ``<name>_bridge.JSN`` (default: next to the input) and return its path."""
    if output_path is None:
        output_path = bridged_output_path(input_path)
    reference = load_pololu_reference(input_path, result_index=result_index)
    reference_states = reference_states_from_pololu_reference(
        reference,
        final_action="zero",
        acceleration="finite-difference",
        dtype=float,
    )
    reference_states = reference_states.copy()
    reference_states[-1, 3:] = 0.0

    wait_states = wait_reference_states(reference_states[-1, :3], wait_time, reference.dt)
    bridge_states = bridge_reference_states(
        start_pose=reference_states[-1, :3],
        goal_pose=reference_states[0, :3],
        bridge_time=bridge_time,
        dt=reference.dt,
    )
    full_reference_states = np.vstack([reference_states, wait_states, bridge_states[1:]])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if plot_path is not None:
        plot_bridged_reference(
            reference_states=reference_states,
            wait_states=wait_states,
            bridge_states=bridge_states,
            dt=reference.dt,
            output_path=plot_path,
        )
    formatted = format_pololu_reference(
        ReferenceTrajectory(
            path=Path(input_path),
            reference_states=full_reference_states,
            dt=reference.dt,
        ),
        cost=cost,
        time_stamp=time_stamp,
        decimals=decimals,
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(formatted, file, indent=2)
        file.write("\n")
    from wmr_simulator.pololu.reference_exporter import check_firmware_limits

    for warning in check_firmware_limits(
        num_states=full_reference_states.shape[0],
        file_bytes=output_path.stat().st_size,
        label=str(output_path),
    ):
        print(f"WARNING: {warning}")
    return output_path


def append_bridge_references_in_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    wait_time: float,
    bridge_time: float,
    result_index: int = 0,
    cost: float = 100.0,
    time_stamp: float = 0.0,
    decimals: int = 6,
) -> list[tuple[Path, Path]]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-bridged references (they live next to the unbridged ones).
    input_paths = sorted(path for path in input_dir.glob("*.JSN") if not path.stem.endswith("_bridge"))
    if not input_paths:
        raise ValueError(f"No unbridged JSN files found in: {input_dir}")

    outputs = []
    for input_path in input_paths:
        output_path = output_dir / f"{input_path.stem}_bridge.JSN"
        plot_path = output_path.with_suffix(".pdf")
        append_bridge_reference(
            input_path,
            output_path,
            wait_time=wait_time,
            bridge_time=bridge_time,
            result_index=result_index,
            cost=cost,
            time_stamp=time_stamp,
            decimals=decimals,
            plot_path=plot_path,
        )
        outputs.append((output_path, plot_path))
    return outputs


def plot_bridged_reference(
    reference_states: np.ndarray,
    wait_states: np.ndarray,
    bridge_states: np.ndarray,
    dt: float,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_reference_states = np.vstack([reference_states, wait_states, bridge_states[1:]])
    time = np.arange(full_reference_states.shape[0], dtype=float) * dt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), gridspec_kw={"height_ratios": [1.3, 1.0]})
    ax_xy, ax_state = axes

    ax_xy.plot(reference_states[:, 0], reference_states[:, 1], linewidth=1.4, label="Reference")
    if len(wait_states):
        ax_xy.scatter(wait_states[:, 0], wait_states[:, 1], s=12, label="Wait")
    ax_xy.plot(bridge_states[:, 0], bridge_states[:, 1], linewidth=1.4, label="Bridge")
    ax_xy.scatter(reference_states[0, 0], reference_states[0, 1], marker="o", s=35, color="black", label="Start")
    ax_xy.scatter(reference_states[-1, 0], reference_states[-1, 1], marker="x", s=45, color="black", label="Original goal")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Bridged Pololu Reference")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True)
    ax_xy.legend(loc="best")

    ax_state.plot(time, full_reference_states[:, 0], label="x")
    ax_state.plot(time, full_reference_states[:, 1], label="y")
    ax_state.plot(time, full_reference_states[:, 2], label="theta")
    ax_state.set_xlabel("time [s]")
    ax_state.set_ylabel("state")
    ax_state.grid(True)
    ax_state.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append a wait and bridge-back path to a Pololu reference JSN.")
    parser.add_argument("--input", type=str, default="Pololu Data/References/dt comparison/")
    parser.add_argument("--output", type=str, default="Pololu Data/References/dt comparison/bridged/")
    parser.add_argument("--plot-output", type=str, default=None)
    parser.add_argument("--wait-time", type=float, default=2.0)
    parser.add_argument("--bridge-time", type=float, default=10.0)
    parser.add_argument("--result-index", type=int, default=0)
    parser.add_argument("--cost", type=float, default=100.0)
    parser.add_argument("--time-stamp", type=float, default=0.0)
    parser.add_argument("--decimals", type=int, default=6)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if input_path.is_dir():
        if args.plot_output is not None:
            parser.error("--plot-output can only be used when --input points to a single file.")
        output_dir = Path(args.output) if args.output is not None else input_path
        saved_paths = append_bridge_references_in_directory(
            input_path,
            output_dir,
            wait_time=args.wait_time,
            bridge_time=args.bridge_time,
            result_index=args.result_index,
            cost=args.cost,
            time_stamp=args.time_stamp,
            decimals=args.decimals,
        )
        for output_path, plot_path in saved_paths:
            print(output_path)
            print(plot_path)
        return 0

    if not input_path.is_file():
        parser.error(f"--input must point to a JSN file or directory: {input_path}")
    output_path = Path(args.output) if args.output is not None else bridged_output_path(input_path)
    plot_path = Path(args.plot_output) if args.plot_output is not None else output_path.with_suffix(".pdf")
    saved_path = append_bridge_reference(
        input_path,
        output_path,
        wait_time=args.wait_time,
        bridge_time=args.bridge_time,
        result_index=args.result_index,
        cost=args.cost,
        time_stamp=args.time_stamp,
        decimals=args.decimals,
        plot_path=plot_path,
    )
    print(saved_path)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
