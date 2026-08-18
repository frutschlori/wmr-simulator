"""Render a video of one MuJoCo deployment of an active-learning trajectory.

The run is the real thing, not a re-simulation for the camera: the hidden
MuJoCo plant driven by the firmware port, exactly what the ``simulate-deployment``
and ``benchmark`` stages put on the (simulated) SD card. The iteration the
trajectory came from is found by walking up to its ``ROBOTCFG.CFG``, so the
controller and the identified robot parameters are the ones that trajectory was
actually deployed under.

``--trajectory`` takes either a designed reference pickle or a Pololu ``.JSN``
(bridged or not) -- a pickle is converted through the same exporter the robot's
JSN is written by, so the two drive identically.

``--gains both`` puts the iteration's two controller options in one video, on
the **same seed**, so the hand placement and the sensor noise are common random
numbers and the difference between the two traces is the controller and nothing
else. Both variants are independent closed-loop deployments -- own plant, own
firmware, own EKF -- but a MuJoCo scene holds one robot, so only the
parametrized run can be the rendered chassis; the static one is drawn beside it
as a marker from its own recorded true pose. The traces keep their colours in
single-variant videos too: parametrized orange, static pink.

The camera is static and looks down from ``--elevation`` degrees (80 by
default, i.e. just off vertical, so the chassis and its heading stay readable),
framed to fit the whole reference. The reference is drawn in full from the
first frame; the robot's trace grows behind it as it drives.

Examples
--------
    # the iteration's identification trajectory, deployed controller
    uv run python scripts/render_deployment_video.py \
        --trajectory "Pololu Data/exp02/iteration_05/identification_trajectory/identification_trajectory_20260817_190841_bridge.JSN"

    # a tuning trajectory under the static-gain baseline, flatter camera
    uv run python scripts/render_deployment_video.py \
        --trajectory "Pololu Data/exp02/iteration_05/tuning_trajectories/tuning_trajectory_00_20260817_190928.pkl" \
        --gains static --elevation 65 --out visualize/tuning_00_static.mp4

    # both controllers in one video, paired seed
    ... --gains both

    # force software rendering if EGL is unavailable
    ... --gl osmesa
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", default="Pololu Data/exp02/iteration_01/identification_trajectory/identification_trajectory_20260817_190112_bridge.JSN")
    parser.add_argument(
        "--gains",
        choices=("parametrized", "static", "both"),
        default="static",
        help="Controller to drive it under: the iteration's deployed gain parametrization "
        "(ROBOTCFG.CFG + GAINMLP.JSN), its static-gain baseline (ROBOTCFG_static.CFG), or both in one "
        "video on the same seed. Default both.",
    )
    parser.add_argument(
        "--config",
        default="Pololu Data/exp02/iteration_03/ROBOTCFG_static.CFG",
        help="ROBOTCFG.CFG / robot_config.yaml to drive instead of the iteration's own (skips --gains)",
    )
    parser.add_argument("--out", default=None, help="Output mp4 (default: <iteration>/visualize/<name>_<gains>.mp4)")
    parser.add_argument("--seed", type=int, default=0, help="Hand-placement draw and sensor noise (default 0)")
    parser.add_argument("--elevation", type=float, default=80.0, help="Camera elevation in degrees (default 80)")
    parser.add_argument("--azimuth", type=float, default=90.0, help="Camera azimuth in degrees (default 90)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--log-dir", default=None, help="Keep the run's TRxx log here (default: discard it)")
    parser.add_argument("--gl", default=None, help="MUJOCO_GL backend: egl | osmesa | glfw")
    args = parser.parse_args(argv)

    # MUJOCO_GL is read when mujoco is first imported, so it has to be set
    # before anything that pulls the plant in.
    os.environ.setdefault("MUJOCO_GL", args.gl or "egl")
    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl

    import tempfile

    from wmr_simulator.mujoco_sim.render import (
        find_iteration_root,
        render_deployment,
        render_deployment_comparison,
        resolve_controller,
    )

    trajectory = Path(args.trajectory)
    if not trajectory.is_file():
        raise SystemExit(f"No such trajectory: {trajectory}")
    if args.gains == "both" and args.config is not None:
        raise SystemExit("--gains both compares an iteration's two controllers; it cannot take a --config override.")

    iteration_root = None if args.config is not None else find_iteration_root(trajectory)
    out_path = Path(args.out) if args.out is not None else _default_out_path(trajectory, iteration_root, args.gains)

    print(f"trajectory: {trajectory}")
    if iteration_root is not None:
        print(f"iteration:  {iteration_root}")
    print(f"camera:     elevation {args.elevation:.0f} deg, azimuth {args.azimuth:.0f} deg")

    shot = dict(
        seed=args.seed,
        elevation=args.elevation,
        azimuth=args.azimuth,
        fps=args.fps,
        width=args.width,
        height=args.height,
        log_dir=args.log_dir,
    )

    if args.gains == "both":
        runs, num_frames = render_deployment_comparison(iteration_root, trajectory, out_path, **shot)
        for run in runs:
            role = "chassis" if run.live else "marker"
            label = f"{run.controller.variant} ({role}, {run.controller.colour_name})"
            print(f"controller: {label} -- {run.controller.description}")
        print(f"{out_path}  ({num_frames} frames, {runs[0].result.duration:.2f} s @ {args.fps} fps)")
        _print_offset(runs[0].result)
        print("--- ground truth (not in the log), same seed for both ---")
        for run in runs:
            print(f"  {run.controller.variant:<13} {_truth_line(run.result)}")
        return 0

    with tempfile.TemporaryDirectory() as staging:
        if iteration_root is None:
            config_path = Path(args.config)
            style = {}
            print(f"controller: {config_path.name}")
        else:
            controller = resolve_controller(iteration_root, args.gains, staging)
            config_path = controller.config_path
            style = {"trace_rgba": controller.rgba, "trace_label": controller.label}
            print(f"controller: {controller.description}")

        result, num_frames = render_deployment(config_path, trajectory, out_path, **shot, **style)

    print(f"{out_path}  ({num_frames} frames, {result.duration:.2f} s @ {args.fps} fps)")
    _print_offset(result)
    print("--- ground truth (not in the log) ---")
    print(f"  {_truth_line(result)}")
    return 0


def _default_out_path(trajectory: Path, iteration_root: Path | None, gains: str) -> Path:
    if iteration_root is None:
        return Path("visualize") / f"{trajectory.stem}.mp4"
    return iteration_root / "visualize" / f"{trajectory.stem}_{gains}.mp4"


def _print_offset(result) -> None:
    offset = result.start_offset
    print(f"start offset: {1000 * offset[0]:+.0f} mm, {1000 * offset[1]:+.0f} mm, {offset[2]:+.3f} rad")


def _truth_line(result) -> str:
    return (
        f"tracking RMSE {result.tracking_rmse:.4f} m, max {result.tracking_max:.4f} m, "
        f"final pose error {result.final_pose_error:.4f} m, "
        f"max |duty| {result.max_duty:.3f} ({100 * result.duty_saturated_fraction:.1f}% saturated)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
