"""Render a video of one MuJoCo deployment of a trajectory.

The run is the hidden MuJoCo plant driven by the firmware port, as in the
``simulate-deployment`` and ``benchmark`` stages. The iteration a trajectory
belongs to is found by walking up to its ``ROBOTCFG.CFG``; ``--config`` drives
another config instead. ``--trajectory`` takes a reference pickle or a JSN.

``--gains both`` drives the iteration's static and parametrized controllers on
the same seed; the static run is drawn as a marker beside the rendered robot.

Cameras: ``--camera-mode static`` looks down from ``--elevation`` on the whole
reference; ``--camera-mode follow`` is a chase camera at ``--follow-angle``
round the robot's heading (0 behind, 90 abeam), ``--follow-distance`` and
``--follow-height`` away, or at ``--follow-elevation`` (negative: from under
the floor). ``--layout split`` shows the overview on the left and one chase
pane per ``--follow-angle`` on the right. ``--slowmo-factor`` samples the
plant that many times more often at the same ``--fps``.

Examples
--------
    # a benchmark reference under an iteration's static controller
    python scripts/render_deployment_video.py \
        --trajectory experiments/exp01/iteration_03/benchmark/eight_fast.JSN --gains static

    # chase camera, quarter speed
    ... --layout single --camera-mode follow --follow-angle 45 --slowmo-factor 4

    # split screen: overview left, chase from behind and abeam stacked right
    ... --layout split --follow-angle 0 90

    # force software rendering if EGL is unavailable
    ... --gl osmesa
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", required=True, help="Reference pickle or Pololu JSN inside an iteration folder.")
    parser.add_argument(
        "--gains",
        choices=("parametrized", "static", "both"),
        default="static",
        help="Controller to drive it under: the iteration's deployed gain parametrization "
        "(ROBOTCFG.CFG + GAINMLP.JSN), its static-gain baseline (ROBOTCFG_static.CFG), or both in one "
        "video on the same seed. Default static.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="ROBOTCFG.CFG / robot_config.yaml to drive instead of the iteration's own (skips --gains)",
    )
    parser.add_argument("--out", default=None, help="Output mp4 (default: <iteration>/visualize/<name>_<gains>.mp4)")
    parser.add_argument("--seed", type=int, default=0, help="Hand-placement draw and sensor noise (default 0)")
    parser.add_argument(
        "--layout",
        choices=("single", "split"),
        default="split",
        help="single: one camera fills the frame, chosen by --camera-mode. split: a split screen of the "
        "same run -- the overhead shot on the left, one chase pane per --follow-angle stacked down the "
        "right, tiling the frame exactly (default split)",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("static", "follow"),
        default="follow",
        help="static: one overhead shot framed on the whole reference. follow: a chase camera riding "
        "at the robot's own height, framed on the robot. Ignored under --layout split, which uses both "
        "(default follow)",
    )
    parser.add_argument("--elevation", type=float, default=80.0, help="Static camera elevation in degrees (default 80)")
    parser.add_argument("--azimuth", type=float, default=90.0, help="Static camera azimuth in degrees (default 90)")
    parser.add_argument(
        "--follow-angle",
        type=float,
        nargs="+",
        default=[0, 90],
        help="Where the chase camera sits, in degrees round the robot's own heading: 0 directly behind, "
        "90 abeam, 45 between the two; negative swings it the other way. Under --layout split, one pane "
        "per angle given (default: 30 alone, or 0 90 under --layout split)",
    )
    parser.add_argument(
        "--follow-distance",
        type=float,
        nargs="+",
        default=[0.15, 0.25],
        help="How far back the chase camera rides, in metres along the floor. Under --layout split, "
        "one value shared by every chase pane or one per --follow-angle (default 0.18)",
    )
    parser.add_argument(
        "--follow-height",
        type=float,
        default=0.045,
        help="Chase camera height above the floor in metres. The robot is ~0.025 m tall; the default "
        "rakes in just low enough that the caster ball clears the chassis skirt, so it and the wheels "
        "show together. Drop to ~0.03 for a plainer view of the caster (default 0.045)",
    )
    parser.add_argument(
        "--follow-elevation",
        type=float,
        nargs="+",
        default=[-35, 0],
        help="Chase camera elevation in degrees, stated outright instead of following from "
        "--follow-height: 0 level with the robot, positive above it, negative BELOW THE FLOOR, which "
        "MuJoCo renders one-sided so the camera looks up through it at the caster and the contact "
        "patches. --follow-distance is then the straight-line range. Under --layout split, one value "
        "shared by every chase pane or one per --follow-angle (default -35)",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--slowmo-factor",
        type=float,
        default=3.0,
        help="Stretch the run out by this factor, e.g. 4 for quarter speed. The plant is sampled that "
        "many times more often, so the extra frames really resolve the motion (default 1 = real time)",
    )
    parser.add_argument("--log-dir", default=None, help="Keep the run's TRxx log here (default: discard it)")
    parser.add_argument("--gl", default=None, help="MUJOCO_GL backend: egl | osmesa | glfw")
    args = parser.parse_args(argv)

    # MUJOCO_GL has to be set before mujoco is imported.
    os.environ.setdefault("MUJOCO_GL", args.gl or "egl")
    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl

    import tempfile

    from wmr_simulator.mujoco_sim.render import (
        View,
        find_iteration_root,
        pane_rects,
        render_deployment,
        render_deployment_comparison,
        resolve_controller,
        split_screen_views,
    )

    trajectory = Path(args.trajectory)
    if not trajectory.is_file():
        raise SystemExit(f"No such trajectory: {trajectory}")
    if args.gains == "both" and args.config is not None:
        raise SystemExit("--gains both compares an iteration's two controllers; it cannot take a --config override.")

    iteration_root = None if args.config is not None else find_iteration_root(trajectory)
    out_path = (
        Path(args.out) if args.out is not None else _default_out_path(trajectory, iteration_root, args.gains, args.layout)
    )

    print(f"trajectory: {trajectory}")
    if iteration_root is not None:
        print(f"iteration:  {iteration_root}")
    if args.slowmo_factor <= 0:
        raise SystemExit(f"--slowmo-factor must be positive, got {args.slowmo_factor}")

    views = _build_views(args, View, split_screen_views)
    for view, rect in zip(views, pane_rects(len(views), args.width, args.height)):
        where = f"{rect[2]}x{rect[3]} at ({rect[0]}, {rect[1]})" if len(views) > 1 else f"{args.width}x{args.height}"
        print(f"camera:     {view.label}  [{where}]")
    if args.slowmo_factor != 1.0:
        print(f"slow motion: {args.slowmo_factor:g}x, sampled at {round(args.fps * args.slowmo_factor)} Hz")

    shot = dict(
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
        log_dir=args.log_dir,
        views=views,
        slowmo_factor=args.slowmo_factor,
    )

    if args.gains == "both":
        runs, num_frames = render_deployment_comparison(iteration_root, trajectory, out_path, **shot)
        for run in runs:
            role = "chassis" if run.live else "marker"
            label = f"{run.controller.variant} ({role}, {run.controller.colour_name})"
            print(f"controller: {label} -- {run.controller.description}")
        _print_written(out_path, num_frames, args, runs[0].result)
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

    _print_written(out_path, num_frames, args, result)
    _print_offset(result)
    print("--- ground truth (not in the log) ---")
    print(f"  {_truth_line(result)}")
    return 0


def _build_views(args, View, split_screen_views) -> list:
    """The cameras this run is rendered from, one per pane (the chase
    settings take one shared value or one per ``--follow-angle``)."""
    if args.layout == "split":
        try:
            return split_screen_views(
                args.follow_angle,
                elevation=args.elevation,
                azimuth=args.azimuth,
                follow_distance=args.follow_distance,
                follow_height=args.follow_height,
                follow_elevation=args.follow_elevation,
            )
        except ValueError as error:
            raise SystemExit(f"{error}: --follow-angle {' '.join(f'{angle:g}' for angle in args.follow_angle)}")

    if args.camera_mode == "static":
        return [View(camera_mode="static", elevation=args.elevation, azimuth=args.azimuth)]
    # A single camera takes the first value of each list.
    extra = [
        flag
        for flag, values in (
            ("--follow-angle", args.follow_angle),
            ("--follow-distance", args.follow_distance),
            ("--follow-elevation", args.follow_elevation),
        )
        if len(values) > 1
    ]
    if extra:
        print(f"note:       one camera, so only the first {', '.join(extra)} is used (--layout split uses them all)")
    return [
        View(
            camera_mode="follow",
            follow_angle=args.follow_angle[0],
            follow_distance=args.follow_distance[0],
            follow_height=args.follow_height,
            follow_elevation=args.follow_elevation[0],
        )
    ]


def _print_written(out_path, num_frames: int, args, result) -> None:
    played = num_frames / max(1, args.fps)
    line = f"{out_path}  ({num_frames} frames, {played:.2f} s @ {args.fps} fps"
    if args.slowmo_factor != 1.0:
        line += f", {args.slowmo_factor:g}x slow motion of a {result.duration:.2f} s run"
    print(line + ")")


def _default_out_path(trajectory: Path, iteration_root: Path | None, gains: str, layout: str) -> Path:
    suffix = "_split" if layout == "split" else ""
    if iteration_root is None:
        return Path("visualize") / f"{trajectory.stem}{suffix}.mp4"
    return iteration_root / "visualize" / f"{trajectory.stem}_{gains}{suffix}.mp4"


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
