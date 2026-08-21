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

Two camera modes. ``--camera-mode static`` is the overhead shot: it looks down
from ``--elevation`` degrees, framed once to fit the whole reference.
``--camera-mode follow`` is a chase camera that rides at the robot's own height
and keeps it in frame, which is the only view that shows the caster and both
wheels at once -- ``--follow-angle`` swings it round the robot's *own heading*
(0 directly behind, 90 abeam, 45 between the two), with ``--follow-distance``
and ``--follow-height`` setting how far back and how high it rides. The camera
lags the robot slightly rather than locking to it, so the robot moves within
the frame instead of the world swinging around it.

``--follow-elevation`` states the chase camera's angle outright instead of
letting it follow from the height and stand-off: 0 is level with the robot,
positive looks down on it, and **negative goes under the floor**. MuJoCo renders
the ground plane one-sided, so a camera below it looks straight up through the
floor at the caster ball, both tires and their contact patches -- there is no
other way to watch the caster roll. ``--follow-distance`` is then the
straight-line range rather than a distance along the floor.

``--layout split`` puts several cameras on the *same* run in one frame: the
overhead shot on the left, and one chase pane per ``--follow-angle`` stacked
down the right. ``--follow-angle``, ``--follow-distance`` and
``--follow-elevation`` are all per pane -- each takes one value that every
chase pane shares, or one value per angle -- so a split screen can hold two
genuinely different shots rather than the same shot at two bearings. One deployment, one video -- the panes are rendered together
and tiled, not stitched afterwards, so they cannot drift out of step. The panes
tile the frame exactly, which at the default 1920x1080 means a 960x1080
overview beside two 960x540 chase views; one ``--follow-angle`` gives two
960x1080 panes instead. It costs one render per pane, so a three-pane video
takes about three times as long as a single one.

``--slowmo-factor`` stretches the run out: the plant is sampled that many times
more often and the frames are played at ``--fps``, so a 4x video is four times
as long *and* four times as finely resolved -- which is what makes the wheel
hops and the yaw wobble visible rather than just slower.

Examples
--------
    # the iteration's identification trajectory, deployed controller
    uv run python scripts/render_deployment_video.py \
        --trajectory "Pololu Data/exp02/iteration_05/identification_trajectory/identification_trajectory_20260817_190841_bridge.JSN"

    # a tuning trajectory under the static-gain baseline, flatter camera
    uv run python scripts/render_deployment_video.py \
        --trajectory "Pololu Data/exp02/iteration_05/tuning_trajectories/tuning_trajectory_00_20260817_190928.pkl" \
        --gains static --elevation 65 --out visualize/tuning_00_static.mp4

    # chase camera over the robot's shoulder, quarter speed to see it hop
    uv run python scripts/render_deployment_video.py \
        --trajectory "Pololu Data/exp02/iteration_05/identification_trajectory/identification_trajectory_20260817_190841_bridge.JSN" \
        --camera-mode follow --follow-angle 45 --slowmo-factor 4

    # split screen: overview left, chase from behind and abeam stacked right
    ... --layout split --follow-angle 0 90

    # split screen with one chase pane, looking up through the floor at the caster
    ... --layout split --follow-angle 90 --follow-elevation -35

    # two chase panes from genuinely different places: close in from behind,
    # and abeam from under the floor
    ... --layout split --follow-angle 0 90 --follow-elevation 10 -60 --follow-distance 0.12 0.20

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
    parser.add_argument("--trajectory", default="Pololu Data/mujcoco tweaked/iteration_01/identification_trajectory/identification_trajectory_20260820_113707_bridge.JSN")
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
        "right, tiling the frame exactly (default single)",
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

    # MUJOCO_GL is read when mujoco is first imported, so it has to be set
    # before anything that pulls the plant in.
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
    """The cameras this run is rendered from, one per pane.

    The three chase settings are per pane: each takes one value that every pane
    shares, or one value per ``--follow-angle``. That is what lets a split
    screen hold two genuinely different shots -- abeam from under the floor
    beside close-in from behind -- rather than the same shot at two bearings.
    A count that is neither 1 nor the number of angles is a typo, and is
    refused rather than zipped short.
    """
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
    # One camera fills the frame, so it takes the first of each list. Said out
    # loud rather than dropped quietly, since the defaults are a split screen's.
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
    # The layout is in the name so a split screen does not overwrite the
    # single-camera video of the same run.
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
