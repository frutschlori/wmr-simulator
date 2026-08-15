"""Render a MuJoCo rollout of the Pololu plant to an mp4.

Offscreen rendering needs an OpenGL backend, chosen with the MUJOCO_GL
environment variable *before* mujoco is imported:

    egl     GPU, headless        (default here; needs libEGL)
    osmesa  CPU software         (slow, but works anywhere)
    glfw    on-screen window     (needs a display)

Examples
--------
    # 6 s of driving a 1.2 m radius arc, tracking camera
    uv run python scripts/render_mujoco_video.py \
        --duty-left 0.36 --duty-right 0.42 --duration 6 \
        --out visualize/mujoco_arc.mp4

    # force software rendering if EGL is unavailable
    MUJOCO_GL=osmesa uv run python scripts/render_mujoco_video.py

    # drive it from a wheel-speed callback instead of constant duty
    #   (this is what the firmware port will do in step 4 of the plan)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/pololu_calibrated.xml")
    parser.add_argument("--out", default="visualize/mujoco_rollout.mp4")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds of simulated time")
    parser.add_argument("--settle", type=float, default=0.5, help="seconds to settle before driving")
    parser.add_argument("--duty-left", type=float, default=0.25)
    parser.add_argument("--duty-right", type=float, default=0.27)
    parser.add_argument("--omega-max", type=float, default=220.0, help="wheel speed at duty=1 (ctrl = duty*omega_max)")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera", default="track_cam", help="camera name, or 'free' for the default view")
    parser.add_argument("--gl", default=None, help="MUJOCO_GL backend: egl | osmesa | glfw")
    args = parser.parse_args(argv)

    if args.gl:
        os.environ["MUJOCO_GL"] = args.gl
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    # The offscreen framebuffer is a *model* property (<visual><global offwidth>),
    # and mujoco.Renderer refuses to render larger than it. Size it from the
    # request so any resolution works regardless of what the XML declares.
    # h264 also needs even dimensions, so round up to even.
    width = args.width + (args.width % 2)
    height = args.height + (args.height % 2)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    # Contacts and actuator forces are the interesting part of this plant:
    # the wheels losing ground contact is exactly the failure mode to watch.
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    camera = -1 if args.camera == "free" else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera == -1 and args.camera != "free":
        raise SystemExit(f"No camera named {args.camera!r} in {args.model}")

    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    for _ in range(int(args.settle / model.opt.timestep)):
        mujoco.mj_step(model, data)

    data.ctrl[:] = [args.duty_left * args.omega_max, args.duty_right * args.omega_max]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps_per_frame = max(1, round(1.0 / (args.fps * model.opt.timestep)))
    n_frames = int(args.duration * args.fps)

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with imageio.get_writer(out_path, fps=args.fps, macro_block_size=1) as writer:
            for _ in range(n_frames):
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                writer.append_data(renderer.render())

    print(f"{out_path}  ({n_frames} frames, {args.duration:.1f} s @ {args.fps} fps, {width}x{height})")
    print(f"backend: MUJOCO_GL={os.environ['MUJOCO_GL']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
