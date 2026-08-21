"""Render a MuJoCo deployment to an mp4: the run itself, not a re-simulation.

The rollout is exactly the one ``deploy.run_deployment`` drives -- hidden plant
plus the firmware port, the same thing the ``simulate-deployment`` and
``benchmark`` stages put on the (simulated) SD card. This module only watches
it: the recorders here are ``run_deployment`` observers, so there is one copy
of the control loop and the video cannot drift from what the pipeline records.

The overlay is drawn straight into MuJoCo's scene as decor geoms, which is why
the reference is visible in full from the first frame while the robot's trace
grows behind it: the reference is a fixed polyline, the trace is appended one
point per frame from the plant's *true* pose. The trace is ground truth and is
therefore only ever a picture -- nothing here feeds the pipeline.

**Comparing two controllers in one video** (``render_deployment_comparison``)
drives the iteration's two controller options on the **same seed**, so the hand
placement and the sensor noise are common random numbers and the difference
between the two traces is the controller and nothing else -- the same pairing
the benchmark stage records under. Each variant is its own full
``run_deployment``: its own plant, its own firmware and EKF, its own closed
loop. Nothing here replays anybody's trajectory into a robot.

There is one plant and one robot in a MuJoCo scene, so only one of the two can
be the *rendered chassis*: the parametrized run is driven while the frames are
being written, and the static run -- driven first, by ``record_pose_track``, in
exactly the same way -- is drawn beside it from its own recorded true pose as a
marker in its trace's colour. That is a drawing order, not a difference in how
the two were simulated.

A MuJoCo scene has no text, so the legend is composited onto the rendered
frames with Pillow. It is built once and blended in per frame, since nothing
about it changes during a run.

Offscreen rendering needs an OpenGL backend, chosen through MUJOCO_GL *before*
mujoco is imported: ``egl`` (headless GPU, the default here), ``osmesa`` (CPU
software, slow but works anywhere), ``glfw`` (needs a display).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wmr_simulator.mujoco_sim.deploy import DeploymentResult
from wmr_simulator.mujoco_sim.plant import MujocoPlant

# Straight down reads as a floor plan and loses the robot's body entirely, so
# the default leans the camera just off vertical: enough to see the chassis and
# which way it is pointing, flat enough that the path geometry is undistorted.
DEFAULT_ELEVATION = 80.0
# Azimuth 90 puts world +x to the right and +y up the frame, i.e. the same
# orientation as every trajectory plot in visualize/.
DEFAULT_AZIMUTH = 90.0
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Fraction of headroom left around the reference when framing the shot.
CAMERA_MARGIN = 1.2

STATIC_CAMERA = "static"
FOLLOW_CAMERA = "follow"
CAMERA_MODES = (STATIC_CAMERA, FOLLOW_CAMERA)

# The chase camera rides at roughly the robot's own height and rakes in almost
# flat, which is the point of it: the caster ball (12.7 mm, 40 mm behind the
# axle) and both wheels are only ever visible together from down there, and a
# top-down shot cannot show contact geometry at all. The height is measured:
# the chassis skirt hides the caster from 0.06 m up, and it clears the skirt
# from about 0.045 m down (0.03 m puts it in plain view, at a very high
# horizon). Both wheels show at any of these; a wheel is clearest abeam.
DEFAULT_FOLLOW_DISTANCE = 0.18
DEFAULT_FOLLOW_HEIGHT = 0.045
# 0 sits the camera squarely behind the robot, 90 abeam of it, 45 between the
# two -- measured round the robot's own heading, so it holds through every turn.
DEFAULT_FOLLOW_ANGLE = 0.0
# The robot spans about 0-25 mm above the floor (axle 16 mm, deck top 20 mm,
# caster centre 6.6 mm), so aiming here keeps body and contacts in frame.
FOLLOW_LOOKAT_HEIGHT = 0.015
# A camera nailed to the raw pose holds the robot perfectly still and swings
# the world around it instead, hiding exactly the wobble this view exists to
# show; a camera that lags too much loses the robot out of frame. The two axes
# want very different amounts, so they get their own time constants: heading is
# what whips the whole world sideways and is smoothed hard, position only needs
# enough lag to let the robot shift within the frame. Position lag is roughly
# tau * speed, so at 1 m/s this is ~8 cm -- under one robot length, against a
# stand-off of DEFAULT_FOLLOW_DISTANCE. In simulated seconds, so the framing is
# identical at every --slowmo-factor.
FOLLOW_POSITION_TAU = 0.08
FOLLOW_YAW_TAU = 0.30
# The reference and trace capsules are sized for a shot framed on a whole
# trajectory; from 0.18 m away a 7 mm tube is wider than the robot is tall and
# simply buries it, so the chase camera draws them thinner.
FOLLOW_LINE_SCALE = 0.3

# >1 stretches the run out: the plant is sampled that many times more often and
# the extra frames are played at the same rate, so the motion is genuinely
# resolved rather than the same frames merely held longer.
DEFAULT_SLOWMO_FACTOR = 1.0

STATIC_VARIANT = "static"
PARAMETRIZED_VARIANT = "parametrized"

# Deliberately not white: the chassis is a white cylinder, and a white
# reference under it is indistinguishable from the robot at the zoom a whole
# trajectory has to be framed at.
REFERENCE_RGBA = (0.55, 0.95, 0.35, 1.0)
# One colour per controller, the same in a single-variant video and in a
# comparison, so a trace means the same thing across every video of a run.
VARIANT_RGBA = {
    PARAMETRIZED_VARIANT: (0.98, 0.40, 0.10, 1.0),
    STATIC_VARIANT: (0.98, 0.30, 0.78, 1.0),
}
# What to call those colours when a run is reported on the terminal.
VARIANT_COLOUR_NAMES = {PARAMETRIZED_VARIANT: "orange", STATIC_VARIANT: "pink"}
# How a run names itself in the video's legend.
VARIANT_LABELS = {
    PARAMETRIZED_VARIANT: "parametrized gains",
    STATIC_VARIANT: "static gains",
}
REFERENCE_LABEL = "reference"
DEFAULT_TRACE_LABEL = "robot"
DEFAULT_TRACE_RGBA = VARIANT_RGBA[PARAMETRIZED_VARIANT]

REFERENCE_RADIUS = 0.006
TRACE_RADIUS = 0.007
# Above the floor plane and stacked, so nothing z-fights with the ground or
# with the layer under it.
REFERENCE_HEIGHT = 0.002
REPLAY_TRACE_HEIGHT = 0.004
TRACE_HEIGHT = 0.006
# The replayed robot is a marker, not a body: a ring around where its chassis
# is, with a heading tick. A ring rather than a disc, and wider than the
# chassis, because the two runs spend most of a good comparison nearly on top
# of each other -- a disc at the chassis's own radius is simply hidden under
# the white body from a top-down camera, which is exactly the case where the
# viewer most needs to see that both are still there.
# The legend is sized off the frame height rather than fixed, so it stays
# legible at 640x360 and does not swell at 4K.
LEGEND_FONT_DIVISOR = 34
LEGEND_MIN_FONT_SIZE = 13
LEGEND_MARGIN_FRACTION = 0.02
LEGEND_PANEL_RGBA = (14, 20, 28, 195)
LEGEND_TEXT_RGB = (238, 241, 245)

REPLAY_MARKER_RADIUS_FACTOR = 1.35
REPLAY_MARKER_SEGMENTS = 28
REPLAY_MARKER_HEIGHT = 0.012
FALLBACK_CHASSIS_RADIUS = 0.0485


@dataclass(frozen=True)
class DeploymentController:
    """Which controller of an iteration a run is driven under.

    ``config_path`` is the config as the firmware reads it *off its own card*:
    ``FirmwareConfig.from_file`` picks up a ``GAINMLP.JSN`` sitting next to it,
    mirroring ``sdlog.rs``, so the static variant cannot be driven from the
    iteration root -- the deployed network is right there and would be applied
    to it. It is staged into its own directory instead, which is exactly the
    swap the real SD card needs.
    """

    variant: str
    config_path: Path
    iteration_root: Path
    description: str

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return VARIANT_RGBA[self.variant]

    @property
    def colour_name(self) -> str:
        return VARIANT_COLOUR_NAMES[self.variant]

    @property
    def label(self) -> str:
        """How this controller names itself in the video's legend."""
        return VARIANT_LABELS[self.variant]


@dataclass(frozen=True)
class VariantRun:
    """One controller's run, as it appears in a comparison video."""

    controller: DeploymentController
    result: DeploymentResult
    rgba: tuple[float, float, float, float]
    # Which run carries the rendered chassis; the other is drawn as a marker
    # from its own recorded pose. Both are independent closed-loop deployments
    # of the same plant on the same seed -- this is a drawing role only.
    live: bool


def find_iteration_root(trajectory: str | Path) -> Path:
    """The active-learning iteration a trajectory file belongs to.

    Identified by its ``ROBOTCFG.CFG``, which every iteration has and no
    trajectory subdirectory does, so this works from a pickle in
    ``tuning_trajectories/``, a bridged JSN in ``identification_trajectory/``
    and the baseline in ``benchmark/`` alike.
    """
    trajectory = Path(trajectory).resolve()
    for candidate in trajectory.parents:
        if (candidate / "ROBOTCFG.CFG").is_file():
            return candidate
    raise FileNotFoundError(
        f"No active-learning iteration above {trajectory}: no directory among its parents holds a "
        "ROBOTCFG.CFG. Pass --config to drive a trajectory that is not part of an experiment."
    )


def ships_two_controllers(iteration_root: str | Path) -> bool:
    """Whether this iteration has a static baseline distinct from its deployed controller.

    ``ROBOTCFG_static.CFG`` is written by ``finalize`` only when the previous
    tuning ran an independent static tune beside a parametrized one, so its
    presence *is* the question "was the parametrization worth it". Without it
    there is one controller -- iteration 1, or an experiment with the
    parametrization off -- and nothing to compare it against.
    """
    from wmr_simulator.active_learning.experiment import IterationPaths

    return IterationPaths(root=Path(iteration_root)).robotcfg_static_cfg.is_file()


def resolve_controller(iteration_root: str | Path, variant: str, staging_dir: str | Path) -> DeploymentController:
    """The iteration's deployed (``parametrized``) or static-gain controller.

    Mirrors what the benchmark stage compares. An iteration without a
    ``ROBOTCFG_static.CFG`` shipped only one controller, so both variants
    resolve to the same config there, and the run is labelled as what it is.
    """
    from wmr_simulator.active_learning.experiment import IterationPaths

    if variant not in VARIANT_RGBA:
        raise ValueError(f"variant must be one of {sorted(VARIANT_RGBA)}, got {variant!r}")
    paths = IterationPaths(root=Path(iteration_root))
    if not paths.robotcfg_cfg.is_file():
        raise FileNotFoundError(f"Missing {paths.robotcfg_cfg}")

    if variant == PARAMETRIZED_VARIANT or not paths.robotcfg_static_cfg.is_file():
        network = " + GAINMLP.JSN" if paths.gainmlp_jsn.is_file() else ""
        if variant == STATIC_VARIANT:
            description = (
                f"{paths.robotcfg_cfg.name}{network} (identity) -- this iteration ships no separate "
                "ROBOTCFG_static.CFG, so its deployed controller is the static one"
            )
        else:
            description = f"{paths.robotcfg_cfg.name}{network}"
        return DeploymentController(variant, paths.robotcfg_cfg, paths.root, description)

    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = Path(shutil.copy2(paths.robotcfg_static_cfg, staging_dir / paths.robotcfg_cfg.name))
    return DeploymentController(
        variant,
        staged,
        paths.root,
        f"{paths.robotcfg_static_cfg.name} (staged without a GAINMLP.JSN beside it)",
    )


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------


class FrameObserver:
    """A ``run_deployment`` observer that samples the run at video cadence.

    The frame schedule is derived from the plant on the first call rather than
    up front: the plant is constructed inside ``run_deployment``, and its
    timestep is what a frame interval has to be a whole number of.
    """

    def __init__(self, fps: int = DEFAULT_FPS) -> None:
        self.fps = int(fps)
        self.poses: list[np.ndarray] = []
        self._steps_per_frame: int | None = None

    def __call__(self, tick, plant: MujocoPlant) -> None:
        if self._steps_per_frame is None:
            self._steps_per_frame = max(1, round(1.0 / (self.fps * plant.timestep)))
            self._start(plant)
        if tick.index % self._steps_per_frame:
            return
        self.poses.append(np.asarray(plant.pose(), dtype=float))
        self._frame(plant)

    @property
    def num_frames(self) -> int:
        return len(self.poses)

    def pose_track(self) -> np.ndarray:
        """``(frames, 3)`` of true ``(x, y, yaw)``, one row per video frame."""
        return np.asarray(self.poses, dtype=float).reshape(-1, 3)

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameObserver":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- hooks ------------------------------------------------------------

    def _start(self, plant: MujocoPlant) -> None:
        """Called once, with the plant, before the first frame is sampled."""

    def _frame(self, plant: MujocoPlant) -> None:
        """Called once per frame, after the pose has been appended."""


class DeploymentVideoRecorder(FrameObserver):
    """A ``run_deployment`` observer that writes the run out as an mp4.

    ``replay`` overlays another already-recorded run of the same reference: its
    trace grows frame for frame beside this one's and its robot is drawn as a
    marker. It is what puts two controllers in one video.
    """

    def __init__(
        self,
        reference_xy: np.ndarray,
        dt: float,
        out_path: str | Path,
        *,
        elevation: float = DEFAULT_ELEVATION,
        azimuth: float = DEFAULT_AZIMUTH,
        fps: int = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        trace_rgba=DEFAULT_TRACE_RGBA,
        trace_label: str = DEFAULT_TRACE_LABEL,
        replay_poses: np.ndarray | None = None,
        replay_rgba=VARIANT_RGBA[STATIC_VARIANT],
        replay_label: str = VARIANT_LABELS[STATIC_VARIANT],
        camera_mode: str = STATIC_CAMERA,
        follow_distance: float = DEFAULT_FOLLOW_DISTANCE,
        follow_height: float = DEFAULT_FOLLOW_HEIGHT,
        follow_angle: float = DEFAULT_FOLLOW_ANGLE,
        slowmo_factor: float = DEFAULT_SLOWMO_FACTOR,
    ) -> None:
        if camera_mode not in CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {CAMERA_MODES}, got {camera_mode!r}")
        if slowmo_factor <= 0.0:
            raise ValueError(f"slowmo_factor must be positive, got {slowmo_factor}")
        # Slow motion is a *sampling* rate, not a playback rate: the plant is
        # watched slowmo times more often and the frames are written at the
        # requested fps, so a 2x video is twice as long and twice as finely
        # resolved. Dropping the writer's fps instead would give a video of the
        # same length with fewer frames in it.
        super().__init__(fps=max(1, round(fps * float(slowmo_factor))))
        self.output_fps = max(1, int(fps))
        import imageio.v2 as imageio
        import mujoco

        self._mujoco = mujoco
        self._imageio = imageio
        self.reference_xy = np.asarray(reference_xy, dtype=float)[:, :2]
        self.dt = float(dt)
        self.out_path = Path(out_path)
        self.elevation = float(elevation)
        self.azimuth = float(azimuth)
        self.camera_mode = str(camera_mode)
        self.follow_distance = float(follow_distance)
        self.follow_height = float(follow_height)
        self.follow_angle = float(follow_angle)
        self.slowmo_factor = float(slowmo_factor)
        # Height and stand-off are the two knobs; the elevation that makes the
        # camera actually look at the robot from there follows from them, so
        # the two can never be set to disagree with each other.
        rise = self.follow_height - FOLLOW_LOOKAT_HEIGHT
        self._follow_elevation = math.degrees(math.atan2(rise, self.follow_distance))
        self._follow_range = math.hypot(self.follow_distance, rise)
        self._follow_position_alpha = 1.0 - math.exp(-1.0 / (self.fps * FOLLOW_POSITION_TAU))
        self._follow_yaw_alpha = 1.0 - math.exp(-1.0 / (self.fps * FOLLOW_YAW_TAU))
        self._follow_pose: tuple[float, float, float] | None = None
        self._line_scale = FOLLOW_LINE_SCALE if self.camera_mode == FOLLOW_CAMERA else 1.0
        # h264 needs even dimensions.
        self.width = int(width) + int(width) % 2
        self.height = int(height) + int(height) % 2
        self.trace_rgba = tuple(trace_rgba)
        self.trace_label = str(trace_label)
        self.replay_poses = None if replay_poses is None else np.asarray(replay_poses, dtype=float).reshape(-1, 3)
        self.replay_rgba = tuple(replay_rgba)
        self.replay_label = str(replay_label)

        self._renderer = None
        self._writer = None
        self._camera = None
        self._scene_option = None
        self._chassis_radius = FALLBACK_CHASSIS_RADIUS
        self._legend = _legend_overlay(self.legend_entries(), self.width, self.height)

    def legend_entries(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        """``(label, rgba)`` per thing drawn, in the order they are drawn."""
        entries = [(REFERENCE_LABEL, REFERENCE_RGBA)]
        if self.replay_poses is not None:
            entries.append((self.replay_label, self.replay_rgba))
        entries.append((self.trace_label, self.trace_rgba))
        return entries

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- hooks ------------------------------------------------------------

    def _start(self, plant: MujocoPlant) -> None:
        mujoco = self._mujoco
        model = plant.model
        # mujoco.Renderer refuses to render larger than the model's offscreen
        # framebuffer, which is a <visual><global> property of the XML.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)

        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width, max_geom=self._geom_budget(model))
        self._scene_option = mujoco.MjvOption()
        self._camera = self._build_camera(model)
        self._chassis_radius = _chassis_radius(mujoco, model)

        if self.camera_mode == STATIC_CAMERA:
            lookat, distance = _frame_shot(
                self.reference_xy,
                azimuth=self.azimuth,
                elevation=self.elevation,
                fovy=float(model.vis.global_.fovy),
                aspect=self.width / self.height,
            )
            self._camera.lookat[:] = lookat
            self._camera.distance = distance

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = self._imageio.get_writer(self.out_path, fps=self.output_fps, macro_block_size=1)

    def _frame(self, plant: MujocoPlant) -> None:
        mujoco = self._mujoco
        if self.camera_mode == FOLLOW_CAMERA:
            self._update_follow_camera(plant.pose())
        self._renderer.update_scene(plant.data, camera=self._camera, scene_option=self._scene_option)
        scene = self._renderer.scene
        scale = self._line_scale
        _add_polyline(
            mujoco, scene, self.reference_xy, REFERENCE_HEIGHT, scale * REFERENCE_RADIUS, REFERENCE_RGBA
        )
        if self.replay_poses is not None and len(self.replay_poses):
            # The two runs drive the same reference for the same duration, so
            # they share a frame index; clamp anyway rather than let a
            # one-frame rounding difference truncate the replay.
            index = min(self.num_frames, len(self.replay_poses))
            _add_polyline(
                mujoco,
                scene,
                self.replay_poses[:index, :2],
                REPLAY_TRACE_HEIGHT,
                scale * TRACE_RADIUS,
                self.replay_rgba,
            )
            _add_robot_marker(
                mujoco, scene, self.replay_poses[index - 1], self._chassis_radius, self.replay_rgba, scale
            )
        _add_polyline(
            mujoco, scene, self.pose_track()[:, :2], TRACE_HEIGHT, scale * TRACE_RADIUS, self.trace_rgba
        )
        frame = self._renderer.render()
        margin = max(1, round(LEGEND_MARGIN_FRACTION * self.height))
        _composite(frame, self._legend, margin, margin)
        self._writer.append_data(frame)

    # -- internals --------------------------------------------------------

    def _geom_budget(self, model) -> int:
        """One geom per polyline segment, so the budget has to cover the traces.

        The scene's ``maxgeom`` is fixed when the ``Renderer`` is built, and a
        polyline that runs into it is silently truncated -- a trace that stops
        growing mid-video and looks like the robot stopped. The run cannot last
        longer than the reference does, so size the traces from that and double
        it.
        """
        frames = int(self.fps * len(self.reference_xy) * self.dt)
        traces = 2 if self.replay_poses is not None else 1
        marker = REPLAY_MARKER_SEGMENTS + 2
        return len(self.reference_xy) + 2 * traces * frames + marker + model.ngeom + 1000

    def _build_camera(self, model):
        mujoco = self._mujoco
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        if self.camera_mode == STATIC_CAMERA:
            camera.azimuth = self.azimuth
            # MuJoCo measures elevation downward-negative, so a request of 80 deg
            # "looking down from above" is -80 in the camera's own convention.
            camera.elevation = -self.elevation
            # lookat and distance are framed against the reference in _start,
            # once the model's fovy is available.
        else:
            # Only the azimuth and the lookat move with the robot; how far back
            # and how high the camera rides is fixed for the whole run.
            camera.elevation = -self._follow_elevation
            camera.distance = self._follow_range
        return camera

    def _update_follow_camera(self, pose) -> None:
        """Point the chase camera at the robot for this frame.

        MuJoCo's free camera sits at
        ``lookat - distance * [cos(az)cos(el), sin(az)cos(el), sin(el)]``, so the
        camera is placed entirely by its azimuth: ``azimuth = yaw`` puts it
        squarely behind the robot, and ``+ follow_angle`` swings it round toward
        the robot's right -- 90 abeam, 45 between the two. Because the azimuth
        is measured off the robot's *own* heading it holds through every turn,
        which a fixed azimuth cannot do.
        """
        x, y, yaw = (float(value) for value in pose)
        if self._follow_pose is None:
            self._follow_pose = (x, y, yaw)
        else:
            prev_x, prev_y, prev_yaw = self._follow_pose
            # Take the short way round: a raw difference across the +-pi
            # branch cut would swing the camera a full turn the wrong way.
            delta_yaw = (yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
            move = self._follow_position_alpha
            turn = self._follow_yaw_alpha
            self._follow_pose = (
                prev_x + move * (x - prev_x),
                prev_y + move * (y - prev_y),
                prev_yaw + turn * delta_yaw,
            )

        cam_x, cam_y, cam_yaw = self._follow_pose
        self._camera.azimuth = math.degrees(cam_yaw) + self.follow_angle
        self._camera.elevation = -self._follow_elevation
        self._camera.lookat[:] = [cam_x, cam_y, FOLLOW_LOOKAT_HEIGHT]
        self._camera.distance = self._follow_range


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def record_pose_track(
    robot_config: str | Path,
    trajectory: str | Path,
    *,
    seed: int = 0,
    fps: int = DEFAULT_FPS,
    log_dir: str | Path | None = None,
) -> tuple[DeploymentResult, np.ndarray]:
    """Drive one deployment without rendering; returns its true pose per frame."""
    observer = FrameObserver(fps=fps)
    result = _drive(robot_config, trajectory, observer, seed=seed, log_dir=log_dir)
    return result, observer.pose_track()


def render_deployment(
    robot_config: str | Path,
    trajectory: str | Path,
    out_path: str | Path,
    *,
    seed: int = 0,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    log_dir: str | Path | None = None,
    trace_rgba=DEFAULT_TRACE_RGBA,
    trace_label: str = DEFAULT_TRACE_LABEL,
    replay_poses: np.ndarray | None = None,
    replay_rgba=VARIANT_RGBA[STATIC_VARIANT],
    replay_label: str = VARIANT_LABELS[STATIC_VARIANT],
    camera_mode: str = STATIC_CAMERA,
    follow_distance: float = DEFAULT_FOLLOW_DISTANCE,
    follow_height: float = DEFAULT_FOLLOW_HEIGHT,
    follow_angle: float = DEFAULT_FOLLOW_ANGLE,
    slowmo_factor: float = DEFAULT_SLOWMO_FACTOR,
) -> tuple[DeploymentResult, int]:
    """Drive one deployment and write the mp4.

    Returns the run's ``DeploymentResult`` and the number of frames written.

    ``log_dir`` keeps the ``TRxx`` the run produces. It defaults to a temporary
    directory that is thrown away: a video is not data collection, and dropping
    a log into an iteration's ``data/`` would hand identification a run nobody
    asked for.

    ``camera_mode="follow"`` swaps the static overhead shot for a chase camera
    riding at ``follow_height`` above the floor, ``follow_distance`` back, and
    ``follow_angle`` degrees round the robot's own heading (0 behind, 90 abeam).
    ``slowmo_factor`` above 1 samples the plant that much more often and plays
    the frames at ``fps``, so the video is that many times longer.
    """
    from wmr_simulator.pololu.reference_importer import load_reference

    reference = load_reference(trajectory)
    with DeploymentVideoRecorder(
        reference.states[:, :2],
        reference.dt,
        out_path,
        elevation=elevation,
        azimuth=azimuth,
        fps=fps,
        width=width,
        height=height,
        trace_rgba=trace_rgba,
        trace_label=trace_label,
        replay_poses=replay_poses,
        replay_rgba=replay_rgba,
        replay_label=replay_label,
        camera_mode=camera_mode,
        follow_distance=follow_distance,
        follow_height=follow_height,
        follow_angle=follow_angle,
        slowmo_factor=slowmo_factor,
    ) as recorder:
        result = _drive(robot_config, trajectory, recorder, seed=seed, log_dir=log_dir)
        num_frames = recorder.num_frames
    return result, num_frames


def render_deployment_comparison(
    iteration_root: str | Path,
    trajectory: str | Path,
    out_path: str | Path,
    *,
    seed: int = 0,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    log_dir: str | Path | None = None,
    camera_mode: str = STATIC_CAMERA,
    follow_distance: float = DEFAULT_FOLLOW_DISTANCE,
    follow_height: float = DEFAULT_FOLLOW_HEIGHT,
    follow_angle: float = DEFAULT_FOLLOW_ANGLE,
    slowmo_factor: float = DEFAULT_SLOWMO_FACTOR,
) -> tuple[list[VariantRun], int]:
    """Both of an iteration's controllers, in one video, on the same seed.

    Paired seeds make the hand placement and the sensor noise common random
    numbers, so the two traces differ by the controller and nothing else -- the
    same pairing the benchmark stage records the two variants under.

    Refuses an iteration that ships only one controller: there both variants
    resolve to the same config, and the video would be one run drawn twice.
    """
    import tempfile

    iteration_root = Path(iteration_root)
    if not ships_two_controllers(iteration_root):
        raise FileNotFoundError(
            f"{iteration_root} has no ROBOTCFG_static.CFG, so it ships a single controller and there "
            "is nothing to compare. Render one variant instead."
        )

    with tempfile.TemporaryDirectory() as staging:
        replay = resolve_controller(iteration_root, STATIC_VARIANT, staging)
        live = resolve_controller(iteration_root, PARAMETRIZED_VARIANT, staging)

        # The marker is indexed frame-for-frame against the rendered run, so
        # it has to be sampled on the cadence that run is *rendered* at -- under
        # slow motion that is fps * slowmo_factor, not fps, or the replay track
        # runs out partway through and its robot freezes.
        replay_result, replay_poses = record_pose_track(
            replay.config_path,
            trajectory,
            seed=seed,
            fps=max(1, round(fps * slowmo_factor)),
            log_dir=log_dir,
        )
        live_result, num_frames = render_deployment(
            live.config_path,
            trajectory,
            out_path,
            seed=seed,
            elevation=elevation,
            azimuth=azimuth,
            fps=fps,
            width=width,
            height=height,
            log_dir=log_dir,
            trace_rgba=live.rgba,
            trace_label=live.label,
            replay_poses=replay_poses,
            replay_rgba=replay.rgba,
            replay_label=replay.label,
            camera_mode=camera_mode,
            follow_distance=follow_distance,
            follow_height=follow_height,
            follow_angle=follow_angle,
            slowmo_factor=slowmo_factor,
        )

    return (
        [
            VariantRun(live, live_result, live.rgba, live=True),
            VariantRun(replay, replay_result, replay.rgba, live=False),
        ],
        num_frames,
    )


def _drive(robot_config, trajectory, observer, *, seed: int, log_dir) -> DeploymentResult:
    import tempfile

    from wmr_simulator.mujoco_sim.deploy import run_deployment

    if log_dir is not None:
        return run_deployment(robot_config, trajectory, log_dir, seed=seed, observer=observer)
    with tempfile.TemporaryDirectory() as scratch:
        return run_deployment(robot_config, trajectory, scratch, seed=seed, observer=observer)


# ---------------------------------------------------------------------------
# Scene decoration
# ---------------------------------------------------------------------------


def _add_polyline(mujoco, scene, points_xy: np.ndarray, z: float, radius: float, rgba) -> None:
    """Append a polyline to the scene as connector capsules."""
    points_xy = np.asarray(points_xy, dtype=float)
    if len(points_xy) < 2:
        return
    points = np.column_stack([points_xy, np.full(len(points_xy), z)])
    for start, end in zip(points[:-1], points[1:]):
        if not _connect(mujoco, scene, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end, rgba):
            return


def _add_robot_marker(mujoco, scene, pose, chassis_radius: float, rgba, line_scale: float = 1.0) -> None:
    """The other run's robot: a ring around its chassis, with a heading tick.

    A ring rather than a look-alike chassis for legibility, not for honesty --
    both runs are real deployments. The two spend most of a good comparison
    nearly on top of each other, and an opaque body at the chassis's own radius
    would simply hide under the rendered one from a top-down camera.
    """
    x, y, yaw = (float(value) for value in pose)
    radius = REPLAY_MARKER_RADIUS_FACTOR * chassis_radius
    angles = np.linspace(0.0, 2.0 * math.pi, REPLAY_MARKER_SEGMENTS + 1)
    ring = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles)])
    _add_polyline(mujoco, scene, ring, REPLAY_MARKER_HEIGHT, line_scale * 0.6 * TRACE_RADIUS, rgba)
    # Which way it is pointing, since a ring alone cannot say. Drawn outside
    # the ring so it clears the live chassis when the two runs coincide.
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    tick = np.array([[x, y]]) + np.outer([1.0, 1.5], radius * heading)
    _add_polyline(mujoco, scene, tick, REPLAY_MARKER_HEIGHT, line_scale * 0.6 * TRACE_RADIUS, rgba)


def _legend_overlay(entries, width: int, height: int) -> np.ndarray:
    """The legend as an ``(h, w, 4)`` RGBA patch, drawn once per video.

    A MuJoCo scene carries no text, so this is composited onto the rendered
    frames rather than added to the scene. Nothing in it changes during a run,
    which is why it is built up front instead of per frame.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_size = max(LEGEND_MIN_FONT_SIZE, round(height / LEGEND_FONT_DIVISOR))
    font = ImageFont.load_default(size=font_size)
    pad = round(0.7 * font_size)
    row_gap = round(0.55 * font_size)
    swatch_width = round(2.4 * font_size)
    swatch_height = max(3, round(0.24 * font_size))
    label_gap = round(0.6 * font_size)

    boxes = [font.getbbox(label) for label, _ in entries]
    text_height = max(box[3] - box[1] for box in boxes)
    row_height = max(text_height, swatch_height)
    panel_width = 2 * pad + swatch_width + label_gap + max(box[2] - box[0] for box in boxes)
    panel_height = 2 * pad + len(entries) * row_height + (len(entries) - 1) * row_gap
    # A legend wider than the frame is a sizing bug, not something to crop.
    panel_width = min(panel_width, width)
    panel_height = min(panel_height, height)

    image = Image.new("RGBA", (panel_width, panel_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [0, 0, panel_width - 1, panel_height - 1], radius=round(0.45 * font_size), fill=LEGEND_PANEL_RGBA
    )

    centre = pad + row_height / 2
    for label, rgba in entries:
        colour = tuple(int(round(255 * channel)) for channel in rgba[:3]) + (255,)
        draw.rounded_rectangle(
            [pad, centre - swatch_height / 2, pad + swatch_width, centre + swatch_height / 2],
            radius=swatch_height / 2,
            fill=colour,
        )
        draw.text((pad + swatch_width + label_gap, centre), label, font=font, fill=LEGEND_TEXT_RGB, anchor="lm")
        centre += row_height + row_gap

    return np.asarray(image)


def _composite(frame: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend an RGBA patch into an RGB frame, in place."""
    height = min(overlay.shape[0], frame.shape[0] - y)
    width = min(overlay.shape[1], frame.shape[1] - x)
    if height <= 0 or width <= 0:
        return
    patch = overlay[:height, :width]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    region = frame[y : y + height, x : x + width]
    region[:] = np.round(region * (1.0 - alpha) + patch[..., :3] * alpha).astype(np.uint8)


def _connect(mujoco, scene, geom_type, radius: float, start, end, rgba) -> bool:
    """One decor geom spanning ``start`` to ``end``; False when the scene is full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, geom_type, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32)
    )
    mujoco.mjv_connector(geom, geom_type, radius, start, end)
    scene.ngeom += 1
    return True


def _chassis_radius(mujoco, model) -> float:
    """The plant's own body radius, so a marker is the size of the real robot."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis_deck")
    if geom_id < 0:
        return FALLBACK_CHASSIS_RADIUS
    return float(model.geom_size[geom_id][0])


def _frame_shot(
    points_xy: np.ndarray,
    *,
    azimuth: float,
    elevation: float,
    fovy: float,
    aspect: float,
) -> tuple[np.ndarray, float]:
    """``(lookat, distance)`` putting every point of the reference in frame.

    Exact rather than an approximation of the foreshortening: each point is
    projected onto the camera's own right/up axes and the distance is whatever
    makes the worst one fit, in both screen directions.
    """
    points = np.column_stack([np.asarray(points_xy, dtype=float)[:, :2], np.zeros(len(points_xy))])
    lookat = 0.5 * (points.min(axis=0) + points.max(axis=0))

    az = math.radians(azimuth)
    el = math.radians(-elevation)
    forward = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    tan_v = math.tan(0.5 * math.radians(fovy))
    tan_h = aspect * tan_v

    offsets = points - lookat
    depth_at_lookat = offsets @ forward
    # A point fits when |offset . up| <= tan_v * depth, with depth = distance +
    # offset . forward; solving for the distance gives one bound per point.
    needed = np.maximum(
        np.abs(offsets @ up) / tan_v - depth_at_lookat,
        np.abs(offsets @ right) / tan_h - depth_at_lookat,
    )
    return lookat, float(CAMERA_MARGIN * max(needed.max(), 1e-3))
