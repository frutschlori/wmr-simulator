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

**Split screen** (``views``) puts several cameras on the same run. MuJoCo has
no notion of a multi-pane video, but it does not need one: a ``Renderer`` draws
whatever ``mjData`` it is handed, so one pane per view means one renderer per
view all rendering the *same* physics step, tiled into a single frame with
numpy. There is still exactly one deployment behind it, and no second library
stitching videos together afterwards -- the panes are frames of one video from
the moment they are written.

``pane_rects`` sizes the panes so they tile the frame exactly, which is why a
1920x1080 split screen is a 960x1080 overview beside two 960x540 chase views:
nothing is scaled and nothing is letterboxed.

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
# The chase camera's elevation normally *follows* from how high and how far
# back it rides, so the two can never be set to disagree. ``follow_elevation``
# overrides that and states the angle outright instead, with follow_distance
# read as the straight-line range: 0 is level with the robot, negative goes
# under the floor. MuJoCo renders the ground plane one-sided, so a camera below
# it looks straight up through it at the caster ball, both tires and their
# contact patches -- the one view that shows what the robot stands on.
DEFAULT_FOLLOW_ELEVATION = None
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


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class View:
    """One camera of a video: where it sits and what it follows.

    A video is a sequence of these. A single view fills the frame, which is
    what every video was before split screens existed; several are tiled by
    ``pane_rects`` and each gets its own renderer over the same physics step.

    ``camera_mode="static"`` is the overhead shot, framed once on the whole
    reference from ``elevation``/``azimuth``. ``"follow"`` is the chase camera,
    placed ``follow_angle`` degrees round the robot's *own* heading so the
    bearing holds through every turn.
    """

    camera_mode: str = STATIC_CAMERA
    elevation: float = DEFAULT_ELEVATION
    azimuth: float = DEFAULT_AZIMUTH
    follow_angle: float = DEFAULT_FOLLOW_ANGLE
    follow_distance: float = DEFAULT_FOLLOW_DISTANCE
    follow_height: float = DEFAULT_FOLLOW_HEIGHT
    follow_elevation: float | None = DEFAULT_FOLLOW_ELEVATION

    def __post_init__(self) -> None:
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {CAMERA_MODES}, got {self.camera_mode!r}")

    @property
    def is_follow(self) -> bool:
        return self.camera_mode == FOLLOW_CAMERA

    @property
    def elevation_above_lookat(self) -> float:
        """Degrees the camera sits above what it looks at; negative is below it.

        For a chase camera this normally *follows* from the height and stand-off
        rather than being a third setting that could disagree with them --
        unless ``follow_elevation`` states it outright, which is the only way to
        get under the floor.
        """
        if not self.is_follow:
            return float(self.elevation)
        if self.follow_elevation is not None:
            return float(self.follow_elevation)
        return math.degrees(math.atan2(self.follow_height - FOLLOW_LOOKAT_HEIGHT, self.follow_distance))

    @property
    def follow_range(self) -> float:
        """Straight-line distance the chase camera holds from its lookat point."""
        if self.follow_elevation is not None:
            return float(self.follow_distance)
        return math.hypot(self.follow_distance, self.follow_height - FOLLOW_LOOKAT_HEIGHT)

    @property
    def line_scale(self) -> float:
        """How much to thin the reference and trace tubes for this view.

        From 0.18 m away a 7 mm tube is wider than the robot is tall and simply
        buries it, so a chase view draws them thinner than an overview does.
        """
        return FOLLOW_LINE_SCALE if self.is_follow else 1.0

    @property
    def label(self) -> str:
        """How this view names itself when a run is reported on the terminal."""
        elevation = self.elevation_above_lookat
        if not self.is_follow:
            return f"top view, {elevation:.0f} deg"
        return (
            f"chase {self.follow_angle:.0f} deg ({follow_bearing(self.follow_angle)}), "
            f"elev {elevation:+.0f} deg, {self.follow_distance:.2f} m"
        )


def follow_bearing(angle: float) -> str:
    """What a ``follow_angle`` looks like from the robot: behind, abeam, head-on."""
    folded = abs(float(angle)) % 360.0
    folded = min(folded, 360.0 - folded)
    if folded < 15.0:
        return "behind"
    if folded > 165.0:
        return "head-on"
    if 75.0 <= folded <= 105.0:
        return "abeam"
    return "oblique"


def pane_rects(num_views: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """``(x, y, w, h)`` per view, tiling the frame exactly.

    One view fills the frame. Otherwise the first takes the left half and the
    rest stack down the right, which is the layout the split screen exists for:
    the whole trajectory on the left for context, the chase views beside it for
    what the robot is actually doing under it. At 1920x1080 that is a 960x1080
    overview and two 960x540 chase panes -- each rendered at its own size, so
    nothing is scaled and nothing is letterboxed.
    """
    if num_views < 1:
        raise ValueError(f"a video needs at least one view, got {num_views}")
    if num_views == 1:
        return [(0, 0, int(width), int(height))]

    left = int(width) // 2
    rects = [(0, 0, left, int(height))]
    rows = num_views - 1
    top = 0
    for row in range(rows):
        # Any remainder goes to the last row rather than being spread, so the
        # panes tile the frame exactly whatever the height divides into.
        row_height = int(height) - top if row == rows - 1 else int(height) // rows
        rects.append((left, top, int(width) - left, row_height))
        top += row_height
    return rects


def per_chase_view(values, count: int, name: str) -> list:
    """Broadcast a chase-camera setting over ``count`` panes.

    A setting is either one value, which every pane shares, or one per pane.
    ``None`` broadcasts too, since that is how a setting says "derive me" --
    there is no way to spell a per-pane None on a command line anyway.
    Anything else is a mismatch worth refusing rather than silently zipping
    short: two panes and three distances is a typo, not a request.
    """
    if values is None or isinstance(values, (int, float)):
        return [values] * count
    values = list(values)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} takes one value or one per follow angle ({count}), got {len(values)}")
    return values


def split_screen_views(
    follow_angles,
    *,
    elevation: float = DEFAULT_ELEVATION,
    azimuth: float = DEFAULT_AZIMUTH,
    follow_distance=DEFAULT_FOLLOW_DISTANCE,
    follow_height=DEFAULT_FOLLOW_HEIGHT,
    follow_elevation=DEFAULT_FOLLOW_ELEVATION,
) -> list[View]:
    """An overview plus one chase view per angle, in ``pane_rects`` order.

    The chase settings are per pane: each takes one value shared by every pane,
    or one value per angle. That is what lets a split screen watch the same run
    from two genuinely different places -- abeam from under the floor beside
    close-in from behind -- rather than the same shot twice at two bearings.
    """
    follow_angles = [float(angle) for angle in follow_angles]
    if not follow_angles:
        raise ValueError("a split screen needs at least one follow angle")
    count = len(follow_angles)
    distances = per_chase_view(follow_distance, count, "follow_distance")
    heights = per_chase_view(follow_height, count, "follow_height")
    elevations = per_chase_view(follow_elevation, count, "follow_elevation")

    overview = View(camera_mode=STATIC_CAMERA, elevation=elevation, azimuth=azimuth)
    return [overview] + [
        View(
            camera_mode=FOLLOW_CAMERA,
            follow_angle=angle,
            follow_distance=distance,
            follow_height=height,
            follow_elevation=chase_elevation,
        )
        for angle, distance, height, chase_elevation in zip(follow_angles, distances, heights, elevations)
    ]


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
        fps: int = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        trace_rgba=DEFAULT_TRACE_RGBA,
        trace_label: str = DEFAULT_TRACE_LABEL,
        replay_poses: np.ndarray | None = None,
        replay_rgba=VARIANT_RGBA[STATIC_VARIANT],
        replay_label: str = VARIANT_LABELS[STATIC_VARIANT],
        views=None,
        slowmo_factor: float = DEFAULT_SLOWMO_FACTOR,
    ) -> None:
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
        # One view fills the frame; several are tiled into it by pane_rects,
        # each rendered at its own pane's size rather than scaled to fit.
        self.views = (View(),) if views is None else tuple(views)
        self.slowmo_factor = float(slowmo_factor)
        # h264 needs even dimensions on the *composed* frame; the panes are
        # numpy slices of it and are free to be any size that tiles it.
        self.width = int(width) + int(width) % 2
        self.height = int(height) + int(height) % 2
        self.trace_rgba = tuple(trace_rgba)
        self.trace_label = str(trace_label)
        self.replay_poses = None if replay_poses is None else np.asarray(replay_poses, dtype=float).reshape(-1, 3)
        self.replay_rgba = tuple(replay_rgba)
        self.replay_label = str(replay_label)

        self._panes: list[_ViewPane] = []
        self._writer = None
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
        for pane in self._panes:
            pane.close()
        self._panes = []

    # -- hooks ------------------------------------------------------------

    def _start(self, plant: MujocoPlant) -> None:
        mujoco = self._mujoco
        model = plant.model
        rects = pane_rects(len(self.views), self.width, self.height)
        # mujoco.Renderer refuses to render larger than the model's offscreen
        # framebuffer, which is a <visual><global> property of the XML. It is
        # the largest *pane* that has to fit, not the composed frame.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, max(rect[2] for rect in rects))
        model.vis.global_.offheight = max(model.vis.global_.offheight, max(rect[3] for rect in rects))
        self._chassis_radius = _chassis_radius(mujoco, model)

        budget = self._geom_budget(model)
        self._panes = [
            _ViewPane(
                mujoco, model, view, rect, reference_xy=self.reference_xy, fps=self.fps, max_geom=budget
            )
            for view, rect in zip(self.views, rects)
        ]

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = self._imageio.get_writer(self.out_path, fps=self.output_fps, macro_block_size=1)

    def _frame(self, plant: MujocoPlant) -> None:
        # Every pane draws the same physics step, so the panes of a frame are
        # the same instant of one deployment seen from several cameras.
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        for pane in self._panes:
            frame[pane.top : pane.top + pane.height, pane.left : pane.left + pane.width] = pane.render(
                plant, self._decorate
            )
        margin = max(1, round(LEGEND_MARGIN_FRACTION * self.height))
        _composite(frame, self._legend, margin, margin)
        self._writer.append_data(frame)

    def _decorate(self, scene, scale: float) -> None:
        """Add the reference, the traces and the replayed robot to one pane's scene.

        Called once per pane per frame: each renderer owns its own scene, and
        the tube radii are scaled per view, since a tube sized for an overview
        buries the robot from 0.18 m away.
        """
        mujoco = self._mujoco
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


class _ViewPane:
    """One ``View``'s renderer and camera, and where its frame lands.

    A pane is not a separate video: it renders whatever ``mjData`` it is handed,
    so all the panes of a frame are the same physics step of the same
    deployment, and the tiling in ``DeploymentVideoRecorder._frame`` is the only
    thing that makes them a split screen.
    """

    def __init__(
        self,
        mujoco,
        model,
        view: View,
        rect: tuple[int, int, int, int],
        *,
        reference_xy: np.ndarray,
        fps: int,
        max_geom: int,
    ) -> None:
        self._mujoco = mujoco
        self.view = view
        self.left, self.top, self.width, self.height = (int(value) for value in rect)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width, max_geom=max_geom)
        self.scene_option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        # MuJoCo measures elevation downward-negative, so a camera 80 deg above
        # what it looks at is -80 in the camera's own convention -- and a
        # follow_elevation of -90, under the floor looking up, is +90.
        self.camera.elevation = -view.elevation_above_lookat

        if view.is_follow:
            # Only the azimuth and the lookat move with the robot; how far back
            # and how high the camera rides is fixed for the whole run.
            self.camera.distance = view.follow_range
        else:
            self.camera.azimuth = view.azimuth
            # Framed against this pane's own aspect ratio, not the composed
            # frame's: a 960x1080 overview is a different shot from a 1920x1080
            # one and has to be framed as one.
            lookat, distance = _frame_shot(
                reference_xy,
                azimuth=view.azimuth,
                elevation=view.elevation,
                fovy=float(model.vis.global_.fovy),
                aspect=self.width / self.height,
            )
            self.camera.lookat[:] = lookat
            self.camera.distance = distance

        self._position_alpha = 1.0 - math.exp(-1.0 / (fps * FOLLOW_POSITION_TAU))
        self._yaw_alpha = 1.0 - math.exp(-1.0 / (fps * FOLLOW_YAW_TAU))
        self._pose: tuple[float, float, float] | None = None

    def render(self, plant: MujocoPlant, decorate) -> np.ndarray:
        """This pane's ``(h, w, 3)`` frame of the plant's current state."""
        if self.view.is_follow:
            self._track(plant.pose())
        self.renderer.update_scene(plant.data, camera=self.camera, scene_option=self.scene_option)
        decorate(self.renderer.scene, self.view.line_scale)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()

    def _track(self, pose) -> None:
        """Point the chase camera at the robot for this frame.

        MuJoCo's free camera sits at
        ``lookat - distance * [cos(az)cos(el), sin(az)cos(el), sin(el)]``, so the
        camera is placed entirely by its azimuth: ``azimuth = yaw`` puts it
        squarely behind the robot, and ``+ follow_angle`` swings it round toward
        the robot's right -- 90 abeam, 45 between the two. Because the azimuth
        is measured off the robot's *own* heading it holds through every turn,
        which a fixed azimuth cannot do.

        The camera lags the robot rather than locking to it: one nailed to the
        raw pose holds the robot perfectly still and swings the world around it
        instead, hiding exactly the wobble this view exists to show.
        """
        x, y, yaw = (float(value) for value in pose)
        if self._pose is None:
            self._pose = (x, y, yaw)
        else:
            prev_x, prev_y, prev_yaw = self._pose
            # Take the short way round: a raw difference across the +-pi
            # branch cut would swing the camera a full turn the wrong way.
            delta_yaw = (yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi
            move, turn = self._position_alpha, self._yaw_alpha
            self._pose = (
                prev_x + move * (x - prev_x),
                prev_y + move * (y - prev_y),
                prev_yaw + turn * delta_yaw,
            )

        cam_x, cam_y, cam_yaw = self._pose
        self.camera.azimuth = math.degrees(cam_yaw) + self.view.follow_angle
        self.camera.lookat[:] = [cam_x, cam_y, FOLLOW_LOOKAT_HEIGHT]


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
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    log_dir: str | Path | None = None,
    trace_rgba=DEFAULT_TRACE_RGBA,
    trace_label: str = DEFAULT_TRACE_LABEL,
    replay_poses: np.ndarray | None = None,
    replay_rgba=VARIANT_RGBA[STATIC_VARIANT],
    replay_label: str = VARIANT_LABELS[STATIC_VARIANT],
    views=None,
    slowmo_factor: float = DEFAULT_SLOWMO_FACTOR,
) -> tuple[DeploymentResult, int]:
    """Drive one deployment and write the mp4.

    Returns the run's ``DeploymentResult`` and the number of frames written.

    ``log_dir`` keeps the ``TRxx`` the run produces. It defaults to a temporary
    directory that is thrown away: a video is not data collection, and dropping
    a log into an iteration's ``data/`` would hand identification a run nobody
    asked for.

    ``views`` is the cameras to render it from: one fills the frame (the
    default, an overhead shot from ``elevation``/``azimuth``), several are tiled
    into a split screen by ``pane_rects``. They all watch the same deployment,
    so a split screen costs one render per pane and nothing else.
    ``slowmo_factor`` above 1 samples the plant that much more often and plays
    the frames at ``fps``, so the video is that many times longer.
    """
    from wmr_simulator.pololu.reference_importer import load_reference

    reference = load_reference(trajectory)
    with DeploymentVideoRecorder(
        reference.states[:, :2],
        reference.dt,
        out_path,
        fps=fps,
        width=width,
        height=height,
        trace_rgba=trace_rgba,
        trace_label=trace_label,
        replay_poses=replay_poses,
        replay_rgba=replay_rgba,
        replay_label=replay_label,
        views=views,
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
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    log_dir: str | Path | None = None,
    views=None,
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
            fps=fps,
            width=width,
            height=height,
            log_dir=log_dir,
            trace_rgba=live.rgba,
            trace_label=live.label,
            replay_poses=replay_poses,
            replay_rgba=replay.rgba,
            replay_label=replay.label,
            views=views,
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
