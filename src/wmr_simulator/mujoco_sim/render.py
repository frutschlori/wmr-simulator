"""Render a MuJoCo deployment to an mp4.

The recorders are observers of ``deploy.run_deployment``, so the video shows
exactly the run the pipeline would record (hidden plant plus firmware port).
The reference and the robot's true trace are drawn into the MuJoCo scene as
decor geoms; the legend is composited onto the frames with Pillow.

``render_deployment_comparison`` drives an iteration's static and parametrized
controllers on the same seed: the parametrized run is rendered, the static run
(driven first by ``record_pose_track``) is drawn as a marker from its recorded
true pose. ``views`` renders several cameras of the same run into one
split-screen frame (``pane_rects``).

Offscreen rendering needs an OpenGL backend, chosen through MUJOCO_GL before
mujoco is imported: ``egl`` (headless GPU, the default), ``osmesa`` (CPU) or
``glfw`` (needs a display).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wmr_simulator.mujoco_sim.deploy import DeploymentResult
from wmr_simulator.mujoco_sim.plant import MujocoPlant

# Overview camera: just off vertical.
DEFAULT_ELEVATION = 80.0
# Azimuth 90: world +x to the right, +y up the frame.
DEFAULT_AZIMUTH = 90.0
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Fraction of headroom left around the reference when framing the shot.
CAMERA_MARGIN = 1.2

STATIC_CAMERA = "static"
FOLLOW_CAMERA = "follow"
CAMERA_MODES = (STATIC_CAMERA, FOLLOW_CAMERA)

# Chase camera: low, at about the robot's height, so wheels and caster show.
DEFAULT_FOLLOW_DISTANCE = 0.18
DEFAULT_FOLLOW_HEIGHT = 0.045
# Bearing round the robot's heading: 0 behind, 90 abeam.
DEFAULT_FOLLOW_ANGLE = 0.0
# None: the elevation follows from height and distance. Otherwise the angle
# itself, with follow_distance as the range (negative: from under the floor).
DEFAULT_FOLLOW_ELEVATION = None
# Height of the chase camera's lookat point [m].
FOLLOW_LOOKAT_HEIGHT = 0.015
# Low-pass time constants [simulated s] of the chase camera's heading and position.
FOLLOW_POSITION_TAU = 0.08
FOLLOW_YAW_TAU = 0.30
# Tube radius factor of the chase view.
FOLLOW_LINE_SCALE = 0.3

# > 1 samples the plant that many times more often at the same playback fps.
DEFAULT_SLOWMO_FACTOR = 1.0

STATIC_VARIANT = "static"
PARAMETRIZED_VARIANT = "parametrized"

REFERENCE_RGBA = (0.55, 0.95, 0.35, 1.0)
# One colour per controller.
VARIANT_RGBA = {
    PARAMETRIZED_VARIANT: (0.98, 0.40, 0.10, 1.0),
    STATIC_VARIANT: (0.98, 0.30, 0.78, 1.0),
}
VARIANT_COLOUR_NAMES = {PARAMETRIZED_VARIANT: "orange", STATIC_VARIANT: "pink"}
VARIANT_LABELS = {
    PARAMETRIZED_VARIANT: "parametrized gains",
    STATIC_VARIANT: "static gains",
}
REFERENCE_LABEL = "reference"
DEFAULT_TRACE_LABEL = "robot"
DEFAULT_TRACE_RGBA = VARIANT_RGBA[PARAMETRIZED_VARIANT]

REFERENCE_RADIUS = 0.006
TRACE_RADIUS = 0.007
# Heights of the decor layers above the floor.
REFERENCE_HEIGHT = 0.002
REPLAY_TRACE_HEIGHT = 0.004
TRACE_HEIGHT = 0.006
# The replayed robot is a ring with a heading tick, wider than the chassis.
# The legend scales with the frame height.
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
    """One camera of a video.

    ``camera_mode="static"`` is the overview, framed on the whole reference
    from ``elevation``/``azimuth``. ``"follow"`` is the chase camera, placed
    ``follow_angle`` degrees round the robot's heading.
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
        """Degrees the camera sits above what it looks at (negative: below)."""
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
        """Radius factor of the reference and trace tubes in this view."""
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
    """``(x, y, w, h)`` per view, tiling the frame exactly: one view fills it,
    otherwise the first takes the left half and the rest stack down the right."""
    if num_views < 1:
        raise ValueError(f"a video needs at least one view, got {num_views}")
    if num_views == 1:
        return [(0, 0, int(width), int(height))]

    left = int(width) // 2
    rects = [(0, 0, left, int(height))]
    rows = num_views - 1
    top = 0
    for row in range(rows):
        # The last row takes the remainder.
        row_height = int(height) - top if row == rows - 1 else int(height) // rows
        rects.append((left, top, int(width) - left, row_height))
        top += row_height
    return rects


def per_chase_view(values, count: int, name: str) -> list:
    """Broadcast a chase-camera setting (one value, None, or one per pane)
    over ``count`` panes."""
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
    """An overview plus one chase view per angle, in ``pane_rects`` order;
    each chase setting is shared or given per angle."""
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

    The static variant's config is staged into its own directory, since the
    firmware picks up a ``GAINMLP.JSN`` next to the config.
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
    # Whether this run is the rendered chassis (else drawn as a marker).
    live: bool


def find_iteration_root(trajectory: str | Path) -> Path:
    """The active-learning iteration a trajectory file belongs to (the first
    parent directory with a ``ROBOTCFG.CFG``)."""
    trajectory = Path(trajectory).resolve()
    for candidate in trajectory.parents:
        if (candidate / "ROBOTCFG.CFG").is_file():
            return candidate
    raise FileNotFoundError(
        f"No active-learning iteration above {trajectory}: no directory among its parents holds a "
        "ROBOTCFG.CFG. Pass --config to drive a trajectory that is not part of an experiment."
    )


def ships_two_controllers(iteration_root: str | Path) -> bool:
    """Whether this iteration ships a static controller (``ROBOTCFG_static.CFG``)
    beside its deployed one."""
    from wmr_simulator.active_learning.experiment import IterationPaths

    return IterationPaths(root=Path(iteration_root)).robotcfg_static_cfg.is_file()


def resolve_controller(iteration_root: str | Path, variant: str, staging_dir: str | Path) -> DeploymentController:
    """The iteration's deployed (``parametrized``) or static-gain controller
    (the same config when the iteration ships only one)."""
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
    """A ``run_deployment`` observer that samples the true pose at video cadence."""

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
    """A ``run_deployment`` observer that writes the run out as an mp4;
    ``replay`` overlays another recorded run of the same reference."""

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
        # Slow motion samples the plant more often; the playback fps stays.
        super().__init__(fps=max(1, round(fps * float(slowmo_factor))))
        self.output_fps = max(1, int(fps))
        import imageio.v2 as imageio
        import mujoco

        self._mujoco = mujoco
        self._imageio = imageio
        self.reference_xy = np.asarray(reference_xy, dtype=float)[:, :2]
        self.dt = float(dt)
        self.out_path = Path(out_path)
        self.views = (View(),) if views is None else tuple(views)
        self.slowmo_factor = float(slowmo_factor)
        # h264 needs even frame dimensions.
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
        # The offscreen framebuffer has to fit the largest pane.
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
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        for pane in self._panes:
            frame[pane.top : pane.top + pane.height, pane.left : pane.left + pane.width] = pane.render(
                plant, self._decorate
            )
        margin = max(1, round(LEGEND_MARGIN_FRACTION * self.height))
        _composite(frame, self._legend, margin, margin)
        self._writer.append_data(frame)

    def _decorate(self, scene, scale: float) -> None:
        """Add the reference, the traces and the replayed robot to one pane's scene."""
        mujoco = self._mujoco
        _add_polyline(
            mujoco, scene, self.reference_xy, REFERENCE_HEIGHT, scale * REFERENCE_RADIUS, REFERENCE_RGBA
        )
        if self.replay_poses is not None and len(self.replay_poses):
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
        """Scene geom budget: one geom per polyline segment, sized from the
        reference length with a factor of two."""
        frames = int(self.fps * len(self.reference_xy) * self.dt)
        traces = 2 if self.replay_poses is not None else 1
        marker = REPLAY_MARKER_SEGMENTS + 2
        return len(self.reference_xy) + 2 * traces * frames + marker + model.ngeom + 1000


class _ViewPane:
    """One ``View``'s renderer and camera, and where its frame lands."""

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
        # MuJoCo's elevation is negative above the lookat point.
        self.camera.elevation = -view.elevation_above_lookat

        if view.is_follow:
            self.camera.distance = view.follow_range
        else:
            self.camera.azimuth = view.azimuth
            # Framed for this pane's aspect ratio.
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
        """Point the chase camera at the low-passed robot pose:
        ``azimuth = yaw + follow_angle``."""
        x, y, yaw = (float(value) for value in pose)
        if self._pose is None:
            self._pose = (x, y, yaw)
        else:
            prev_x, prev_y, prev_yaw = self._pose
            # Wrap the heading difference to [-pi, pi).
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
    """Drive one deployment and write the mp4; returns the run's
    ``DeploymentResult`` and the number of frames written.

    ``log_dir`` keeps the run's ``TRxx`` log (default: a temporary directory).
    ``views`` are the cameras (default: one overview from
    ``elevation``/``azimuth``). ``slowmo_factor`` above 1 makes the video that
    many times longer.
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
    """Both of an iteration's controllers in one video, on the same seed
    (refuses an iteration that ships only one controller)."""
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

        # Sampled at the rendered cadence, fps * slowmo_factor.
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
    """The other run's robot: a ring around its chassis, with a heading tick."""
    x, y, yaw = (float(value) for value in pose)
    radius = REPLAY_MARKER_RADIUS_FACTOR * chassis_radius
    angles = np.linspace(0.0, 2.0 * math.pi, REPLAY_MARKER_SEGMENTS + 1)
    ring = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles)])
    _add_polyline(mujoco, scene, ring, REPLAY_MARKER_HEIGHT, line_scale * 0.6 * TRACE_RADIUS, rgba)
    # Heading tick, outside the ring.
    heading = np.array([math.cos(yaw), math.sin(yaw)])
    tick = np.array([[x, y]]) + np.outer([1.0, 1.5], radius * heading)
    _add_polyline(mujoco, scene, tick, REPLAY_MARKER_HEIGHT, line_scale * 0.6 * TRACE_RADIUS, rgba)


def _legend_overlay(entries, width: int, height: int) -> np.ndarray:
    """The legend as an ``(h, w, 4)`` RGBA patch, drawn once per video."""
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
    """``(lookat, distance)`` putting every point of the reference in frame
    (each point projected onto the camera's right/up axes)."""
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
