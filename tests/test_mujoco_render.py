"""The deployment video: reference loading, iteration resolution, framing.

Rendering itself needs an OpenGL backend, so it is not exercised here; what is
tested is everything that decides *what* gets rendered, plus the observer hook
the recorder rides on.
"""

from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pytest

from wmr_simulator.mujoco_sim.render import (
    DEFAULT_TRACE_LABEL,
    PARAMETRIZED_VARIANT,
    STATIC_VARIANT,
    REFERENCE_LABEL,
    REFERENCE_RGBA,
    VARIANT_LABELS,
    VARIANT_RGBA,
    DeploymentVideoRecorder,
    FrameObserver,
    View,
    _composite,
    _frame_shot,
    _legend_overlay,
    find_iteration_root,
    follow_bearing,
    pane_rects,
    per_chase_view,
    record_pose_track,
    render_deployment_comparison,
    resolve_controller,
    ships_two_controllers,
    split_screen_views,
)
from wmr_simulator.pololu.reference_importer import load_pololu_reference, load_reference


def _reference_states(num: int = 21, dt: float = 0.05) -> np.ndarray:
    """An arc as the simulator's 8-column reference states."""
    time = np.arange(num) * dt
    speed, yaw_rate = 0.5, 0.4
    yaw = yaw_rate * time
    radius = speed / yaw_rate
    return np.column_stack(
        [
            radius * np.sin(yaw),
            radius * (1.0 - np.cos(yaw)),
            yaw,
            speed * np.cos(yaw),
            speed * np.sin(yaw),
            np.full(num, yaw_rate),
            np.zeros(num),
            np.zeros(num),
        ]
    ).astype(np.float32)


def test_pickle_and_its_exported_jsn_load_to_the_same_reference(tmp_path):
    """The pickle route is the export route, so the robot drives the same curve."""
    from wmr_simulator.pololu.reference_exporter import export_reference_trajectory, load_reference_trajectory

    pickle_path = tmp_path / "trajectory.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump({"reference_states": _reference_states(), "dt": 0.05}, file)
    jsn_path = export_reference_trajectory(load_reference_trajectory(pickle_path), tmp_path / "export")

    from_pickle = load_reference(pickle_path)
    from_jsn = load_pololu_reference(jsn_path)

    np.testing.assert_allclose(from_pickle.states, from_jsn.states)
    np.testing.assert_allclose(from_pickle.actions, from_jsn.actions)
    assert from_pickle.dt == from_jsn.dt


def test_load_reference_still_reads_a_jsn(tmp_path):
    path = tmp_path / "TRJ0001.JSN"
    states = _reference_states(num=5)
    path.write_text(
        json.dumps(
            {
                "result": [
                    {
                        "dt": 0.05,
                        "states": states[:, :3].tolist(),
                        "actions": np.zeros((4, 2)).tolist(),
                    }
                ]
            }
        )
    )
    assert load_reference(path).states.shape == (5, 3)


def _iteration(tmp_path, *, static: bool, gain_mlp: bool = True):
    root = tmp_path / "exp" / "iteration_03"
    (root / "tuning_trajectories").mkdir(parents=True)
    (root / "ROBOTCFG.CFG").write_text("WHEEL_RADIUS 0.016\n")
    if gain_mlp:
        (root / "GAINMLP.JSN").write_text("{}")
    if static:
        (root / "ROBOTCFG_static.CFG").write_text("WHEEL_RADIUS 0.016\n")
    trajectory = root / "tuning_trajectories" / "tuning_trajectory_00.pkl"
    trajectory.write_bytes(b"")
    return root, trajectory


def test_find_iteration_root_walks_up_to_the_robotcfg(tmp_path):
    root, trajectory = _iteration(tmp_path, static=True)
    assert find_iteration_root(trajectory) == root.resolve()


def test_find_iteration_root_refuses_a_trajectory_outside_an_experiment(tmp_path):
    orphan = tmp_path / "somewhere" / "trajectory.pkl"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="ROBOTCFG.CFG"):
        find_iteration_root(orphan)


def test_static_variant_is_staged_away_from_the_deployed_network(tmp_path):
    """FirmwareConfig.from_file picks up a GAINMLP.JSN beside the config, so the
    static baseline cannot be driven from the iteration root."""
    root, _ = _iteration(tmp_path, static=True)
    staging = tmp_path / "staging"

    controller = resolve_controller(root, STATIC_VARIANT, staging)
    assert controller.config_path.parent == staging
    assert controller.config_path.name == "ROBOTCFG.CFG"
    assert not (staging / "GAINMLP.JSN").exists()
    assert controller.config_path.read_text() == (root / "ROBOTCFG_static.CFG").read_text()


def test_parametrized_variant_is_driven_where_it_lives(tmp_path):
    root, _ = _iteration(tmp_path, static=True)
    controller = resolve_controller(root, PARAMETRIZED_VARIANT, tmp_path / "staging")
    assert controller.config_path == root / "ROBOTCFG.CFG"
    assert "GAINMLP.JSN" in controller.description


def test_an_iteration_without_a_static_config_resolves_both_variants_to_it(tmp_path):
    """Iteration 1, or an experiment with the parametrization off: one controller."""
    root, _ = _iteration(tmp_path, static=False)
    for variant in (STATIC_VARIANT, PARAMETRIZED_VARIANT):
        assert resolve_controller(root, variant, tmp_path / "staging").config_path == root / "ROBOTCFG.CFG"
    assert "identity" in resolve_controller(root, STATIC_VARIANT, tmp_path / "staging").description


@pytest.mark.parametrize("elevation", [90.0, 80.0, 45.0])
@pytest.mark.parametrize("azimuth", [90.0, 0.0, 215.0])
def test_frame_shot_puts_the_whole_reference_inside_the_frustum(elevation, azimuth):
    points = _reference_states(num=41)[:, :2] + np.array([3.0, -2.0])
    fovy, aspect = 45.0, 1280 / 720
    lookat, distance = _frame_shot(points, azimuth=azimuth, elevation=elevation, fovy=fovy, aspect=aspect)

    az, el = math.radians(azimuth), math.radians(-elevation)
    forward = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_v = math.tan(0.5 * math.radians(fovy))

    offsets = np.column_stack([points, np.zeros(len(points))]) - lookat
    depth = offsets @ forward + distance
    assert np.all(depth > 0.0)
    assert np.all(np.abs(offsets @ up) <= tan_v * depth)
    assert np.all(np.abs(offsets @ right) <= aspect * tan_v * depth)


@pytest.fixture(scope="module")
def robot_config(tmp_path_factory):
    from types import SimpleNamespace

    from wmr_simulator.pololu.robot_config import export_robot_config

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0)
    return export_robot_config(
        tmp_path_factory.mktemp("config") / "ROBOTCFG.CFG",
        physical_params=physical,
        controller_gains=[2.0, 6.7, 7.4, 3.2, 0.0],
    )


@pytest.fixture(scope="module")
def short_trajectory(tmp_path_factory):
    """1 s of the same arc, as a reference pickle -- the other input format."""
    path = tmp_path_factory.mktemp("trajectory") / "trajectory.pkl"
    with path.open("wb") as file:
        pickle.dump({"reference_states": _reference_states(num=21), "dt": 0.05}, file)
    return path


def test_run_deployment_calls_the_observer_with_the_plant(robot_config, short_trajectory, tmp_path):
    """And it drives a pickle straight, which is what the video script needs."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment

    seen = []

    def observer(tick, plant):
        seen.append((tick.index, tuple(plant.pose())))

    result = run_deployment(robot_config, short_trajectory, tmp_path / "logs", seed=0, observer=observer)

    assert len(seen) > 100
    assert [index for index, _ in seen] == sorted(index for index, _ in seen)
    # The last call sees the pose the result reports, i.e. the run's real end.
    np.testing.assert_allclose(seen[-1][1], result.final_pose)


def test_the_two_variants_get_different_trace_colours(tmp_path):
    root, _ = _iteration(tmp_path, static=True)
    staging = tmp_path / "staging"
    live = resolve_controller(root, PARAMETRIZED_VARIANT, staging)
    replay = resolve_controller(root, STATIC_VARIANT, staging)
    assert live.rgba != replay.rgba
    assert {live.rgba, replay.rgba} == set(VARIANT_RGBA.values())


def test_ships_two_controllers_tracks_the_static_config(tmp_path):
    assert ships_two_controllers(_iteration(tmp_path / "two", static=True)[0])
    assert not ships_two_controllers(_iteration(tmp_path / "one", static=False)[0])


def test_comparison_refuses_an_iteration_with_a_single_controller(tmp_path):
    """Both variants resolve to the same config there, so it would be one run drawn twice."""
    root, trajectory = _iteration(tmp_path, static=False)
    with pytest.raises(FileNotFoundError, match="nothing to compare"):
        render_deployment_comparison(root, trajectory, tmp_path / "out.mp4")


def test_record_pose_track_returns_one_true_pose_per_frame(robot_config, short_trajectory, tmp_path):
    result, poses = record_pose_track(robot_config, short_trajectory, seed=0, fps=30)

    assert poses.ndim == 2 and poses.shape[1] == 3
    # 1 s of reference at 30 fps, give or take the trailing partial frame.
    assert 25 <= len(poses) <= 35
    # The frames land on a fixed step cadence, so the last one is up to a frame
    # of motion short of where the run actually ended.
    np.testing.assert_allclose(poses[0][:2], result.start_pose[:2], atol=0.02)
    np.testing.assert_allclose(poses[-1], result.final_pose, atol=0.05)
    assert not (tmp_path / "TR00").exists()  # the log went to a scratch directory


def test_frame_observer_samples_at_the_requested_rate(robot_config, short_trajectory):
    fast, slow = FrameObserver(fps=50), FrameObserver(fps=10)
    from wmr_simulator.mujoco_sim.deploy import run_deployment
    import tempfile

    for observer in (fast, slow):
        with tempfile.TemporaryDirectory() as scratch:
            run_deployment(robot_config, short_trajectory, scratch, seed=0, observer=observer)
    assert fast.num_frames == pytest.approx(5 * slow.num_frames, rel=0.15)


def test_the_two_runs_of_a_comparison_are_paired_on_the_seed(robot_config, short_trajectory):
    """Same seed means the same hand placement, so only the controller differs."""
    first, _ = record_pose_track(robot_config, short_trajectory, seed=3, fps=30)
    second, _ = record_pose_track(robot_config, short_trajectory, seed=3, fps=30)
    assert first.start_offset == second.start_offset


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def _recorder(tmp_path, **kwargs):
    """A recorder built far enough to hold a legend; no renderer, so no GL."""
    return DeploymentVideoRecorder(
        _reference_states(num=21)[:, :2], 0.05, tmp_path / "out.mp4", width=1280, height=720, **kwargs
    )


def test_legend_lists_the_reference_and_the_single_run(tmp_path):
    entries = _recorder(tmp_path).legend_entries()
    assert entries == [(REFERENCE_LABEL, REFERENCE_RGBA), (DEFAULT_TRACE_LABEL, VARIANT_RGBA[PARAMETRIZED_VARIANT])]


def test_legend_lists_both_runs_in_the_order_they_are_drawn(tmp_path):
    recorder = _recorder(
        tmp_path,
        trace_rgba=VARIANT_RGBA[PARAMETRIZED_VARIANT],
        trace_label=VARIANT_LABELS[PARAMETRIZED_VARIANT],
        replay_poses=np.zeros((5, 3)),
        replay_rgba=VARIANT_RGBA[STATIC_VARIANT],
        replay_label=VARIANT_LABELS[STATIC_VARIANT],
    )
    labels = [label for label, _ in recorder.legend_entries()]
    assert labels == [REFERENCE_LABEL, VARIANT_LABELS[STATIC_VARIANT], VARIANT_LABELS[PARAMETRIZED_VARIANT]]


def test_legend_overlay_carries_every_entry_colour(tmp_path):
    entries = [("reference", REFERENCE_RGBA), ("static baseline", VARIANT_RGBA[STATIC_VARIANT])]
    overlay = _legend_overlay(entries, 1280, 720)

    assert overlay.ndim == 3 and overlay.shape[2] == 4
    assert overlay.shape[0] < 720 and overlay.shape[1] < 1280
    opaque = overlay[overlay[..., 3] > 250][:, :3].astype(int)
    for _, rgba in entries:
        swatch = np.array([round(255 * channel) for channel in rgba[:3]])
        assert np.abs(opaque - swatch).sum(axis=1).min() <= 3, f"no {swatch} pixels in the legend"


def test_legend_scales_with_the_frame_and_never_overflows_it():
    entries = [("reference", REFERENCE_RGBA), (VARIANT_LABELS[PARAMETRIZED_VARIANT], VARIANT_RGBA[PARAMETRIZED_VARIANT])]
    small = _legend_overlay(entries, 640, 360)
    large = _legend_overlay(entries, 2560, 1440)
    assert large.shape[0] > small.shape[0] and large.shape[1] > small.shape[1]
    for overlay, (width, height) in ((small, (640, 360)), (large, (2560, 1440))):
        assert overlay.shape[0] <= height and overlay.shape[1] <= width


def test_composite_blends_in_place_and_leaves_the_rest_of_the_frame_alone():
    frame = np.full((40, 60, 3), 100, dtype=np.uint8)
    overlay = np.zeros((10, 20, 4), dtype=np.uint8)
    overlay[..., :3] = 200
    overlay[..., 3] = 255
    overlay[0, 0, 3] = 0  # fully transparent pixel stays as the frame was

    _composite(frame, overlay, 5, 3)

    assert frame[3, 5].tolist() == [100, 100, 100]
    assert frame[4, 6].tolist() == [200, 200, 200]
    assert frame[0, 0].tolist() == [100, 100, 100]
    assert frame[39, 59].tolist() == [100, 100, 100]


def test_composite_clips_an_overlay_that_runs_off_the_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay = np.full((8, 8, 4), 255, dtype=np.uint8)
    _composite(frame, overlay, 6, 6)  # only a 4x4 corner fits
    assert frame[9, 9].tolist() == [255, 255, 255]
    assert frame[5, 5].tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# Views and the split screen
# ---------------------------------------------------------------------------


def test_a_single_view_fills_the_frame():
    assert pane_rects(1, 1920, 1080) == [(0, 0, 1920, 1080)]


def test_split_screen_tiles_1920x1080_into_the_panes_it_promises():
    """An overview beside stacked chase views, at sizes that need no scaling."""
    assert pane_rects(2, 1920, 1080) == [(0, 0, 960, 1080), (960, 0, 960, 1080)]
    assert pane_rects(3, 1920, 1080) == [(0, 0, 960, 1080), (960, 0, 960, 540), (960, 540, 960, 540)]


@pytest.mark.parametrize("num_views", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("size", [(1920, 1080), (1281, 723), (640, 360)])
def test_panes_tile_the_frame_exactly_and_never_overlap(num_views, size):
    """A gap or an overlap would leave uninitialized pixels in the frame buffer."""
    width, height = size
    covered = np.zeros((height, width), dtype=int)
    for left, top, pane_width, pane_height in pane_rects(num_views, width, height):
        assert pane_width > 0 and pane_height > 0
        covered[top : top + pane_height, left : left + pane_width] += 1
    assert np.all(covered == 1)


def test_a_view_refuses_a_camera_mode_it_cannot_render():
    with pytest.raises(ValueError, match="camera_mode"):
        View(camera_mode="orbit")


def test_a_chase_camera_elevation_follows_from_its_height_and_stand_off():
    """Height and stand-off are the two knobs, so they cannot disagree with a third."""
    view = View(camera_mode="follow", follow_height=0.045, follow_distance=0.18)
    rise = 0.045 - 0.015  # FOLLOW_LOOKAT_HEIGHT
    assert view.elevation_above_lookat == pytest.approx(math.degrees(math.atan2(rise, 0.18)))
    assert view.follow_range == pytest.approx(math.hypot(0.18, rise))


def test_follow_elevation_overrides_the_height_and_reads_distance_as_the_range():
    """Which is the only way under the floor: the height cannot go negative usefully."""
    view = View(camera_mode="follow", follow_elevation=-90.0, follow_distance=0.16, follow_height=0.045)
    assert view.elevation_above_lookat == -90.0
    assert view.follow_range == pytest.approx(0.16)


def test_an_overview_reports_its_own_elevation_and_full_width_lines():
    view = View(camera_mode="static", elevation=80.0)
    assert view.elevation_above_lookat == 80.0
    assert view.line_scale == 1.0
    # A tube sized for a whole-trajectory shot buries the robot from 0.18 m away.
    assert View(camera_mode="follow").line_scale < 1.0


@pytest.mark.parametrize(
    "angle,bearing", [(0.0, "behind"), (-5.0, "behind"), (90.0, "abeam"), (-90.0, "abeam"), (180.0, "head-on"), (45.0, "oblique")]
)
def test_follow_bearing_names_where_the_camera_sits(angle, bearing):
    assert follow_bearing(angle) == bearing


def test_split_screen_views_puts_the_overview_first_and_one_chase_pane_per_angle():
    views = split_screen_views([0.0, 90.0], elevation=70.0, follow_elevation=-20.0)
    assert [view.camera_mode for view in views] == ["static", "follow", "follow"]
    assert views[0].elevation == 70.0
    assert [view.follow_angle for view in views[1:]] == [0.0, 90.0]
    # A single chase setting is shared by every pane.
    assert all(view.follow_elevation == -20.0 for view in views[1:])


def test_each_chase_pane_can_have_its_own_distance_and_elevation():
    """Otherwise a split screen is the same shot at two bearings."""
    views = split_screen_views([0.0, 90.0], follow_elevation=[10.0, -60.0], follow_distance=[0.12, 0.20])
    assert [view.follow_elevation for view in views[1:]] == [10.0, -60.0]
    assert [view.follow_distance for view in views[1:]] == [0.12, 0.20]
    # follow_elevation is set, so the range is the distance as given.
    assert [view.follow_range for view in views[1:]] == [0.12, 0.20]
    assert [view.elevation_above_lookat for view in views[1:]] == [10.0, -60.0]


def test_a_chase_setting_with_the_wrong_count_is_refused_not_zipped_short():
    with pytest.raises(ValueError, match="one per follow angle"):
        split_screen_views([0.0, 90.0], follow_distance=[0.12, 0.16, 0.20])


@pytest.mark.parametrize(
    "values,expected",
    [
        (0.18, [0.18, 0.18]),
        ([0.18], [0.18, 0.18]),
        (None, [None, None]),
        ([0.12, 0.20], [0.12, 0.20]),
    ],
)
def test_per_chase_view_broadcasts_a_shared_setting_and_keeps_a_per_pane_one(values, expected):
    assert per_chase_view(values, 2, "follow_distance") == expected


def test_two_views_differing_only_in_stand_off_report_themselves_differently():
    """The terminal line is how a run says which shots it rendered."""
    near = View(camera_mode="follow", follow_angle=90.0, follow_elevation=-35.0, follow_distance=0.12)
    far = View(camera_mode="follow", follow_angle=90.0, follow_elevation=-35.0, follow_distance=0.20)
    assert near.label != far.label


def test_split_screen_needs_at_least_one_chase_angle():
    with pytest.raises(ValueError, match="follow angle"):
        split_screen_views([])


def test_a_recorder_defaults_to_one_overview_view(tmp_path):
    assert _recorder(tmp_path).views == (View(),)

