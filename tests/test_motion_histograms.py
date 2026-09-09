"""Velocity/acceleration distributions of a reference set or a recorded run.

These figures exist to answer one question: does the motion the gains were tuned
on cover the motion they are judged on? That only works if every channel is
derived the same way whatever it came from, which is what is tested here --
plus the two places the derivation can quietly go wrong (a log's reference
carries a *body* forward speed, and a log's timestamps can repeat).
"""

from __future__ import annotations

import numpy as np
import pytest

from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
from wmr_simulator.visualization.motion_histograms import (
    measured_motion_channels_from_log,
    motion_channels,
    motion_channels_from_reference_states,
    plot_motion_histograms,
    pool_motion_channels,
    reference_motion_channels_from_log,
)

from test_active_learning_baseline_runs import CIRCLE_RADIUS, CIRCLE_RATE, write_run_csv


def arc_reference_states(speed=0.8, rate=0.5, num_samples=101, dt=0.05):
    """A constant-speed, constant-curvature reference in the designed form
    ``[x, y, theta, vx, vy, omega, ax, ay]`` (global velocity)."""
    time = dt * np.arange(num_samples)
    heading = rate * time
    return np.column_stack(
        [
            speed / rate * np.sin(heading),
            speed / rate * (1.0 - np.cos(heading)),
            heading,
            speed * np.cos(heading),
            speed * np.sin(heading),
            rate * np.ones_like(time),
            np.zeros_like(time),
            np.zeros_like(time),
        ]
    )


def test_designed_reference_states_give_forward_speed_and_yaw_rate():
    channels = motion_channels_from_reference_states(arc_reference_states(), dt=0.05)

    assert np.allclose(channels["v"], 0.8)
    assert np.allclose(channels["omega"], 0.5)
    # Constant speed on a constant-curvature arc: both derivatives are zero.
    assert np.allclose(channels["a"], 0.0, atol=1e-9)
    assert np.allclose(channels["alpha"], 0.0, atol=1e-9)


def test_a_stacked_set_is_pooled_without_differentiating_across_the_seam():
    """Two trajectories at different speeds pool into one sample set, and the
    step between the end of one and the start of the next is not an
    acceleration -- each trajectory is differentiated on its own."""
    slow = arc_reference_states(speed=0.2)
    fast = arc_reference_states(speed=2.0)

    channels = motion_channels_from_reference_states(np.stack([slow, fast]), dt=0.05)

    assert len(channels["v"]) == 2 * len(slow)
    assert set(np.round(channels["v"], 6)) == {0.2, 2.0}
    assert np.allclose(channels["a"], 0.0, atol=1e-9)


def test_a_reversing_reference_keeps_its_sign():
    states = arc_reference_states()
    states[:, 3:5] *= -1.0

    channels = motion_channels_from_reference_states(states, dt=0.05)

    assert np.allclose(channels["v"], -0.8)


def test_repeated_timestamps_do_not_become_infinite_accelerations():
    """Log streams carry duplicate timestamps; a zero interval in the
    derivative would put an infinity into every histogram."""
    time = np.array([0.0, 0.1, 0.1, 0.2, 0.3])

    channels = motion_channels(time, np.array([0.0, 1.0, 1.0, 2.0, 3.0]), np.zeros(5))

    assert np.all(np.isfinite(channels["a"]))
    assert np.allclose(channels["a"], 10.0)


def test_a_logs_reference_speed_is_read_off_not_projected(tmp_path):
    """A Pololu log's reference carries ``v_ff``/``w_ff`` -- a body forward
    speed in the global-vx slot -- so projecting it onto the heading the way a
    designed trajectory is projected would scale it by cos(theta)."""
    path = tmp_path / "TR00.csv"
    write_run_csv(path)
    log = load_pololu_traj_control_log(path)

    channels = reference_motion_channels_from_log(log)

    assert np.allclose(channels["v"], CIRCLE_RADIUS * CIRCLE_RATE)
    assert np.allclose(channels["omega"], CIRCLE_RATE)


def test_measured_channels_come_off_the_mocap_twist(tmp_path):
    """The run is a circle of ``radius + offset`` at the reference's rate, so
    the measured speed is the offset circle's, not the reference's."""
    path = tmp_path / "TR00.csv"
    write_run_csv(path, radial_offset=0.1)
    log = load_pololu_traj_control_log(path)

    channels = measured_motion_channels_from_log(log)

    assert np.median(channels["v"]) == pytest.approx((CIRCLE_RADIUS + 0.1) * CIRCLE_RATE, rel=0.05)
    assert np.median(channels["omega"]) == pytest.approx(CIRCLE_RATE, rel=0.05)


def test_pooling_skips_empty_channel_sets():
    empty = {name: np.zeros(0) for name in ("v", "omega", "a", "alpha")}
    channels = motion_channels_from_reference_states(arc_reference_states(), dt=0.05)

    pooled = pool_motion_channels([empty, channels, empty])

    assert np.allclose(pooled["v"], channels["v"])


def test_the_figure_is_written_and_an_empty_set_is_skipped(tmp_path):
    channels = motion_channels_from_reference_states(arc_reference_states(), dt=0.05)

    path = plot_motion_histograms(
        [("designed", channels)],
        out_dir=tmp_path,
        robot_cfg={"v_max": 2.5, "omega_max": 10.0, "a_max": 5.0, "alpha_max": 20.0},
    )

    assert path == tmp_path / "motion_histograms.pdf"
    assert path.is_file()
    assert plot_motion_histograms([], out_dir=tmp_path) is None
