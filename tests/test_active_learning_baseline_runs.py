"""Scoring of benchmark runs: tracking RMSE and yaw ringing."""

from __future__ import annotations

import numpy as np
import pytest

from wmr_simulator.active_learning.baseline_runs import (
    load_run_logs,
    run_tracking_rmse,
    run_yaw_ringing,
)
from wmr_simulator.active_learning.experiment import Experiment
from wmr_simulator.pololu.log_loader import POLOLU_TRAJ_CONTROL_COLUMNS

CIRCLE_RADIUS = 0.5
CIRCLE_RATE = 1.0


def write_run_csv(
    path,
    *,
    radial_offset=0.0,
    duration=6.5,
    dt=0.02,
    yaw_ringing_amplitude=None,
    ringing_frequency_hz=4.0,
    reference_rate_wobble=0.0,
    departure_time=None,
):
    """A traj-control recording of a circle, tracked with a radial offset.

    The reference is the exact circle and the mocap track the same circle at
    ``radius + radial_offset``, so the run's position RMSE against its reference
    is ``|radial_offset|`` by construction. ``yaw_ringing_amplitude`` [rad/s]
    adds an IMU stream: the constant circle yaw rate plus an oscillation of that
    amplitude at ``ringing_frequency_hz`` (0 for a clean turn, None for no IMU
    samples at all). ``reference_rate_wobble`` [rad/s] adds a 1.5 Hz swing to
    the reference yaw rate that the gyro follows exactly (a perfectly tracked
    reference that is not constant). From ``departure_time`` on the robot is a
    metre off the track and spinning: the gyro reads a broadband 15 rad/s.
    """
    spin = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in np.arange(0.0, duration, dt):
        yaw = CIRCLE_RATE * t
        reference_rate = CIRCLE_RATE + reference_rate_wobble * np.sin(2.0 * np.pi * 1.5 * t)
        row = {name: "" for name in POLOLU_TRAJ_CONTROL_COLUMNS}
        row["ts"] = f"{1000.0 * t:.0f}"
        departed = departure_time is not None and t >= departure_time
        measured = CIRCLE_RADIUS + radial_offset + (1.0 if departed else 0.0)
        row.update(
            x_raw=f"{measured * np.cos(yaw):.6f}",
            y_raw=f"{measured * np.sin(yaw):.6f}",
            yaw_raw=f"{np.arctan2(np.sin(yaw + 0.5 * np.pi), np.cos(yaw + 0.5 * np.pi)):.6f}",
        )
        row.update(
            x_des=f"{CIRCLE_RADIUS * np.cos(yaw):.6f}",
            y_des=f"{CIRCLE_RADIUS * np.sin(yaw):.6f}",
            yaw_des=f"{np.arctan2(np.sin(yaw + 0.5 * np.pi), np.cos(yaw + 0.5 * np.pi)):.6f}",
            v_ff=f"{CIRCLE_RADIUS * CIRCLE_RATE:.6f}",
            w_ff=f"{reference_rate:.6f}",
        )
        row.update(omega_l_meas="10.0", omega_r_meas="12.0", omega_l_cmd="10.0", omega_r_cmd="12.0")
        row.update(v_actual="0.5", w_actual="1.0", duty_l="0.3", duty_r="0.35")
        if yaw_ringing_amplitude is not None:
            yaw_rate = reference_rate + yaw_ringing_amplitude * np.sin(2.0 * np.pi * ringing_frequency_hz * t)
            if departed:
                yaw_rate = spin.normal(0.0, 15.0)
            row["gyro_z"] = f"{np.rad2deg(yaw_rate):.6f}"
        rows.append(row)

    lines = [",".join(POLOLU_TRAJ_CONTROL_COLUMNS)]
    lines += [",".join(row[name] for name in POLOLU_TRAJ_CONTROL_COLUMNS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def experiment(tmp_path):
    """An experiment with empty iteration folders, ready to have runs dropped in.

    Only the directory layout matters here, so the iterations are created
    directly rather than by running the loop.
    """

    def build(num_iterations=2):
        exp = Experiment.create(tmp_path / "exp", {"num_iterations": num_iterations})
        for index in range(1, num_iterations + 1):
            exp.paths(index).create_directories()
        return exp

    return build


def test_the_tracking_error_is_measured_against_the_reference_the_run_tracked(experiment):
    """The number the legend ranks iterations by: a run that held a constant
    0.04 m offset from its reference scores exactly that."""
    exp = experiment(1)
    paths = exp.paths(1)
    write_run_csv(paths.benchmark_data_dir / "TR00.csv", radial_offset=0.04)

    (run,) = load_run_logs(paths.benchmark_data_dir)

    assert run_tracking_rmse(run.log) == pytest.approx(0.04, abs=2e-3)


def test_yaw_ringing_sees_the_oscillation_and_not_the_turn(experiment):
    """The smoothness metric beside the RMSE: a steady turn scores ~0 however
    fast it turns, a 4 Hz yaw oscillation scores on the order of its RMS, and a
    log without IMU samples has no score rather than a zero one."""
    exp = experiment(1)
    paths = exp.paths(1)
    write_run_csv(paths.benchmark_data_dir / "TR00.csv", yaw_ringing_amplitude=0.0)
    write_run_csv(paths.benchmark_data_dir / "TR01.csv", yaw_ringing_amplitude=2.0)
    write_run_csv(paths.benchmark_data_dir / "TR02.csv")

    clean, ringing, no_imu = load_run_logs(paths.benchmark_data_dir)

    assert run_yaw_ringing(clean) == pytest.approx(0.0, abs=1e-3)
    assert run_yaw_ringing(ringing) == pytest.approx(2.0 / np.sqrt(2.0), rel=0.25)
    assert run_yaw_ringing(no_imu) is None


def test_yaw_ringing_does_not_score_the_references_own_yaw_rate(experiment):
    """A reference whose yaw rate swings inside the band, tracked perfectly,
    scores ~0: the reference yaw rate is subtracted before filtering."""
    exp = experiment(1)
    paths = exp.paths(1)
    write_run_csv(paths.benchmark_data_dir / "TR00.csv", yaw_ringing_amplitude=0.0, reference_rate_wobble=2.0)

    (run,) = load_run_logs(paths.benchmark_data_dir)

    assert run_yaw_ringing(run) == pytest.approx(0.0, abs=1e-2)


def test_yaw_ringing_stops_where_the_robot_leaves_the_track(experiment):
    """A spin-out after the robot left the track is not ringing: the score is
    the tracked part's, and a run that leaves too early has none."""
    exp = experiment(1)
    paths = exp.paths(1)
    write_run_csv(paths.benchmark_data_dir / "TR00.csv", yaw_ringing_amplitude=2.0)
    write_run_csv(paths.benchmark_data_dir / "TR01.csv", yaw_ringing_amplitude=2.0, departure_time=4.0)
    write_run_csv(paths.benchmark_data_dir / "TR02.csv", yaw_ringing_amplitude=2.0, departure_time=0.5)

    tracked, departed, early = load_run_logs(paths.benchmark_data_dir)

    assert run_yaw_ringing(departed) == pytest.approx(run_yaw_ringing(tracked), rel=0.1)
    assert run_yaw_ringing(early) is None
