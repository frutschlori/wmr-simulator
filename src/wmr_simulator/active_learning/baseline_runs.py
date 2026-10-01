"""Cross-iteration collection of an experiment's baseline comparison runs.

A baseline run is a recording of a *held-out* reference -- one the iteration
neither identified on nor tuned against -- so it is the only thing in the loop
that stays fixed while the controller moves. Overlaying every iteration's runs
of one such reference in a single figure is what shows whether the loop is
actually improving the controller.

Two layouts produce them, and both are read here:

- the ``benchmark`` stage, which drives ``benchmark.trajectory`` in the MuJoCo
  plant under each controller the iteration ships, on paired seeds: the
  static-gain baseline into ``data/benchmark_static/`` and the deployed
  parametrized controller into ``data/benchmark/``. Iteration 1 records only the
  static set, because its deployed controller *is* the stock static one (its
  parametrization is still the identity) and nothing tuned exists to compare it
  against yet;
- hand-recorded real-robot runs, where a shape sits directly in ``data/<shape>/``
  when it was driven on static gains and in ``data/with gain MLP/<shape>/`` when
  it was driven with the exported network (both spellings of that directory are
  accepted, as the recorded experiments carry both).

Runs are grouped by *shape* -- the reference that was driven -- because two
shapes are not comparable, and within a shape by controller *variant*, which is
what the comparison is about. At most ``MAX_RUNS_PER_ITERATION`` runs of a group
are kept: the recordings are repeats of one reference, so beyond a handful more
lines only thicken the same band.

The heavy lifting (decoding, loading, the tracking error) lives here;
``visualization.baseline_runs`` only renders what this returns, the same split
``progress`` and ``visualization.pipeline_progress`` use.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Repeats of one reference under one controller; past a handful the extra lines
# land on top of each other.
MAX_RUNS_PER_ITERATION = 5

# Controller variants a baseline run can have been recorded under. The static
# one is the gain-parametrization-free baseline (ROBOTCFG_static.CFG on the
# robot), the parametrized one the controller the iteration actually deployed.
STATIC_VARIANT = "static"
PARAMETRIZED_VARIANT = "parametrized"
VARIANT_TITLES = {STATIC_VARIANT: "Static gains", PARAMETRIZED_VARIANT: "Gain parametrization"}
VARIANT_ORDER = (STATIC_VARIANT, PARAMETRIZED_VARIANT)

# data/ subdirectories the benchmark stage writes, one per variant. Their shape
# is benchmark.trajectory, which the directory name does not carry.
BENCHMARK_DIRECTORY_NAMES = {PARAMETRIZED_VARIANT: "benchmark", STATIC_VARIANT: "benchmark_static"}
BENCHMARK_SHAPE = "benchmark"

# Where the hand-recorded robot runs of a shape live when the gain network was
# on the card. Anything else directly under data/ is a static-gain recording.
GAIN_MLP_DIRECTORY_NAMES = ("with gain MLP", "with_gain_MLP")

# Yaw ringing: the RMS of the IMU yaw-rate tracking error (gyro minus the
# reference yaw rate) inside this band, filtered forward and backward with a
# 2nd-order Butterworth so the response is flat and phase-free. The ringing that
# precedes duty saturation on the fast shapes sits at 1.9-3.6 Hz on the robot
# (median 3.0) and 2.2-3.9 Hz in MuJoCo (median 3.8), so the band holds both
# with margin. The reference is subtracted first because a perfectly tracked
# lemniscate or designed curve carries 0.08-0.22 rad/s of its own yaw-rate
# changes in the band, as much as a quiet controller scores. Until 2026-10-01
# the metric was the gyro minus its 0.3 s centered moving average, whose gain
# runs from 0.52 at 2 Hz to 1.2 at 5 Hz and so read MuJoCo's ringing about 1.25x
# high against the robot's.
YAW_RINGING_BAND_HZ = (1.5, 8.0)

# A run is scored only up to the first reference instant its mocap position is
# further than this from the reference (the default ``benchmark.
# divergence_radius``). A spin-out past that point is broadband and scored
# 1.1-1.9 rad/s, more than most genuine ringing, while a run that rang itself
# off the track keeps the ringing that took it there.
YAW_RINGING_DEPARTURE_RADIUS_M = 0.25

# Less tracked time than this before the departure leaves no score: a second
# holds three ringing periods.
YAW_RINGING_MIN_DURATION_S = 1.0


@dataclass(frozen=True)
class BaselineRun:
    """One recorded run of a baseline reference."""

    name: str
    poses: np.ndarray  # (N, 3) smoothed mocap pose track
    time_s: np.ndarray  # (N,) pose timestamps
    tracking_rmse: float | None
    yaw_ringing: float | None


@dataclass(frozen=True)
class BaselineIterationRuns:
    """One iteration's runs of one baseline shape, per controller variant."""

    index: int
    runs: dict[str, list[BaselineRun]]
    reference: np.ndarray | None  # (M, 3) reference the runs tracked


def collect_baseline_runs(
    experiment, max_runs: int = MAX_RUNS_PER_ITERATION
) -> dict[str, list[BaselineIterationRuns]]:
    """Every iteration's baseline runs, keyed by shape then ordered by iteration.

    Shapes with no runs at all are absent, and so are iterations that recorded
    none of a shape -- an experiment that only started benchmarking halfway
    through plots the iterations it has.
    """
    clip = bool(experiment.config["log_loading"]["clip_after_first_trajectory"])
    collected: dict[str, list[BaselineIterationRuns]] = {}
    for index in experiment.iteration_indices():
        paths = experiment.paths(index)
        for shape, directories in baseline_run_directories(paths).items():
            runs = {}
            reference = None
            for variant, directory in directories.items():
                logs = load_run_logs(directory, max_runs=max_runs, clip_after_first_trajectory=clip)
                if not logs:
                    continue
                runs[variant] = [_baseline_run(run) for run in logs]
                if reference is None:
                    reference = np.asarray(logs[0].log.reference.states, dtype=float)[:, :3]
            if runs:
                collected.setdefault(shape, []).append(
                    BaselineIterationRuns(index=index, runs=runs, reference=reference)
                )
    return collected


def baseline_run_directories(paths) -> dict[str, dict[str, Path]]:
    """``{shape: {variant: directory}}`` of one iteration's baseline recordings.

    Only ``data/`` subdirectories are considered: everything directly in
    ``data/`` is identification data, which the iteration was fitted on and so
    is not held out.
    """
    directories: dict[str, dict[str, Path]] = {}
    if not paths.data_dir.is_dir():
        return directories

    for variant, name in BENCHMARK_DIRECTORY_NAMES.items():
        directory = paths.data_dir / name
        if _holds_runs(directory):
            # One reference: the runs sit straight in the variant directory and
            # the shape is not recoverable from the path, so it is just
            # "benchmark".
            directories.setdefault(BENCHMARK_SHAPE, {})[variant] = directory
            continue
        # A benchmark *set*: one subdirectory per reference, named after it, so
        # the shape is the directory name and each gets its own figure.
        for shape_dir in sorted(path for path in directory.glob("*") if path.is_dir()):
            if _holds_runs(shape_dir):
                directories.setdefault(shape_dir.name, {})[variant] = shape_dir

    for child in sorted(path for path in paths.data_dir.iterdir() if path.is_dir()):
        if child.name in BENCHMARK_DIRECTORY_NAMES.values():
            continue
        if child.name in GAIN_MLP_DIRECTORY_NAMES:
            for shape_dir in sorted(path for path in child.iterdir() if path.is_dir()):
                if _holds_runs(shape_dir):
                    directories.setdefault(shape_dir.name, {})[PARAMETRIZED_VARIANT] = shape_dir
        elif _holds_runs(child):
            directories.setdefault(child.name, {})[STATIC_VARIANT] = child
    return directories


def _holds_runs(directory: Path) -> bool:
    """Whether ``directory`` holds recordings directly (nested ones belong to a
    separate set, e.g. the gain-MLP tree beside a static shape)."""
    return directory.is_dir() and any(path.is_file() for path in directory.glob("TR*"))


class RunLog(NamedTuple):
    """One recording: its name, the loaded log and the IMU yaw-rate stream
    (``pololu.log_loader.load_imu_gyro_z``; empty arrays when not logged)."""

    name: str
    log: object
    imu_time_s: np.ndarray
    imu_gyro_z: np.ndarray


def load_run_logs(
    directory: Path,
    max_runs: int | None = None,
    clip_after_first_trajectory: bool = True,
) -> list[RunLog]:
    """A ``RunLog`` for each ``TR*`` recording directly in ``directory``.

    Benchmark recordings are normally kept in the firmware's binary SD-card
    format, so anything that is not already a csv is decoded into a temporary
    directory -- reading an experiment for a plot must not write into it.
    """
    from wmr_simulator.pololu.decode_binary import decode_file
    from wmr_simulator.pololu.log_loader import load_imu_gyro_z, load_pololu_traj_control_log

    directory = Path(directory)
    if not directory.is_dir():
        return []
    log_paths = sorted(path for path in directory.glob("TR*") if path.is_file())
    # A decoded csv sits next to its binary as TRxx.csv, so both spellings of
    # the same run are in the glob; the csv is the one to read.
    decoded = {path.stem for path in log_paths if path.suffix.lower() == ".csv"}
    log_paths = [path for path in log_paths if path.suffix.lower() == ".csv" or path.name not in decoded]
    if max_runs is not None:
        log_paths = log_paths[:max_runs]

    logs: list[RunLog] = []
    with tempfile.TemporaryDirectory(prefix="baseline_runs_") as temp_name:
        temporary_dir = Path(temp_name)
        for log_path in log_paths:
            csv_path = log_path if log_path.suffix.lower() == ".csv" else temporary_dir / f"{log_path.name}.csv"
            if csv_path != log_path:
                with contextlib.redirect_stdout(io.StringIO()):
                    if not decode_file(str(log_path), str(csv_path)):
                        print(f"Baseline run skipped, unreadable log: {log_path}")
                        continue
            try:
                log = load_pololu_traj_control_log(
                    csv_path, clip_after_first_trajectory=clip_after_first_trajectory
                )
                imu_time, imu_gyro_z = load_imu_gyro_z(
                    csv_path, clip_after_first_trajectory=clip_after_first_trajectory
                )
                logs.append(RunLog(log_path.stem, log, imu_time, imu_gyro_z))
            except ValueError as error:
                print(f"Baseline run skipped {log_path}: {error}")
    return logs


def _baseline_run(run: RunLog) -> BaselineRun:
    return BaselineRun(
        name=run.name,
        poses=np.asarray(run.log.pose.states, dtype=float),
        time_s=np.asarray(run.log.pose.time_s, dtype=float),
        tracking_rmse=run_tracking_rmse(run.log),
        yaw_ringing=run_yaw_ringing(run),
    )


def run_tracking_rmse(log) -> float | None:
    """Position RMSE of a recorded run against the reference it tracked.

    Aligned on the reference timestamps, the same way ``progress`` scores a
    benchmark run: the reference is what the controller was asked for at those
    instants, and the mocap track is sampled on its own clock.
    """
    errors = _position_errors(log)
    if errors is None:
        return None
    return float(np.sqrt(np.mean(errors[1] ** 2)))


def _position_errors(log) -> tuple[np.ndarray, np.ndarray] | None:
    """``(time, distance)`` of the mocap position from the reference at every
    reference instant after the first, or None without enough of either."""
    reference_states = np.asarray(log.reference.states, dtype=float)
    reference_time = np.asarray(log.reference.time_s, dtype=float)
    if len(reference_states) < 2:
        return None
    target_time = reference_time[1:]
    pose_time = np.asarray(log.pose.time_s, dtype=float)
    pose_states = np.asarray(log.pose.states, dtype=float)
    if len(pose_time) < 2:
        return None
    error_x = np.interp(target_time, pose_time, pose_states[:, 0]) - reference_states[1:, 0]
    error_y = np.interp(target_time, pose_time, pose_states[:, 1]) - reference_states[1:, 1]
    return target_time, np.hypot(error_x, error_y)


def run_yaw_ringing(
    run: RunLog,
    band_hz: tuple[float, float] = YAW_RINGING_BAND_HZ,
    departure_radius: float = YAW_RINGING_DEPARTURE_RADIUS_M,
) -> float | None:
    """RMS of the band-passed IMU yaw-rate tracking error [rad/s].

    The smoothness counterpart to ``run_tracking_rmse``: position RMSE cannot
    tell a controller that tracks through a 3-4 Hz yaw oscillation from one that
    does not, and that oscillation is what drives the wheels into duty
    saturation and the robot off the fast circle. Read off the gyro rather than
    the mocap twist, which the Savitzky-Golay smoothing attenuates, or the
    encoders, which stop measuring the body once the wheels slip. The reference
    yaw rate is subtracted and the error band-passed (``YAW_RINGING_BAND_HZ``)
    over the reference's time span, cut at the first instant the robot is
    ``departure_radius`` off the reference (``YAW_RINGING_DEPARTURE_RADIUS_M``).
    The stream is resampled onto its median sample period first. None when the
    log carries no IMU samples, or less than ``YAW_RINGING_MIN_DURATION_S`` of
    them before the departure.

    Ringing only appears on fast references, so compare it per shape (or on the
    worst shape), never pooled over a set of slow and fast ones.
    """
    from scipy import signal

    reference_time = np.asarray(run.log.reference.time_s, dtype=float)
    reference_rate = np.asarray(run.log.reference.states, dtype=float)[:, 5]
    imu_time = np.asarray(run.imu_time_s, dtype=float)
    gyro = np.asarray(run.imu_gyro_z, dtype=float)
    if len(reference_time) < 2 or len(imu_time) < 3:
        return None
    end_time = reference_time[-1]
    errors = _position_errors(run.log)
    if errors is not None:
        departed = np.nonzero(errors[1] > departure_radius)[0]
        if len(departed):
            end_time = errors[0][departed[0]]
    inside = (imu_time >= reference_time[0]) & (imu_time <= end_time)
    if int(inside.sum()) < 3:
        return None
    imu_time, gyro = imu_time[inside], gyro[inside]
    period = float(np.median(np.diff(imu_time)))
    if period <= 0.0 or band_hz[1] >= 0.5 / period:
        return None
    grid = np.arange(imu_time[0], imu_time[-1], period)
    if grid[-1] - grid[0] < YAW_RINGING_MIN_DURATION_S:
        return None
    rate_error = np.interp(grid, imu_time, gyro) - np.interp(grid, reference_time, reference_rate)
    sos = signal.butter(2, band_hz, btype="bandpass", fs=1.0 / period, output="sos")
    ringing = signal.sosfiltfilt(sos, rate_error)
    return float(np.sqrt(np.mean(ringing**2)))


def variant_panels(records: list[BaselineIterationRuns]) -> tuple[tuple[str, str], ...]:
    """``(variant, panel title)`` for the variants any iteration recorded.

    An experiment with no static-gain baseline (iteration 1, or one with the
    parametrization off) has one variant and gets one panel rather than an empty
    second one.
    """
    present = {variant for record in records for variant in record.runs}
    return tuple(
        (variant, VARIANT_TITLES[variant]) for variant in VARIANT_ORDER if variant in present
    )
