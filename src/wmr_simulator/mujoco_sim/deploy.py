"""Run one deployment: config + trajectory in, an SD-card log out.

    ROBOTCFG.CFG / robot_config.yaml -> firmware.FirmwareConfig
    trajectory .JSN / .pkl           -> pololu.reference_importer
    plant.MujocoPlant  <-> firmware.Firmware  -> binlog.BinaryLogWriter

The start pose is drawn like a hand placement (uniform over a disk of
``init_offset_radius`` and +-``init_offset_angle``), or given as ``start_pose``
for a chained repeat run. Only the log is written to the target directory; the
ground truth is returned in :class:`DeploymentResult`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from wmr_simulator.mujoco_sim.binlog import BinaryLogWriter
from wmr_simulator.mujoco_sim.firmware import OUTER_OUTPUT_STAMP_DELAY_MS, Firmware, FirmwareConfig
from wmr_simulator.mujoco_sim.plant import HiddenPlantConfig, MujocoPlant

# Log timestamps start at the firmware's uptime.
DEFAULT_BOOT_TIME_MS = 15_000

# Hand-placement spread (gain_tuning.defaults).
DEFAULT_START_OFFSET_RADIUS = 0.05
DEFAULT_START_OFFSET_ANGLE = 0.2

MAX_LOG_INDEX = 100  # sdlog.rs::open_new_file tries TR00..TR99


@dataclass(frozen=True)
class DeploymentResult:
    """Where the log went, plus ground truth the pipeline is not allowed to see."""

    log_path: Path
    num_records: int
    duration: float
    start_pose: tuple[float, float, float]
    start_offset: tuple[float, float, float]
    # Where the robot ended up (the start of a chained next run).
    final_pose: tuple[float, float, float]
    # Scored over the first ``scored_duration`` seconds of the trajectory.
    tracking_rmse: float
    tracking_max: float
    final_pose_error: float
    max_duty: float
    duty_saturated_fraction: float
    scored_duration: float
    # Position RMSE over the whole run, bridge included.
    tracking_rmse_full: float


def run_deployment(
    robot_config: str | Path,
    trajectory: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 0,
    log_name: str | None = None,
    start_pose: Sequence[float] | None = None,
    start_offset_radius: float = DEFAULT_START_OFFSET_RADIUS,
    start_offset_angle: float = DEFAULT_START_OFFSET_ANGLE,
    plant_config: HiddenPlantConfig | None = None,
    boot_time_ms: int = DEFAULT_BOOT_TIME_MS,
    observer: Callable[["object", MujocoPlant], None] | None = None,
    score_duration: float | None = None,
) -> DeploymentResult:
    """Execute ``trajectory`` under ``robot_config`` and write a ``TRxx`` log.

    ``start_pose`` places the robot at ``(x, y, yaw)`` instead of drawing a
    hand placement (a chained repeat run). ``observer`` is called with
    ``(tick, plant)`` after every physics step (used by the video renderer).
    ``score_duration`` limits the ground-truth scores to the first seconds of
    the trajectory (None: the whole run).
    """
    from wmr_simulator.pololu.reference_importer import load_reference

    if score_duration is not None and score_duration <= 0.0:
        raise ValueError(f"score_duration must be positive, got {score_duration}")
    config = FirmwareConfig.from_file(robot_config)
    reference = load_reference(trajectory)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / (log_name if log_name is not None else next_log_name(output_dir))
    if log_path.exists():
        raise FileExistsError(f"{log_path} already exists; the firmware never overwrites a log either.")

    plant = MujocoPlant(plant_config, seed=seed)
    reference_start = np.asarray(reference.states[0], dtype=float)
    if start_pose is None:
        offset = sample_start_offset(np.random.default_rng(seed), start_offset_radius, start_offset_angle)
        start_pose = reference_start + offset
    else:
        start_pose = np.asarray(start_pose, dtype=float)
        if start_pose.shape != (3,):
            raise ValueError(f"start_pose must be (x, y, yaw), got shape {start_pose.shape}")
        offset = start_pose - reference_start
        offset[2] = math.atan2(math.sin(offset[2]), math.cos(offset[2]))
    plant.reset(*start_pose)

    # The EKF is initialized from the first (calibrated) mocap frame.
    first_mocap = config.mocap_calibration.correct(plant.read_mocap())
    firmware = Firmware(config, reference.states, reference.actions, _pose_of(first_mocap))
    firmware.reset_encoders(plant.encoder_counts())

    errors: list[float] = []
    duties: list[float] = []
    errors_full: list[float] = []
    score_end = math.inf if score_duration is None else float(score_duration) + 1e-9
    with BinaryLogWriter(log_path) as writer:
        for tick in firmware.clock(plant.timestep, boot_time_ms).ticks(firmware.duration):
            if tick.mocap:
                mocap = firmware.correct_mocap(plant.read_mocap())
                firmware.receive_mocap(_pose_of(mocap))
                writer.mocap(tick.t_ms, mocap)
            if tick.imu:
                writer.imu(tick.t_ms, *plant.read_imu())
            if tick.inner:
                duty = firmware.inner_tick(plant.encoder_counts())
                plant.set_duty(*duty)
                writer.motor(tick.t_ms, *duty)
                writer.encoder(tick.t_ms, *firmware.inner.omega_lp)
                if tick.time <= score_end:
                    duties.append(float(np.abs(duty).max()))
            if tick.outer:
                outputs = firmware.outer_tick(tick.time)
                setpoint = outputs.setpoint
                writer.setpoint(
                    tick.t_ms, setpoint.x_des, setpoint.y_des, setpoint.yaw_des, setpoint.v_ff, setpoint.w_ff
                )
                # The robot's own records land ~2 ms after the setpoint.
                output_t_ms = tick.t_ms + OUTER_OUTPUT_STAMP_DELAY_MS
                writer.wheel_cmd(output_t_ms, outputs.omega_left, outputs.omega_right)
                writer.tracking_error(output_t_ms, outputs.x_err, outputs.y_err, outputs.yaw_err)
                true_pose = plant.pose()
                error = math.hypot(setpoint.x_des - true_pose[0], setpoint.y_des - true_pose[1])
                errors_full.append(error)
                if tick.time <= score_end:
                    errors.append(error)
            plant.step()
            if observer is not None:
                observer(tick, plant)
        num_records = writer.num_records

    error_array = np.asarray(errors)
    duty_array = np.asarray(duties)
    if error_array.size == 0 or duty_array.size == 0:
        raise ValueError(f"score_duration {score_duration} s leaves no tick to score.")
    error_full_array = np.asarray(errors_full)
    return DeploymentResult(
        log_path=log_path,
        num_records=num_records,
        duration=firmware.duration,
        start_pose=tuple(float(value) for value in start_pose),
        start_offset=tuple(float(value) for value in offset),
        final_pose=tuple(float(value) for value in plant.pose()),
        tracking_rmse=float(np.sqrt((error_array**2).mean())),
        tracking_max=float(error_array.max()),
        final_pose_error=float(np.linalg.norm(plant.pose()[:2] - reference.states[-1, :2])),
        max_duty=float(duty_array.max()),
        duty_saturated_fraction=float(np.mean(duty_array >= 0.999)),
        scored_duration=float(firmware.duration if score_duration is None else min(score_duration, firmware.duration)),
        tracking_rmse_full=float(np.sqrt((error_full_array**2).mean())),
    )


def next_log_name(output_dir: str | Path) -> str:
    """First free ``TRxx``, the way ``sdlog.rs::open_new_file`` picks one."""
    output_dir = Path(output_dir)
    for index in range(MAX_LOG_INDEX):
        name = f"TR{index:02d}"
        # A decoded csv counts as taken.
        if not (output_dir / name).exists() and not (output_dir / f"{name}.csv").exists():
            return name
    raise RuntimeError(f"No free TRxx log slot in {output_dir} (TR00-TR{MAX_LOG_INDEX - 1:02d} all taken)")


def sample_start_offset(rng: np.random.Generator, offset_radius: float, offset_angle: float) -> np.ndarray:
    """``[dx, dy, dtheta]``, uniform over the disk by area and over +-angle
    (as ``trajectory_optimization.start_offsets``)."""
    if offset_radius <= 0.0 and offset_angle <= 0.0:
        return np.zeros(3)
    radius = offset_radius * math.sqrt(rng.uniform())
    bearing = rng.uniform(-math.pi, math.pi)
    return np.array([radius * math.cos(bearing), radius * math.sin(bearing), rng.uniform(-offset_angle, offset_angle)])


def _pose_of(mocap: Sequence[float]) -> tuple[float, float, float]:
    """``(x, y, yaw)`` out of a 6-DoF mocap record."""
    return (float(mocap[0]), float(mocap[1]), float(mocap[5]))
