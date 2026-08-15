"""Run one deployment: config + trajectory in, an SD-card log out.

This is the piece that stands in for carrying the SD card to the robot and
back. It wires the three halves together and does nothing else of its own:

    ROBOTCFG.CFG / robot_config.yaml -> firmware.FirmwareConfig
    trajectory .JSN                  -> pololu.reference_importer
    plant.MujocoPlant  <-> firmware.Firmware  -> binlog.BinaryLogWriter

The only randomization is the start pose: the plant's parameters stay fixed so
the identification benchmark has one true parameter vector. The offset
distribution is the pipeline's own (``gain_tuning.defaults``): uniform over a
disk of ``init_offset_radius`` and uniform over +-``init_offset_angle``, which
is how the robot is placed by hand.
A repeat run instead passes ``start_pose`` and begins where the last one ended,
the way a bridged trajectory is repeated on the robot without touching it.

**Nothing but the log may be written into the target directory.** The
pipeline's ``decode-logs`` stage tries to decode every non-csv, non-CFG file it
finds in ``data/``, so a ground-truth sidecar next to the log would break it -
and would leak the hidden truth into the pipeline besides. The ground-truth
diagnostics come back in :class:`DeploymentResult` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from wmr_simulator.mujoco_sim.binlog import BinaryLogWriter
from wmr_simulator.mujoco_sim.firmware import OUTER_OUTPUT_STAMP_DELAY_MS, Firmware, FirmwareConfig
from wmr_simulator.mujoco_sim.plant import HiddenPlantConfig, MujocoPlant

# Real logs start at the firmware's uptime, not at zero (59 s in the sample
# log this was checked against). Any plausible offset works; a fixed one keeps
# runs comparable.
DEFAULT_BOOT_TIME_MS = 15_000

# gain_tuning.defaults: measured across the exp04/exp05 logs, where the robot
# was placed 31-100 mm and up to 9.5 deg off the reference start.
DEFAULT_START_OFFSET_RADIUS = 0.1
DEFAULT_START_OFFSET_ANGLE = 0.3

MAX_LOG_INDEX = 100  # sdlog.rs::open_new_file tries TR00..TR99


@dataclass(frozen=True)
class DeploymentResult:
    """Where the log went, plus ground truth the pipeline is not allowed to see."""

    log_path: Path
    num_records: int
    duration: float
    start_pose: tuple[float, float, float]
    start_offset: tuple[float, float, float]
    # Where the robot ended up. Chaining runs starts the next one here instead
    # of drawing a new placement, which is what driving a bridged trajectory
    # buys on the real robot: nobody carries it back to the start.
    final_pose: tuple[float, float, float]
    tracking_rmse: float
    tracking_max: float
    final_pose_error: float
    max_duty: float
    duty_saturated_fraction: float


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
) -> DeploymentResult:
    """Execute ``trajectory`` under ``robot_config`` and write a ``TRxx`` log.

    ``start_pose`` places the robot at a given ``(x, y, yaw)`` instead of
    drawing a hand placement -- how a *repeat* run starts, from wherever the
    previous one's bridge path left the robot. The reported ``start_offset`` is
    then how far off the trajectory's start that happened to be, which is the
    same quantity as a drawn offset and comparable with it.
    """
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    config = FirmwareConfig.from_file(robot_config)
    reference = load_pololu_reference(trajectory)

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

    # The firmware waits for the first mocap frame before starting the outer
    # loop, and initializes the EKF from it (`wait_for_ekf_init`).
    first_mocap = plant.read_mocap()
    firmware = Firmware(config, reference.states, reference.actions, _pose_of(first_mocap))
    firmware.reset_encoders(plant.encoder_counts())

    errors: list[float] = []
    duties: list[float] = []
    with BinaryLogWriter(log_path) as writer:
        for tick in firmware.clock(plant.timestep, boot_time_ms).ticks(firmware.duration):
            if tick.mocap:
                mocap = plant.read_mocap()
                firmware.receive_mocap(_pose_of(mocap))
                writer.mocap(tick.t_ms, mocap)
            if tick.imu:
                writer.imu(tick.t_ms, *plant.read_imu())
            if tick.inner:
                duty = firmware.inner_tick(plant.encoder_counts())
                plant.set_duty(*duty)
                writer.motor(tick.t_ms, *duty)
                writer.encoder(tick.t_ms, *firmware.inner.omega_lp)
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
                errors.append(math.hypot(setpoint.x_des - true_pose[0], setpoint.y_des - true_pose[1]))
            plant.step()
        num_records = writer.num_records

    error_array = np.asarray(errors)
    duty_array = np.asarray(duties)
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
    )


def next_log_name(output_dir: str | Path) -> str:
    """First free ``TRxx``, the way ``sdlog.rs::open_new_file`` picks one."""
    output_dir = Path(output_dir)
    for index in range(MAX_LOG_INDEX):
        name = f"TR{index:02d}"
        # The decoded csv counts as taken too: the pipeline writes it next to
        # the binary, and reusing the name would silently mix two runs.
        if not (output_dir / name).exists() and not (output_dir / f"{name}.csv").exists():
            return name
    raise RuntimeError(f"No free TRxx log slot in {output_dir} (TR00-TR{MAX_LOG_INDEX - 1:02d} all taken)")


def sample_start_offset(rng: np.random.Generator, offset_radius: float, offset_angle: float) -> np.ndarray:
    """``[dx, dy, dtheta]``, uniform over the disk by area and over +-angle.

    Same distribution as ``trajectory_optimization.start_offsets`` draws for the
    gain tuner, so a deployment starts the way the tuner assumed it would.
    """
    if offset_radius <= 0.0 and offset_angle <= 0.0:
        return np.zeros(3)
    radius = offset_radius * math.sqrt(rng.uniform())
    bearing = rng.uniform(-math.pi, math.pi)
    return np.array([radius * math.cos(bearing), radius * math.sin(bearing), rng.uniform(-offset_angle, offset_angle)])


def _pose_of(mocap: Sequence[float]) -> tuple[float, float, float]:
    """``(x, y, yaw)`` out of a 6-DoF mocap record."""
    return (float(mocap[0]), float(mocap[1]), float(mocap[5]))
