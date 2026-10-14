"""Calibration of the mocap rigid body against the robot's wheel axle.

The mocap rigid body's origin and orientation are arbitrary, while the model,
controller and firmware assume the pose of the wheel-axle midpoint. The mocap
body frame is the robot frame turned by ``yaw_offset`` about the vertical, and
the rigid body's origin sits at ``offset`` = (dx, dy) from the axle midpoint,
in the mocap body frame:

    p_axle = p_mocap - Rz(yaw_mocap) offset,    yaw_robot = yaw_mocap - yaw_offset.

Estimation (``estimate_mocap_calibration``): without sideways slip, the mocap
origin's velocity in the mocap body frame is

    (vx_m, vy_m) = s v_enc (cos yaw_offset, -sin yaw_offset) + omega (-dy, dx),

linear in (s cos yaw_offset, s sin yaw_offset, dx, dy), with the encoder speed
scale ``s`` fitted. Straight runs determine the heading offset, turns on the
spot the origin offset; ``calibration_reference_states`` does both.

The firmware applies the correction to every mocap frame (``mocap_*`` keys of
ROBOTCFG.CFG); ``mujoco_sim.firmware`` mirrors it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

# ROBOTCFG.CFG keys (firmware read_robot_config_from_sd.rs); absent means zero.
CONFIG_KEYS = ("mocap_yaw_offset", "mocap_offset_x", "mocap_offset_y")

# Calibration samples are used only below this encoder speed and lateral
# acceleration (no wheel slip).
MAX_SPEED = 1.0
MAX_LATERAL_ACCELERATION = 1.5


@dataclass(frozen=True)
class MocapCalibration:
    """``yaw_offset`` [rad] = mocap yaw - robot yaw; ``offset`` [m] = rigid-body
    origin minus axle midpoint, in the mocap body frame."""

    yaw_offset: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)

    def config_values(self) -> dict[str, float]:
        return dict(zip(CONFIG_KEYS, (self.yaw_offset, *self.offset)))


    def to_mapping(self) -> dict:
        return {
            "yaw_offset_rad": float(self.yaw_offset),
            "yaw_offset_deg": round(math.degrees(self.yaw_offset), 4),
            "offset_m": [float(value) for value in self.offset],
        }

    @classmethod
    def from_mapping(cls, values: Mapping) -> "MocapCalibration":
        return cls(float(values["yaw_offset_rad"]), tuple(float(v) for v in values["offset_m"]))

    def save(self, path: str | Path, **extra) -> Path:
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({**self.to_mapping(), **extra}, sort_keys=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MocapCalibration":
        import yaml

        return cls.from_mapping(yaml.safe_load(Path(path).read_text()))

    def correct(self, poses: np.ndarray) -> np.ndarray:
        """Axle poses from mocap frames ``(..., 6)`` = ``(x, y, z, roll, pitch,
        yaw)``; same layout back, z, roll and pitch passed through."""
        poses = np.asarray(poses, dtype=float)
        corrected = poses.copy()
        dx, dy = self.offset
        cos_yaw, sin_yaw = np.cos(poses[..., 5]), np.sin(poses[..., 5])
        corrected[..., 0] = poses[..., 0] - (cos_yaw * dx - sin_yaw * dy)
        corrected[..., 1] = poses[..., 1] - (sin_yaw * dx + cos_yaw * dy)
        corrected[..., 5] = _wrap(poses[..., 5] - self.yaw_offset)
        return corrected

    def compose(self, residual: "MocapCalibration") -> "MocapCalibration":
        """The calibration equivalent to applying ``self`` and then ``residual``
        (estimated on logs that ``self`` had already corrected)."""
        c, s = math.cos(self.yaw_offset), math.sin(self.yaw_offset)
        rx, ry = residual.offset
        # The residual's offset is in the frame of the corrected yaw, which is
        # the mocap yaw turned back by self.yaw_offset.
        offset = (self.offset[0] + c * rx + s * ry, self.offset[1] - s * rx + c * ry)
        return MocapCalibration(float(_wrap(self.yaw_offset + residual.yaw_offset)), offset)


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


# ------------------------------------------------------------------ trajectory


def calibration_reference_states(
    dt: float = 0.05,
    *,
    # Along the arena's long axis (y).
    start=(0.0, 0.0, 0.5 * math.pi),
    length: float = 1.4,
    peak_speed: float = 0.6,
    spin_angle: float = 3.0 * math.pi,
    spin_rate: float = 5.0,
) -> np.ndarray:
    """``[x, y, theta, vx, vy, omega, ax, ay]`` of the calibration run: a slow
    straight run, a turn on the spot of 1.5 revolutions, and the same back to
    the start pose. Every piece is a rest-to-rest minimum-jerk profile.
    """
    from wmr_simulator.trajectory_optimization.reference_extension import (
        _minimum_jerk_path,
        _path_states,
        straight_line,
    )

    width = 8
    rows = [np.array([[*start, 0.0, 0.0, 0.0, 0.0, 0.0]])]

    def pose():
        return rows[-1][-1, :3]

    def spin(angle: float):
        turned, rate, _ = _minimum_jerk_path(abs(angle), 1.875 * abs(angle) / spin_rate, dt)
        sign = 1.0 if angle >= 0.0 else -1.0
        x0, y0, theta0 = (float(value) for value in pose())
        zeros = np.zeros_like(turned)
        rows.append(_path_states(zeros + x0, zeros + y0, theta0 + sign * turned, zeros, sign * rate, zeros, width))

    for direction in (1.0, -1.0):
        rows.append(straight_line(pose(), length, dt, width, peak_speed=peak_speed, min_duration=0.0))
        spin(direction * spin_angle)
    return np.vstack(rows)


def export_calibration_reference(output_dir: str | Path, name: str = "mocap_calibration", dt: float = 0.05) -> tuple[Path, Path]:
    """Write the calibration run as a reference pickle and a compact firmware
    JSN (``<name>.pkl``, ``<name>.JSN``); refuses a JSN over the firmware limits."""
    import json
    import pickle

    from wmr_simulator.pololu.reference_exporter import (
        ReferenceTrajectory,
        check_firmware_limits,
        format_pololu_reference,
    )

    states = calibration_reference_states(dt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / f"{name}.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump({"reference_states": states, "dt": dt}, file)
    jsn_path = output_dir / f"{name}.JSN"
    payload = format_pololu_reference(ReferenceTrajectory(pickle_path, states, dt))
    jsn_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    problems = check_firmware_limits(len(states), jsn_path.stat().st_size, label=str(jsn_path))
    if problems:
        raise ValueError("; ".join(problems))
    return pickle_path, jsn_path


# ------------------------------------------------------------------ estimation


def calibration_samples(
    path: str | Path,
    wheel_radius: float,
    *,
    max_speed: float = MAX_SPEED,
    max_lateral_acceleration: float = MAX_LATERAL_ACCELERATION,
    edge_s: float = 0.3,
    yaw_rate_source: str = "mocap",
) -> np.ndarray:
    """``(N, 4)`` = mocap body-frame velocity ``(vx_m, vy_m)``, encoder speed and
    yaw rate of one decoded log, at the mocap samples (the log loader's
    smoothed mocap twist and its lag-aligned encoder speeds).

    ``yaw_rate_source`` ``"gyro"`` takes the yaw rate from the IMU instead of
    the differentiated mocap heading (the (vx_m, vy_m) targets stay mocap). The
    gyro bias is the mean gyro-minus-mocap difference over the usable samples,
    which cancels for the run's symmetric CCW/CW turns; the gyro scale is not
    corrected (see ``gyro_scale`` in the diagnostics of the estimate).
    """
    from wmr_simulator.pololu.log_loader import load_imu_gyro_z, load_pololu_traj_control_log

    if yaw_rate_source not in ("mocap", "gyro"):
        raise ValueError(f"yaw_rate_source must be 'mocap' or 'gyro', got {yaw_rate_source!r}")

    log = load_pololu_traj_control_log(path)
    pose_time = np.asarray(log.pose.time_s, dtype=float)
    twist = np.asarray(log.pose.twists, dtype=float)  # (vx, vy, omega) in the mocap body frame
    wheel_time = np.asarray(log.wheel.time_s, dtype=float)
    speeds = np.asarray(log.wheel.speeds, dtype=float)  # (right, left)
    speed = wheel_radius * 0.5 * np.interp(pose_time, wheel_time, speeds.sum(axis=1))
    usable = pose_time >= max(pose_time[0], wheel_time[0]) + edge_s
    usable &= pose_time <= min(pose_time[-1], wheel_time[-1]) - edge_s
    usable &= np.abs(speed) < max_speed
    usable &= np.abs(speed * twist[:, 2]) < max_lateral_acceleration
    omega = twist[:, 2]
    if yaw_rate_source == "gyro":
        gyro_time, gyro = load_imu_gyro_z(path)
        if len(gyro) < 2:
            raise ValueError(f"{path} has no gyro samples.")
        omega = np.interp(pose_time, gyro_time, gyro)
        omega = omega - float(np.mean(omega[usable] - twist[usable, 2]))
    return np.column_stack([twist[usable, 0], twist[usable, 1], speed[usable], omega[usable]])


def _solve(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares for (s cos psi, s sin psi, dx, dy): solution, residuals,
    standard errors."""
    vx, vy, speed, omega = samples.T
    zeros = np.zeros_like(speed)
    design = np.vstack([
        np.column_stack([speed, zeros, zeros, -omega]),
        np.column_stack([zeros, -speed, omega, zeros]),
    ])
    target = np.concatenate([vx, vy])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ solution
    covariance = np.linalg.pinv(design.T @ design) * float(residual @ residual) / max(len(target) - 4, 1)
    return solution, residual, np.sqrt(np.diag(covariance))


def _calibration(solution: np.ndarray) -> tuple[MocapCalibration, float]:
    cos_part, sin_part, dx, dy = (float(value) for value in solution)
    return MocapCalibration(math.atan2(sin_part, cos_part), (dx, dy)), math.hypot(cos_part, sin_part)


def estimate_mocap_calibration(
    log_paths: Iterable[str | Path], wheel_radius: float, **sample_options
) -> tuple[MocapCalibration, dict]:
    """Least-squares calibration from decoded logs of the calibration run, and
    diagnostics (speed scale ``s``, residual RMS, standard errors, per-log
    estimates)."""
    log_paths = [Path(path) for path in log_paths]
    samples = [calibration_samples(path, wheel_radius, **sample_options) for path in log_paths]
    if not samples:
        raise ValueError("No calibration logs.")
    solution, residual, standard_error = _solve(np.vstack(samples))
    calibration, scale = _calibration(solution)
    per_log = []
    for path, sample in zip(log_paths, samples):
        single, _ = _calibration(_solve(sample)[0])
        per_log.append({
            "log": str(path),
            "samples": int(len(sample)),
            "yaw_offset_deg": round(math.degrees(single.yaw_offset), 3),
            "offset_mm": [round(1000.0 * value, 2) for value in single.offset],
        })
    diagnostics = {
        "samples": int(sum(len(sample) for sample in samples)),
        "speed_scale": round(scale, 4),
        "residual_rms_m_s": round(float(np.sqrt(np.mean(residual**2))), 5),
        "yaw_offset_standard_error_deg": round(math.degrees(float(np.hypot(*standard_error[:2])) / scale), 4),
        "offset_standard_error_mm": [round(1000.0 * float(value), 3) for value in standard_error[2:]],
        "per_log": per_log,
    }
    return calibration, diagnostics
