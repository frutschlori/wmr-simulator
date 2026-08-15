"""Writer for the firmware's Compact Tagged Binary SD-card log.

The deployment has to hand the pipeline a file that is indistinguishable from
one pulled off the SD card, so this is the exact inverse of
``pololu/decode_binary.py`` - which is why the magic header and the per-tag
payload widths are imported from there rather than restated: one spec, two
directions.

Layout: the 4-byte magic, then records of ``<u32 t_ms><u8 tag><N x f32 LE>``
(``sdlog.rs::log_event_compact``). The decoder buckets records by *exact*
millisecond into CSV rows, in order of first appearance, so the caller's
timestamps decide the row structure.
"""

from __future__ import annotations

import math
import struct
from enum import IntEnum
from pathlib import Path
from typing import Sequence

from wmr_simulator.pololu.decode_binary import FLOAT_COUNTS, MAGIC_HEADER


class LogTag(IntEnum):
    """The firmware's ``LogEvent`` discriminants."""

    EKF_STATE = 1
    SETPOINT = 2
    WHEEL_CMD = 3
    TRACKING_ERROR = 4
    MOTOR = 5
    ENCODER = 6
    MOCAP = 7
    ODOM = 8
    IMU = 9


# What the robot actually produces. EkfState and Odom exist in the firmware's
# enum but nothing ever queues them, so a real log leaves the csv's x/y/yaw and
# v_actual/w_actual columns empty - writing them here would make a deployment
# log recognisable at a glance, and would hand identification a pose stream the
# real pipeline never sees.
WRITTEN_TAGS = frozenset(
    {LogTag.SETPOINT, LogTag.WHEEL_CMD, LogTag.TRACKING_ERROR, LogTag.MOTOR, LogTag.ENCODER, LogTag.MOCAP, LogTag.IMU}
)

MAX_TIMESTAMP_MS = 2**32 - 1


class BinaryLogWriter:
    """Append records to one ``TRxx`` log file.

    Use it as a context manager; the magic header is written on open. The
    per-tag helpers take the payloads in the same shape the plant hands them
    out, so the firmware port can pass sensor readings straight through.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "wb")
        self._file.write(MAGIC_HEADER)
        self.num_records = 0

    def __enter__(self) -> "BinaryLogWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    # --------------------------------------------------------------- records

    def write(self, t_ms: int, tag: LogTag, values: Sequence[float]) -> None:
        """Write one record. ``t_ms`` is the firmware's uptime in milliseconds."""
        tag = LogTag(tag)
        if tag not in WRITTEN_TAGS:
            raise ValueError(
                f"{tag.name} is never written by the real firmware; writing it would fill csv columns "
                "that a real log leaves empty."
            )
        expected = FLOAT_COUNTS[int(tag)]
        if len(values) != expected:
            raise ValueError(f"{tag.name} takes {expected} floats, got {len(values)}")
        if not all(math.isfinite(value) for value in values):
            # The decoder renders NaN as an empty csv cell, so a non-finite
            # value would silently vanish instead of failing loudly.
            raise ValueError(f"{tag.name} payload must be finite, got {list(values)}")
        if not isinstance(t_ms, (int, float)) or not 0 <= t_ms <= MAX_TIMESTAMP_MS:
            raise ValueError(f"Timestamp {t_ms} is not a u32 millisecond count")

        self._file.write(struct.pack("<IB", int(t_ms), int(tag)))
        self._file.write(struct.pack(f"<{expected}f", *(float(value) for value in values)))
        self.num_records += 1

    def setpoint(self, t_ms: int, x_des: float, y_des: float, yaw_des: float, v_ff: float, w_ff: float) -> None:
        self.write(t_ms, LogTag.SETPOINT, (x_des, y_des, yaw_des, v_ff, w_ff))

    def wheel_cmd(self, t_ms: int, omega_left: float, omega_right: float) -> None:
        self.write(t_ms, LogTag.WHEEL_CMD, (omega_left, omega_right))

    def tracking_error(self, t_ms: int, x_err: float, y_err: float, yaw_err: float) -> None:
        self.write(t_ms, LogTag.TRACKING_ERROR, (x_err, y_err, yaw_err))

    def motor(self, t_ms: int, duty_left: float, duty_right: float) -> None:
        self.write(t_ms, LogTag.MOTOR, (duty_left, duty_right))

    def encoder(self, t_ms: int, omega_left_lp: float, omega_right_lp: float) -> None:
        """The *low-passed* wheel speeds - that is what the inner loop logs."""
        self.write(t_ms, LogTag.ENCODER, (omega_left_lp, omega_right_lp))

    def mocap(self, t_ms: int, pose: Sequence[float]) -> None:
        """``(x, y, z, roll, pitch, yaw)``, as ``MujocoPlant.read_mocap`` returns it."""
        self.write(t_ms, LogTag.MOCAP, pose)

    def imu(self, t_ms: int, accel: Sequence[float], gyro: Sequence[float]) -> None:
        """Accelerometer in g and gyro in deg/s, as ``MujocoPlant.read_imu`` returns them."""
        self.write(t_ms, LogTag.IMU, (*accel, *gyro))
