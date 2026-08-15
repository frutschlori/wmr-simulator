"""The hidden MuJoCo plant that stands in for the real robot.

This module owns the ground truth: it is the only place
``models/pololu_calibrated.xml`` and ``models/pololu_hidden.yaml`` are read.
Everything downstream (the firmware port, the deployment driver) sees the plant
exclusively through duty commands in and *sensor* readings out - integer
encoder counts, a noisy mocap pose and a quantized IMU - the same interface the
firmware has to the real robot.

The true parameters are applied by patching the XML and compiling with
``from_xml_string``: structural elements (the IMU site and its sensors, and the
caster ``<pair>``) cannot be added to an already compiled ``MjModel``.

Units at the boundary match the hardware, not MuJoCo: the accelerometer reports
g and the gyro deg/s, because that is what the LSM6DSO driver returns and what
the SD-card logs contain.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# The hidden plant is a repo asset, not user config, so its two files resolve
# against the repo root and not the caller's cwd: a deployment is driven from
# wherever the experiment lives, which is rarely the checkout.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIDDEN_CONFIG_PATH = REPO_ROOT / "models/pololu_hidden.yaml"

# Standard gravity, for the accelerometer's m/s^2 -> g conversion.
GRAVITY = 9.80665

IMU_SITE_NAME = "imu"
ACCELEROMETER_SENSOR_NAME = "imu_acc"
GYRO_SENSOR_NAME = "imu_gyro"


@dataclass(frozen=True)
class MotorSpec:
    """Velocity actuator, parametrized by three physical quantities."""

    omega_max: float
    stall_torque: float
    time_constant: float

    @property
    def kv(self) -> float:
        """Actuator gain: torque = kv * (ctrl - qvel), i.e. (kt/R) of a DC motor."""
        return self.stall_torque / self.omega_max

    @property
    def armature(self) -> float:
        """Reflected inertia giving the requested free wheel-speed time constant."""
        return self.kv * self.time_constant

    @property
    def ctrl_limit(self) -> float:
        return 1.2 * self.omega_max


@dataclass(frozen=True)
class GeometrySpec:
    wheel_radius: float
    wheel_width: float
    half_track: float


@dataclass(frozen=True)
class FrictionSpec:
    floor: float
    wheel: float
    caster: float


@dataclass(frozen=True)
class EncoderSpec:
    counts_per_revolution: float


@dataclass(frozen=True)
class ImuSpec:
    site_pos: tuple[float, float, float]
    site_euler: tuple[float, float, float]
    accel_lsb_g: float
    gyro_lsb_dps: float
    accel_noise_g: float
    gyro_noise_dps: float
    accel_bias_g: tuple[float, float, float]
    gyro_bias_dps: tuple[float, float, float]


@dataclass(frozen=True)
class MocapSpec:
    offset_xyz: tuple[float, float, float]
    position_noise_std: float
    angle_noise_std: float


@dataclass(frozen=True)
class HiddenPlantConfig:
    """Everything the plant knows and the pipeline does not."""

    model_path: Path
    motor: MotorSpec
    geometry: GeometrySpec
    friction: FrictionSpec
    encoder: EncoderSpec
    imu: ImuSpec
    mocap: MocapSpec
    settle_time: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HiddenPlantConfig":
        sections = {
            "motor": MotorSpec,
            "geometry": GeometrySpec,
            "friction": FrictionSpec,
            "encoder": EncoderSpec,
            "imu": ImuSpec,
            "mocap": MocapSpec,
        }
        known = set(sections) | {"model_path", "settle_time"}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown hidden-plant config keys: {unknown}")
        missing = sorted(known - set(values))
        if missing:
            raise ValueError(f"Hidden-plant config is missing keys: {missing}")
        return cls(
            model_path=_resolve_repo_path(values["model_path"]),
            settle_time=float(values["settle_time"]),
            **{name: _build_spec(spec, values[name], name) for name, spec in sections.items()},
        )


def _resolve_repo_path(value: str | Path) -> Path:
    """A relative asset path is repo-relative; an absolute one is left alone."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_hidden_plant_config(path: str | Path = DEFAULT_HIDDEN_CONFIG_PATH) -> HiddenPlantConfig:
    """Read the hidden ground-truth yaml."""
    import yaml

    with open(path, "r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    return HiddenPlantConfig.from_mapping(values)


def build_plant_xml(config: HiddenPlantConfig) -> str:
    """Apply the hidden parameters to the model XML and return the patched source.

    The yaml is authoritative: every value it carries is written into the XML
    here, so the two can never silently disagree.
    """
    root = ET.parse(config.model_path).getroot()

    motor = config.motor
    for name in ("left_motor", "right_motor"):
        actuator = _require(root, f".//velocity[@name='{name}']", config.model_path)
        actuator.set("kv", repr(motor.kv))
        actuator.set("ctrlrange", f"{-motor.ctrl_limit} {motor.ctrl_limit}")

    geometry = config.geometry
    for side, sign in (("left", 1.0), ("right", -1.0)):
        body = _require(root, f".//body[@name='{side}_wheel']", config.model_path)
        body.set("pos", f"0 {sign * geometry.half_track} 0")
        joint = _require(root, f".//joint[@name='{side}_wheel_joint']", config.model_path)
        joint.set("armature", repr(motor.armature))
        geom = _require(root, f".//geom[@name='{side}_wheel_geom']", config.model_path)
        geom.set("size", f"{geometry.wheel_radius} {geometry.wheel_width}")
        _set_sliding_friction(geom, config.friction.wheel)

    _set_sliding_friction(_require(root, ".//geom[@name='floor']", config.model_path), config.friction.floor)
    # The caster's friction lives on the <pair>, which is the whole point of
    # that block: MuJoCo otherwise combines geom friction as the elementwise
    # maximum and the "frictionless" ball inherits the floor's.
    pair = _require(root, ".//pair[@geom2='caster_geom']", config.model_path)
    _set_sliding_friction(pair, config.friction.caster, num_sliding=2)

    # The IMU site and its sensors are not in the delivered model, and cannot be
    # added after compilation - hence the XML-patch path.
    chassis = _require(root, ".//body[@name='chassis']", config.model_path)
    ET.SubElement(
        chassis,
        "site",
        {
            "name": IMU_SITE_NAME,
            "pos": " ".join(repr(value) for value in config.imu.site_pos),
            "euler": " ".join(repr(value) for value in config.imu.site_euler),
            "size": "0.002",
            "rgba": "1 0 0 0",
        },
    )
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "accelerometer", {"name": ACCELEROMETER_SENSOR_NAME, "site": IMU_SITE_NAME})
    ET.SubElement(sensor, "gyro", {"name": GYRO_SENSOR_NAME, "site": IMU_SITE_NAME})

    return ET.tostring(root, encoding="unicode")


class MujocoPlant:
    """The robot as physics plus sensors: duty in, sensor readings out.

    Nothing here reads a pipeline config. The only inputs are the two duty
    values the firmware's inner loop produces, and the only outputs are what the
    firmware can actually measure, plus explicitly-named ground-truth accessors
    for validation (``pose``, ``twist``, ``wheel_speeds``) that the firmware port
    must not use.
    """

    def __init__(
        self,
        config: HiddenPlantConfig | None = None,
        seed: int = 0,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.config = load_hidden_plant_config() if config is None else config
        self.model = mujoco.MjModel.from_xml_string(build_plant_xml(self.config))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        root_joint = self._joint_id("root")
        self._root_qpos = int(self.model.jnt_qposadr[root_joint])
        self._root_dof = int(self.model.jnt_dofadr[root_joint])
        self._wheel_qpos = tuple(
            int(self.model.jnt_qposadr[self._joint_id(f"{side}_wheel_joint")]) for side in ("left", "right")
        )
        self._wheel_dof = tuple(
            int(self.model.jnt_dofadr[self._joint_id(f"{side}_wheel_joint")]) for side in ("left", "right")
        )
        self._accel_adr = self._sensor_slice(ACCELEROMETER_SENSOR_NAME)
        self._gyro_adr = self._sensor_slice(GYRO_SENSOR_NAME)

        self.reset()

    # ------------------------------------------------------------------ clock

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def steps_for(self, duration: float) -> int:
        """Physics steps closest to ``duration`` seconds."""
        return max(1, int(round(duration / self.timestep)))

    # ------------------------------------------------------------ actuation

    def reset(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        """Settle onto the caster, then place the robot at ``(x, y, yaw)`` at rest.

        Settling first and placing afterwards keeps the contact penetration at
        its steady value: dropping the chassis in at its nominal height and
        driving away immediately would start every run with a bounce.
        """
        mujoco = self._mujoco
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        for _ in range(self.steps_for(self.config.settle_time)):
            mujoco.mj_step(self.model, self.data)

        quat = np.array(self.data.qpos[self._root_qpos + 3 : self._root_qpos + 7])
        settled_yaw = _quat_to_euler(quat)[2]
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.array([0.0, 0.0, 1.0]), yaw - settled_yaw)
        rotated = np.zeros(4)
        mujoco.mju_mulQuat(rotated, turn, quat)

        self.data.qpos[self._root_qpos + 0] = x
        self.data.qpos[self._root_qpos + 1] = y
        self.data.qpos[self._root_qpos + 3 : self._root_qpos + 7] = rotated
        for adr in self._wheel_qpos:
            self.data.qpos[adr] = 0.0
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_duty(self, duty_left: float, duty_right: float) -> None:
        """Apply the inner loop's duty command, clamped exactly as the H-bridge is."""
        omega_max = self.config.motor.omega_max
        self.data.ctrl[0] = float(np.clip(duty_left, -1.0, 1.0)) * omega_max
        self.data.ctrl[1] = float(np.clip(duty_right, -1.0, 1.0)) * omega_max

    def step(self, num_steps: int = 1) -> None:
        for _ in range(num_steps):
            self._mujoco.mj_step(self.model, self.data)

    # --------------------------------------------------------------- sensors

    def encoder_counts(self) -> tuple[int, int]:
        """Integer quadrature counts, at the true CPR and in the wiring's sign.

        The firmware turns count differences back into rad/s with the CPR from
        its own config file; the two agree by construction (see the hidden
        yaml), so this is a quantizer, not an unknown parameter.
        """
        scale = self.config.encoder.counts_per_revolution / (2.0 * math.pi)
        angles = self.wheel_angles()
        return (int(round(angles[0] * scale)), int(round(angles[1] * scale)))

    def read_mocap(self) -> np.ndarray:
        """Noisy marker pose ``(x, y, z, roll, pitch, yaw)``, as tag 7 logs it."""
        spec = self.config.mocap
        pose = self.marker_pose()
        noise = np.concatenate(
            (
                self.rng.normal(0.0, spec.position_noise_std, size=3),
                self.rng.normal(0.0, spec.angle_noise_std, size=3),
            )
        )
        return pose + noise

    def read_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Accelerometer in g and gyro in deg/s, biased, noisy and quantized.

        Degrees, because the LSM6DSO driver returns dps and the logs carry it
        raw - the rest of the codebase converts on load.
        """
        spec = self.config.imu
        accel = np.array(self.data.sensordata[self._accel_adr]) / GRAVITY
        gyro = np.rad2deg(self.data.sensordata[self._gyro_adr])
        accel = accel + np.asarray(spec.accel_bias_g) + self.rng.normal(0.0, spec.accel_noise_g, size=3)
        gyro = gyro + np.asarray(spec.gyro_bias_dps) + self.rng.normal(0.0, spec.gyro_noise_dps, size=3)
        return _quantize(accel, spec.accel_lsb_g), _quantize(gyro, spec.gyro_lsb_dps)

    # ---------------------------------------------------------- ground truth

    def pose(self) -> np.ndarray:
        """True ``(x, y, yaw)`` of the wheel-axle midpoint."""
        position = self.data.qpos[self._root_qpos : self._root_qpos + 3]
        yaw = _quat_to_euler(self.data.qpos[self._root_qpos + 3 : self._root_qpos + 7])[2]
        return np.array([position[0], position[1], yaw])

    def marker_pose(self) -> np.ndarray:
        """True ``(x, y, z, roll, pitch, yaw)`` of the mocap marker frame."""
        position = np.array(self.data.qpos[self._root_qpos : self._root_qpos + 3])
        quat = np.array(self.data.qpos[self._root_qpos + 3 : self._root_qpos + 7])
        rotation = np.zeros(9)
        self._mujoco.mju_quat2Mat(rotation, quat)
        position = position + rotation.reshape(3, 3) @ np.asarray(self.config.mocap.offset_xyz)
        return np.concatenate((position, _quat_to_euler(quat)))

    def twist(self) -> tuple[float, float]:
        """True body-frame ``(v, w)``."""
        velocity = self.data.qvel[self._root_dof : self._root_dof + 3]
        yaw = self.pose()[2]
        forward = velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw)
        return float(forward), float(self.data.qvel[self._root_dof + 5])

    def wheel_angles(self) -> tuple[float, float]:
        return (float(self.data.qpos[self._wheel_qpos[0]]), float(self.data.qpos[self._wheel_qpos[1]]))

    def wheel_speeds(self) -> tuple[float, float]:
        return (float(self.data.qvel[self._wheel_dof[0]]), float(self.data.qvel[self._wheel_dof[1]]))

    # --------------------------------------------------------------- private

    def _joint_id(self, name: str) -> int:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"No joint named {name!r} in {self.config.model_path}")
        return joint_id

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"No sensor named {name!r} in the patched model")
        start = int(self.model.sensor_adr[sensor_id])
        return slice(start, start + int(self.model.sensor_dim[sensor_id]))


def _build_spec(spec_type: type, values: Mapping[str, Any], section: str):
    names = {field.name for field in fields(spec_type)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown keys in hidden-plant config section {section!r}: {unknown}")
    missing = sorted(names - set(values))
    if missing:
        raise ValueError(f"Hidden-plant config section {section!r} is missing keys: {missing}")
    return spec_type(**{name: _coerce(value) for name, value in values.items()})


def _coerce(value: Any):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(entry) for entry in value)
    return float(value)


def _require(root: ET.Element, path: str, source: Path) -> ET.Element:
    element = root.find(path)
    if element is None:
        raise ValueError(f"{source} has no element matching {path!r}")
    return element


def _set_sliding_friction(element: ET.Element, value: float, num_sliding: int = 1) -> None:
    """Overwrite only the sliding components, keeping torsional/rolling as declared."""
    components = element.get("friction", "").split()
    if len(components) <= num_sliding:
        raise ValueError(f"Element {element.get('name', element.tag)!r} has no friction to patch")
    components[:num_sliding] = [repr(value)] * num_sliding
    element.set("friction", " ".join(components))


def _quantize(values: np.ndarray, lsb: float) -> np.ndarray:
    if lsb <= 0.0:
        return values
    return np.rint(values / lsb) * lsb


def _quat_to_euler(quat: Sequence[float]) -> np.ndarray:
    """MuJoCo ``(w, x, y, z)`` to ``(roll, pitch, yaw)``, the mocap convention."""
    w, x, y, z = (float(value) for value in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])
