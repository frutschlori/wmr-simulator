"""Port of the Pololu firmware's trajectory-following control stack.

This is the robot's *belief* about itself: everything here is driven by
``ROBOTCFG.CFG`` / ``robot_config.yaml`` and by sensor readings, never by the
hidden plant. The gap between the config's ``wheel_radius``/``wheel_base`` and
the plant's true geometry is exactly what the system identification has to
close.

Ported one-to-one from the sibling firmware repo (`~/…/pololu-rs/firmware`),
branch **`experimental_gain_mlp`** — which is what the robot runs, and which
differs from `main` in ways that matter here (feedforward, integral limit,
where the odometry twist comes from):

- ``inner_controller.rs`` — 100 Hz: counts → rad/s → 3 Hz low-pass →
  feedforward + PI → duty. The *low-passed* speed is what gets logged as
  ``omega_*_meas``, and on this branch it is also what the body twist for the
  EKF is built from: ``odometry.rs`` is deleted and the inner loop publishes
  ``(v, w)`` itself, from the **filtered** speeds and the config's geometry.
- ``ekf.rs`` — 3-state EKF, ``Q``/``R``/``P0`` copied from ``default_at_origin``.
- ``trajectory_control.rs`` — 20 Hz: predict → mocap update if fresh →
  setpoint lookup → gain-MLP factors → Kanayama law → wheel speeds. No
  clipping of the wheel command; the only saturation is the duty clamp inside
  the inner loop.
- ``gain_mlp_store.rs`` — the outer loop overrides ``kx/ky/ktheta`` for its own
  tick and *publishes* the two motor-gain factors, which the inner loop then
  applies at 100 Hz until the next outer tick replaces them. They start at 1.0,
  so the first 50 ms of a run is unscaled.

Deviations from the firmware, all deliberate:

- **float64, not float32.** The differences are far below the plant's own
  noise, and f32 everywhere would obscure the control logic.
- **The clock is exact.** Real logs show ±5 ms of scheduler jitter on the
  100 Hz streams and 39–62 ms on the outer loop; here every task is on its
  nominal period. The firmware's ``dt_sample`` (measured elapsed time, with a
  guard that skips intervals under half a period) therefore always equals its
  nominal ``dt`` here, and the guard never fires.
- **The gain MLP is evaluated by the pipeline's own numpy replica.** The
  firmware runs ``libs/gain_mlp``; ``pololu.gain_mlp_exporter.reference_forward``
  is the numpy mirror of that crate's ``factors``, and it is what the crate's
  golden test is generated from, so a third implementation here would only add
  a way for the three to disagree. It reads ``GAINMLP.JSN`` itself -- the same
  SD-card file the robot parses -- so the port still learns nothing the card
  does not carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

# --- firmware constants (bin/programm_entrance.rs, inner_controller.rs) -----

INNER_PERIOD_S = 0.010
MOCAP_PERIOD_S = 0.010
IMU_PERIOD_S = 0.010

# The inner ticker is deliberately started half a period late so it never
# collides with the outer one (inner_controller.rs, "Shift inner controller
# ticker half a period").
INNER_PHASE_S = 0.005

WHEEL_LP_CUTOFF_HZ = 3.0
INTEGRAL_LIMIT = 0.8
DUTY_LIMIT = 1.0

# ekf.rs::default_at_origin
EKF_INITIAL_COVARIANCE = (1.0, 1.0, 1.0)
EKF_PROCESS_NOISE = (0.001, 0.001, 0.01)
EKF_MEASUREMENT_NOISE = (0.0001, 0.0001, 0.001)

# Measured off real logs: the outer loop's WheelCmd/TrackingError records are
# stamped ~2 ms after its Setpoint, because that is how long its body takes.
OUTER_OUTPUT_STAMP_DELAY_MS = 2


GAIN_MLP_FILENAME = "GAINMLP.JSN"


@dataclass(frozen=True)
class FirmwareConfig:
    """What the firmware reads off the SD card - and all it is allowed to know."""

    wheel_radius: float
    wheel_base: float
    encoder_cpr: float
    kp_inner: float
    ki_inner: float
    kd_inner: float
    kx_traj: float
    ky_traj: float
    ktheta_traj: float
    motor_direction_left: float
    motor_direction_right: float
    traj_following_dt_s: float
    # The feedforward's motor gain. Note this is the config's own `wheel_max`
    # (250 by default), NOT the simulator's identified `max_wheel_speed`.
    wheel_max: float
    # The parsed GAINMLP.JSN, when one sat next to the config on the card.
    # Not a scalar like the rest, so it is excluded from the mapping parsing.
    gain_mlp: dict | None = None

    @property
    def base_gains(self) -> np.ndarray:
        """``[kx, ky, ktheta, kp_inner, ki_inner]`` - the five the MLP scales."""
        return np.array(
            [self.kx_traj, self.ky_traj, self.ktheta_traj, self.kp_inner, self.ki_inner],
            dtype=float,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, float], gain_mlp: dict | None = None) -> "FirmwareConfig":
        """Build from a firmware ``key=value`` mapping, ignoring keys we do not model."""
        scalars = {field for field in cls.__dataclass_fields__} - {"gain_mlp"}
        missing = sorted(scalars - set(values))
        if missing:
            raise ValueError(f"Robot config is missing keys: {missing}")
        return cls(gain_mlp=gain_mlp, **{field: float(values[field]) for field in scalars})

    @classmethod
    def from_file(cls, path: str | Path) -> "FirmwareConfig":
        """Read a ``ROBOTCFG.CFG`` or a simulator ``robot_config.yaml`` (by extension).

        A ``GAINMLP.JSN`` sitting next to it is picked up the same way the
        firmware picks it up off the SD card (``sdlog.rs``): present means the
        gains are scheduled, absent means static. A robot_config.yaml that
        *enables* a parametrization without the exported network next to it is
        refused - on the robot that combination silently runs static gains, and
        reproducing that silently here would be worse.
        """
        path = Path(path)
        gain_mlp = load_gain_mlp(path.parent / GAIN_MLP_FILENAME)
        if path.suffix.lower() in {".yaml", ".yml"}:
            values = _firmware_values_from_robot_config_yaml(path, has_gain_mlp=gain_mlp is not None)
            return cls.from_mapping(values, gain_mlp=gain_mlp)
        from wmr_simulator.pololu.robot_config import load_robot_config_file

        return cls.from_mapping(load_robot_config_file(path), gain_mlp=gain_mlp)


def load_gain_mlp(path: str | Path) -> dict | None:
    """Parse a ``GAINMLP.JSN``; ``None`` when the card does not carry one.

    Only the fields ``reference_forward`` needs are checked, and only for
    presence: the file is written by ``pololu.gain_mlp_exporter``, which is also
    what validates it against the firmware's capacity limits.
    """
    import json

    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted({"sizes", "scheduled_indices", "bound", "feature_scale", "weights"} - set(payload))
    if missing:
        raise ValueError(f"{path} is not a gain-MLP network: missing {missing}")
    return payload


@dataclass(frozen=True)
class Tick:
    """One physics step, and which firmware tasks fire on it."""

    index: int
    time: float
    t_ms: int
    inner: bool
    outer: bool
    mocap: bool
    imu: bool


class FirmwareClock:
    """The multi-rate schedule, in physics steps.

    Every task is an embassy ``Ticker``, so each fires first after one full
    period, not at t = 0. The inner loop additionally starts half a period late
    so it never lands on the same instant as the outer loop; the wheel command
    the outer loop produces therefore reaches the motors on the next inner tick,
    5 ms later - the actuation latency the channel gives on the robot.
    """

    def __init__(self, timestep: float, outer_period_s: float, boot_time_ms: int = 0) -> None:
        self.timestep = float(timestep)
        self.outer_period_s = float(outer_period_s)
        self.boot_time_ms = int(boot_time_ms)

    def ticks(self, duration: float) -> Iterator[Tick]:
        periods = (INNER_PERIOD_S, self.outer_period_s, MOCAP_PERIOD_S, IMU_PERIOD_S)
        deadlines = [INNER_PHASE_S + INNER_PERIOD_S, self.outer_period_s, MOCAP_PERIOD_S, IMU_PERIOD_S]
        index = 0
        # Half a step of slack, so a deadline landing exactly on a step boundary
        # fires on that step rather than on the next one by floating-point luck.
        slack = 0.5 * self.timestep
        while True:
            index += 1
            time = index * self.timestep
            if time > duration + slack:
                return
            fired = []
            for slot, period in enumerate(periods):
                due = time + slack >= deadlines[slot]
                if due:
                    deadlines[slot] += period
                fired.append(due)
            yield Tick(
                index=index,
                time=time,
                t_ms=self.boot_time_ms + int(round(time * 1000.0)),
                inner=fired[0],
                outer=fired[1],
                mocap=fired[2],
                imu=fired[3],
            )


class InnerLoop:
    """``inner_controller.rs``: counts -> rad/s -> low-pass -> feedforward + PI -> duty.

    It also publishes the body twist the EKF predicts from. That used to be a
    separate 100 Hz ``odometry.rs`` task working on the *raw* count differences;
    on the deployed branch that file is gone and the twist is built from the
    **low-passed** speeds, here, with the config's geometry.
    """

    def __init__(self, config: FirmwareConfig, dt: float = INNER_PERIOD_S) -> None:
        self.config = config
        self.dt = dt
        tau = 1.0 / (2.0 * math.pi * WHEEL_LP_CUTOFF_HZ)
        self.alpha = dt / (tau + dt)
        self.command = np.zeros(2)
        self.reset((0, 0))

    def reset(self, counts: Sequence[int]) -> None:
        self.previous_counts = np.asarray(counts, dtype=np.int64)
        self.omega_lp = np.zeros(2)
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.twist = (0.0, 0.0)
        # gain_mlp_store::reset_inner_gain_scales - 1.0 until the first outer
        # tick publishes factors, and again once the trajectory ends.
        self.gain_scales = (1.0, 1.0)

    def set_gain_scales(self, kp_scale: float, ki_scale: float) -> None:
        """``gain_mlp_store::set_inner_gain_scales``, published by the outer loop.

        They stay in force for the ~5 inner ticks until the next outer tick, so
        the inner loop runs a piecewise-constant gain, exactly as on the robot.
        """
        self.gain_scales = (float(kp_scale), float(ki_scale))

    def set_command(self, omega_left: float, omega_right: float) -> None:
        self.command = np.array([omega_left, omega_right], dtype=float)

    def update(self, counts: Sequence[int]) -> np.ndarray:
        """One 100 Hz tick. Returns the duty pair; ``omega_lp`` is what gets logged."""
        config = self.config
        counts = np.asarray(counts, dtype=np.int64)
        omega_raw = 2.0 * math.pi * (counts - self.previous_counts) / (config.encoder_cpr * self.dt)
        self.previous_counts = counts

        self.omega_lp = self.omega_lp + self.alpha * (omega_raw - self.omega_lp)
        error = self.command - self.omega_lp
        kp_scale, ki_scale = self.gain_scales
        kp = config.kp_inner * kp_scale
        ki = config.ki_inner * ki_scale
        self.integral = np.clip(self.integral + ki * self.dt * error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error

        # The feedforward divides by the config's `wheel_max`, not by the
        # identified motor gain - so it is a ~12% under-estimate at the stock
        # numbers, which the integral term is left to make up.
        feedforward = self.command / config.wheel_max
        # Only kp and ki are scheduled; kd is not one of the five gains.
        output = np.clip(
            feedforward + kp * error + self.integral + config.kd_inner * derivative,
            -DUTY_LIMIT,
            DUTY_LIMIT,
        )

        left, right = float(self.omega_lp[0]), float(self.omega_lp[1])
        self.twist = (
            config.wheel_radius * (right + left) / 2.0,
            config.wheel_radius * (right - left) / config.wheel_base,
        )
        return output * np.array([config.motor_direction_left, config.motor_direction_right])


class Ekf:
    """``ekf.rs``: 3-state pose EKF, unicycle prediction, ``H = I`` mocap update."""

    def __init__(self, pose: Sequence[float]) -> None:
        self.x = np.asarray(pose, dtype=float).copy()
        self.p = np.diag(EKF_INITIAL_COVARIANCE).astype(float)
        self.q = np.diag(EKF_PROCESS_NOISE).astype(float)
        self.r = np.diag(EKF_MEASUREMENT_NOISE).astype(float)

    def predict(self, v: float, w: float, dt: float) -> None:
        theta = self.x[2]
        # "Stabilized prediction": x,y propagate on the midpoint heading.
        theta_mid = wrap_angle(theta + 0.5 * w * dt)
        self.x[0] += v * math.cos(theta_mid) * dt
        self.x[1] += v * math.sin(theta_mid) * dt
        self.x[2] = wrap_angle(theta + w * dt)

        jacobian = np.array(
            [
                [1.0, 0.0, -v * math.sin(theta_mid) * dt],
                [0.0, 1.0, v * math.cos(theta_mid) * dt],
                [0.0, 0.0, 1.0],
            ]
        )
        self.p = jacobian @ self.p @ jacobian.T + self.q

    def update(self, measurement: Sequence[float]) -> None:
        innovation = np.asarray(measurement, dtype=float) - self.x
        innovation[2] = wrap_angle(innovation[2])
        gain = self.p @ np.linalg.inv(self.p + self.r)
        self.x = self.x + gain @ innovation
        self.x[2] = wrap_angle(self.x[2])
        self.p = (np.eye(3) - gain) @ self.p

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()


@dataclass(frozen=True)
class Setpoint:
    x_des: float
    y_des: float
    yaw_des: float
    v_ff: float
    w_ff: float


@dataclass(frozen=True)
class OuterOutputs:
    setpoint: Setpoint
    omega_left: float
    omega_right: float
    x_err: float
    y_err: float
    yaw_err: float


class TrajectoryFollower:
    """``trajectory_control.rs`` + ``setpoint/mod.rs``: lookup and the Kanayama law."""

    def __init__(self, config: FirmwareConfig, states: np.ndarray, actions: np.ndarray) -> None:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError(f"Trajectory states must have shape (N, 3), got {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(f"Trajectory actions must have shape (M, 2), got {actions.shape}")
        if states.shape[0] < 2:
            raise ValueError("Trajectory must contain at least two states")
        self.config = config
        self.states = states
        self.actions = actions
        self.dt = config.traj_following_dt_s

    @property
    def duration(self) -> float:
        """``SetpointFinder::duration`` - states, not actions, set the run length."""
        return self.states.shape[0] * self.dt

    def setpoint(self, t: float) -> Setpoint:
        """``idx = floor(t/dt)``, pose from ``states[idx + 1]`` and feedforward from ``actions[idx]``."""
        index = int(t / self.dt)
        # The firmware clamps to len(actions) - 1 only. Exports carry N-1
        # actions for N states so the two agree, but a file with N actions
        # would index states out of bounds there; clamp on both here.
        index = max(0, min(index, self.actions.shape[0] - 1, self.states.shape[0] - 2))
        pose = self.states[index + 1]
        action = self.actions[index]
        return Setpoint(float(pose[0]), float(pose[1]), float(pose[2]), float(action[0]), float(action[1]))

    def control(
        self,
        pose: Sequence[float],
        setpoint: Setpoint,
        gains: Sequence[float] | None = None,
    ) -> OuterOutputs:
        """``gains`` is ``(kx, ky, ktheta)`` for this tick; None uses the config's.

        The firmware overwrites ``controller.k*`` in place every tick when a
        gain MLP is loaded, so the scheduled values are what the law below runs
        on - not a correction applied afterwards.
        """
        config = self.config
        kx_traj, ky_traj, ktheta_traj = (
            (config.kx_traj, config.ky_traj, config.ktheta_traj) if gains is None else
            (float(gains[0]), float(gains[1]), float(gains[2]))
        )
        x, y, theta = (float(value) for value in pose)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        dx = setpoint.x_des - x
        dy = setpoint.y_des - y
        x_err = cos_theta * dx + sin_theta * dy
        y_err = -sin_theta * dx + cos_theta * dy
        yaw_err = wrap_angle(setpoint.yaw_des - theta)

        v = setpoint.v_ff * math.cos(yaw_err) + kx_traj * x_err
        w = setpoint.w_ff + setpoint.v_ff * (ky_traj * y_err + ktheta_traj * math.sin(yaw_err))

        # No clipping here: saturation happens only at the duty clamp.
        omega_right = (2.0 * v + config.wheel_base * w) / (2.0 * config.wheel_radius)
        omega_left = (2.0 * v - config.wheel_base * w) / (2.0 * config.wheel_radius)
        return OuterOutputs(setpoint, omega_left, omega_right, x_err, y_err, yaw_err)


class Firmware:
    """The whole stack, ticked by :class:`FirmwareClock`.

    The only inputs are encoder counts and mocap poses; the only outputs are
    duty values and the records that go into the log.
    """

    def __init__(
        self,
        config: FirmwareConfig,
        states: np.ndarray,
        actions: np.ndarray,
        initial_pose: Sequence[float],
    ) -> None:
        self.config = config
        self.inner = InnerLoop(config)
        self.follower = TrajectoryFollower(config, states, actions)
        self.ekf = Ekf(initial_pose)
        self.mocap_pose: np.ndarray | None = None
        self.mocap_fresh = False
        # Last factors the gain MLP produced, for diagnostics; the firmware
        # only logs them to defmt, so nothing downstream may read them.
        self.gain_factors = np.ones(5)

    def reset_encoders(self, counts: Sequence[int]) -> None:
        """Latch the starting counts, as the inner loop does when it spins up."""
        self.inner.reset(counts)

    @property
    def duration(self) -> float:
        return self.follower.duration

    def clock(self, timestep: float, boot_time_ms: int = 0) -> FirmwareClock:
        return FirmwareClock(timestep, self.config.traj_following_dt_s, boot_time_ms)

    def receive_mocap(self, pose: Sequence[float]) -> None:
        """``mocap_update_task``: store the pose ``(x, y, yaw)`` and raise the fresh flag."""
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,):
            raise ValueError(f"Mocap pose must be (x, y, yaw), got shape {pose.shape}")
        self.mocap_pose = pose.copy()
        self.mocap_fresh = True

    def inner_tick(self, counts: Sequence[int]) -> np.ndarray:
        return self.inner.update(counts)

    def outer_tick(self, t: float) -> OuterOutputs:
        """Estimate, look up the setpoint, run the law, hand the command to the inner loop."""
        v, w = self.inner.twist
        self.ekf.predict(v, w, self.config.traj_following_dt_s)
        if self.mocap_fresh and self.mocap_pose is not None:
            self.ekf.update(self.mocap_pose)
            self.mocap_fresh = False

        setpoint = self.follower.setpoint(t)
        pose = self.ekf.state
        self.gain_factors = self._gain_factors(setpoint, pose, (v, w))
        gains = self.config.base_gains * self.gain_factors
        self.inner.set_gain_scales(self.gain_factors[3], self.gain_factors[4])
        outputs = self.follower.control(pose, setpoint, gains[:3])
        self.inner.set_command(outputs.omega_left, outputs.omega_right)
        return outputs

    def _gain_factors(self, setpoint: Setpoint, pose: Sequence[float], twist: Sequence[float]) -> np.ndarray:
        """The five multiplicative gain factors for this tick, all 1.0 without a network.

        Inputs are the firmware's: the setpoint it just wrote, the EKF pose and
        the *odometry* twist of this same tick - the one the EKF predicted from,
        which on this branch comes off the low-passed wheel speeds.
        """
        if self.config.gain_mlp is None:
            return np.ones(5)
        from wmr_simulator.pololu.gain_mlp_exporter import reference_forward

        return np.asarray(
            reference_forward(
                self.config.gain_mlp,
                ref=[setpoint.x_des, setpoint.y_des, setpoint.yaw_des, setpoint.v_ff, setpoint.w_ff],
                pose=[float(value) for value in pose],
                twist=[float(value) for value in twist],
            ),
            dtype=float,
        )


def wrap_angle(angle: float) -> float:
    """``math.rs::wrap_angle`` - atan2 of the sine and cosine."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _firmware_values_from_robot_config_yaml(path: Path, has_gain_mlp: bool = False) -> dict[str, float]:
    """Convert a simulator ``robot_config.yaml`` into firmware config values.

    The unit conversion (inner gains are duty/(rad/s) on the robot, wheel-speed
    feedback in the simulator) is the exporter's, reused rather than restated.
    """
    from types import SimpleNamespace

    import yaml

    from wmr_simulator.pololu.robot_config import robot_config_values

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    parametrization = config.get("controller", {}).get("gain_parametrization", {})
    if parametrization.get("enabled", False) and not has_gain_mlp:
        raise ValueError(
            f"{path} enables a gain parametrization but no {GAIN_MLP_FILENAME} sits next to it. "
            "Export one (pololu.gain_mlp_exporter) or disable the parametrization: the robot "
            "would run static gains here, and doing that silently would hide the difference."
        )

    robot = config["robot"]
    physical_params = SimpleNamespace(
        wheel_radius=robot["wheel_radius"],
        base_diameter=robot["base_diameter"],
        max_wheel_speed=robot["max_wheel_speed"],
    )
    return robot_config_values(
        physical_params=physical_params,
        controller_gains=config["controller"]["gains"],
    )
