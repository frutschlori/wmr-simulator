from typing import NamedTuple

import jax
import jax.numpy as np

from wmr_simulator.residual_model.burnout import traction_limited_ground_speeds
from wmr_simulator.residual_model.residual import apply_residual_model, residual_features


class DiffDriveState(NamedTuple):
    # States for simulation
    pose: jax.Array           # [x, y, theta]
    wheel_speeds: jax.Array   # motor-side wheel speeds (what the encoders measure)
    key: jax.Array
    ground_wheel_speeds: jax.Array  # traction-limited ground-contact wheel speeds

    # States for logs
    vel_omega: jax.Array      #  [v, w]
    duty_cycle: jax.Array     # motor duty cycles in [-1, 1]
    wheel_speed_cmd: jax.Array  # desired wheel speed command for diagnostics
    # Lateral body velocity (scalar). Always 0 (a true two-wheel differential
    # drive has no chassis side-slip, and the residual only corrects v/omega);
    # kept separate from vel_omega so the logged [v, omega] shape is unaffected.
    vel_lateral: jax.Array = np.zeros((), dtype=np.float32)


def body_velocities(wheel_speeds, wheel_radius, base_diameter):
    """Body velocities (v, omega) from wheel speeds.

    ``base_diameter`` is the *effective* wheelbase (see DiffDrive.__init__).
    This is ideal differential-drive kinematics (no lateral body velocity: a
    true two-wheel differential drive has no chassis side-slip).
    """
    ur, ul = wheel_speeds[0], wheel_speeds[1]
    v = 0.5 * wheel_radius * (ur + ul)
    omega = wheel_radius * (ur - ul) / base_diameter
    return v, omega


def integrate_planar_pose(pose, v, omega, dt):
    """Euler-integrate a planar pose from forward and angular body velocity."""
    x, y, theta = pose
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array(
        [
            x + v * cos_t * dt,
            y + v * sin_t * dt,
            _wrap_to_pi(theta + omega * dt),
        ]
    )


class DiffDrive:
    def __init__(self, robot_cfg, dt=0.01):
        # static model physical parameters
        self.r = robot_cfg['wheel_radius']
        # Effective wheelbase: tire scrub in turns acts as a constant multiplicative
        # wheelbase correction that is not separately identifiable from the geometric
        # wheelbase (Borenstein & Feng 1996, E_b). Configure/identify the effective
        # value directly.
        self.L = robot_cfg['base_diameter']
        # max wheel speed
        self.max_wheel_speed = robot_cfg['max_wheel_speed']
        # robot time step
        self.dt = dt
        # motor time constant
        self.tau = robot_cfg['time_constant']
        if self.tau >= 1e-3:
            self.alpha = np.exp(-self.dt / self.tau)
        else:
            self.alpha = 0.0
        # Traction limit ~ mu*g (m/s^2); 0 disables (burnout model, see
        # residual_model.burnout).
        self.a_slip_max = float(robot_cfg.get('a_slip_max', 0.0))

    def _resolve_params(
        self,
        wheel_radius=None,
        base_diameter=None,
        max_wheel_speed=None,
        time_constant=None,
        a_slip_max=None,
    ):
        # optionally accept explicit physical parameters s.t. SI-loop can differentiate through module
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        max_speed = self.max_wheel_speed if max_wheel_speed is None else max_wheel_speed
        tau = self.tau if time_constant is None else time_constant
        a_slip_max = self.a_slip_max if a_slip_max is None else a_slip_max
        return r, L, max_speed, tau, a_slip_max

    def step(
        self,
        state,
        duty_cycle,
        wheel_radius=None,
        base_diameter=None,
        max_wheel_speed=None,
        time_constant=None,
        a_slip_max=None,
        dt=None,
        residual_model=None,
        wheel_speed_cmd=None,
    ):
        duty_cycle = np.array(duty_cycle, dtype=np.float32)
        r, L, max_speed, tau, a_slip_max = self._resolve_params(
            wheel_radius, base_diameter, max_wheel_speed, time_constant, a_slip_max
        )
        dt = self.dt if dt is None else dt
        safe_tau = np.maximum(tau, 1e-3)
        alpha = np.where(tau >= 1e-3, np.exp(-dt / safe_tau), 0.0)
        # 1) Saturate duty cycle commands
        duty_cycle = np.clip(duty_cycle, min=-1.0, max=1.0)

        # 2) First-order wheel dynamics (discrete), motor side (what the encoders
        #    see). This is the *nominal* lag prediction; a learned residual may
        #    correct it below.
        target_wheel_speeds = max_speed * duty_cycle
        nominal_lag_wheel_speeds = alpha * state.wheel_speeds + (1.0 - alpha) * target_wheel_speeds

        logged_wheel_speed_cmd = target_wheel_speeds if wheel_speed_cmd is None else wheel_speed_cmd
        next_wheel_speeds = nominal_lag_wheel_speeds
        # 3) Traction limit ("burnout"): ground speeds follow the motor side rate-limited
        next_ground_speeds = traction_limited_ground_speeds(
            state.ground_wheel_speeds, next_wheel_speeds, a_slip_max, r, dt
        )
        # 4) Body velocities (effective wheelbase, ideal differential-drive kinematics)
        v_nom, w_nom = body_velocities(next_ground_speeds, r, L)

        if residual_model is not None:
            # 5) Learned additive residual (see residual_model.residual):
            #    conditioned on the (state, action) pair -- the nominal predicted
            #    twist and the commanded twist the lag is heading toward -- it
            #    corrects the forward speed and yaw rate the ideal kinematics get
            #    wrong. The action half is what lets it express transient errors
            #    (motor time constant, delay, backlash), which are proportional to
            #    command minus state. A mixture-of-experts gate zeroes the
            #    correction in operating regimes the training data never covered.
            #    No lateral side-slip in nominal operation, so v_y stays 0.
            v_cmd, w_cmd = body_velocities(target_wheel_speeds, r, L)
            delta = apply_residual_model(
                residual_model, residual_features(v_nom, w_nom, v_cmd, w_cmd)
            )
            v = v_nom + delta[0]
            w = w_nom + delta[1]
        else:
            v = v_nom
            w = w_nom
        # 6) Pose integration (ideal differential-drive kinematics, v_y = 0).
        next_pose = integrate_planar_pose(state.pose, v, w, dt)

        # Logged vel_omega stays [v_x_body, omega].
        next_vel_omega = np.array([v, w])

        return DiffDriveState(
            next_pose,
            next_wheel_speeds,
            state.key,
            next_ground_speeds,
            next_vel_omega,
            duty_cycle,
            logged_wheel_speed_cmd,
            np.zeros((), dtype=np.float32),
        )

    def step_kinematic(
        self,
        state,
        wheel_speeds,
        duty_cycle,
        wheel_radius=None,
        base_diameter=None,
        a_slip_max=None,
        dt=None,
        wheel_speed_cmd=None,
    ):
        """Propagates the robot state from measured (motor-side) wheel speeds
        without motor dynamics (used by replay-based identification).

        The traction limit (a_slip_max) is applied so that replay-based
        identification and FIM computations are sensitive to it.
        """
        wheel_speeds = np.array(wheel_speeds, dtype=np.float32)
        if wheel_speed_cmd is None:
            wheel_speed_cmd = np.zeros_like(wheel_speeds)
        r, L, _, _, a_slip_max = self._resolve_params(
            wheel_radius, base_diameter, a_slip_max=a_slip_max
        )
        dt = self.dt if dt is None else dt

        ground_speeds = traction_limited_ground_speeds(
            state.ground_wheel_speeds, wheel_speeds, a_slip_max, r, dt
        )
        v, w = body_velocities(ground_speeds, r, L)
        next_pose = integrate_planar_pose(state.pose, v, w, dt)
        next_vel_omega = np.array([v, w])

        return DiffDriveState(
            next_pose,
            wheel_speeds,
            state.key,
            ground_speeds,
            next_vel_omega,
            duty_cycle,
            wheel_speed_cmd,
            np.zeros((), dtype=np.float32),
        )

    # getters
    @staticmethod
    def get_init_state(key, init_pose=(0.0, 0.0, 0.0)):
        return DiffDriveState(
            pose=np.array(init_pose, dtype=np.float32),
            wheel_speeds=np.array((0.0, 0.0), dtype=np.float32),
            key=key,
            ground_wheel_speeds=np.array((0.0, 0.0), dtype=np.float32),
            vel_omega=np.array((0.0, 0.0), dtype=np.float32),
            duty_cycle=np.array((0.0, 0.0), dtype=np.float32),
            wheel_speed_cmd=np.array((0.0, 0.0), dtype=np.float32),
            vel_lateral=np.zeros((), dtype=np.float32),
        )

    @staticmethod
    def get_pose(state):
        return state.pose

    @staticmethod
    def get_wheel_speeds(state):
        return state.wheel_speeds

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
