from typing import NamedTuple

import jax
import jax.numpy as np

from wmr_simulator.residual_model.residual import apply_residual_model, residual_features


class DiffDriveState(NamedTuple):
    # States for simulation
    pose: jax.Array           # [x, y, theta]
    wheel_speeds: jax.Array   # motor-side wheel speeds (what the encoders measure)
    key: jax.Array

    # States for logs
    vel_omega: jax.Array      #  [v, w]
    duty_cycle: jax.Array     # motor duty cycles in [-1, 1]
    wheel_speed_cmd: jax.Array  # desired wheel speed command for diagnostics
    # Lateral body velocity (scalar, always 0).
    vel_lateral: jax.Array = np.zeros((), dtype=np.float32)


def body_velocities(wheel_speeds, wheel_radius, base_diameter):
    """Body velocities (v, omega) from wheel speeds (ideal differential-drive
    kinematics with the effective wheelbase ``base_diameter``)."""
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
        # Effective wheelbase (absorbs tire scrub in turns).
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

    def _resolve_params(
        self,
        wheel_radius=None,
        base_diameter=None,
        max_wheel_speed=None,
        time_constant=None,
    ):
        # Explicit physical parameters, so identification can differentiate through them.
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        max_speed = self.max_wheel_speed if max_wheel_speed is None else max_wheel_speed
        tau = self.tau if time_constant is None else time_constant
        return r, L, max_speed, tau

    def step(
        self,
        state,
        duty_cycle,
        wheel_radius=None,
        base_diameter=None,
        max_wheel_speed=None,
        time_constant=None,
        dt=None,
        residual_model=None,
        wheel_speed_cmd=None,
    ):
        duty_cycle = np.array(duty_cycle, dtype=np.float32)
        r, L, max_speed, tau = self._resolve_params(
            wheel_radius, base_diameter, max_wheel_speed, time_constant
        )
        dt = self.dt if dt is None else dt
        safe_tau = np.maximum(tau, 1e-3)
        alpha = np.where(tau >= 1e-3, np.exp(-dt / safe_tau), 0.0)
        # 1) Saturate duty cycle commands
        duty_cycle = np.clip(duty_cycle, min=-1.0, max=1.0)

        # 2) First-order wheel dynamics (discrete), motor side.
        target_wheel_speeds = max_speed * duty_cycle
        nominal_lag_wheel_speeds = alpha * state.wheel_speeds + (1.0 - alpha) * target_wheel_speeds

        logged_wheel_speed_cmd = target_wheel_speeds if wheel_speed_cmd is None else wheel_speed_cmd
        next_wheel_speeds = nominal_lag_wheel_speeds
        # 3) Body velocities (ideal differential-drive kinematics).
        v_nom, w_nom = body_velocities(next_wheel_speeds, r, L)

        if residual_model is not None:
            # 4) Additive residual on [v, omega] from the nominal and the
            #    commanded twist (residual_model.residual).
            v_cmd, w_cmd = body_velocities(target_wheel_speeds, r, L)
            delta = apply_residual_model(
                residual_model, residual_features(v_nom, w_nom, v_cmd, w_cmd)
            )
            v = v_nom + delta[0]
            w = w_nom + delta[1]
        else:
            v = v_nom
            w = w_nom
        # 5) Pose integration.
        next_pose = integrate_planar_pose(state.pose, v, w, dt)

        # Logged vel_omega stays [v_x_body, omega].
        next_vel_omega = np.array([v, w])

        return DiffDriveState(
            next_pose,
            next_wheel_speeds,
            state.key,
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
        dt=None,
        wheel_speed_cmd=None,
    ):
        """Propagates the robot state from measured (motor-side) wheel speeds
        without motor dynamics (used by replay-based identification)."""
        wheel_speeds = np.array(wheel_speeds, dtype=np.float32)
        if wheel_speed_cmd is None:
            wheel_speed_cmd = np.zeros_like(wheel_speeds)
        r, L, _, _ = self._resolve_params(wheel_radius, base_diameter)
        dt = self.dt if dt is None else dt

        v, w = body_velocities(wheel_speeds, r, L)
        next_pose = integrate_planar_pose(state.pose, v, w, dt)
        next_vel_omega = np.array([v, w])

        return DiffDriveState(
            next_pose,
            wheel_speeds,
            state.key,
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
            vel_omega=np.array((0.0, 0.0), dtype=np.float32),
            duty_cycle=np.array((0.0, 0.0), dtype=np.float32),
            wheel_speed_cmd=np.array((0.0, 0.0), dtype=np.float32),
            vel_lateral=np.zeros((), dtype=np.float32),
        )

    @staticmethod
    def get_pose(state):
        return state.pose


    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
