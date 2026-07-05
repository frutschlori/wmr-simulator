from typing import NamedTuple

import jax
import jax.numpy as np

from wmr_simulator.slip import (
    apply_wheel_slip,
    ar1_slip_update,
    backlash_transmission,
    integrate_planar_pose,
    slip_body_velocities,
    traction_limited_ground_speeds,
)


class DiffDriveState(NamedTuple):
    # States for simulation
    pose: jax.Array           # [x, y, theta]
    wheel_speeds: jax.Array   # motor-side wheel speeds (what the encoders measure)
    key: jax.Array
    slip_noise: jax.Array     # AR(1) slip fractions [eta_r, eta_l]
    ground_wheel_speeds: jax.Array  # traction-limited ground-contact wheel speeds
    gear_gap_offset: jax.Array      # backlash tooth position in [-b, +b] per wheel (rad)

    # States for logs
    vel_omega: jax.Array      #  [v, w]
    duty_cycle: jax.Array     # motor duty cycles in [-1, 1]
    wheel_speed_cmd: jax.Array  # desired wheel speed command for diagnostics


class DiffDrive:
    def __init__(self, robot_cfg, dt=0.01):
        # static model physical parameters
        self.r = robot_cfg['wheel_radius']
        # Effective wheelbase: tire scrub in turns acts as a constant multiplicative
        # wheelbase correction that is not separately identifiable from the geometric
        # wheelbase (Borenstein & Feng 1996, E_b). Configure/identify the effective
        # value directly; see wmr_simulator.slip module docstring.
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
        # Slip model parameters (see wmr_simulator.slip for formulas and references)
        self.a_slip_max = float(robot_cfg.get('a_slip_max', 0.0))  # traction limit ~ mu*g (m/s^2); 0 disables
        self.b_backlash = float(robot_cfg.get('b_backlash', 0.0))  # gear play half-width at wheel output (rad)
        self.slip_sigma = float(robot_cfg.get('slip_sigma', 0.0))  # AR(1) slip std
        self.slip_tau = float(robot_cfg.get('slip_tau', 0.0))      # AR(1) correlation time (s)

    def _resolve_params(
        self,
        wheel_radius=None,
        base_diameter=None,
        max_wheel_speed=None,
        time_constant=None,
        a_slip_max=None,
        b_backlash=None,
        slip_sigma=None,
        slip_tau=None,
    ):
        # optionally accept explicit physical parameters s.t. SI-loop can differentiate through module
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        max_speed = self.max_wheel_speed if max_wheel_speed is None else max_wheel_speed
        tau = self.tau if time_constant is None else time_constant
        a_slip_max = self.a_slip_max if a_slip_max is None else a_slip_max
        b_backlash = self.b_backlash if b_backlash is None else b_backlash
        slip_sigma = self.slip_sigma if slip_sigma is None else slip_sigma
        slip_tau = self.slip_tau if slip_tau is None else slip_tau
        return r, L, max_speed, tau, a_slip_max, b_backlash, slip_sigma, slip_tau

    def step(
        self,
        state,
        duty_cycle,
        wheel_radius=None,
        base_diameter=None,
        max_wheel_speed=None,
        time_constant=None,
        a_slip_max=None,
        b_backlash=None,
        slip_sigma=None,
        slip_tau=None,
        dt=None,
    ):
        duty_cycle = np.array(duty_cycle, dtype=np.float32)
        r, L, max_speed, tau, a_slip_max, b_backlash, slip_sigma, slip_tau = self._resolve_params(
            wheel_radius, base_diameter, max_wheel_speed, time_constant,
            a_slip_max, b_backlash, slip_sigma, slip_tau,
        )
        dt = self.dt if dt is None else dt
        safe_tau = np.maximum(tau, 1e-3)
        alpha = np.where(tau >= 1e-3, np.exp(-dt / safe_tau), 0.0)
        # 1) Saturate duty cycle commands
        duty_cycle = np.clip(duty_cycle, min=-1.0, max=1.0)

        # 2) First-order wheel dynamics (discrete), motor side (what the encoders see)
        target_wheel_speeds = max_speed * duty_cycle
        next_wheel_speeds = alpha * state.wheel_speeds + (1.0 - alpha) * target_wheel_speeds

        # 3) Gearbox backlash: motion absorbed by the gear play never reaches the wheel
        next_gap_offset, gear_output_speeds = backlash_transmission(
            state.gear_gap_offset, next_wheel_speeds, b_backlash, dt
        )

        # 4) Traction limit ("burnout"): ground speeds follow gear output rate-limited
        next_ground_speeds = traction_limited_ground_speeds(
            state.ground_wheel_speeds, gear_output_speeds, a_slip_max, r, dt
        )

        # 5) Correlated (first-order Gauss-Markov) multiplicative wheel slip
        key, slip_key = jax.random.split(state.key, 2)
        next_slip_noise = ar1_slip_update(state.slip_noise, slip_key, slip_sigma, slip_tau, dt)
        effective_wheel_speeds = apply_wheel_slip(next_ground_speeds, next_slip_noise)

        # 6) Body velocities (effective wheelbase, ideal differential-drive kinematics)
        v, w = slip_body_velocities(effective_wheel_speeds, r, L)

        # 7) Pose integration
        next_pose = integrate_planar_pose(state.pose, v, w, dt)

        next_vel_omega = np.array([v, w])

        return DiffDriveState(
            next_pose,
            next_wheel_speeds,
            key,
            next_slip_noise,
            next_ground_speeds,
            next_gap_offset,
            next_vel_omega,
            duty_cycle,
            target_wheel_speeds,
        )

    def step_kinematic(
        self,
        state,
        wheel_speeds,
        duty_cycle,
        wheel_radius=None,
        base_diameter=None,
        a_slip_max=None,
        b_backlash=None,
        dt=None,
        wheel_speed_cmd=None,
    ):
        """Propagates the robot state from measured (motor-side) wheel speeds without
        motor dynamics or stochastic slip.

        The deterministic slip components -- gearbox backlash (b_backlash) and the
        traction limit (a_slip_max) -- are applied so that replay-based
        identification and FIM computations are sensitive to them.
        """
        wheel_speeds = np.array(wheel_speeds, dtype=np.float32)
        if wheel_speed_cmd is None:
            wheel_speed_cmd = np.zeros_like(wheel_speeds)
        r, L, _, _, a_slip_max, b_backlash, _, _ = self._resolve_params(
            wheel_radius, base_diameter, a_slip_max=a_slip_max, b_backlash=b_backlash
        )
        dt = self.dt if dt is None else dt

        next_gap_offset, gear_output_speeds = backlash_transmission(
            state.gear_gap_offset, wheel_speeds, b_backlash, dt
        )
        ground_speeds = traction_limited_ground_speeds(
            state.ground_wheel_speeds, gear_output_speeds, a_slip_max, r, dt
        )
        v, w = slip_body_velocities(ground_speeds, r, L)
        next_pose = integrate_planar_pose(state.pose, v, w, dt)
        next_vel_omega = np.array([v, w])

        return DiffDriveState(
            next_pose,
            wheel_speeds,
            state.key,
            state.slip_noise,
            ground_speeds,
            next_gap_offset,
            next_vel_omega,
            duty_cycle,
            wheel_speed_cmd,
        )

    # getters
    @staticmethod
    def get_init_state(key, init_pose=(0.0, 0.0, 0.0)):
        return DiffDriveState(
            pose=np.array(init_pose, dtype=np.float32),
            wheel_speeds=np.array((0.0, 0.0), dtype=np.float32),
            key=key,
            slip_noise=np.array((0.0, 0.0), dtype=np.float32),
            ground_wheel_speeds=np.array((0.0, 0.0), dtype=np.float32),
            gear_gap_offset=np.array((0.0, 0.0), dtype=np.float32),
            vel_omega=np.array((0.0, 0.0), dtype=np.float32),
            duty_cycle=np.array((0.0, 0.0), dtype=np.float32),
            wheel_speed_cmd=np.array((0.0, 0.0), dtype=np.float32),
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
