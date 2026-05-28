from typing import NamedTuple
import jax
import jax.numpy as np


class DiffDriveState(NamedTuple):
    # States for simulation
    pose: jax.Array           # [x, y, theta]
    wheel_speeds: jax.Array   # actual wheel speeds
    key: jax.Array

    # States for logs
    vel_omega: jax.Array      #  [v, w]
    wheel_cmd: jax.Array      # commanded wheel speeds


class DiffDrive:
    def __init__(self, robot_cfg, dt=0.01):
        # static model physical parameters
        self.r = robot_cfg['wheel_radius']
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
        # NEW: slip parameters (dimensionless std dev)
        self.slip_r = float(robot_cfg.get('slip_r', 0.0))
        self.slip_l = float(robot_cfg.get('slip_l', 0.0))

    def _resolve_geometry(self, wheel_radius=None, base_diameter=None):
        # optionally accept explicit physical parameters s.t. SI-loop can differentiate through module
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        return r, L

    def step(self, state, wheel_cmd, wheel_radius=None, base_diameter=None):
        wheel_cmd = np.array(wheel_cmd, dtype=np.float32)
        r, L = self._resolve_geometry(wheel_radius, base_diameter)
        # 1) Saturate wheel commands
        wheel_cmd = np.clip(wheel_cmd, min=-self.max_wheel_speed, max=self.max_wheel_speed)

        # 2) First-order wheel dynamics (discrete)
        next_wheel_speeds = (self.alpha * state.wheel_speeds + (1.0 - self.alpha) * wheel_cmd)
        # add slip
        key, key_r, key_l = jax.random.split(state.key, 3)
        slip_r = jax.random.uniform(key_r, minval=-self.slip_r, maxval=self.slip_r)
        slip_l = jax.random.uniform(key_l, minval=-self.slip_l, maxval=self.slip_l)
        ur_eff = next_wheel_speeds[0] * (1.0 - slip_r)
        ul_eff = next_wheel_speeds[1] * (1.0 - slip_l)

        # 3) update (from ur, ul to v, w)
        v = 0.5 * r * (ur_eff + ul_eff)
        w = (r / L) * (ur_eff - ul_eff)

        # 4) Compute updated robot state and return it
        x, y, theta = state.pose
        next_pose = np.array([x + v * np.cos(theta) * self.dt,
                              y + v * np.sin(theta) * self.dt,
                              self._wrap_to_pi(theta + w * self.dt)])

        next_vel_omega = np.array([v, w])

        # 6) Log states are included in DiffDriveState
        return DiffDriveState(next_pose, next_wheel_speeds, key, next_vel_omega, wheel_cmd)

    # getters
    @staticmethod
    def get_init_state(key, init_pose=(0.0, 0.0, 0.0)):
        return DiffDriveState(
            pose=np.array(init_pose, dtype=np.float32),
            wheel_speeds=np.array((0.0, 0.0), dtype=np.float32),
            key=key,
            vel_omega=np.array((0.0, 0.0), dtype=np.float32),
            wheel_cmd=np.array((0.0, 0.0), dtype=np.float32),
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
