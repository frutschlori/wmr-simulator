from typing import NamedTuple
import jax.numpy as np


class DiffDriveState(NamedTuple):
    # States for simulation
    pose: np.ndarray           # [x, y, theta]
    wheel_speeds: np.ndarray   # actual wheel speeds

    # States for logs only
    vel_omega: np.ndarray      #  [v, w]
    wheel_cmd: np.ndarray      # commanded wheel speeds


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

    def step(self, state, wheel_cmd):
        wheel_cmd = np.array(wheel_cmd, dtype=np.float32)
        # 1) Saturate wheel commands
        wheel_cmd = np.clip(wheel_cmd, -self.max_wheel_speed, self.max_wheel_speed)

        # 2) First-order wheel dynamics (discrete)
        next_wheel_speeds = (self.alpha * state.wheel_speeds + (1.0 - self.alpha) * wheel_cmd)
        # Slip is skipped for now -> implement later
        ur_eff = next_wheel_speeds[0]
        ul_eff = next_wheel_speeds[1]

        # 3) update (from ur, ul to v, w)
        v = 0.5 * self.r * (ur_eff + ul_eff)
        w = (self.r / self.L) * (ur_eff - ul_eff)

        # 4) Compute updated robot state and return it
        x, y, theta = state.pose
        next_pose = np.array(
            [
                x + v * np.cos(theta) * self.dt,
                y + v * np.sin(theta) * self.dt,
                self._wrap_to_pi(theta + w * self.dt),
            ],
            dtype=np.float32,
        )

        next_vel_omega = np.array([v, w])

        # 6) Log states are included in DiffDriveState
        return DiffDriveState(next_pose, next_wheel_speeds, next_vel_omega, wheel_cmd)

    # getters
    @staticmethod
    def init_state(init_pose=(0.0, 0.0, 0.0)):
        return DiffDriveState(
            pose=np.array(init_pose, dtype=np.float32),
            wheel_speeds=np.array((0.0, 0.0), dtype=np.float32),
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
