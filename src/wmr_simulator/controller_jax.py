import jax.numpy as np


class Controller:
    def __init__(self, robot_param, gains=None, cmd_limits=None, dt=0.1):
        if gains is None:
            self.gains = [5, 2, 2, 1.0, 1.0, 0.1, 0.1]
        else:
            self.gains = gains
        self.kx, self.ky, self.kth, self.kprmotor, self.kplmotor, self.kirmotor, self.kilmotor = self.gains
        self.cmd_limit = cmd_limits
        self.dt = dt # timestep (s) used in integral calculation
        self.r = robot_param['wheel_radius']  # wheel radius
        self.L = robot_param['base_diameter']  # wheelbase
        self.cmd_limits = cmd_limits
        self.dt = dt

    def _resolve_geometry(self, wheel_radius=None, base_diameter=None):
        # optionally accept explicit physical parameters s.t. SI-loop can differentiate through module
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        return r, L

    def compute(self, ctrl_state, ref_state, pose_state, wheel_meas, wheel_radius=None, base_diameter=None):
        """
        ctrl_state: (ir, il)
        ref_state: [px_d, py_d, vx_d, vy_d, ax_d, ay_d]
        pose_state: (px, py, th)
        wheel_meas: (ur_meas, ul_meas) from encoders
        Returns: (ur_cmd, ul_cmd)
        """
        r, L = self._resolve_geometry(wheel_radius, base_diameter)
        # 1) wheel references -> (reference traj - > (v_ref, w_ref) -> (ur_ref, ul_ref))
        ur_ref, ul_ref = self._pose_control(ref_state, pose_state, r, L)
        # 2) Wheel-speed control (PI)
        ir, il, ur_cmd, ul_cmd = self._wheel_speed_control(ctrl_state, (ur_ref, ul_ref), wheel_meas)
        # 3) saturation on commands here
        if self.cmd_limits is not None:
            umin, umax = self.cmd_limits
            ur_cmd = np.clip(ur_cmd, umin, umax).astype(float)
            ul_cmd = np.clip(ul_cmd, umin, umax).astype(float)

        return np.asarray((ir, il)), np.asarray((ur_cmd, ul_cmd))

    def _pose_control(self, refstate, state, r, L):
        px, py, th = state[0:3]
        px_d, py_d, th_d = refstate[0:3]
        vx_d, vy_d, w_d = refstate[3:6]
        v_d = np.linalg.norm(np.asarray([vx_d, vy_d]))
        # ax_d, ay_d = refstate[6:8]

        x_e = (px_d - px) * np.cos(th) + (py_d - py) * np.sin(th)
        y_e = -(px_d - px) * np.sin(th) + (py_d - py) * np.cos(th)
        th_e = self._wrap_to_pi(th_d - th)
        v = v_d * np.cos(th_e) + self.kx * x_e
        w = w_d + v_d * (self.ky * y_e + self.kth * np.sin(th_e)) + self.kth * th_e
        ur_ref, ul_ref = self._vw_to_wheels(v, w, r, L)
        return ur_ref, ul_ref

    def _vw_to_wheels(self, v, w, r, L):
        ur_ref = (2*v + L*w) / (2*r)
        ul_ref = (2*v - L*w) / (2*r)
        return ur_ref, ul_ref

    def _wheel_speed_control(self, ctrl_state, wheel_ref, wheel_true):
        ur_ref, ul_ref = wheel_ref
        ur_true, ul_true = wheel_true
        # Errors
        er = ur_ref - ur_true
        el = ul_ref - ul_true
        # integral errors
        ir, il = ctrl_state
        ir += er * self.dt
        il += el * self.dt
        # PI control
        ur_cmd = ur_ref + self.kprmotor * er + self.kirmotor * ir
        ul_cmd = ul_ref + self.kplmotor * el + self.kilmotor * il
        return ir, il, ur_cmd, ul_cmd

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
