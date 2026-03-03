import numpy as np
class Controller:
    def __init__(self, robot_param, gains=None, cmd_limits=None, dt=0.1):
        if gains is None:
            self.gains = [5, 2, 2, 1.0, 1.0, 0.1, 0.1]
        else: 
            self.gains = gains
        self.kx, self.ky, self.kth, self.kprmotor, self.kplmotor, self.kirmotor, self.kilmotor = self.gains
        self.cmd_limit = cmd_limits
        self.ir = 0.0
        self.il = 0.0
        self.dt = dt # timestep (s) used in integral calculation
        self.r = robot_param['wheel_radius']  # wheel radius
        self.L = robot_param['base_diameter']  # wheelbase
        self.cmd_limits = cmd_limits
        self.dt = dt

    def compute(self, refstate, pose_state, wheel_meas):
        """
        refstate: [px_d, py_d, vx_d, vy_d, ax_d, ay_d]
        pose_state: (px, py, th)
        wheel_meas: (ur_meas, ul_meas) from encoders
        dt: timestep (s)
        Returns: (ur_cmd, ul_cmd)
        """
        # 1) wheel references -> (reference traj - > (v_ref, w_ref) -> (ur_ref, ul_ref))
        ur_ref, ul_ref = self._pose_control(refstate, pose_state)
        # 2) Wheel-speed control (PI)
        ur_cmd, ul_cmd = self._wheel_speed_control((ur_ref, ul_ref), wheel_meas)
        # 3) saturation on commands here 
        if self.cmd_limits is not None:
            umin, umax = self.cmd_limits
            ur_cmd = float(np.clip(ur_cmd, umin, umax))
            ul_cmd = float(np.clip(ul_cmd, umin, umax))

        return ur_cmd, ul_cmd

    def _pose_control(self, refstate, state):
        px = state[0]
        py = state[1]
        th = state[2]
        px_d, py_d, th_d = refstate[0:3]
        vx_d, vy_d, w_d = refstate[3:6]
        ax_d, ay_d = refstate[6:8]
        v_d = np.linalg.norm([vx_d, vy_d]) 

        x_e = (px_d - px) * np.cos(th) + (py_d - py) * np.sin(th)
        y_e = -(px_d - px) * np.sin(th) + (py_d - py) * np.cos(th)
        th_e = self._wrap_to_pi(th_d - th)
        v = v_d * np.cos(th_e) + self.kx * x_e
        w = w_d + v_d * (self.ky * y_e + self.kth * np.sin(th_e)) + self.kth * th_e
        ur_ref, ul_ref = self._vw_to_wheels(v, w)
        return ur_ref, ul_ref

    def _vw_to_wheels(self, v, w):
        ur_ref = (2*v + self.L*w)/ (2*self.r)
        ul_ref = (2*v - self.L*w)/ (2*self.r)
        return ur_ref, ul_ref

    def _wheel_speed_control(self, wheel_ref, wheel_true):
        ur_ref, ul_ref = wheel_ref
        ur_true, ul_true = wheel_true
        # Errors
        er = ur_ref - ur_true
        el = ul_ref - ul_true
        # integral errors
        self.ir += er * self.dt
        self.il += el * self.dt
        # PI control
        ur_cmd = ur_ref + self.kprmotor * er + self.kirmotor * self.ir
        ul_cmd = ul_ref + self.kplmotor * el + self.kilmotor * self.il
        return ur_cmd, ul_cmd

    def _wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

