import jax.numpy as np


class Controller:
    def __init__(self, robot_param, gains, duty_limits=None, dt=0.1):
        self.gains = gains
        self.kx, self.ky, self.kth, self.kprmotor, self.kplmotor, self.kirmotor, self.kilmotor = self.gains
        self.duty_limits = duty_limits
        self.dt = dt # timestep (s) used in integral calculation
        self.r = robot_param['wheel_radius']  # wheel radius
        self.L = robot_param['base_diameter']  # wheelbase
        self.max_wheel_speed = robot_param.get('max_wheel_speed', 1.0)
        self.dt = dt

    def compute(self, ctrl_state, ref_state, pose_state, wheel_meas, gains=None,
                wheel_radius=None, base_diameter=None, max_wheel_speed=None):
        """
        ctrl_state: (ir, il)
        ref_state: [px_d, py_d, vx_d, vy_d, ax_d, ay_d]
        pose_state: (px, py, th)
        wheel_meas: (ur_meas, ul_meas) from encoders
        Returns: motor duty cycles (right, left) in [-1, 1]
        """

        # resolve robot parameters and gains
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        motor_gain = self.robot_param_max_wheel_speed(robot_param_max=max_wheel_speed)
        gain_values = self.gains if gains is None else gains

        # 1) wheel references -> (reference traj - > (v_ref, w_ref) -> (ur_ref, ul_ref))
        wheel_ref = self._pose_control(ref_state, pose_state, r, L, gain_values)
        # 2) Wheel-speed control (PI) in wheel-speed units, then normalize to duty.
        ir, il, duty_r, duty_l = self._wheel_speed_control(ctrl_state, wheel_ref, wheel_meas, gain_values, motor_gain)
        # 3) saturation on duty cycles here
        if self.duty_limits is not None:
            umin, umax = self.duty_limits
            duty_r = np.clip(duty_r, min=umin, max=umax)
            duty_l = np.clip(duty_l, min=umin, max=umax)

        return np.asarray((ir, il)), np.asarray((duty_r, duty_l))

    def robot_param_max_wheel_speed(self, robot_param_max=None):
        return self.max_wheel_speed if robot_param_max is None else robot_param_max

    def _pose_control(self, refstate, state, r, L, gains):
        kx, ky, kth, _, _, _, _ = gains
        px, py, th = state[0:3]
        px_d, py_d, th_d = refstate[0:3]
        vx_d, vy_d, w_d = refstate[3:6]
        v_d = np.sqrt(vx_d**2 + vy_d**2 + 1e-12)
        # ax_d, ay_d = refstate[6:8]

        x_e = (px_d - px) * np.cos(th) + (py_d - py) * np.sin(th)
        y_e = -(px_d - px) * np.sin(th) + (py_d - py) * np.cos(th)
        th_e = self._wrap_to_pi(th_d - th)
        v = v_d * np.cos(th_e) + kx * x_e
        w = w_d + v_d * (ky * y_e + kth * np.sin(th_e)) + kth * th_e
        ur_ref, ul_ref = self._vw_to_wheels(v, w, r, L)
        wheel_ref = (ur_ref, ul_ref)
        return wheel_ref

    def _vw_to_wheels(self, v, w, r, L):
        ur_ref = (2*v + L*w) / (2*r)
        ul_ref = (2*v - L*w) / (2*r)
        return ur_ref, ul_ref

    def _wheel_speed_control(self, ctrl_state, wheel_ref, wheel_meas, gains, motor_gain):
        _, _, _, kprmotor, kplmotor, kirmotor, kilmotor = gains
        ur_ref, ul_ref = wheel_ref
        ur_meas, ul_meas = wheel_meas
        # Errors
        er = ur_ref - ur_meas
        el = ul_ref - ul_meas
        # integral errors
        ir, il = ctrl_state
        ir += er * self.dt
        il += el * self.dt
        # Keep motor PI gains in wheel-speed units for numerically stable gradients.
        ur_cmd = ur_ref + kprmotor * er + kirmotor * ir
        ul_cmd = ul_ref + kplmotor * el + kilmotor * il
        # Map to duty cycles for the motor model.
        duty_r = ur_cmd / motor_gain
        duty_l = ul_cmd / motor_gain
        return ir, il, duty_r, duty_l

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
