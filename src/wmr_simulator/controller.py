import jax.numpy as np


# Firmware `inner_controller.rs` clamps the integrator state, in duty units.
INTEGRAL_DUTY_LIMIT = 0.8

# The flat controller gain vector. kdmotor came last (2026-09-30) so the first
# five slots keep their meaning in every config, log and GAINMLP.JSN written
# before it; controller_gains_array reads those with kdmotor = 0.
GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor", "kdmotor")
NUM_GAINS = len(GAIN_NAMES)
KDMOTOR_INDEX = GAIN_NAMES.index("kdmotor")
# [ir, il, er_prev, el_prev]: integrators (duty units) and the previous
# wheel-speed errors (rad/s) the D-term differentiates against.
CONTROLLER_STATE_SIZE = 4


def controller_gains_array(gains):
    """Gain vector (or batch ``(..., n)``) as float32 ``(..., NUM_GAINS)``.

    Gain vectors from before the D-term have five entries; they are read with
    kdmotor = 0, which is the controller that recorded them (the firmware's
    ``kd_inner`` was always shipped at 0).
    """
    gains = np.asarray(gains, dtype=np.float32)
    if gains.shape[-1] == NUM_GAINS - 1:
        gains = np.concatenate([gains, np.zeros((*gains.shape[:-1], 1), dtype=np.float32)], axis=-1)
    if gains.shape[-1] != NUM_GAINS:
        raise ValueError(f"Expected {NUM_GAINS} controller gains {list(GAIN_NAMES)}, got shape {gains.shape}.")
    return gains


def controller_gains_list(gains) -> list[float]:
    """One gain vector as Python floats, padded like :func:`controller_gains_array`.

    For exports and yaml, which must not round through float32.
    """
    values = [float(gain) for gain in gains]
    if len(values) == NUM_GAINS - 1:
        values.append(0.0)
    if len(values) != NUM_GAINS:
        raise ValueError(f"Expected {NUM_GAINS} controller gains {list(GAIN_NAMES)}, got {len(values)}.")
    return values


def initial_controller_state():
    return np.zeros(CONTROLLER_STATE_SIZE, dtype=np.float32)


class Controller:
    def __init__(self, robot_param, gains, duty_limits=None, dt=0.1):
        self.gains = controller_gains_array(gains)
        self.duty_limits = duty_limits
        self.dt = dt # timestep (s) used in integral calculation
        self.r = robot_param['wheel_radius']  # wheel radius
        self.L = robot_param['base_diameter']  # wheelbase
        self.max_wheel_speed = robot_param.get('max_wheel_speed', 1.0)
        self.dt = dt

    def compute(self, ctrl_state, ref_state, pose_state, wheel_meas, gains=None,
                wheel_radius=None, base_diameter=None, max_wheel_speed=None):
        """
        ctrl_state: (ir, il, er_prev, el_prev), integrators in duty units and
            the previous wheel-speed errors
        ref_state: [x, y, theta, vx, vy, omega, a, alpha]
        pose_state: (x, y, theta)
        wheel_meas: (ur_meas, ul_meas) from encoders
        Returns: motor duty cycles (right, left) in [-1, 1]
        """

        wheel_ref = self.compute_wheel_reference(
            ref_state,
            pose_state,
            gains=gains,
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        )
        return self.compute_duty(
            ctrl_state,
            wheel_ref,
            wheel_meas,
            gains=gains,
            max_wheel_speed=max_wheel_speed,
        )

    def compute_wheel_reference(self, ref_state, pose_state, gains=None, wheel_radius=None, base_diameter=None):
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        gain_values = self.gains if gains is None else gains
        return np.asarray(self._pose_control(ref_state, pose_state, r, L, gain_values))

    def compute_duty(self, ctrl_state, wheel_ref, wheel_meas, gains=None, max_wheel_speed=None):
        motor_gain = self.robot_param_max_wheel_speed(robot_param_max=max_wheel_speed)
        gain_values = self.gains if gains is None else gains
        next_ctrl_state, duty_r, duty_l = self._wheel_speed_control(
            ctrl_state, wheel_ref, wheel_meas, gain_values, motor_gain
        )
        if self.duty_limits is not None:
            umin, umax = self.duty_limits
            duty_r = np.clip(duty_r, min=umin, max=umax)
            duty_l = np.clip(duty_l, min=umin, max=umax)
        return next_ctrl_state, np.asarray((duty_r, duty_l))

    def robot_param_max_wheel_speed(self, robot_param_max=None):
        return self.max_wheel_speed if robot_param_max is None else robot_param_max

    def _pose_control(self, refstate, state, r, L, gains):
        kx, ky, kth = gains[0], gains[1], gains[2]
        px, py, th = state[0:3]
        px_d, py_d, th_d = refstate[0:3]
        vx_d, vy_d, w_d = refstate[3:6]
        v_d = np.sqrt(vx_d**2 + vy_d**2 + 1e-12)
        # ax_d, ay_d = refstate[6:8]

        x_e = (px_d - px) * np.cos(th) + (py_d - py) * np.sin(th)
        y_e = -(px_d - px) * np.sin(th) + (py_d - py) * np.cos(th)
        th_e = self.SO2_dist(th_d, th)
        v = v_d * np.cos(th_e) + kx * x_e
        w = w_d + v_d * (ky * y_e + kth * np.sin(th_e))
        ur_ref, ul_ref = self._vw_to_wheels(v, w, r, L)
        wheel_ref = (ur_ref, ul_ref)
        return wheel_ref

    def _vw_to_wheels(self, v, w, r, L):
        ur_ref = (2*v + L*w) / (2*r)
        ul_ref = (2*v - L*w) / (2*r)
        return ur_ref, ul_ref

    def _wheel_speed_control(self, ctrl_state, wheel_ref, wheel_meas, gains, motor_gain):
        kp, ki, kd = gains[3], gains[4], gains[5]
        ur_ref, ul_ref = wheel_ref
        ur_meas, ul_meas = wheel_meas
        # Errors
        er = ur_ref - ur_meas
        el = ul_ref - ul_meas
        # Integrator state lives in duty units and is clamped like the firmware's.
        ir, il, er_prev, el_prev = ctrl_state
        ki_duty = ki / motor_gain
        ir = np.clip(ir + ki_duty * self.dt * er, -INTEGRAL_DUTY_LIMIT, INTEGRAL_DUTY_LIMIT)
        il = np.clip(il + ki_duty * self.dt * el, -INTEGRAL_DUTY_LIMIT, INTEGRAL_DUTY_LIMIT)
        # D-term on the error, as in the firmware: the previous error starts at 0
        # (and is reset there on a trajectory resume), so the reference steps the
        # outer loop issues every few inner ticks kick it like on the robot.
        dr = (er - er_prev) / self.dt
        dl = (el - el_prev) / self.dt
        # Map to duty cycles for the motor model. kp and kd are in wheel-speed
        # units like the feedforward; the firmware's are these / wheel_max.
        duty_r = (ur_ref + kp * er + kd * dr) / motor_gain + ir
        duty_l = (ul_ref + kp * el + kd * dl) / motor_gain + il
        return np.stack([ir, il, er, el]), duty_r, duty_l

    @staticmethod
    def SO2_dist(angle1, angle2):
        error = angle1 - angle2
        return (error + np.pi) % (2 * np.pi) - np.pi
