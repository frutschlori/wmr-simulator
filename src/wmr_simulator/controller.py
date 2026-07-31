import jax.numpy as np

# Geometry-level control laws selectable via `controller.type` in the problem yaml.
KANAYAMA = "kanayama"
DYNAMIC_FEEDBACK = "dynamic_feedback"
CONTROLLER_TYPES = (KANAYAMA, DYNAMIC_FEEDBACK)

# The controller gains live in three separate lists in the config yamls and are
# concatenated, in this order, into the one flat gain vector the code passes
# around (gain tuning, gain parametrization indices, ROBOTCFG/GAINMLP export).
# The first five entries are the firmware-exportable ones; keep them first.
GEOMETRIC_GAIN_NAMES = ("kx", "ky", "kth")
MOTOR_GAIN_NAMES = ("kpmotor", "kimotor")
DYNAMIC_FEEDBACK_GAIN_NAMES = ("kp_x", "kd_x", "kp_y", "kd_y")
GAIN_GROUPS = (
    ("geometric_gains", GEOMETRIC_GAIN_NAMES),
    ("motor_gains", MOTOR_GAIN_NAMES),
    ("dynamic_feedback_gains", DYNAMIC_FEEDBACK_GAIN_NAMES),
)
GAIN_NAMES = GEOMETRIC_GAIN_NAMES + MOTOR_GAIN_NAMES + DYNAMIC_FEEDBACK_GAIN_NAMES
NUM_GAINS = len(GAIN_NAMES)

# Slices of the flat vector, by group.
GEOMETRIC_GAIN_SLICE = slice(0, 3)
MOTOR_GAIN_SLICE = slice(3, 5)
DYNAMIC_FEEDBACK_GAIN_SLICE = slice(5, 9)
# The gains the firmware knows (kx, ky, kth, kpmotor, kimotor); the firmware
# implements the Kanayama law only.
FIRMWARE_GAIN_SLICE = slice(0, 5)

# Which gains each control law actually reads. Gains outside the active set have
# no effect on the rollout, so the gain tuner holds them fixed.
ACTIVE_GAIN_INDICES = {
    KANAYAMA: (0, 1, 2, 3, 4),
    DYNAMIC_FEEDBACK: (3, 4, 5, 6, 7, 8),
}
# Gains that may legitimately be exactly 0 (integral action off). Every other
# gain must stay strictly positive for the closed loop to be stable, which is
# what lets the tuner search them in log space.
ZERO_ALLOWED_GAIN_INDICES = (GAIN_NAMES.index("kimotor"),)

# Dynamic-feedback constants mirrored from the firmware
# (pololu-rs firmware/src/control_types.rs, dynamic_feedback_control).
DYNAMIC_FEEDBACK_EPS_V = 0.1  # regularizes the 1/v inversion at standstill
DYNAMIC_FEEDBACK_V_SEED = 0.05  # first-step velocity guess, signed by the along-track error


def controller_type_from_cfg(controller_cfg) -> str:
    controller_type = controller_cfg.get("type", KANAYAMA)
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError(f"Unknown controller type {controller_type!r}; expected one of {CONTROLLER_TYPES}.")
    return controller_type


def gains_from_cfg(controller_cfg) -> list[float]:
    """The flat gain vector (GAIN_NAMES order) from the split config lists."""
    gains: list[float] = []
    for key, names in GAIN_GROUPS:
        gains.extend(_gain_list(controller_cfg, key, names))
    return gains


def set_gains(controller_cfg: dict, gains) -> dict:
    """Write a flat gain vector back into the split config lists (in place)."""
    gains = [float(gain) for gain in gains]
    if len(gains) != NUM_GAINS:
        raise ValueError(f"Expected {NUM_GAINS} controller gains {GAIN_NAMES}, got {len(gains)}.")
    offset = 0
    for key, names in GAIN_GROUPS:
        controller_cfg[key] = gains[offset:offset + len(names)]
        offset += len(names)
    return controller_cfg


def active_gain_mask(controller_type: str):
    """Boolean mask over the flat gain vector: which gains this law reads."""
    indices = ACTIVE_GAIN_INDICES[controller_type]
    return [index in indices for index in range(NUM_GAINS)]


def _gain_list(controller_cfg, key: str, names) -> list[float]:
    if key not in controller_cfg:
        raise KeyError(f"Missing controller.{key} ({', '.join(names)}) in the config.")
    gains = [float(gain) for gain in controller_cfg[key]]
    if len(gains) != len(names):
        raise ValueError(f"Expected {len(names)} controller.{key} [{', '.join(names)}], got {len(gains)}.")
    return gains


class Controller:
    def __init__(
        self,
        robot_param,
        gains,
        duty_limits=None,
        dt=0.1,
        geometry_dt=None,
        controller_type=KANAYAMA,
    ):
        # Flat gain vector in GAIN_NAMES order; each control law slices out its
        # own group, so every gain the tuner touches stays traceable.
        self.gains = gains
        self.duty_limits = duty_limits
        self.dt = dt  # timestep (s) of the wheel loop, used in the integral calculation
        self.r = robot_param['wheel_radius']  # wheel radius
        self.L = robot_param['base_diameter']  # wheelbase
        self.max_wheel_speed = robot_param.get('max_wheel_speed', 1.0)
        # Timestep (s) of the geometry loop; the dynamic-feedback law integrates
        # its commanded velocity at this rate.
        self.geometry_dt = dt if geometry_dt is None else geometry_dt
        if controller_type not in CONTROLLER_TYPES:
            raise ValueError(f"Unknown controller type {controller_type!r}; expected one of {CONTROLLER_TYPES}.")
        self.controller_type = controller_type

    def compute(self, ctrl_state, ref_state, pose_state, wheel_meas, geometry_state=None, gains=None,
                wheel_radius=None, base_diameter=None, max_wheel_speed=None):
        """
        ctrl_state: (ir, il)
        ref_state: [x, y, theta, vx, vy, omega, a, alpha]
        pose_state: (x, y, theta)
        wheel_meas: (ur_meas, ul_meas) from encoders
        geometry_state: geometry-loop controller state (see initial_geometry_state)
        Returns: (ctrl_state, duty cycles (right, left) in [-1, 1], geometry_state)
        """

        wheel_ref, next_geometry_state = self.compute_wheel_reference(
            ref_state,
            pose_state,
            geometry_state=geometry_state,
            gains=gains,
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        )
        next_ctrl_state, duty = self.compute_duty(
            ctrl_state,
            wheel_ref,
            wheel_meas,
            gains=gains,
            max_wheel_speed=max_wheel_speed,
        )
        return next_ctrl_state, duty, next_geometry_state

    def initial_geometry_state(self):
        """State carried between geometry steps: the dynamic-feedback velocity
        integrator. Unused (passed through) by the Kanayama law."""
        return np.zeros((), dtype=np.float32)

    def compute_wheel_reference(self, ref_state, pose_state, geometry_state=None, gains=None,
                                wheel_radius=None, base_diameter=None):
        r = self.r if wheel_radius is None else wheel_radius
        L = self.L if base_diameter is None else base_diameter
        gain_values = self.gains if gains is None else gains
        geometry_state = self.initial_geometry_state() if geometry_state is None else geometry_state
        if self.controller_type == DYNAMIC_FEEDBACK:
            wheel_ref, next_geometry_state = self._dynamic_feedback_control(
                ref_state, pose_state, geometry_state, r, L, gain_values
            )
        else:
            wheel_ref = self._pose_control(ref_state, pose_state, r, L, gain_values)
            next_geometry_state = geometry_state
        return np.asarray(wheel_ref), next_geometry_state

    def compute_duty(self, ctrl_state, wheel_ref, wheel_meas, gains=None, max_wheel_speed=None):
        motor_gain = self.robot_param_max_wheel_speed(robot_param_max=max_wheel_speed)
        gain_values = self.gains if gains is None else gains
        ir, il, duty_r, duty_l = self._wheel_speed_control(ctrl_state, wheel_ref, wheel_meas, gain_values, motor_gain)
        if self.duty_limits is not None:
            umin, umax = self.duty_limits
            duty_r = np.clip(duty_r, min=umin, max=umax)
            duty_l = np.clip(duty_l, min=umin, max=umax)
        return np.asarray((ir, il)), np.asarray((duty_r, duty_l))

    def robot_param_max_wheel_speed(self, robot_param_max=None):
        return self.max_wheel_speed if robot_param_max is None else robot_param_max

    def _pose_control(self, refstate, state, r, L, gains):
        kx, ky, kth = gains[GEOMETRIC_GAIN_SLICE]
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

    def _dynamic_feedback_control(self, refstate, state, prev_v, r, L, gains):
        """Dynamic feedback linearization of the unicycle (firmware port).

        The singular 1/v of the exact feedback linearization is dodged the same
        way the firmware does it: the commanded body velocity ``v`` becomes the
        controller's own integrator state, driven by the along-heading part of a
        PD law on the world-frame position, and the yaw rate uses a
        Tikhonov-regularized inverse of that velocity.

        Returns the wheel reference and the updated integrator state.
        """
        kp_x, kd_x, kp_y, kd_y = gains[DYNAMIC_FEEDBACK_GAIN_SLICE]
        px, py, th = state[0:3]
        px_d, py_d, th_d = refstate[0:3]
        vx_d, vy_d, w_d = refstate[3:6]
        v_d = np.sqrt(vx_d**2 + vy_d**2 + 1e-12)

        x_e = (px_d - px) * np.cos(th) + (py_d - py) * np.sin(th)
        # The firmware seeds the integrator on its first call with a small push
        # in the direction of the along-track error; prev_v == 0 is that call.
        prev_v = np.where(prev_v == 0.0, DYNAMIC_FEEDBACK_V_SEED * np.sign(x_e), prev_v)

        # World-frame velocity implied by the integrator state and the heading.
        xd = prev_v * np.cos(th)
        yd = prev_v * np.sin(th)

        # Reference velocity/acceleration in the world frame. Like the firmware
        # this ignores the reference's tangential acceleration (refstate[6:8]).
        xd_d = v_d * np.cos(th_d)
        yd_d = v_d * np.sin(th_d)
        xdd_d = -v_d * np.sin(th_d) * w_d
        ydd_d = v_d * np.cos(th_d) * w_d

        u1 = xdd_d + kp_x * (px_d - px) + kd_x * (xd_d - xd)
        u2 = ydd_d + kp_y * (py_d - py) + kd_y * (yd_d - yd)

        a = np.cos(th) * u1 + np.sin(th) * u2
        inv_v = prev_v / (prev_v**2 + DYNAMIC_FEEDBACK_EPS_V**2)
        w = (-np.sin(th) * u1 + np.cos(th) * u2) * inv_v
        v = prev_v + a * self.geometry_dt
        return self._vw_to_wheels(v, w, r, L), v

    def _vw_to_wheels(self, v, w, r, L):
        ur_ref = (2*v + L*w) / (2*r)
        ul_ref = (2*v - L*w) / (2*r)
        return ur_ref, ul_ref

    def _wheel_speed_control(self, ctrl_state, wheel_ref, wheel_meas, gains, motor_gain):
        kp, ki = gains[MOTOR_GAIN_SLICE]
        ur_ref, ul_ref = wheel_ref
        ur_meas, ul_meas = wheel_meas
        # Errors
        er = ur_ref - ur_meas
        el = ul_ref - ul_meas
        # integral errors
        ir, il = ctrl_state
        ir += er * self.dt
        il += el * self.dt
        # control law
        ur_cmd = ur_ref + kp * er + ki * ir
        ul_cmd = ul_ref + kp * el + ki * il
        # Map to duty cycles for the motor model.
        duty_r = ur_cmd / motor_gain
        duty_l = ul_cmd / motor_gain
        return ir, il, duty_r, duty_l

    @staticmethod
    def SO2_dist(angle1, angle2):
        error = angle1 - angle2
        return (error + np.pi) % (2 * np.pi) - np.pi
