from typing import NamedTuple
import jax
import jax.numpy as np


class EstimatorState(NamedTuple):
    pose_hat: np.ndarray
    pose_meas: np.ndarray
    u_hat: np.ndarray
    u_true: np.ndarray
    P: np.ndarray
    key: jax.Array

class DiffDriveEstimator:
    def __init__(self, estimator_cfg, dt):
        """
        Unified estimator for differential-drive robot.

        Modes:
          - type == "dr": dead-reckoning (integrate wheel speeds)
          - type == "kf": extended Kalman filter on [x, y, theta]
        """

        self.filter_type = estimator_cfg.get("type", "dr").lower()  # "dr" or "kf"

        # estimator model parameters
        self.r_est = float(estimator_cfg["wheel_radius"])
        self.L_est = float(estimator_cfg["base_diameter"])

        self.dt = float(dt)

        # sensor noise parameters (mocap + IMU + encoders)
        self.noise_pos = float(estimator_cfg.get("noise_pos", 0.0))      # x,y measurement noise
        self.noise_angle = float(estimator_cfg.get("noise_angle", 0.0))  # theta measurement noise
        self.enc_angle_noise = float(estimator_cfg.get("enc_angle_noise", 0.0))

        # process noise (for KF) and slip statistics
        self.proc_pos_std = float(estimator_cfg.get("proc_pos_std", 0.0))
        self.proc_theta_std = float(estimator_cfg.get("proc_theta_std", 0.0))
        self.slip_r_var = (float(estimator_cfg.get("slip_r", 0.0)) ** 2) / 3
        self.slip_l_var = (float(estimator_cfg.get("slip_l", 0.0)) ** 2) / 3

    # ------------------------------------------------------------------ #
    # Initialization utilities
    # ------------------------------------------------------------------ #
    def get_init_state(self, key, start_pose):
        # Internal estimate
        pose_hat = np.array(start_pose)

        # Last estimated wheel speeds
        u_hat = np.array([0.0, 0.0])
        u_true = np.copy(u_hat)

        # Last noisy measurement (mocap + IMU)
        pose_meas = np.copy(pose_hat)

        # KF covariance etc. (only used in KF mode)
        # Initial covariance: a bit uncertain
        P = np.diag(np.asarray([0.05**2, 0.05**2, np.deg2rad(5)**2]))

        # Process noise
        qx2 = self.proc_pos_std ** 2
        qy2 = self.proc_pos_std ** 2
        qth2 = self.proc_theta_std ** 2
        self.Q = np.diag(np.asarray([qx2, qy2, qth2]))

        # Encoder noise covariance
        self.M_encoder = np.eye(2) * (self.enc_angle_noise ** 2) # encoder angle noise

        # Measurement noise (mocap + IMU)
        rx2 = self.noise_pos ** 2
        ry2 = self.noise_pos ** 2
        rth2 = self.noise_angle ** 2
        self.R = np.diag(np.asarray([rx2, ry2, rth2]))

        self.I3 = np.eye(3)

        return EstimatorState(pose_hat, pose_meas, u_hat, u_true, P, key)

    # ------------------------------------------------------------------ #
    # Main update
    # ------------------------------------------------------------------ #
    def update(self, est_state: EstimatorState, ur_true: float, ul_true: float, pose_true=None):
        """
        Update estimator.

        est_state        : estimator state from last time step (containing last state and covariances estimates)
        ur_true, ul_true : true wheel speeds from robot (used to simulate encoders)
        pose_true        : true pose (x,y,theta) from robot, used to simulate mocap+IMU
                           If None and filter_type == "kf", we do only prediction.
        """
        # Log true wheel speeds for later comparison
        u_true = np.array([ur_true, ul_true])

        # Generate keys for PRNG
        key, k_enc_r, k_enc_l, k_x_meas, k_y_meas, k_th_meas = jax.random.split(est_state.key, 6)

        # 1) Simulate encoder increments
        dphi_r_true = ur_true * self.dt
        dphi_l_true = ul_true * self.dt

        dphi_r_meas = dphi_r_true + self.enc_angle_noise * jax.random.normal(k_enc_r)
        dphi_l_meas = dphi_l_true + self.enc_angle_noise * jax.random.normal(k_enc_l)

        # 2) Estimated wheel angular velocities from increments
        ur_hat = dphi_r_meas / self.dt
        ul_hat = dphi_l_meas / self.dt
        u_hat = np.array([ur_hat, ul_hat])

        # 3) Propagate pose (prediction step)
        v_hat = 0.5 * self.r_est * (ur_hat + ul_hat)
        w_hat = (self.r_est / self.L_est) * (ur_hat - ul_hat)

        if self.filter_type == "dr":
            # ----- DEAD-RECKONING: simple integration + noisy measurement
            x_hat, y_hat, theta_hat = est_state.pose_hat
            x_hat += v_hat * np.cos(theta_hat) * self.dt
            y_hat += v_hat * np.sin(theta_hat) * self.dt
            theta_hat = self._wrap_to_pi(theta_hat + w_hat * self.dt) # changed order to match robot model and EKF
            pose_hat = np.array([x_hat, y_hat, theta_hat])

            # simulate one noisy pose measurement (mocap + IMU)
            x_meas = x_hat + self.noise_pos * jax.random.normal(k_x_meas)
            y_meas = y_hat + self.noise_pos * jax.random.normal(k_y_meas)
            th_meas = self._wrap_to_pi(
                theta_hat + self.noise_angle * jax.random.normal(k_th_meas)
            )
            pose_meas = np.array([x_meas, y_meas, th_meas])

            P = np.zeros((3,3)) # only to comply with est_state structure

        elif self.filter_type == "kf":
            # ----- EKF prediction -----
            # state x = [x_hat, y_hat, theta_hat]
            x = est_state.pose_hat
            th = x[2]

            # Nonlinear prediction
            x_pred = np.array([
                x[0] + v_hat * np.cos(th) * self.dt,
                x[1] + v_hat * np.sin(th) * self.dt,
                self._wrap_to_pi(th + w_hat * self.dt),
                ])

            # Jacobian F = df/dx
            Fx = np.array([[1, 0, -v_hat * np.sin(th) * self.dt],
                           [0, 1, v_hat * np.cos(th) * self.dt],
                           [0, 0, 1]])

            # Input Jacobian L = df/dphi
            ct = np.cos(th)
            st = np.sin(th)
            rot = self.r_est / self.L_est
            r = self.r_est
            Lx = np.array([[r/2 * ct, r/2 * ct],
                           [r/2 * st, r/2 * st],
                           [rot     , -rot]])

            # Slip-induced covariance on measured wheel increments
            # since slip is multiplicative its var needs to be scaled with dphi_true^2
            # because in practice we only know dphi_meas, which also contains enc noise, we subtract its var
            # to not inflate slip var (max just ensures non negativity for small speeds)
            dphi_l_slip_var = np.maximum(dphi_l_meas**2 - self.enc_angle_noise**2, 0.0) * self.slip_l_var
            dphi_r_slip_var = np.maximum(dphi_r_meas**2 - self.enc_angle_noise**2, 0.0) * self.slip_r_var
            M_slip = np.diag(np.array([dphi_r_slip_var, dphi_l_slip_var]))

            # Total input covariance: slip contribution + encoder increment noise
            M = M_slip + self.M_encoder

            # Covariance prediction
            P_pred = Fx @ est_state.P @ Fx.T + Lx @ M @ Lx.T + self.Q # added input noise L M L^T

            # ----- Measurement simulation (mocap + IMU) -----
            if pose_true is not None:
                x_true, y_true, th_true = pose_true

                z = np.array([
                    x_true + self.noise_pos * jax.random.normal(k_x_meas),
                    y_true + self.noise_pos * jax.random.normal(k_y_meas),
                    self._wrap_to_pi(
                        th_true + self.noise_angle * jax.random.normal(k_th_meas)
                    ),
                ])
                pose_meas = z.copy()

                # Measurement model: h(x) = x (identity)
                H = self.I3
                # innovation
                y_res = z - x_pred
                # wrap angle difference
                y_res = y_res.at[2].set(self._wrap_to_pi(y_res[2]))

                S = H @ P_pred @ H.T + self.R
                S += 1e-6 * self.I3  # regularization
                K = P_pred @ np.linalg.inv(S)
                # K = np.linalg.solve(S.T, P_pred.T).T

                x_upd = x_pred + K @ y_res
                x_upd = x_upd.at[2].set(self._wrap_to_pi(x_upd[2]))

                P_upd = (self.I3 - K @ H) @ P_pred

                pose_hat = x_upd
                P = P_upd
            else:
                # no measurement → pure prediction
                pose_hat = x_pred
                P = P_pred
                # no new pose_meas in this step
                pose_meas = np.zeros(3)

        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

        return EstimatorState(pose_hat, pose_meas, u_hat, u_true, P, key)

    # ------------------------------------------------------------------ #
    # Outputs to controller
    # ------------------------------------------------------------------ #
    def get_est_pose(self, est_state):
        """
        Return pose used by the controller.

        - In DR mode: we return the *noisy measurement* (mocap-like).
        - In KF mode: we return the *filtered estimate* x_hat.
        """
        if self.filter_type == "kf":
            return est_state.pose_hat
        else:
            return est_state.pose_meas

    @staticmethod
    def get_est_wheel_speeds(est_state):
        """
        Return estimated wheel speeds (from encoders): ur_hat, ul_hat.
        """
        return est_state.u_hat

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
