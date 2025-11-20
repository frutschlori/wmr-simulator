import numpy as np

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

        # measurement noise (mocap + IMU + encoders)
        self.noise_pos = float(estimator_cfg.get("noise_pos", 0.0))      # x,y measurement noise
        self.noise_angle = float(estimator_cfg.get("noise_angle", 0.0))  # theta measurement noise
        self.enc_angle_noise = float(estimator_cfg.get("enc_angle_noise", 0.0))

        # process noise (for KF)
        self.proc_pos_std = float(estimator_cfg.get("proc_pos_std", 0.0))
        self.proc_theta_std = float(estimator_cfg.get("proc_theta_std", 0.0))

        # Logs: estimated internal state, noisy measurement, wheel speeds
        self.log_pose_hat = []    # [x_hat, y_hat, theta_hat]
        self.log_pose_meas = []   # noisy measurement [x_mocap, y_mocap, theta_imu]
        self.log_wheel_hat = []   # [ur_hat, ul_hat] from encoders
        self.log_wheel_true = []  # [ur_true, ul_true]

        # Initial pose estimate from YAML (if provided)
        x0, y0, th0 = estimator_cfg["start"]
        self._init_state(x0, y0, th0)

    # ------------------------------------------------------------------ #
    # Initialization utilities
    # ------------------------------------------------------------------ #
    def _init_state(self, x0, y0, theta0):
        # Internal estimate
        self.x_hat = float(x0)
        self.y_hat = float(y0)
        self.theta_hat = float(theta0)

        # Encoder angles
        self.enc_r = 0.0
        self.enc_l = 0.0

        # Last estimated wheel speeds
        self.ur_hat = 0.0
        self.ul_hat = 0.0

        # Last noisy measurement (mocap + IMU)
        self.pose_meas = np.array([self.x_hat, self.y_hat, self.theta_hat])

        # Logs
        self.log_pose_hat = []
        self.log_pose_meas = []
        self.log_wheel_hat = []
        self.log_wheel_true = []

        # KF covariance etc. (only used in KF mode)
        if self.filter_type == "kf":
            # State x = [x, y, theta]
            # Initial covariance: a bit uncertain
            self.P = np.diag([0.05**2, 0.05**2, (5.0 * np.pi/180.0)**2])

            # Process noise
            qx2 = self.proc_pos_std ** 2
            qy2 = self.proc_pos_std ** 2
            qth2 = self.proc_theta_std ** 2
            self.Q = np.diag([qx2, qy2, qth2])

            # Measurement noise (mocap + IMU)
            rx2 = self.noise_pos ** 2
            ry2 = self.noise_pos ** 2
            rth2 = self.noise_angle ** 2
            self.R = np.diag([rx2, ry2, rth2])

            self.I3 = np.eye(3)
        self.log_pose_hat.append([self.x_hat, self.y_hat, self.theta_hat])

    def reset(self, x0: float, y0: float, theta0: float):
        """Public reset."""
        self._init_state(x0, y0, theta0)

    # ------------------------------------------------------------------ #
    # Main update
    # ------------------------------------------------------------------ #
    def update(self, ur_true: float, ul_true: float, pose_true=None):
        """
        Update estimator.

        ur_true, ul_true : true wheel speeds from robot (used to simulate encoders)
        pose_true        : true pose (x,y,theta) from robot, used to simulate mocap+IMU
                           If None and filter_type == "kf", we do only prediction.
        """
        # Log true wheel speeds for later comparison
        self.log_wheel_true.append([ur_true, ul_true])

        # 1) Simulate encoder increments
        dphi_r_true = float(ur_true) * self.dt
        dphi_l_true = float(ul_true) * self.dt

        dphi_r_meas = dphi_r_true + np.random.normal(0.0, self.enc_angle_noise)
        dphi_l_meas = dphi_l_true + np.random.normal(0.0, self.enc_angle_noise)

        self.enc_r += dphi_r_meas
        self.enc_l += dphi_l_meas

        # 2) Estimated wheel angular velocities from increments
        self.ur_hat = dphi_r_meas / self.dt
        self.ul_hat = dphi_l_meas / self.dt
        self.log_wheel_hat.append([self.ur_hat, self.ul_hat])

        # 3) Propagate pose (prediction step)
        v_hat = 0.5 * self.r_est * (self.ur_hat + self.ul_hat)
        w_hat = (self.r_est / self.L_est) * (self.ur_hat - self.ul_hat)

        if self.filter_type == "dr":
            # ----- DEAD-RECKONING: simple integration + noisy measurement
            self.theta_hat = self._wrap_to_pi(self.theta_hat + w_hat * self.dt)
            self.x_hat += v_hat * np.cos(self.theta_hat) * self.dt
            self.y_hat += v_hat * np.sin(self.theta_hat) * self.dt

            self.log_pose_hat.append([self.x_hat, self.y_hat, self.theta_hat])

            # simulate one noisy pose measurement (mocap + IMU)
            x_meas = self.x_hat + np.random.normal(0.0, self.noise_pos)
            y_meas = self.y_hat + np.random.normal(0.0, self.noise_pos)
            th_meas = self._wrap_to_pi(
                self.theta_hat + np.random.normal(0.0, self.noise_angle)
            )

            self.pose_meas = np.array([x_meas, y_meas, th_meas])
            self.log_pose_meas.append(self.pose_meas.copy())

        elif self.filter_type == "kf":
            # ----- EKF prediction -----
            # state x = [x, y, theta]
            x = np.array([self.x_hat, self.y_hat, self.theta_hat])

            th = self.theta_hat
            dt = self.dt

            # Nonlinear prediction
            x_pred = np.empty_like(x)
            x_pred[0] = x[0] + v_hat * np.cos(th) * dt
            x_pred[1] = x[1] + v_hat * np.sin(th) * dt
            x_pred[2] = self._wrap_to_pi(x[2] + w_hat * dt)

            # Jacobian F = df/dx
            Fx = np.eye(3)
            Fx[0, 2] = -v_hat * np.sin(th) * dt
            Fx[1, 2] =  v_hat * np.cos(th) * dt
            # Fx[2,2] = 1 already

            # Covariance prediction
            P_pred = Fx @ self.P @ Fx.T + self.Q

            # ----- Measurement simulation (mocap + IMU) -----
            if pose_true is not None:
                x_true, y_true, th_true = pose_true

                z = np.zeros(3)
                z[0] = x_true + np.random.normal(0.0, self.noise_pos)
                z[1] = y_true + np.random.normal(0.0, self.noise_pos)
                z[2] = self._wrap_to_pi(
                    th_true + np.random.normal(0.0, self.noise_angle)
                )
                self.pose_meas = z.copy()
                self.log_pose_meas.append(z.copy())

                # Measurement model: h(x) = x (identity)
                H = self.I3
                # innovation
                y_res = z - x_pred
                # wrap angle difference
                y_res[2] = self._wrap_to_pi(y_res[2])

                S = H @ P_pred @ H.T + self.R
                K = P_pred @ np.linalg.inv(S)

                x_upd = x_pred + K @ y_res
                x_upd[2] = self._wrap_to_pi(x_upd[2])

                P_upd = (self.I3 - K @ H) @ P_pred

                self.x_hat, self.y_hat, self.theta_hat = x_upd
                self.P = P_upd
            else:
                # no measurement → pure prediction
                self.x_hat, self.y_hat, self.theta_hat = x_pred
                self.P = P_pred
                # no new pose_meas in this step

            self.log_pose_hat.append([self.x_hat, self.y_hat, self.theta_hat])

        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

    # ------------------------------------------------------------------ #
    # Outputs to controller
    # ------------------------------------------------------------------ #
    def get_est_pose(self):
        """
        Return pose used by the controller.

        - In DR mode: we return the *noisy measurement* (mocap-like).
        - In KF mode: we return the *filtered estimate* x_hat.
        """
        if self.filter_type == "kf":
            return np.array([self.x_hat, self.y_hat, self.theta_hat])
        else:
            return self.pose_meas

    def get_est_wheel_speeds(self):
        """
        Return estimated wheel speeds (from encoders): ur_hat, ul_hat.
        """
        return float(self.ur_hat), float(self.ul_hat)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _wrap_to_pi(self, angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
