import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class QuinticSplineFitter:
    """
    Quintic spline trajectory optimizer.
    Minimizes ∫ (p'')² dt over all segments, subject to:
      - position constraints at waypoints
      - C² continuity (position, velocity, acceleration)
      - optional derivative constraints p'(u_k) = v_waypoints[k] at all knots k
      - optional boundary second derivatives p''(0)=a_start, p''(T)=a_end
    """

    def __init__(self, waypoints, total_time=5.0,
                 v_start=None, v_end=None,
                 a_start=0.0, a_end=0.0,
                 v_waypoints=None):
        """
        waypoints   : 1D array of positions at knots
        total_time  : parameter horizon (here used as u ∈ [0, total_time])
        v_start     : optional derivative at u=0 (ignored if v_waypoints given)
        v_end       : optional derivative at u=total_time (ignored if v_waypoints given)
        a_start     : p''(0)
        a_end       : p''(T)
        v_waypoints : list/array of length num_waypoints (num_segments+1) with
                      desired p'(u_k) at each knot k, or None to leave unconstrained.
                      Entries can be None to leave specific knots free.
        """
        self.waypoints = np.array(waypoints).flatten()
        self.num_segments = len(self.waypoints) - 1
        self.total_time = total_time
        self.T_segment = total_time / self.num_segments
        self.coeffs = None
        self.T = None

        # old-style boundary derivative specs (used if v_waypoints is None)
        self.v_start = v_start
        self.v_end   = v_end
        # boundary second derivatives
        self.a_start = a_start
        self.a_end   = a_end
        # per-knot derivative constraints
        self.v_waypoints = v_waypoints  # list or None

    def _compute_Q(self, T):
        # Not used here but kept for completeness
        return np.array([
            [400/7 * T**7, 40 * T**6, 24 * T**5, 10 * T**4],
            [40 * T**6, 28.8 * T**5, 18 * T**4, 8 * T**3],
            [24 * T**5, 18 * T**4, 12 * T**3, 6 * T**2],
            [10 * T**4, 8 * T**3, 6 * T**2, 4 * T]
        ], dtype=float)

    def fit(self):
        n = self.num_segments
        T = self.T_segment
        coeffs = [cp.Variable(6) for _ in range(n)]  # a,b,c,d,e,f

        # cost matrix for jerk
        Qj = np.array([
            [720*T**5, 360*T**4, 120*T**3],
            [360*T**4, 192*T**3,  72*T**2],
            [120*T**3,  72*T**2,  36*T   ],
        ], dtype=float)

        cost = 0
        constraints = []

        for i in range(n):
            a, b, c, d, e, f = coeffs[i]
            cost += 0.5 * cp.quad_form(cp.hstack([a, b, c]), Qj)

        # --- position constraints (start/end of each segment) ---
        for i in range(n):
            a, b, c, d, e, f = coeffs[i]
            constraints += [
                f == self.waypoints[i],
                a*T**5 + b*T**4 + c*T**3 + d*T**2 + e*T + f == self.waypoints[i+1],
            ]

        # --- continuity constraints (C² between segments) ---
        for i in range(n - 1):
            a1, b1, c1, d1, e1, f1 = coeffs[i]
            a2, b2, c2, d2, e2, f2 = coeffs[i + 1]

            # velocity continuity: p'(T) == p'(0)
            constraints.append(
                5*a1*T**4 + 4*b1*T**3 + 3*c1*T**2 + 2*d1*T + e1 == e2
            )
            # acceleration continuity: p''(T) == p''(0)
            constraints.append(
                20*a1*T**3 + 12*b1*T**2 + 6*c1*T + 2*d1 == 2*d2
            )
            # position continuity
            constraints.append(
                a1*T**5 + b1*T**4 + c1*T**3 + d1*T**2 + e1*T + f1 == f2
            )

        # --- derivative constraints at knots (including boundaries) ---

        # Build v_wp list of length n+1 with desired p'(u_k) at each knot
        if self.v_waypoints is not None:
            v_wp = list(self.v_waypoints)
            assert len(v_wp) == n + 1, "v_waypoints must have length num_segments+1"
        else:
            # fallback to old boundary-only API
            v_wp = [None] * (n + 1)
            v_wp[0]  = self.v_start
            v_wp[-1] = self.v_end

        # boundary coefficients
        a0, b0, c0, d0, e0, f0 = coeffs[0]
        aN, bN, cN, dN, eN, fN = coeffs[-1]

        # start derivative
        if v_wp[0] is None:
            constraints.append(e0 == 0)          # p'(0) = 0
        else:
            constraints.append(e0 == v_wp[0])    # p'(0) = specified
        # start acceleration
        constraints.append(d0 == self.a_start)   # p''(0) = a_start (usually 0)

        # internal waypoint derivatives k = 1..n-1
        for k in range(1, n):
            if v_wp[k] is None:
                continue
            # derivative at end of segment k-1
            a, b, c, d, e, f = coeffs[k-1]
            p_prime_end = 5*a*T**4 + 4*b*T**3 + 3*c*T**2 + 2*d*T + e
            constraints.append(p_prime_end == v_wp[k])

        # end derivatives
        pT_prime = 5*aN*T**4 + 4*bN*T**3 + 3*cN*T**2 + 2*dN*T + eN
        pT_ddot  = 20*aN*T**3 + 12*bN*T**2 + 6*cN*T + 2*dN

        if v_wp[-1] is None:
            constraints.append(pT_prime == 0)        # p'(T) = 0
        else:
            constraints.append(pT_prime == v_wp[-1]) # p'(T) = specified

        constraints.append(pT_ddot == self.a_end)    # p''(T) = a_end

        # --- solve ---
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Quintic spline solve failed: {problem.status}")

        self.coeffs = coeffs
        self.T = T

    def eval(self, t):
        t = np.asarray(t)
        y = []
        for ti in t:
            seg = min(int(ti // self.T), self.num_segments - 1)
            tau = ti - seg * self.T
            a, b, c, d, e, f = [x.value for x in self.coeffs[seg]]
            y.append(a*tau**5 + b*tau**4 + c*tau**3 + d*tau**2 + e*tau + f)
        return np.array(y)

    def evald(self, t):
        t = np.asarray(t)
        yd = []
        for ti in t:
            seg = min(int(ti // self.T), self.num_segments - 1)
            tau = ti - seg * self.T
            a, b, c, d, e, f = [x.value for x in self.coeffs[seg]]
            yd.append(5*a*tau**4 + 4*b*tau**3 + 3*c*tau**2 + 2*d*tau + e)
        return np.array(yd)

    def evaldd(self, t):
        t = np.asarray(t)
        ydd = []
        for ti in t:
            seg = min(int(ti // self.T), self.num_segments - 1)
            tau = ti - seg * self.T
            a, b, c, d, e, f = [x.value for x in self.coeffs[seg]]
            ydd.append(20*a*tau**3 + 12*b*tau**2 + 6*c*tau + 2*d)
        return np.array(ydd)


class PathTimeTrajectory:
    """
    Path-then-time trajectory generator for a differential drive / unicycle.

    - Accepts desired heading at *all* waypoints via theta_waypoints (optional).
    - Enforces geometric tangent direction at each waypoint u_k.
    - Physical velocity is still zero at t=0 and t=T due to sigma'(0)=sigma'(T)=0.
    """

    def __init__(self,
                 waypoints_x,
                 waypoints_y,
                 total_time,
                 theta_waypoints=None,  # list of θ_k, len = num_waypoints, entries can be None
                 v_param=1.0):          # magnitude of geometric derivative in u

        self.waypoints_x = np.array(waypoints_x).flatten()
        self.waypoints_y = np.array(waypoints_y).flatten()
        assert len(self.waypoints_x) == len(self.waypoints_y), \
            "x and y waypoints must have same length"

        self.total_time = float(total_time)
        self.u_total = 1.0   # path parameter always in [0,1]
        self.v_param = v_param

        num_wps = len(self.waypoints_x)

        # Build theta list (per waypoint)
        if theta_waypoints is None:
            theta_wp = [None] * num_wps
        else:
            assert len(theta_waypoints) == num_wps, \
                "theta_waypoints must have same length as waypoints"
            theta_wp = list(theta_waypoints)

        # Convert heading constraints into derivative constraints wrt u
        if all(th is None for th in theta_wp):
            v_wp_x = None
            v_wp_y = None
        else:
            v_wp_x = []
            v_wp_y = []
            for th in theta_wp:
                if th is None:
                    v_wp_x.append(None)
                    v_wp_y.append(None)
                else:
                    v_wp_x.append(v_param * np.cos(th))
                    v_wp_y.append(v_param * np.sin(th))

        # ---- Build splines with v_waypoints in u ----
        self.spline_x = QuinticSplineFitter(
            self.waypoints_x,
            total_time=1.0,
            v_waypoints=v_wp_x,
            a_start=0.0,
            a_end=0.0,
        )
        self.spline_x.fit()

        self.spline_y = QuinticSplineFitter(
            self.waypoints_y,
            total_time=1.0,
            v_waypoints=v_wp_y,
            a_start=0.0,
            a_end=0.0,
        )
        self.spline_y.fit()

    # -------- S-curve time-scaling sigma, sigma_dot, sigma_ddot --------

    def _sigma(self, t):
        """5th-order S-curve on [0, T] -> [0, 1]."""
        T = self.total_time
        tau = t / T
        return 10*tau**3 - 15*tau**4 + 6*tau**5

    def _sigma_dot(self, t):
        T = self.total_time
        tau = t / T
        return (30*tau**2 - 60*tau**3 + 30*tau**4) / T

    def _sigma_ddot(self, t):
        T = self.total_time
        tau = t / T
        return (60*tau - 180*tau**2 + 120*tau**3) / (T**2)

    # -------- Core trajectory evaluations --------

    def eval(self, t):
        """
        Positions [N, 2] for times t.
        """
        t = np.asarray(t)
        u_t = self._sigma(t) * self.u_total  # u(t) ∈ [0,1]

        x = self.spline_x.eval(u_t)
        y = self.spline_y.eval(u_t)
        return np.stack([x, y], axis=-1)  # (N, 2)

    def evald(self, t):
        """
        Velocities [N, 2] for times t.
        """
        t = np.asarray(t)
        u_t = self._sigma(t) * self.u_total
        u_dot = self._sigma_dot(t) * self.u_total

        xd_u = self.spline_x.evald(u_t)  # dx/du
        yd_u = self.spline_y.evald(u_t)  # dy/du

        vx = u_dot * xd_u
        vy = u_dot * yd_u
        return np.stack([vx, vy], axis=-1)

    def evaldd(self, t):
        """
        Accelerations [N, 2] for times t.
        """
        t = np.asarray(t)
        u_t = self._sigma(t) * self.u_total
        u_dot = self._sigma_dot(t) * self.u_total
        u_ddot = self._sigma_ddot(t) * self.u_total

        xd_u = self.spline_x.evald(u_t)
        xdd_u = self.spline_x.evaldd(u_t)

        yd_u = self.spline_y.evald(u_t)
        ydd_u = self.spline_y.evaldd(u_t)

        ax = u_ddot * xd_u + (u_dot**2) * xdd_u
        ay = u_ddot * yd_u + (u_dot**2) * ydd_u
        return np.stack([ax, ay], axis=-1)

    def eval_theta_omega(self, t):
        """
        Heading th_d(t) and angular velocity w_d(t).
        Uses path tangent in u, plus curvature κ(u) and speed v.
        """
        t = np.asarray(t)
        u_t = self._sigma(t) * self.u_total
        u_dot = self._sigma_dot(t) * self.u_total

        xd_u = self.spline_x.evald(u_t)
        xdd_u = self.spline_x.evaldd(u_t)

        yd_u = self.spline_y.evald(u_t)
        ydd_u = self.spline_y.evaldd(u_t)

        mag2 = xd_u**2 + yd_u**2
        eps_dir = 1e-6
        valid = mag2 > eps_dir

        th_d = np.zeros_like(xd_u)

        if np.any(valid):
            first_idx = np.argmax(valid)
            prev_th = np.arctan2(yd_u[first_idx], xd_u[first_idx])

            # set all times up to first_idx to this heading
            th_d[:first_idx+1] = prev_th

            # from first_idx+1 onward, update with unwrap
            for i in range(first_idx+1, len(t)):
                if mag2[i] > eps_dir:
                    th = np.arctan2(yd_u[i], xd_u[i])
                    d = (th - prev_th + np.pi) % (2*np.pi) - np.pi
                    th = prev_th + d
                    th_d[i] = th
                    prev_th = th
                else:
                    th_d[i] = prev_th
        else:
            # degenerate case: path never moves
            th_d[:] = 0.0

        # translational speed
        vx = u_dot * xd_u
        vy = u_dot * yd_u
        v = np.sqrt(vx**2 + vy**2)

        # curvature κ(u)
        eps = 1e-8
        denom = (xd_u**2 + yd_u**2 + eps)**1.5
        kappa = (xd_u * ydd_u - yd_u * xdd_u) / denom

        # angular velocity: ω = κ * v
        w_d = kappa * v

        return th_d, w_d


def compute_reference_trajectory(start, goal, intermediate_waypoints, time):
    """
    Build a path-time trajectory from start -> intermediate_waypoints -> goal
    and return reference_states + the trajectory object.

    start, goal: arrays like [x, y] or [x, y, theta]
    intermediate_waypoints: list/array of waypoints, each being:
        [x, y]       -> no heading constraint at this waypoint
        [x, y, theta]-> enforce this heading
    time: 1D array of time stamps (e.g. np.linspace(0, T, N+1))

    Returns:
        reference_states: [N, 8] with columns
          [x, y, theta, vx, vy, omega, ax, ay]
        traj: PathTimeTrajectory instance
    """
    waypoints_x = []
    waypoints_y = []
    theta_wp = []

    # start
    waypoints_x.append(start[0])
    waypoints_y.append(start[1])
    theta_start = start[2] if len(start) > 2 else None
    theta_wp.append(theta_start)

    # intermediates
    if intermediate_waypoints is not None and len(intermediate_waypoints) > 0:
        # allow list-of-lists or (N,2)/(N,3) array
        for wp in intermediate_waypoints:
            wp = np.asarray(wp)
            waypoints_x.append(wp[0])
            waypoints_y.append(wp[1])
            if wp.shape[0] > 2:
                theta_wp.append(wp[2])
            else:
                theta_wp.append(None)

    # goal
    waypoints_x.append(goal[0])
    waypoints_y.append(goal[1])
    theta_goal = goal[2] if len(goal) > 2 else None
    theta_wp.append(theta_goal)

    waypoints_x = np.array(waypoints_x)
    waypoints_y = np.array(waypoints_y)

    total_time = float(time[-1])

    traj = PathTimeTrajectory(
        waypoints_x,
        waypoints_y,
        total_time=total_time,
        theta_waypoints=theta_wp,  # may contain None entries
    )

    ref_pos = traj.eval(time)    # [N, 2]
    ref_vel = traj.evald(time)   # [N, 2]
    ref_acc = traj.evaldd(time)  # [N, 2]
    th_d, w_d = traj.eval_theta_omega(time)  # [N], [N]

    # Combine into reference_states: [x, y, theta, vx, vy, omega, ax, ay]
    reference_states = np.column_stack([
        ref_pos[:, 0],     # x
        ref_pos[:, 1],     # y
        th_d,              # theta
        ref_vel[:, 0],     # vx
        ref_vel[:, 1],     # vy
        w_d,               # omega
        ref_acc[:, 0],     # ax
        ref_acc[:, 1]      # ay
    ])

    return reference_states, traj


if __name__ == "__main__":
    dt = 0.01
    T = 5.0
    N = int(T / dt)
    t = np.linspace(0, N * dt, N + 1)

    # 4 waypoints example
    waypoints_x = np.array([0.0, 0.5, 1.0, 2.0])
    waypoints_y = np.array([0.0, 0.0, 0.5, 1.0])

    # start and goal with headings
    start = np.array([waypoints_x[0], waypoints_y[0], 0.0])       # θ = 0
    goal  = np.array([waypoints_x[-1], waypoints_y[-1], np.pi/2]) # θ = π/2

    # Example intermediate waypoints:
    #   first: [x, y]      → no θ constraint
    #   second: [x, y, θ]  → enforce θ = π/4 at that waypoint
    intermediate_waypoints = [
        [waypoints_x[1], waypoints_y[1]],               # unconstrained θ
        [waypoints_x[2], waypoints_y[2], np.pi/4],      # constrained θ
    ]

    reference_states, traj = compute_reference_trajectory(
        start, goal, intermediate_waypoints, t
    )

    # Extract individual components from reference_states
    x   = reference_states[:, 0]
    y   = reference_states[:, 1]
    th_d = reference_states[:, 2]
    vx  = reference_states[:, 3]
    vy  = reference_states[:, 4]
    w_d = reference_states[:, 5]
    ax  = reference_states[:, 6]
    ay  = reference_states[:, 7]

    # For plotting waypoint times, reuse traj and time-scaling
    u_t = traj._sigma(t)   # u(t) in [0,1] for each sample in t
    u_wps = np.linspace(0.0, 1.0, len(waypoints_x))
    t_wps = np.array([t[np.argmin(np.abs(u_t - u_wp))] for u_wp in u_wps])

    # For θ markers at waypoints
    theta_waypoints = [start[2]]
    for wp in intermediate_waypoints:
        wp = np.asarray(wp)
        if wp.shape[0] > 2:
            theta_waypoints.append(wp[2])
        else:
            theta_waypoints.append(None)
    theta_waypoints.append(goal[2])

    # --- Plot results ---
    fig, axes = plt.subplots(5, 1, figsize=(8, 10))

    # positions
    axes[0].plot(t, x, label="x(t)", color="C0")
    axes[0].plot(t, y, label="y(t)", color="C1")
    axes[0].scatter(t_wps, waypoints_x, color="r", zorder=5, label="waypoints_x")
    axes[0].scatter(t_wps, waypoints_y, color="b", zorder=5, label="waypoints_y")
    axes[0].legend()
    axes[0].set_ylabel("position")
    axes[0].set_title("Path-Time Quintic Spline Trajectory (θ at selected WPs)")
    axes[0].grid(True)

    # velocities
    axes[1].plot(t, vx, label="vx", color="C0")
    axes[1].plot(t, vy, label="vy", color="C1")
    axes[1].legend()
    axes[1].set_ylabel("velocity")
    axes[1].grid(True)

    # accelerations
    axes[2].plot(t, ax, label="ax", color="C0")
    axes[2].plot(t, ay, label="ay", color="C1")
    axes[2].legend()
    axes[2].set_ylabel("acceleration")
    axes[2].grid(True)

    # heading
    axes[3].plot(t, th_d, label="th_d (heading)", color="C2")
    # only scatter θ where specified (skip None)
    for twp, th in zip(t_wps, theta_waypoints):
        if th is not None:
            axes[3].scatter(twp, th, color="k", marker="x", zorder=5)
    axes[3].legend()
    axes[3].set_ylabel("th_d")
    axes[3].grid(True)

    # angular velocity
    axes[4].plot(t, w_d, label="w_d (angular vel)", color="C3")
    axes[4].legend()
    axes[4].set_ylabel("w_d")
    axes[4].set_xlabel("time [s]")
    axes[4].grid(True)

    plt.tight_layout()

    pdf_filename = "spline_trajectory.pdf"
    fig.savefig(pdf_filename, bbox_inches='tight', transparent=True)
    print(f"Figure saved as: {pdf_filename}")
