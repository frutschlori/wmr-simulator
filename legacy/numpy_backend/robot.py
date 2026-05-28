import numpy as np
import yaml


class DiffDrive:
    def __init__(self, robot_cfg, init_state=[0.0, 0.0, 0.0], dt=0.01):
        # with open(config_path, 'r') as file:
        #     config = yaml.safe_load(file)
        
        # model physical parameters
        self.r = robot_cfg['wheel_radius']
        self.L = robot_cfg['base_diameter']
        # max wheel speed
        self.max_wheel_speed = robot_cfg['max_wheel_speed']
        # robot time step
        self.dt = dt
        # motor time constant
        self.tau = robot_cfg['time_constant']   
        if self.tau >= 1e-3:
            self.alpha = np.exp(-self.dt / self.tau) # set alpha to zero for no lag
        else:
            self.alpha = 0.0 # no lag
        # initial condition on states and wheel speeds
        self.start = np.array(init_state, dtype=np.float64)  # [x, y, theta]
        self.state = self.start.copy()
        self.ur = 0.0  # right wheel initial speed
        self.ul = 0.0  # left  wheel initial speed
        # NEW: slip parameters (dimensionless std dev)
        self.slip_r = float(robot_cfg.get('slip_r', 0.0))
        self.slip_l = float(robot_cfg.get('slip_l', 0.0))
        # Logs
        self.log_states = []
        self.log_vel_omega = []
        self.log_wheel_true = []
        self.log_wheel_meas = []
        self.log_wheel_cmd = []

    def setGoal(self, goal):
        self.goal = np.array(goal)

    def step(self, u):
        ur_cmd, ul_cmd = u
        # 1) Saturate wheel commands
        ur_cmd = np.clip(ur_cmd, -self.max_wheel_speed, self.max_wheel_speed)
        ul_cmd = np.clip(ul_cmd, -self.max_wheel_speed, self.max_wheel_speed)

        # 2) First-order wheel dynamics (discrete)
        self.ur = self.alpha * self.ur + (1.0 - self.alpha) * ur_cmd
        self.ul = self.alpha * self.ul + (1.0 - self.alpha) * ul_cmd
        # add slip
        ur_eff = self.ur*(1-np.random.uniform(-self.slip_r, self.slip_r))
        ul_eff = self.ul*(1-np.random.uniform(-self.slip_l, self.slip_l))
        # 3) update (from ur, ul to v, w)    
        v = (self.r/2) * (ur_eff + ul_eff)
        w = (self.r/self.L) * (ur_eff - ul_eff)
        
        # 4) Update robot state
        self.state[0] += v * np.cos(self.state[2]) * self.dt
        self.state[1] += v * np.sin(self.state[2]) * self.dt
        self.state[2] += w * self.dt
        self.state[2] = self._wrap_to_pi(self.state[2])
        # 6) Log data
        self.log_states.append(self.state.copy())
        self.log_vel_omega.append([v, w])
        self.log_wheel_true.append([self.ur, self.ul])
        self.log_wheel_cmd.append([ur_cmd, ul_cmd])

    # getters
    def get_pose(self):
        return tuple(self.state)

    def get_wheel_speeds(self):
        return float(self.ur), float(self.ul)

    def _wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    


