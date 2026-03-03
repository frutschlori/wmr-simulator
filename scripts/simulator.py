import numpy as np
from wmr_simulator.controller import Controller
from wmr_simulator.robot import DiffDrive
from wmr_simulator.estimator import DiffDriveEstimator
from wmr_simulator.planner import compute_reference_trajectory
import argparse
import yaml
from wmr_simulator.visualize import visualize, plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/empty.yaml", help="path to problem and robot description yaml file")
    parser.add_argument("--output", type=str, default="simulation_visualization", help="path to output visualization pdf and html file")
    args = parser.parse_args()

    # Initialize robot model
    problem_path = args.problem
    with open(problem_path, 'r') as file:
        problem = yaml.safe_load(file)

    # Simulation parameters
    sim_time = problem["sim_time"]  # simulation duration (s)
    dt = float(problem["time_step"])
    sim_N = int(sim_time / dt)
    sim_time_grid = np.linspace(0, sim_N * dt, sim_N + 1)
    start = problem["start"]
    goal = problem["goal"]

    # Initialize robot and estimator
    robot_cfg = problem["robot"]
    robot = DiffDrive(robot_cfg=robot_cfg, init_state=start, dt=dt)
    # Planning time horizon
    planner_cfg = problem["planner"]
    planner_time = planner_cfg["time"]  # total time for reference trajectory (s)
    planner_N = int(planner_time / dt)
    planner_time_grid = np.linspace(0, planner_N * dt, planner_N + 1)
    
    # Generate reference trajectory over full planner horizon
    waypoints = planner_cfg["waypoints"]
    reference_states, polynomial_traj = compute_reference_trajectory(start, goal, waypoints, planner_time_grid)

    # Initialize estimator
    est_cfg = problem["estimator"]  # may be empty
    # Estimator uses robot+estimator params
    estimator = DiffDriveEstimator(estimator_cfg=est_cfg,
                                dt=dt)    # simulation time horizon
    
    # Initialize controller
    ctrl_cfg = problem["controller"]
    ctrl = Controller(robot_param=est_cfg,
                      gains=ctrl_cfg["gains"],
                      cmd_limits=[-robot_cfg["max_wheel_speed"], robot_cfg["max_wheel_speed"]],
                      dt=dt)    
    
    # Simulate only for the simulation time horizon
    for k in range(len(sim_time_grid)):
        # Update estimator with true wheel speeds
        ur_true, ul_true = robot.get_wheel_speeds()
        pose_true = robot.get_pose()
        estimator.update(ur_true, ul_true, pose_true)
        pose_est = estimator.get_est_pose()
        ur_hat, ul_hat = estimator.get_est_wheel_speeds()
        wheel_est = (ur_hat, ul_hat)
        # Determine which reference state to use
        if k < len(reference_states):
            # Use current reference state if available
            ref_state = reference_states[k]
        else:
            # Use last available reference state if simulation continues beyond planner time
            ref_state = reference_states[-1]

        # Compute control commands [ur_cmd, ul_cmd] using reference at current time step
        u = ctrl.compute(ref_state, pose_est, wheel_est)

        # Step the robot simulation
        robot.step(u)
        
        

        

    plot(robot, estimator, sim_time_grid, reference_states, out_prefix=args.output)
    visualize(problem_path, robot, reference_states, out_prefix=args.output)