import numpy as np
import jax
from wmr_simulator.controller_jax import Controller
from wmr_simulator.robot_jax import DiffDrive
from wmr_simulator.estimator_jax import DiffDriveEstimator
from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.visualize_jax import visualize, plot
import argparse
import yaml
import timeit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/empty.yaml",
                        help="path to problem and robot description yaml file")
    parser.add_argument("--output", type=str, default="simulation_visualization",
                        help="path to output visualization pdf and html file")
    args = parser.parse_args()

    # Initialize robot model
    problem_path = args.problem
    with open(problem_path, 'r') as file:
        problem = yaml.safe_load(file)

    # generate random keys for JAX PRNG
    seed = int(np.random.randint(0, 2**31 - 1))
    master_key = jax.random.PRNGKey(seed)
    robot_key, est_key = jax.random.split(master_key, 2)

    # Simulation parameters
    sim_time = problem["sim_time"]  # simulation duration (s)
    dt = float(problem["time_step"])
    sim_N = int(sim_time / dt)
    sim_time_grid = np.linspace(0, sim_N * dt, sim_N + 1)
    start = problem["start"]
    goal = problem["goal"]

    # Initialize robot
    robot_cfg = problem["robot"]
    robot = DiffDrive(robot_cfg=robot_cfg, dt=dt)
    robot_state0 = robot.get_init_state(key=robot_key, init_pose=start)

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
    estimator = DiffDriveEstimator(estimator_cfg=est_cfg, dt=dt)
    est_state0 = estimator.get_init_state(key=est_key, start_pose=est_cfg["start"])

    # Initialize controller
    ctrl_cfg = problem["controller"]
    ctrl = Controller(robot_param=est_cfg,
                      gains=ctrl_cfg["gains"],
                      cmd_limits=[-robot_cfg["max_wheel_speed"], robot_cfg["max_wheel_speed"]],
                      dt=dt)
    ctrl_state0 = (0.0, 0.0) # Initial I-Error terms of wheel speed controller

    # Initial simulation state
    carry0 = (robot_state0, est_state0, ctrl_state0)

    # Extend reference states if needed to match simulation horizon
    if len(reference_states) < len(sim_time_grid):
        num_extra_steps = len(sim_time_grid) - len(reference_states)
        last_ref_state = reference_states[-1]
        ref_states_sim = np.vstack([
            reference_states,
            np.tile(last_ref_state, (num_extra_steps, 1))
        ])
    else:
        ref_states_sim = reference_states[:len(sim_time_grid)]

    # Routine to simulate one time step
    def sim_step(carry, ref_k):
        robot_state, est_state, ctrl_state = carry

        # Get true wheel speeds and robot pose
        ur_true, ul_true = robot.get_wheel_speeds(robot_state)
        pose_true = robot.get_pose(robot_state)

        # Update estimator
        next_est_state = estimator.update(est_state, ur_true, ul_true, pose_true)
        pose_est = estimator.get_est_pose(next_est_state)
        wheel_est = estimator.get_est_wheel_speeds(next_est_state)

        # Compute control commands [ur_cmd, ul_cmd] using reference at current time step
        next_ctrl_state, u_cmd = ctrl.compute(ctrl_state, ref_k, pose_est, wheel_est)

        # Step the robot simulation
        next_robot_state = robot.step(robot_state, u_cmd)

        return (next_robot_state, next_est_state, next_ctrl_state), (next_robot_state, next_est_state)

    # Jit decorated function that iterates over ref_states with lax.scan()
    @jax.jit
    def scan_sim(carry, ref_states):
        return jax.lax.scan(sim_step, carry, ref_states)

    # Run simulation and take rough measurement of the runtime
    jax.block_until_ready(scan_sim(carry0, ref_states_sim))  # compile + first run
    tic = timeit.default_timer()
    _, traj = scan_sim(carry0, ref_states_sim)
    toc = timeit.default_timer()
    print("Simulation runtime: ", round((toc - tic)*1e3, 2), " ms")

    # Visualization
    robot_poses = traj[0].pose
    plot(traj, estimator, sim_time_grid, reference_states, out_prefix=args.output)
    visualize(problem_path, robot_poses, reference_states, out_prefix=args.output)
