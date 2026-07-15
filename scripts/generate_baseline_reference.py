"""Generate an analytic baseline reference trajectory for comparison runs.

Exports the reference as a pickle (directly digestible by the other pipelines)
and as a Pololu JSN (runnable on the robot), then runs the problem's closed
loop on it and writes the summary plot to visualize/.

Examples:
  python scripts/generate_baseline_reference.py --type circle --radius 0.5 --time 10
  python scripts/generate_baseline_reference.py --type lemniscate --time 12 --max-speed 0.4
  python scripts/generate_baseline_reference.py --type lemniscate-ramp --time 30 \
      --cycles 4 --speed-rate 1.4 --amplitude 0.6
  python scripts/generate_baseline_reference.py --type spin --time 8 --max-omega 3.0
  python scripts/generate_baseline_reference.py --type waypoints --num-waypoints 5 \
      --min-x -1.0 --max-x 1.0 --min-y -1.5 --max-y 1.5 \
      --start-heading 0.0 --sample-heading --seed 7
  python scripts/generate_baseline_reference.py --type waypoints \
      --waypoints '[[1, 1], [1, 0], [0, 0], [1, 0]]'
"""

import argparse
from datetime import datetime
import json
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import yaml

from wmr_simulator.trajectory_optimization.baselines import (
    BASELINE_TYPES,
    export_baseline_reference,
    generate_baseline_reference,
    run_baseline_closed_loop,
    summarize_motion,
)
from wmr_simulator.trajectory_optimization.constraints import motion_limits_from_robot_config


WAYPOINT_BASELINE_TYPE = "waypoints"
SCRIPT_BASELINE_TYPES = (*BASELINE_TYPES, WAYPOINT_BASELINE_TYPE)


def parse_waypoints(value: str) -> list[list[float]]:
    try:
        waypoints = np.asarray(json.loads(value), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "--waypoints must be a JSON list of [x, y] pairs, for example '[[1, 1], [1, 0]]'."
        ) from error
    if waypoints.ndim != 2 or waypoints.shape[0] < 2 or waypoints.shape[1] != 2:
        raise argparse.ArgumentTypeError("--waypoints must contain at least two [x, y] pairs.")
    if not np.all(np.isfinite(waypoints)):
        raise argparse.ArgumentTypeError("--waypoints coordinates must be finite.")
    return waypoints.tolist()


def baseline_params_from_args(args) -> dict:
    if args.type == WAYPOINT_BASELINE_TYPE:
        return {
            "total_time": args.time,
            "num_waypoints": args.num_waypoints,
            "provided_waypoints": args.waypoints,
            "sample_heading": args.sample_heading,
            "start_heading": args.start_heading,
            "seed": args.seed,
            "min_x": args.min_x,
            "max_x": args.max_x,
            "min_y": args.min_y,
            "max_y": args.max_y,
        }
    common = {"center": tuple(args.center)}
    if args.type == "circle":
        return {
            "radius": args.radius,
            "total_time": args.time,
            "start_angle": args.start_angle,
            "clockwise": args.clockwise,
            **common,
        }
    if args.type == "spin":
        if args.max_omega is None:
            raise SystemExit("--max-omega is required for --type spin.")
        return {
            "total_time": args.time,
            "max_omega": args.max_omega,
            "start_angle": args.start_angle,
            "clockwise": args.clockwise,
            **common,
        }
    if args.type == "lemniscate":
        if args.max_speed is None:
            raise SystemExit("--max-speed is required for --type lemniscate.")
        return {
            "total_time": args.time,
            "max_speed": args.max_speed,
            "cycles": args.cycles,
            **common,
        }
    return {
        "total_time": args.time,
        "cycles": args.cycles,
        "speed_rate": args.speed_rate,
        "amplitude": args.amplitude,
        "max_speed": args.max_speed,
        "ramp_up_fraction": args.ramp_up_fraction,
        "ramp_down_fraction": args.ramp_down_fraction,
        **common,
    }


def waypoint_reference(
    *,
    dt: float,
    total_time: float,
    num_waypoints: int,
    provided_waypoints: list[list[float]] | None,
    sample_heading: bool,
    start_heading: float | None,
    seed: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if provided_waypoints is None and num_waypoints < 2:
        raise SystemExit("--num-waypoints must be at least 2 for --type waypoints.")
    if total_time <= 0.0:
        raise SystemExit("--time must be positive.")
    if provided_waypoints is None and min_x >= max_x:
        raise SystemExit("--min-x must be smaller than --max-x.")
    if provided_waypoints is None and min_y >= max_y:
        raise SystemExit("--min-y must be smaller than --max-y.")

    rng = np.random.default_rng(seed)
    if provided_waypoints is None:
        x = rng.uniform(min_x, max_x, size=num_waypoints)
        y = rng.uniform(min_y, max_y, size=num_waypoints)
        positions = np.column_stack([x, y])
    else:
        positions = np.asarray(provided_waypoints, dtype=float)
        num_waypoints = len(positions)
    if sample_heading:
        theta = rng.uniform(-np.pi, np.pi, size=num_waypoints)
        if start_heading is not None:
            theta[0] = start_heading
        waypoints = [waypoint for waypoint in np.column_stack([positions, theta])]
    else:
        waypoints = [waypoint for waypoint in positions]
        if start_heading is not None:
            waypoints[0] = np.append(waypoints[0], start_heading)

    num_steps = int(total_time / dt)
    time_grid = np.linspace(0.0, num_steps * dt, num_steps + 1)
    from wmr_simulator.planner import compute_reference_trajectory

    reference_states, _ = compute_reference_trajectory(
        waypoints[0],
        waypoints[-1],
        waypoints[1:-1],
        time_grid,
    )
    return reference_states, waypoints


def report_motion(reference_states: np.ndarray, dt: float, robot_cfg: dict) -> None:
    summary = summarize_motion(reference_states, dt)
    limits = {name: float(value) for name, value in motion_limits_from_robot_config(robot_cfg).items()}
    print(f"Samples: {int(summary['num_samples'])} at dt={dt} ({summary['duration']:.2f} s)")
    print(f"Speed: peak {summary['v_peak']:.3f} m/s, mean {summary['v_mean']:.3f} m/s")
    checks = [
        ("v_peak", "v_max"),
        ("a_peak", "a_max"),
        ("omega_peak", "omega_max"),
        ("alpha_peak", "alpha_max"),
    ]
    for peak_name, limit_name in checks:
        peak = summary[peak_name]
        limit = limits[limit_name]
        flag = "" if peak <= limit else "  <-- exceeds robot limit!"
        print(f"{peak_name}: {peak:.3f} (limit {limit_name}={limit:.3f}){flag}")


def main():
    parser = argparse.ArgumentParser(description="Generate a baseline reference trajectory.")
    parser.add_argument("--problem", default="problems/pololu_gains.yaml")
    # SCRIPT_BASELINE_TYPES = ("circle", "lemniscate", "lemniscate-ramp", "spin", "waypoints")
    parser.add_argument("--type", choices=list(SCRIPT_BASELINE_TYPES), default="waypoints")
    parser.add_argument("--export-dir", default="trajectory_exports/baselines")
    parser.add_argument("--name", default=None, help="Export stem (default: baseline_<type>_<timestamp>).")
    # Shared parameters
    parser.add_argument("--time", type=float, default=8.0, help="Total trajectory time in seconds.")
    parser.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], help="Path center [x y] in meters.")

    # Circle parameters
    parser.add_argument("--radius", type=float, default=1.0, help="Circle radius in meters.")
    parser.add_argument("--start-angle", type=float, default=0.0, help="Circle/spin start angle in radians.")
    parser.add_argument("--clockwise", action="store_true", default=False)
    # Spin parameters (mocap-delay identification)
    parser.add_argument("--max-omega", type=float, default=None,
                        help="Peak angular rate in rad/s (required for spin).")

    # Lemniscate parameters
    parser.add_argument("--max-speed", type=float, default=2.0,
                        help="Peak speed in m/s (required for lemniscate, optional amplitude override for ramp).")
    parser.add_argument("--cycles", type=int, default=1, help="Figure-eight cycles (lemniscate modes).")

    # Lemniscate ramp parameters
    parser.add_argument("--speed-rate", type=float, default=1.0,
                        help="Phase-rate growth factor per completed cycle (lemniscate-ramp).")
    parser.add_argument("--amplitude", type=float, default=5, help="Lemniscate amplitude in meters (ramp mode).")
    parser.add_argument("--ramp-up-fraction", type=float, default=0.2)
    parser.add_argument("--ramp-down-fraction", type=float, default=0.1)

    # Waypoint spline parameters
    parser.add_argument("--num-waypoints", type=int, default=4,
                        help="Number of random waypoints including start and goal (waypoints mode).")
    parser.add_argument("--start-heading", type=float, default=-3.14,
                        help="Heading constraint for the start waypoint in radians (waypoints mode).")
    # parser.add_argument("--waypoints", type=parse_waypoints, default='[[1, 1], [1, 0], [0, 0], [0, 1]]')
    parser.add_argument("--waypoints", type=parse_waypoints, default=None)
    parser.add_argument("--sample-heading", action="store_true", default=False,
                        help="Sample heading constraints at waypoints other than a specified start heading "
                             "(waypoints mode).")
    parser.add_argument("--min-x", type=float, default=-1.0,
                        help="Minimum randomly sampled x coordinate in meters (waypoints mode).")
    parser.add_argument("--max-x", type=float, default=1.0,
                        help="Maximum randomly sampled x coordinate in meters (waypoints mode).")
    parser.add_argument("--min-y", type=float, default=-1.0,
                        help="Minimum randomly sampled y coordinate in meters (waypoints mode).")
    parser.add_argument("--max-y", type=float, default=1.0,
                        help="Maximum randomly sampled y coordinate in meters (waypoints mode).")
    parser.add_argument("--bridge-wait-time", type=float, default=1.0,
                        help="Time to wait at the goal before returning to the start, in seconds.")
    parser.add_argument("--bridge-time", type=float, default=5.0,
                        help="Duration of the bridge-back trajectory, in seconds.")
    # Simulation settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-simulation", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.problem, "r", encoding="utf-8") as file:
        problem = yaml.safe_load(file)
    dt = float(problem["geometry_controller_dt"])

    params = baseline_params_from_args(args)
    sampled_waypoints = None
    if args.type == WAYPOINT_BASELINE_TYPE:
        reference_states, sampled_waypoints = waypoint_reference(dt=dt, **params)
        params.pop("provided_waypoints")
        params["num_waypoints"] = len(sampled_waypoints)
        params["waypoints"] = [waypoint.tolist() for waypoint in sampled_waypoints]
    else:
        reference_states = generate_baseline_reference(args.type, dt=dt, **params)

    print(f"Baseline type: {args.type}")
    print(f"Parameters: {params}")
    if sampled_waypoints is not None:
        print("Waypoints:")
        print(sampled_waypoints)
    report_motion(reference_states, dt, problem["robot"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name if args.name is not None else f"baseline_{args.type.replace('-', '_')}_{timestamp}"
    pickle_path, jsn_path = export_baseline_reference(
        reference_states,
        dt,
        export_dir=args.export_dir,
        name=name,
        baseline_type=args.type,
        baseline_params={key: value for key, value in params.items() if value is not None},
    )
    print("Saved baseline reference:")
    print(pickle_path)
    print(jsn_path)

    from wmr_simulator.pololu.bridge_exporter import append_bridge_reference

    bridged_path = append_bridge_reference(
        jsn_path,
        wait_time=args.bridge_wait_time,
        bridge_time=args.bridge_time,
    )
    print("Saved bridged baseline reference:")
    print(bridged_path)

    if args.skip_simulation:
        return

    from wmr_simulator.visualization.pololu import plot_logged_summary

    simulation, log = run_baseline_closed_loop(args.problem, reference_states, seed=args.seed)
    reference = np.asarray(log.reference.states, dtype=float)
    true_poses = np.asarray(log.pose.true_states, dtype=float)
    reference_indices = np.arange(len(reference)) * simulation.inner_steps_per_geometry_step
    position_errors = true_poses[reference_indices, :2] - reference[:, :2]
    position_rmse = float(np.sqrt(np.mean(np.sum(position_errors**2, axis=1))))
    print(f"Closed-loop position RMSE: {position_rmse:.4f} m")

    plot_path = plot_logged_summary(log, out_prefix=f"summary_{name}")
    print("Saved summary plot:")
    print(plot_path)


if __name__ == "__main__":
    main()
