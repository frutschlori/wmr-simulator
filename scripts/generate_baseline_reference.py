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
"""

import argparse
from datetime import datetime
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


def baseline_params_from_args(args) -> dict:
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
    # BASELINE_TYPES = ("circle", "lemniscate", "lemniscate-ramp", "spin")
    parser.add_argument("--type", choices=list(BASELINE_TYPES), default="lemniscate-ramp")
    parser.add_argument("--export-dir", default="trajectory_exports/baselines")
    parser.add_argument("--name", default=None, help="Export stem (default: baseline_<type>_<timestamp>).")
    # Shared parameters
    parser.add_argument("--time", type=float, default=15.0, help="Total trajectory time in seconds.")
    parser.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], help="Path center [x y] in meters.")
    # Circle parameters
    parser.add_argument("--radius", type=float, default=0.5, help="Circle radius in meters.")
    parser.add_argument("--start-angle", type=float, default=0.0, help="Circle/spin start angle in radians.")
    parser.add_argument("--clockwise", action="store_true", default=False)
    # Spin parameters (mocap-delay identification)
    parser.add_argument("--max-omega", type=float, default=None,
                        help="Peak angular rate in rad/s (required for spin).")
    # Lemniscate parameters
    parser.add_argument("--max-speed", type=float, default=None,
                        help="Peak speed in m/s (required for lemniscate, optional amplitude override for ramp).")
    parser.add_argument("--cycles", type=int, default=3, help="Figure-eight cycles (lemniscate modes).")
    # Lemniscate ramp parameters
    parser.add_argument("--speed-rate", type=float, default=1.5,
                        help="Phase-rate growth factor per completed cycle (lemniscate-ramp).")
    parser.add_argument("--amplitude", type=float, default=1, help="Lemniscate amplitude in meters (ramp mode).")
    parser.add_argument("--ramp-up-fraction", type=float, default=0.2)
    parser.add_argument("--ramp-down-fraction", type=float, default=0.1)
    # Simulation settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-simulation", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.problem, "r", encoding="utf-8") as file:
        problem = yaml.safe_load(file)
    dt = float(problem["geometry_controller_dt"])

    params = baseline_params_from_args(args)
    reference_states = generate_baseline_reference(args.type, dt=dt, **params)

    print(f"Baseline type: {args.type}")
    print(f"Parameters: {params}")
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
