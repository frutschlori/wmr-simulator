import argparse

import numpy as np

from wmr_simulator.gain_tuning.analysis import run_gain_tracking_error_surface
from wmr_simulator.gain_tuning.pipeline import resolve_gain_robot_params


def main():
    parser = argparse.ArgumentParser(description="Plot closed-loop loss over selected controller gains.")
    parser.add_argument("--problem", type=str, default="problems/figure_eight.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--num-realizations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)
    parser.add_argument("--kx-min", type=float, default=0.0)
    parser.add_argument("--kx-max", type=float, default=20.0)
    parser.add_argument("--kx-points", type=int, default=100)
    parser.add_argument("--ky-min", type=float, default=0.0)
    parser.add_argument("--ky-max", type=float, default=20.0)
    parser.add_argument("--ky-points", type=int, default=100)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output", type=str, default="gain_tracking_error_surface")
    args = parser.parse_args()

    robot_params = resolve_gain_robot_params(
        args.problem,
        args.fixed_wheel_radius,
        args.fixed_base_diameter,
    )
    result = run_gain_tracking_error_surface(
        problem_path=args.problem,
        robot_params=robot_params,
        kx_min=args.kx_min,
        kx_max=args.kx_max,
        kx_points=args.kx_points,
        ky_min=args.ky_min,
        ky_max=args.ky_max,
        ky_points=args.ky_points,
        num_realizations=args.num_realizations,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        reference_trajectories_dir=args.reference_trajectories_dir,
        out_prefix=args.output,
    )
    surface = result["tracking_error_surface"]
    print(f"Reference window: {result['window_start']}:{result['window_end']}")
    print(f"Minimum sampled loss: {float(np.min(surface)):.8f}")


if __name__ == "__main__":
    main()
