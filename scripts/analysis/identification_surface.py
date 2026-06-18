import argparse

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.analysis import run_physical_parameter_surface
from wmr_simulator.types import PhysicalParams


def main():
    parser = argparse.ArgumentParser(description="Plot system-ID replay loss over physical parameters.")
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--reference-trajectory", type=str, default=None)
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.2)
    parser.add_argument("--init-max-wheel-speed", type=float, default=150.0)
    parser.add_argument("--init-time-constant", type=float, default=0.1)
    parser.add_argument("--radius-min", type=float, default=0.01)
    parser.add_argument("--radius-max", type=float, default=0.1)
    parser.add_argument("--radius-points", type=int, default=100)
    parser.add_argument("--base-min", type=float, default=0.01)
    parser.add_argument("--base-max", type=float, default=0.5)
    parser.add_argument("--base-points", type=int, default=100)
    parser.add_argument("--output", type=str, default="si_tracking_error_surface")
    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
        max_wheel_speed=jnp.asarray(args.init_max_wheel_speed, dtype=jnp.float32),
        time_constant=jnp.asarray(args.init_time_constant, dtype=jnp.float32),
    )
    result = run_physical_parameter_surface(
        problem_path=args.problem,
        initial_params=init_params,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        radius_points=args.radius_points,
        base_min=args.base_min,
        base_max=args.base_max,
        base_points=args.base_points,
        seed=args.seed,
        window_length=args.window_length,
        reference_trajectory_path=args.reference_trajectory,
        reference_trajectories_dir=args.reference_trajectories_dir,
        out_prefix=args.output,
    )
    surface = result["tracking_error_surface"]
    print(f"Minimum sampled loss: {float(np.min(surface)):.8f}")
    if result["reference_trajectory_path"] is not None:
        print(f"Reference trajectory: {result['reference_trajectory_path']}")


if __name__ == "__main__":
    main()
