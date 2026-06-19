import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.pololu import load_pololu_traj_control_log
from wmr_simulator.types import PhysicalParams
from wmr_simulator.identification.analysis import run_physical_parameter_surface


def load_motor_params(problem_path: str) -> tuple[float, float]:
    with open(problem_path, "r", encoding="utf-8") as file:
        problem = yaml.safe_load(file)
    robot = problem["robot"]
    return float(robot["max_wheel_speed"]), float(robot["time_constant"])


def main():
    parser = argparse.ArgumentParser(description="Plot system-ID replay loss over physical parameters.")
    # Setup description
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    # Log or reference loading
    parser.add_argument("--pololu-log", type=str, default="Pololu Data/Logs/Event based/TR02")
    # parser.add_argument("--pololu-log", type=str, default=None)
    # parser.add_argument("--reference-trajectory", type=str, default="trajectory_exports/20cp_constrained_vertikal.pkl")
    parser.add_argument("--reference-trajectory", type=str, default=None)
    # Replay options for identification loss
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    # Grid settings
    parser.add_argument("--radius-min", type=float, default=0.01)
    parser.add_argument("--radius-max", type=float, default=0.025)
    parser.add_argument("--base-min", type=float, default=0.04)
    parser.add_argument("--base-max", type=float, default=0.14)
    parser.add_argument("--radius-points", type=int, default=100)
    parser.add_argument("--base-points", type=int, default=100)
    # Output settings
    parser.add_argument("--output", type=str, default="real_id_loss_surface")
    parser.add_argument("--save-pdf", action="store_true", default=True)
    parser.add_argument("--show-plot", action="store_true", default=True)
    args = parser.parse_args()

    max_wheel_speed, time_constant = load_motor_params(args.problem)
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
        max_wheel_speed=jnp.asarray(max_wheel_speed, dtype=jnp.float32),
        time_constant=jnp.asarray(time_constant, dtype=jnp.float32),
    )
    if args.pololu_log is not None and args.reference_trajectory is not None:
        parser.error("--pololu-log cannot be combined with --reference-trajectory.")

    target_log = load_pololu_traj_control_log(args.pololu_log) if args.pololu_log is not None else None
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
        target_log=target_log,
        out_prefix=args.output,
        save_plots=args.save_pdf,
        show_plots=args.show_plot,
    )
    surface = result["tracking_error_surface"]
    print(f"Minimum sampled loss: {float(np.min(surface)):.8f}")
    if args.pololu_log is not None:
        print(f"Experiment log: {args.pololu_log}")
    if result["reference_trajectory_path"] is not None:
        print(f"Reference trajectory: {result['reference_trajectory_path']}")


if __name__ == "__main__":
    main()
