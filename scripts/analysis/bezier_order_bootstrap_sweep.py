import argparse
import csv
import os
import pickle

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.trajectory_optimization.pipeline import (
    TrajectoryOptimizationPipeline,
    reference_states_export_payload,
)
from wmr_simulator.types import PhysicalParams, physical_params_to_array


def make_initial_params(args) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
        max_wheel_speed=jnp.asarray(args.init_max_wheel_speed, dtype=jnp.float32),
        time_constant=jnp.asarray(args.init_time_constant, dtype=jnp.float32),
    )


def save_reference_states(pipeline: TrajectoryOptimizationPipeline, order: int, out_dir: str) -> str:
    order_dir = os.path.join(out_dir, f"order_{order:02d}")
    os.makedirs(order_dir, exist_ok=True)
    out_path = os.path.join(order_dir, f"bezier_order_{order:02d}.pkl")
    with open(out_path, "wb") as file:
        pickle.dump(
            reference_states_export_payload(
                pipeline.reference_states,
                pipeline.problem.dt,
                bezier_order=order,
                control_points=np.asarray(pipeline.control_points, dtype=float),
            ),
            file,
        )
    return out_path


def bootstrap_summary(result) -> dict[str, float]:
    bootstrap = result["bootstrap"]
    hidden = np.asarray(physical_params_to_array(result["pipeline"].hidden_params), dtype=float)
    mean = np.asarray(bootstrap["parameter_mean"], dtype=float)
    covariance = np.asarray(bootstrap["parameter_covariance"], dtype=float)

    geometry_bias_mm = 1000.0 * (mean[:2] - hidden[:2])
    geometry_covariance_mm2 = covariance[:2, :2] * 1_000_000.0
    geometry_std_mm = np.sqrt(np.maximum(np.diag(geometry_covariance_mm2), 0.0))
    return {
        "wheel_radius_bias_mm": geometry_bias_mm[0],
        "base_diameter_bias_mm": geometry_bias_mm[1],
        "average_abs_geometry_bias_mm": float(np.mean(np.abs(geometry_bias_mm))),
        "wheel_radius_std_mm": geometry_std_mm[0],
        "base_diameter_std_mm": geometry_std_mm[1],
        "average_geometry_std_mm": float(np.mean(geometry_std_mm)),
        "mean_geometry_loss": float(np.asarray(result["loss_history"])[-1]),
        "mean_motor_loss": float(np.asarray(result["motor_loss_history"])[-1]),
    }


def write_summary_csv(rows: list[dict], out_path: str):
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bias_std(rows: list[dict], out_path: str):
    orders = np.asarray([row["order"] for row in rows], dtype=float)
    average_bias = np.asarray([row["average_abs_geometry_bias_mm"] for row in rows], dtype=float)
    average_std = np.asarray([row["average_geometry_std_mm"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(orders, average_bias, marker="o", linewidth=1.4, label="Average absolute bias")
    ax.plot(orders, average_std, marker="s", linewidth=1.4, label="Average standard deviation")
    ax.set_xlabel("Bezier order")
    ax.set_ylabel("Geometry parameter error [mm]")
    ax.set_title("Bootstrap Identification vs Bezier Order")
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_order(args, order: int, init_params: PhysicalParams, out_dir: str) -> dict:
    trajectory_pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        time_scaling=args.time_scaling,
    )
    constraint_component_weights = {
        "v": args.constraint_v_weight,
        "a": args.constraint_a_weight,
        "lateral": args.constraint_lateral_weight,
        "omega": args.constraint_omega_weight,
        "alpha": args.constraint_alpha_weight,
    }
    optimized_control_points, loss_history = trajectory_pipeline.optimize_bezier_trajectory(
        order=order,
        num_steps=args.opt_steps,
        learning_rate=args.opt_learning_rate,
        window_length=args.window_length,
        constraint_weight=args.constraint_weight,
        constraint_component_weights=constraint_component_weights,
        constraint_smooth_max_beta=args.constraint_smooth_max_beta,
        tangent_floor_weight=args.tangent_floor_weight,
    )
    trajectory_pipeline.set_bezier_control_points(optimized_control_points)
    reference_path = save_reference_states(trajectory_pipeline, order, out_dir)
    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.identification_steps,
        learning_rate=args.identification_learning_rate,
        seed=args.seed,
        reference_trajectories_dir=os.path.dirname(reference_path),
        window_length=args.window_length,
        bootstrap_samples=args.bootstrap_samples,
        replay_wheel_speed_source=args.replay_wheel_speeds,
    )
    row = {
        "order": order,
        "trajectory_path": reference_path,
        "trajectory_final_loss": float(loss_history[-1]) if loss_history else np.nan,
    }
    row.update(bootstrap_summary(result))
    return row


def main():
    parser = argparse.ArgumentParser(description="Sweep Bezier order and evaluate bootstrap ID statistics.")
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    parser.add_argument("--min-order", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="visualize/analysis_outputs/bezier_order_bootstrap_sweep")
    parser.add_argument("--time-scaling", choices=["s-curve", "linear"], default="s-curve")
    parser.add_argument("--window-length", type=int, default=50)
    parser.add_argument("--opt-steps", type=int, default=5000)
    parser.add_argument("--opt-learning-rate", type=float, default=2e-3)
    parser.add_argument("--identification-steps", type=int, default=1000)
    parser.add_argument("--identification-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--replay-wheel-speeds", choices=("estimated", "true"), default="estimated")
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    parser.add_argument("--init-max-wheel-speed", type=float, default=300.0)
    parser.add_argument("--init-time-constant", type=float, default=0.3)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constraint-v-weight", type=float, default=1.0)
    parser.add_argument("--constraint-a-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lateral-weight", type=float, default=1.0)
    parser.add_argument("--constraint-omega-weight", type=float, default=1.0)
    parser.add_argument("--constraint-alpha-weight", type=float, default=1.0)
    parser.add_argument("--tangent-floor-weight", type=float, default=1.0)
    parser.add_argument("--constraint-smooth-max-beta", type=float, default=20.0)
    args = parser.parse_args()

    if args.min_order < 2:
        parser.error("--min-order must be at least 2.")
    if args.max_order < args.min_order:
        parser.error("--max-order must be greater than or equal to --min-order.")

    os.makedirs(args.output_dir, exist_ok=True)
    init_params = make_initial_params(args)
    rows = []
    for order in range(args.min_order, args.max_order + 1):
        print(f"\n=== Bezier order {order} ===")
        row = run_order(args, order, init_params, args.output_dir)
        rows.append(row)
        print(
            f"order={order}  "
            f"avg_bias={row['average_abs_geometry_bias_mm']:.3f} mm  "
            f"avg_std={row['average_geometry_std_mm']:.3f} mm"
        )
        write_summary_csv(rows, os.path.join(args.output_dir, "bootstrap_summary.csv"))
        plot_bias_std(rows, os.path.join(args.output_dir, "bootstrap_bias_std.pdf"))

    print(f"\nSaved summary: {os.path.join(args.output_dir, 'bootstrap_summary.csv')}")
    print(f"Saved plot:    {os.path.join(args.output_dir, 'bootstrap_bias_std.pdf')}")


if __name__ == "__main__":
    main()
