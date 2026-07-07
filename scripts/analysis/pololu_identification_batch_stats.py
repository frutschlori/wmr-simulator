import argparse
import csv
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.pololu import load_pololu_traj_control_log
from wmr_simulator.pololu.log_loader import list_pololu_log_paths
from wmr_simulator.types import PhysicalParams, physical_params_to_array


PARAMETER_NAMES = (
    "wheel_radius_mm",
    "base_diameter_mm",
    "max_wheel_speed_rad_s",
    "time_constant_s",
)
PARAMETER_LABELS = (
    "Wheel radius [mm]",
    "Base diameter [mm]",
    "Max wheel speed [rad/s]",
    "Time constant [s]",
)
PARAMETER_SCALES = np.asarray([1000.0, 1000.0, 1.0, 1.0], dtype=float)
# The batch report covers the geometry + motor parameters (the entries of
# physical_params_to_array).
_NUM_REPORTED_PARAMS = len(PARAMETER_SCALES)


def make_initial_params(args) -> PhysicalParams:
    return PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius, dtype=jnp.float32),
        base_diameter=jnp.asarray(args.init_base_diameter, dtype=jnp.float32),
        max_wheel_speed=jnp.asarray(args.init_max_wheel_speed, dtype=jnp.float32),
        time_constant=jnp.asarray(args.init_time_constant, dtype=jnp.float32),
    )


def estimate_log_params(args, log_path: Path, init_params: PhysicalParams) -> np.ndarray:
    target_log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
    )
    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        reference_trajectories_dir=None,
        window_length=args.window_length,
        target_log=target_log,
    )
    values = np.asarray(physical_params_to_array(result["estimated_params"]), dtype=float)
    return PARAMETER_SCALES * values[:_NUM_REPORTED_PARAMS]


def summarize_group(group_name: str, estimates: np.ndarray) -> dict[str, float | str | int]:
    mean = np.mean(estimates, axis=0)
    std = np.std(estimates, axis=0, ddof=1) if len(estimates) > 1 else np.zeros(estimates.shape[1], dtype=float)
    row: dict[str, float | str | int] = {
        "group": group_name,
        "num_logs": len(estimates),
        "average_geometry_mean_mm": float(np.mean(mean[:2])),
        "average_geometry_std_mm": float(np.mean(std[:2])),
        "average_motor_mean": float(np.mean(mean[2:])),
        "average_motor_std": float(np.mean(std[2:])),
    }
    for index, name in enumerate(PARAMETER_NAMES):
        row[f"{name}_mean"] = float(mean[index])
        row[f"{name}_std"] = float(std[index])
    return row


def write_summary_csv(rows: list[dict], out_path: Path):
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_group_stats(rows: list[dict], out_path: Path):
    group_names = [str(row["group"]) for row in rows]
    x = np.arange(len(group_names))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for index, ax in enumerate(axes.ravel()):
        name = PARAMETER_NAMES[index]
        means = np.asarray([row[f"{name}_mean"] for row in rows], dtype=float)
        stds = np.asarray([row[f"{name}_std"] for row in rows], dtype=float)
        ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.78)
        ax.set_ylabel(PARAMETER_LABELS[index])
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=25, ha="right")

    fig.suptitle("Pololu System Identification Batch Statistics")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_group(args, log_dir: Path, init_params: PhysicalParams) -> tuple[dict, list[dict]]:
    log_paths = list_pololu_log_paths(log_dir)
    estimates = []
    detail_rows = []
    print(f"\n=== {log_dir} ({len(log_paths)} logs) ===")
    for log_path in log_paths:
        print(f"Identifying {log_path.name}...")
        values = estimate_log_params(args, log_path, init_params)
        estimates.append(values)
        detail_row = {"group": log_dir.name, "log": str(log_path)}
        detail_row.update({name: float(values[index]) for index, name in enumerate(PARAMETER_NAMES)})
        detail_rows.append(detail_row)
        print(
            f"  r={values[0]:.3f} mm, b={values[1]:.3f} mm, "
            f"wmax={values[2]:.3f} rad/s, tau={values[3]:.5f} s"
        )
    return summarize_group(log_dir.name, np.vstack(estimates)), detail_rows


def main():
    parser = argparse.ArgumentParser(description="Run Pololu system identification over one or more log directories.")
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    parser.add_argument("--log-dirs", nargs="+",default=[
                        "Pololu Data/Experiments/2026_06_22/Analysis/opt_slow",
                        "Pololu Data/Experiments/2026_06_22/Analysis/opt_fast",
                        "Pololu Data/Experiments/2026_06_22/Analysis/opt_fast2",
                        "Pololu Data/Experiments/2026_06_22/Analysis/opt_turbo",
                        "Pololu Data/Experiments/2026_06_22/Analysis/circle",
                        "Pololu Data/Experiments/2026_06_22/Analysis/figure_eight",
    ])
    parser.add_argument("--output-dir", type=str, default="visualize/analysis_outputs/pololu_identification_batch")
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    parser.add_argument("--init-max-wheel-speed", type=float, default=300.0)
    parser.add_argument("--init-time-constant", type=float, default=0.3)
    args = parser.parse_args()

    init_params = make_initial_params(args)
    summary_rows = []
    detail_rows = []
    for log_dir in [Path(path) for path in args.log_dirs]:
        summary_row, group_detail_rows = run_group(args, log_dir, init_params)
        summary_rows.append(summary_row)
        detail_rows.extend(group_detail_rows)

    output_dir = Path(args.output_dir)
    write_summary_csv(summary_rows, output_dir / "summary.csv")
    write_summary_csv(detail_rows, output_dir / "estimates.csv")
    plot_group_stats(summary_rows, output_dir / "parameter_stats.pdf")

    print(f"\nSaved summary:   {output_dir / 'summary.csv'}")
    print(f"Saved estimates: {output_dir / 'estimates.csv'}")
    print(f"Saved plot:      {output_dir / 'parameter_stats.pdf'}")


if __name__ == "__main__":
    main()
