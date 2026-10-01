"""Identification side study (thesis Ch. 5, Phase 2b): designed vs fixed
identification trajectory, repeated over noise/placement seeds in MuJoCo.

For each (trajectory, controller) cell the trajectory is planned once
(plan-id-trajectory in a base experiment) and then driven by `--seeds`
independent experiments that differ only in their seed, i.e. in hand placement
and sensor noise: simulate-deployment (mujoco_deployment.num_logs chained logs)
-> decode-logs -> identify. Reported per cell:

- the identified parameters' mean, spread and error against the hidden plant's
  measured truth (mujoco_sim.truth; wheel radius, wheelbase on arcs and in
  spin, max wheel speed, logged time constant);
- the sample covariance of (r, L) in relative coordinates against the
  Cramer-Rao bound of the design's own FIM (identification mode, [r, L],
  relative scaling), evaluated at that trajectory and divided by the number of
  logs a fit uses.

    python scripts/analysis/identification_side_study.py --out "Pololu Data/thesis_ch5/phase2b_identification"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONTROLLERS = {
    # The iteration-1 controller of every study run.
    "unit": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    # Converged static S-A-N gains (Phase 2, generation 5, median over seeds).
    "converged": [2.10, 5.78, 6.99, 2.86, 0.0, 0.0],
}
TRAJECTORIES = ("designed", "fixed")


def _overrides(seed: int, trajectory: str) -> dict:
    return {
        "problem": str(REPO / "problems/pololu_gains.yaml"),
        "baseline_identification_trajectory": str(
            REPO / "trajectory_exports/fixed_sets/identification/fixed_identification.pkl"
        ),
        "seed": int(seed),
        "use_gain_parametrization": False,
        "use_residual_model": False,
        "optimize_identification_trajectory": trajectory == "designed",
        "mujoco_deployment": {"enabled": True},
    }


def _experiment_with_gains(root: Path, seed: int, trajectory: str, gains: list[float]):
    from wmr_simulator.active_learning import stages
    from wmr_simulator.active_learning.experiment import Experiment, load_yaml

    if not (root / "experiment.yaml").is_file():
        stages.stage_init(root, _overrides(seed, trajectory))
    experiment = Experiment.load(root)
    robot_config = load_yaml(experiment.paths(1).robot_config)
    robot_config["controller"]["gains"] = [float(gain) for gain in gains]
    stages._initialize_iteration(experiment, 1, robot_config)
    return experiment


def plan_cell(job: dict) -> str:
    """Plan the cell's identification trajectory once, in a base experiment."""
    from wmr_simulator.active_learning import stages

    root = Path(job["out"]) / f"{job['trajectory']}_{job['controller']}" / "base"
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)  # the stages write their plots to a cwd-relative visualize/
    experiment = _experiment_with_gains(root, 0, job["trajectory"], CONTROLLERS[job["controller"]])
    paths = experiment.paths(1)
    if not any(paths.identification_trajectory_dir.glob("*.pkl")):
        stages.stage_plan_identification_trajectory(experiment, 1)
    return str(paths.identification_trajectory_dir)


def identify_seed(job: dict) -> dict:
    """One seed of one cell: deploy, decode, identify; returns the joint fit."""
    from wmr_simulator.active_learning import stages
    from wmr_simulator.active_learning.experiment import load_yaml

    cell = Path(job["out"]) / f"{job['trajectory']}_{job['controller']}"
    root = cell / f"seed{job['seed']:02d}"
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)  # the stages write their plots to a cwd-relative visualize/
    experiment = _experiment_with_gains(root, job["seed"], job["trajectory"], CONTROLLERS[job["controller"]])
    paths = experiment.paths(1)
    if not any(paths.identification_trajectory_dir.glob("*.pkl")):
        for source in (cell / "base" / "iteration_01" / "identification_trajectory").iterdir():
            if source.is_file():
                shutil.copy2(source, paths.identification_trajectory_dir)
    if not paths.identification_result.is_file():
        if not list(paths.data_dir.glob("TR*")):
            stages.stage_simulate_deployment(experiment, 1)
        stages.stage_decode_logs(experiment, 1)
        stages.stage_identify(experiment, 1)
    result = load_yaml(paths.identification_result)
    return {
        "trajectory": job["trajectory"],
        "controller": job["controller"],
        "seed": job["seed"],
        **{key: float(value) for key, value in result["estimated_params"].items()},
        "num_logs": len(result["logs"]),
        "excluded_logs": len(result["excluded_logs"]),
        "encoder_lag_ms": [lag["lag_ms"] for lag in result["encoder_lags"].values() if lag],
    }


def cramer_rao(job: dict) -> dict:
    """Relative CRB of (r, L) for one log of the cell's trajectory, from the
    design's own FIM (identification mode), and the log count a fit uses."""
    import pickle

    from wmr_simulator.active_learning.experiment import Experiment
    from wmr_simulator.active_learning.stages import _identification_problem
    from wmr_simulator.trajectory_optimization.fim import fim_from_factor
    from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

    root = Path(job["out"]) / f"{job['trajectory']}_{job['controller']}" / "base"
    experiment = Experiment.load(root)
    paths = experiment.paths(1)
    config = experiment.config["identification_trajectory"]
    pipeline = TrajectoryOptimizationPipeline(
        str(_identification_problem(paths, config)),
        time_scaling=config["time_scaling"],
        objective_mode="identification",
        controller_gains=CONTROLLERS[job["controller"]],
        motion_phases=config["phases"],
    )
    pickle_path = sorted(paths.identification_trajectory_dir.glob("*.pkl"))[0]
    payload = pickle.load(open(pickle_path, "rb"))
    # Identification mode reads its measurements off a closed-loop log (the
    # pipeline's own curve unless one is passed), so roll the closed loop out on
    # this cell's trajectory first.
    closed_loop_log = pipeline.run_closed_loop_deployment(
        np.asarray(payload["reference_states"], dtype=np.float32)
    )
    fim = np.asarray(
        fim_from_factor(
            pipeline.compute_fim_factor(window_length=config["window_length"], closed_loop_log=closed_loop_log)
        ),
        dtype=float,
    )
    return {
        "trajectory": job["trajectory"],
        "controller": job["controller"],
        "fim_per_log": fim.tolist(),
        "num_logs": int(experiment.config["mujoco_deployment"]["num_logs"]),
    }


def summarize(results: list[dict], bounds: list[dict], truth) -> str:
    lines = [
        "| trajectory | controller | seeds | r [mm] | L [mm] | u [rad/s] | tau [s] | "
        "std r / CRB r | std L / CRB L | corr(r,L) sample / CRB |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for trajectory in TRAJECTORIES:
        for controller in CONTROLLERS:
            rows = [row for row in results if row["trajectory"] == trajectory and row["controller"] == controller]
            if not rows:
                continue
            values = np.asarray([[row["wheel_radius"], row["base_diameter"]] for row in rows])
            mean = values.mean(axis=0)
            relative = values / mean - 1.0
            sample_cov = np.cov(relative, rowvar=False)
            bound = next(b for b in bounds if b["trajectory"] == trajectory and b["controller"] == controller)
            crb = np.linalg.inv(np.asarray(bound["fim_per_log"])) / bound["num_logs"]

            def stat(key, scale, digits):
                column = np.asarray([row[key] for row in rows]) * scale
                return f"{column.mean():.{digits}f} ± {column.std(ddof=1):.{digits}f}"

            sample_corr = sample_cov[0, 1] / np.sqrt(sample_cov[0, 0] * sample_cov[1, 1])
            crb_corr = crb[0, 1] / np.sqrt(crb[0, 0] * crb[1, 1])
            lines.append(
                f"| {trajectory} | {controller} | {len(rows)} | {stat('wheel_radius', 1000, 3)} | "
                f"{stat('base_diameter', 1000, 2)} | {stat('max_wheel_speed', 1, 1)} | {stat('time_constant', 1, 3)} | "
                f"{100 * np.sqrt(sample_cov[0, 0]):.3f} % / {100 * np.sqrt(crb[0, 0]):.3f} % | "
                f"{100 * np.sqrt(sample_cov[1, 1]):.3f} % / {100 * np.sqrt(crb[1, 1]):.3f} % | "
                f"{sample_corr:+.2f} / {crb_corr:+.2f} |"
            )
    lines.append(
        f"\nMuJoCo truth (mujoco_sim.truth.measure_plant_truth): r {1000 * truth.wheel_radius:.3f} mm, "
        f"L {1000 * truth.wheel_base_arc:.2f} mm on arcs / {1000 * truth.wheel_base_spin:.2f} mm in spin, "
        f"u {truth.max_wheel_speed:.2f} rad/s, tau {truth.time_constant_logged:.3f} s (logged; effective)."
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells = [{"out": str(out), "trajectory": t, "controller": c} for t in TRAJECTORIES for c in CONTROLLERS]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cells))) as pool:
        list(pool.map(plan_cell, cells))
    jobs = [{**cell, "seed": seed} for cell in cells for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(identify_seed, jobs))
        bounds = list(pool.map(cramer_rao, cells))
    from wmr_simulator.mujoco_sim.truth import measure_plant_truth

    truth = measure_plant_truth()
    json.dump({"results": results, "bounds": bounds, "truth": str(truth)}, open(out / "side_study.json", "w"), indent=1)
    table = summarize(results, bounds, truth)
    (out / "summary.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
