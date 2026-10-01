"""Identification side study (thesis Ch. 5, Phase 2b): designed vs fixed vs
random identification trajectories, repeated over noise/placement seeds in MuJoCo.

Trajectories, all with the design's two motion phases and limits:

- ``designed``: the FIM design (plan-id-trajectory);
- ``fixed``: the hand-made reference (two slow arcs, left then right, and a fast
  arc; ``fixed_sets.fixed_identification_reference``);
- ``random_XX``: random draws from the designer's own curve family, projected
  into the limits without the FIM (``fixed_sets.random_identification_reference``).

Each trajectory is driven by ``--seeds`` independent experiments at the
converged controller, which differ only in their seed, i.e. in hand placement
and sensor noise: simulate-deployment (``mujoco_deployment.num_logs`` chained
logs) -> decode-logs -> identify. Reported per trajectory:

- the identified parameters' mean, spread and error against the hidden plant's
  measured truth (``mujoco_sim.truth``);
- the sample covariance of (r, L) in relative coordinates against Cramer-Rao
  bounds of the design's FIM (identification mode, [r, L], relative scaling)
  at that trajectory, divided by the logs a fit uses:
    * ``design``: the measurement noise the design assumes (the yaml's
      noise_pos / noise_angle), inputs exact -- what the designer optimizes;
    * ``mocap``: MuJoCo's actual mocap noise, inputs exact;
    * ``eiv``: MuJoCo's actual mocap noise plus the encoder input noise carried
      through the replay (errors in variables): the replayed poses depend on the
      measured wheel speeds, whose noise -- count quantization, differenced to a
      speed, then the loader's smoothing (the firmware low-pass is inverted
      exactly, so it cancels) -- enters as G Sigma_u G^T with G the replay's
      Jacobian in the wheel speeds.

    python scripts/analysis/identification_side_study.py --out "Pololu Data/thesis_ch5/phase2b_identification_v2"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
# Converged static S-A-N gains (Phase 2, generation 5, median over seeds). At
# the iteration-1 unit gains the robot does not follow the trajectory and every
# trajectory identifies badly (Phase 2b, 2026-10-01).
GAINS = [2.10, 5.78, 6.99, 2.86, 0.0, 0.0]
FIXED_PICKLE = REPO / "trajectory_exports/fixed_sets/identification/fixed_identification.pkl"


def _trajectories(num_random: int) -> list[str]:
    return ["designed", "fixed"] + [f"random_{index:02d}" for index in range(num_random)]


def _overrides(seed: int, trajectory: str, out: Path) -> dict:
    baseline = FIXED_PICKLE if trajectory == "fixed" else out / trajectory / "reference.pkl"
    return {
        "problem": str(REPO / "problems/pololu_gains.yaml"),
        "baseline_identification_trajectory": str(baseline),
        "seed": int(seed),
        "use_gain_parametrization": False,
        "use_residual_model": False,
        "optimize_identification_trajectory": trajectory == "designed",
        "mujoco_deployment": {"enabled": True},
    }


def _experiment(root: Path, seed: int, trajectory: str, out: Path):
    from wmr_simulator.active_learning import stages
    from wmr_simulator.active_learning.experiment import Experiment, load_yaml

    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)  # the stages write their plots to a cwd-relative visualize/
    if not (root / "experiment.yaml").is_file():
        stages.stage_init(root, _overrides(seed, trajectory, out))
    experiment = Experiment.load(root)
    robot_config = load_yaml(experiment.paths(1).robot_config)
    robot_config["controller"]["gains"] = [float(gain) for gain in GAINS]
    stages._initialize_iteration(experiment, 1, robot_config)
    return experiment


def _design_pipeline(experiment):
    from wmr_simulator.active_learning.stages import _identification_problem
    from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

    config = experiment.config["identification_trajectory"]
    return TrajectoryOptimizationPipeline(
        str(_identification_problem(experiment.paths(1), config)),
        time_scaling=config["time_scaling"],
        objective_mode="identification",
        controller_gains=GAINS,
        motion_phases=config["phases"],
    )


def plan_trajectory(job: dict) -> str:
    """Plan (designed), copy (fixed) or draw (random) the trajectory once."""
    from wmr_simulator.active_learning import stages
    from wmr_simulator.trajectory_optimization.fixed_sets import random_identification_reference

    out = Path(job["out"])
    trajectory = job["trajectory"]
    root = out / trajectory / "base"
    experiment = _experiment(root, 0, trajectory, out)
    paths = experiment.paths(1)
    if trajectory.startswith("random_") and not (out / trajectory / "reference.pkl").is_file():
        pipeline = _design_pipeline(experiment)
        config = experiment.config["identification_trajectory"]
        states, identified_duration, control_points = random_identification_reference(
            pipeline, int(config["num_control_points"]), seed=int(trajectory.split("_")[1])
        )
        payload = {
            "reference_states": states,
            "dt": float(pipeline.problem.dt),
            "identified_duration": float(identified_duration),
            "phase_breaks": pipeline.phase_breaks,
            "control_points": control_points,
        }
        with open(out / trajectory / "reference.pkl", "wb") as file:
            pickle.dump(payload, file)
    if not any(paths.identification_trajectory_dir.glob("*.pkl")):
        stages.stage_plan_identification_trajectory(experiment, 1)
    return str(paths.identification_trajectory_dir)


def identify_seed(job: dict) -> dict:
    """One seed of one trajectory: deploy, decode, identify; returns the joint fit."""
    from wmr_simulator.active_learning import stages
    from wmr_simulator.active_learning.experiment import load_yaml

    out = Path(job["out"])
    cell = out / job["trajectory"]
    experiment = _experiment(cell / f"seed{job['seed']:02d}", job["seed"], job["trajectory"], out)
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
        "seed": job["seed"],
        **{key: float(value) for key, value in result["estimated_params"].items()},
        "num_logs": len(result["logs"]),
        "excluded_logs": len(result["excluded_logs"]),
        "encoder_lag_ms": [lag["lag_ms"] for lag in result["encoder_lags"].values() if lag],
    }


def _encoder_speed_covariance(num_samples: int, dt: float, counts_per_revolution: float) -> np.ndarray:
    """Covariance of the loader's wheel-speed series from count quantization.

    Cumulative counts carry independent uniform rounding errors (variance
    q^2/12, q one count in rad), a speed sample is their difference over dt,
    the firmware low-pass is inverted exactly and cancels, and the loader then
    applies its Savitzky-Golay smoothing S: Sigma = S D (q^2/12) D^T S^T / dt^2.
    """
    from wmr_simulator.pololu.measurement_smoothing import (
        DEFAULT_ENCODER_SAVGOL_POLYORDER,
        DEFAULT_ENCODER_SAVGOL_WINDOW,
        savgol_smooth_series,
    )

    count = 2.0 * np.pi / abs(counts_per_revolution)
    difference = np.eye(num_samples) - np.eye(num_samples, k=-1)
    time_s = np.arange(num_samples) * dt
    smoothing = savgol_smooth_series(
        time_s, np.eye(num_samples), window_length=DEFAULT_ENCODER_SAVGOL_WINDOW,
        polyorder=DEFAULT_ENCODER_SAVGOL_POLYORDER,
    )
    chain = smoothing @ difference / dt
    return (count**2 / 12.0) * chain @ chain.T


def cramer_rao(job: dict) -> dict:
    """Per-log (r, L) Cramer-Rao bounds of one trajectory, in relative coordinates."""
    import jax
    import jax.numpy as jnp

    from wmr_simulator.active_learning.experiment import Experiment, load_yaml

    out = Path(job["out"])
    root = out / job["trajectory"] / "base"
    os.chdir(root)
    experiment = Experiment.load(root)
    paths = experiment.paths(1)
    window = int(experiment.config["identification_trajectory"]["window_length"])
    pipeline = _design_pipeline(experiment)
    payload = pickle.load(open(sorted(paths.identification_trajectory_dir.glob("*.pkl"))[0], "rb"))
    log = pipeline.run_closed_loop_deployment(np.asarray(payload["reference_states"], dtype=np.float32))
    params = pipeline.nominal_parameters()
    scaling = pipeline.fim_parameter_scaling(params)

    def measurements(p, speeds):
        wheel = log.wheel._replace(speeds=speeds)
        return pipeline.measurement_vector(p, window_length=window, closed_loop_log=log._replace(wheel=wheel))

    speeds = log.wheel.speeds
    sensitivity = np.asarray(jax.jacfwd(lambda p: measurements(p, speeds))(params), dtype=float) * np.asarray(scaling)
    input_jacobian = np.asarray(
        jax.jacfwd(lambda s: measurements(params, s))(speeds), dtype=float
    ).reshape(sensitivity.shape[0], -1)

    hidden = yaml.safe_load(open(REPO / "models/pololu_hidden.yaml"))
    num_poses = sensitivity.shape[0] // 3

    def fim_with(position_std, angle_std, input_covariance=None):
        variances = np.tile([position_std**2, position_std**2, angle_std**2], num_poses)
        covariance = np.diag(variances)
        if input_covariance is not None:
            covariance = covariance + input_jacobian @ input_covariance @ input_jacobian.T
        return sensitivity.T @ np.linalg.solve(covariance, sensitivity)

    estimator = load_yaml(paths.problem)["estimator"]
    per_wheel = _encoder_speed_covariance(
        int(speeds.shape[0]), float(np.median(np.diff(np.asarray(log.wheel.time_s)))),
        float(hidden["encoder"]["counts_per_revolution"]),
    )
    input_covariance = np.kron(per_wheel, np.eye(2))  # speeds flatten as (sample, wheel)
    mocap = hidden["mocap"]
    fims = {
        "design": fim_with(float(estimator["noise_pos"]), float(estimator["noise_angle"])),
        "mocap": fim_with(float(mocap["position_noise_std"]), float(mocap["angle_noise_std"])),
        "eiv": fim_with(float(mocap["position_noise_std"]), float(mocap["angle_noise_std"]), input_covariance),
    }
    num_logs = int(experiment.config["mujoco_deployment"]["num_logs"])
    return {
        "trajectory": job["trajectory"],
        "num_logs": num_logs,
        "crb": {name: (np.linalg.inv(fim) / num_logs).tolist() for name, fim in fims.items()},
    }


def summarize(results: list[dict], bounds: list[dict], truth, trajectories: list[str]) -> str:
    lines = [
        "| trajectory | seeds | r [mm] | L [mm] | u [rad/s] | tau [s] | std r: sample / design / mocap / eiv CRB | "
        "std L: sample / design / mocap / eiv CRB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for trajectory in trajectories:
        rows = [row for row in results if row["trajectory"] == trajectory]
        if not rows:
            continue
        values = np.asarray([[row["wheel_radius"], row["base_diameter"]] for row in rows])
        sample = np.cov(values / values.mean(axis=0) - 1.0, rowvar=False)
        bound = next(b for b in bounds if b["trajectory"] == trajectory)["crb"]

        def stat(key, scale, digits):
            column = np.asarray([row[key] for row in rows]) * scale
            return f"{column.mean():.{digits}f} ± {column.std(ddof=1):.{digits}f}"

        def stds(index):
            parts = [np.sqrt(sample[index, index])] + [np.sqrt(np.asarray(bound[name])[index, index]) for name in ("design", "mocap", "eiv")]
            return " / ".join(f"{100 * value:.4f}" for value in parts) + " %"

        lines.append(
            f"| {trajectory} | {len(rows)} | {stat('wheel_radius', 1000, 3)} | {stat('base_diameter', 1000, 2)} | "
            f"{stat('max_wheel_speed', 1, 1)} | {stat('time_constant', 1, 3)} | {stds(0)} | {stds(1)} |"
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
    parser.add_argument("--random", type=int, default=5, help="Random trajectories drawn.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trajectories = _trajectories(args.random)
    cells = [{"out": str(out), "trajectory": trajectory} for trajectory in trajectories]
    # One task per process: JAX keeps every compiled function of a process, and
    # long-lived workers fitting logs of many lengths ran the machine out of
    # memory (2026-10-01).
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        list(pool.map(plan_trajectory, cells))
        results = list(pool.map(identify_seed, [{**cell, "seed": seed} for cell in cells for seed in range(args.seeds)]))
        bounds = list(pool.map(cramer_rao, cells))
    from wmr_simulator.mujoco_sim.truth import measure_plant_truth

    truth = measure_plant_truth()
    json.dump({"results": results, "bounds": bounds, "truth": str(truth)}, open(out / "side_study.json", "w"), indent=1)
    table = summarize(results, bounds, truth, trajectories)
    (out / "summary.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
