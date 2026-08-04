"""Paired study of optimized versus frozen-random trajectory start offsets.

The harness is deliberately resumable: trajectory exports and individual
tuning records are cached below ``--output-dir``.  A design seed controls the
initial B-spline batch and constraint jitter; a shuffle seed independently
controls the gain tuner's train/validation split, replay noise, and LHS draw.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.objectives import (
    closed_loop_objective_terms,
    make_realizations,
    split_realization_keys_by_trajectory,
)
from wmr_simulator.gain_tuning.pipeline import (
    ControllerTuningPipeline,
    resolve_gain_robot_params,
    run_gain_tuning_experiment,
)
from wmr_simulator.joint_tuning.benchmark import held_out_trajectories, pose_rmse
from wmr_simulator.trajectory_optimization.objectives import fim_loss, fim_objective_term
from wmr_simulator.trajectory_optimization.pipeline import (
    OBJECTIVE_MODE_GAIN_TUNING,
    TrajectoryOptimizationPipeline,
    reference_states_export_payload,
)


GAIN_NAMES = ("kx", "ky", "kth", "kp_motor", "ki_motor")
MODES = ("random", "optimize")
EVAL_SEED = 1234


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _split_terms(result: dict, validation: bool) -> np.ndarray:
    pipeline = result["pipeline"]
    references = (
        pipeline.validation_reference_trajectories
        if validation
        else pipeline.training_reference_trajectories
    )
    offsets = result["validation_start_offsets"] if validation else result["training_start_offsets"]
    namespace = 1 if validation else 0
    roots = result["realizations"]
    robot_keys = split_realization_keys_by_trajectory(
        roots.robot_keys, int(references.shape[0]), namespace=namespace
    )
    estimator_keys = split_realization_keys_by_trajectory(
        roots.estimator_keys, int(references.shape[0]), namespace=namespace
    )

    def one(reference, robot_key, estimator_key, start_offsets):
        return closed_loop_objective_terms(
            pipeline,
            result["optimized_gains"],
            robot_key,
            estimator_key,
            velocity_tracking_weight=GAIN_TUNING_DEFAULTS["velocity_tracking_weight"],
            input_weight=GAIN_TUNING_DEFAULTS["input_weight"],
            input_delta_weight=GAIN_TUNING_DEFAULTS["input_delta_weight"],
            omega_delta_weight=GAIN_TUNING_DEFAULTS["omega_delta_weight"],
            reference_states=reference,
            initial_pose_offsets=start_offsets,
        )

    terms = jax.vmap(one)(references, robot_keys, estimator_keys, offsets)
    return np.asarray(jnp.mean(terms, axis=0), dtype=float)


def synthesize(args, mode: str, design_seed: int, design_dir: Path) -> dict:
    metrics_path = design_dir / "design.json"
    if metrics_path.exists() and len(list(design_dir.glob("trajectory_*.pkl"))) == args.num_trajectories:
        print(f"[design cached] mode={mode} seed={design_seed}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    print(f"[design start] mode={mode} seed={design_seed}", flush=True)
    started = time.perf_counter()
    pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        objective_mode=OBJECTIVE_MODE_GAIN_TUNING,
        num_realizations=args.num_realizations,
        start_offset_mode=mode,
        offset_displacement_step_factor=args.offset_displacement_step_factor,
        offset_heading_step_factor=args.offset_heading_step_factor,
    )
    control_points, _ = pipeline.optimize_trajectories(
        num_control_points=args.num_control_points,
        num_steps=args.trajectory_steps,
        learning_rate=args.trajectory_learning_rate,
        num_trajectories=args.num_trajectories,
        vectorized=True,
        constraint_weight_jitter=args.constraint_weight_jitter,
        seed=design_seed,
        verbose=False,
    )
    design_dir.mkdir(parents=True, exist_ok=True)
    raw_fim = []
    log_fim = []
    for index, points in enumerate(control_points):
        reference = pipeline.reference_states_from_control_points(points)
        realizations = pipeline.realizations._replace(start_offsets=pipeline.batch_start_offsets[index])
        factor = pipeline.compute_fim_factor(reference_states=reference, realizations=realizations)
        raw_fim.append(float(fim_loss(factor, pipeline.criterion)))
        log_fim.append(float(fim_objective_term(factor, pipeline.criterion)))
        payload = reference_states_export_payload(
            reference,
            pipeline.problem.dt,
            start_offsets=pipeline.batch_start_offsets[index],
            design_seed=design_seed,
            start_offset_mode=mode,
        )
        with (design_dir / f"trajectory_{index:02d}.pkl").open("wb") as file:
            pickle.dump(payload, file)

    record = {
        "mode": mode,
        "design_seed": design_seed,
        "criterion": pipeline.criterion,
        "raw_fim_per_trajectory": raw_fim,
        "log_fim_per_trajectory": log_fim,
        "total_objective_per_trajectory": pipeline.batch_final_losses,
        "mean_raw_fim": float(np.mean(raw_fim)),
        "mean_log_fim": float(np.mean(log_fim)),
        "mean_total_objective": float(np.mean(pipeline.batch_final_losses)),
        "start_offsets": pipeline.batch_start_offsets,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(metrics_path, record)
    print(
        f"[design done] mode={mode} seed={design_seed} "
        f"mean log-A={record['mean_log_fim']:.5f} ({record['elapsed_s']:.1f}s)",
        flush=True,
    )
    return _jsonable(record)


def tune(args, mode: str, design_seed: int, shuffle_seed: int, design_dir: Path, out_path: Path) -> dict:
    if out_path.exists():
        print(f"[tune cached] mode={mode} design={design_seed} shuffle={shuffle_seed}", flush=True)
        return json.loads(out_path.read_text(encoding="utf-8"))
    print(f"[tune start] mode={mode} design={design_seed} shuffle={shuffle_seed}", flush=True)
    started = time.perf_counter()
    result = run_gain_tuning_experiment(
        problem_path=args.problem,
        robot_params=resolve_gain_robot_params(args.problem, None, None),
        num_steps=args.tuning_steps,
        learning_rate=GAIN_TUNING_DEFAULTS["learning_rate"],
        num_realizations=args.num_realizations,
        seed=shuffle_seed,
        reference_trajectories_dir=str(design_dir),
        validation_split=GAIN_TUNING_DEFAULTS["validation_split"],
        velocity_tracking_weight=GAIN_TUNING_DEFAULTS["velocity_tracking_weight"],
        input_weight=GAIN_TUNING_DEFAULTS["input_weight"],
        input_delta_weight=GAIN_TUNING_DEFAULTS["input_delta_weight"],
        omega_delta_weight=GAIN_TUNING_DEFAULTS["omega_delta_weight"],
        k_min_stab=GAIN_TUNING_DEFAULTS["k_min_stab"],
        k_max_stab=GAIN_TUNING_DEFAULTS["k_max_stab"],
        k_max_rest=GAIN_TUNING_DEFAULTS["k_max_rest"],
        num_lhs_points=args.num_lhs_points,
        num_adam_optimizations=GAIN_TUNING_DEFAULTS["num_adam_optimizations"],
        schedule_enabled=False,
        static_tune=False,
        init_offset_radius=GAIN_TUNING_DEFAULTS["init_offset_radius"],
        init_offset_angle=GAIN_TUNING_DEFAULTS["init_offset_angle"],
    )
    training_terms = _split_terms(result, validation=False)
    validation_terms = _split_terms(result, validation=True)
    record = {
        "mode": mode,
        "design_seed": design_seed,
        "shuffle_seed": shuffle_seed,
        "gains": result["optimized_gains"],
        "training_objective": result["final_loss"],
        "validation_objective": result["final_validation_loss"],
        "training_tracking_mse": training_terms[0],
        "validation_tracking_mse": validation_terms[0],
        "training_tracking_rms": np.sqrt(training_terms[0]),
        "validation_tracking_rms": np.sqrt(validation_terms[0]),
        "training_terms": training_terms,
        "validation_terms": validation_terms,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(out_path, record)
    print(
        f"[tune done] mode={mode} design={design_seed} shuffle={shuffle_seed} "
        f"val={record['validation_objective']:.6f} tracking={record['validation_tracking_mse']:.6f} "
        f"({record['elapsed_s']:.1f}s)",
        flush=True,
    )
    return _jsonable(record)


def add_common_evaluation(args, tunings: list[dict]) -> None:
    """Score every yielded gain vector on one mode-independent test bundle."""
    gain_pipeline = ControllerTuningPipeline(
        args.problem,
        robot_params=resolve_gain_robot_params(args.problem, None, None),
        seed=EVAL_SEED,
    )
    trajectory_pipeline = TrajectoryOptimizationPipeline(
        args.problem,
        objective_mode=OBJECTIVE_MODE_GAIN_TUNING,
        num_realizations=args.num_realizations,
    )
    references = held_out_trajectories(
        trajectory_pipeline,
        num_control_points=args.num_control_points,
        num_trajectories=args.num_trajectories,
        seed=EVAL_SEED,
    )
    realizations = make_realizations(
        gain_pipeline.robot_key,
        gain_pipeline.estimator_key,
        args.num_realizations,
        GAIN_TUNING_DEFAULTS["init_offset_radius"],
        GAIN_TUNING_DEFAULTS["init_offset_angle"],
    )

    @jax.jit
    def evaluate(gains):
        def terms(reference):
            return closed_loop_objective_terms(
                gain_pipeline,
                gains,
                realizations.robot_keys,
                realizations.estimator_keys,
                velocity_tracking_weight=GAIN_TUNING_DEFAULTS["velocity_tracking_weight"],
                input_weight=GAIN_TUNING_DEFAULTS["input_weight"],
                input_delta_weight=GAIN_TUNING_DEFAULTS["input_delta_weight"],
                omega_delta_weight=GAIN_TUNING_DEFAULTS["omega_delta_weight"],
                reference_states=reference,
                initial_pose_offsets=realizations.start_offsets,
            )

        objective_terms = jnp.mean(jax.vmap(terms)(references), axis=0)
        position_rmse = jnp.mean(
            jax.vmap(lambda reference: pose_rmse(gain_pipeline, gains, reference, realizations))(
                references
            )
        )
        return objective_terms, position_rmse

    print("[common evaluation] fixed held-out trajectories/offsets/noise", flush=True)
    for row in tunings:
        terms, position_rmse = evaluate(jnp.asarray(row["gains"], dtype=jnp.float32))
        row["common_objective"] = float(jnp.sum(terms))
        row["common_tracking_mse"] = float(terms[0])
        row["common_position_rmse_m"] = float(position_rmse)


def summarize(config: dict, designs: list[dict], tunings: list[dict]) -> dict:
    summary = {"config": config, "by_mode": {}, "by_design_seed": []}
    for mode in MODES:
        mode_designs = [row for row in designs if row["mode"] == mode]
        mode_tunings = [row for row in tunings if row["mode"] == mode]
        gains = np.asarray([row["gains"] for row in mode_tunings], dtype=float)
        summary["by_mode"][mode] = {
            "num_designs": len(mode_designs),
            "num_tunings": len(mode_tunings),
            "design_raw_fim_mean": np.mean([row["mean_raw_fim"] for row in mode_designs]),
            "design_raw_fim_std": np.std([row["mean_raw_fim"] for row in mode_designs], ddof=1),
            "design_log_fim_mean": np.mean([row["mean_log_fim"] for row in mode_designs]),
            "design_log_fim_std": np.std([row["mean_log_fim"] for row in mode_designs], ddof=1),
            "design_total_mean": np.mean([row["mean_total_objective"] for row in mode_designs]),
            "design_total_std": np.std([row["mean_total_objective"] for row in mode_designs], ddof=1),
            "validation_objective_mean": np.mean([row["validation_objective"] for row in mode_tunings]),
            "validation_objective_std": np.std([row["validation_objective"] for row in mode_tunings], ddof=1),
            "validation_tracking_mse_mean": np.mean([row["validation_tracking_mse"] for row in mode_tunings]),
            "validation_tracking_mse_std": np.std([row["validation_tracking_mse"] for row in mode_tunings], ddof=1),
            "validation_tracking_rms_mean": np.mean([row["validation_tracking_rms"] for row in mode_tunings]),
            "validation_tracking_rms_std": np.std([row["validation_tracking_rms"] for row in mode_tunings], ddof=1),
            "common_objective_mean": np.mean([row["common_objective"] for row in mode_tunings]),
            "common_objective_std": np.std([row["common_objective"] for row in mode_tunings], ddof=1),
            "common_tracking_mse_mean": np.mean([row["common_tracking_mse"] for row in mode_tunings]),
            "common_tracking_mse_std": np.std([row["common_tracking_mse"] for row in mode_tunings], ddof=1),
            "common_position_rmse_m_mean": np.mean([row["common_position_rmse_m"] for row in mode_tunings]),
            "common_position_rmse_m_std": np.std([row["common_position_rmse_m"] for row in mode_tunings], ddof=1),
            "gain_mean": dict(zip(GAIN_NAMES, np.mean(gains, axis=0))),
            "gain_std": dict(zip(GAIN_NAMES, np.std(gains, axis=0, ddof=1))),
            "gain_min": dict(zip(GAIN_NAMES, np.min(gains, axis=0))),
            "gain_max": dict(zip(GAIN_NAMES, np.max(gains, axis=0))),
        }
        for design_seed in config["design_seeds"]:
            design = next(row for row in mode_designs if row["design_seed"] == design_seed)
            rows = [row for row in mode_tunings if row["design_seed"] == design_seed]
            summary["by_design_seed"].append(
                {
                    "mode": mode,
                    "design_seed": design_seed,
                    "design_log_fim": design["mean_log_fim"],
                    "design_raw_fim": design["mean_raw_fim"],
                    "validation_objective_mean": np.mean([row["validation_objective"] for row in rows]),
                    "validation_objective_std": np.std([row["validation_objective"] for row in rows], ddof=1),
                    "validation_tracking_mse_mean": np.mean([row["validation_tracking_mse"] for row in rows]),
                    "validation_tracking_mse_std": np.std([row["validation_tracking_mse"] for row in rows], ddof=1),
                    "common_objective_mean": np.mean([row["common_objective"] for row in rows]),
                    "common_objective_std": np.std([row["common_objective"] for row in rows], ddof=1),
                    "common_position_rmse_m_mean": np.mean([row["common_position_rmse_m"] for row in rows]),
                    "common_position_rmse_m_std": np.std([row["common_position_rmse_m"] for row in rows], ddof=1),
                }
            )
    return _jsonable(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="problems/pololu_gains.yaml")
    parser.add_argument("--output-dir", default="results/offset_efficacy_study")
    parser.add_argument("--design-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--num-control-points", type=int, default=5)
    parser.add_argument("--num-realizations", type=int, default=GAIN_TUNING_DEFAULTS["num_realizations"])
    parser.add_argument("--trajectory-steps", type=int, default=250)
    parser.add_argument("--trajectory-learning-rate", type=float, default=2e-3)
    parser.add_argument("--constraint-weight-jitter", type=float, default=0.3)
    parser.add_argument("--offset-displacement-step-factor", type=float, default=30.0)
    parser.add_argument("--offset-heading-step-factor", type=float, default=60.0)
    parser.add_argument("--tuning-steps", type=int, default=GAIN_TUNING_DEFAULTS["steps"])
    parser.add_argument("--num-lhs-points", type=int, default=GAIN_TUNING_DEFAULTS["num_lhs_points"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config = vars(args).copy()
    config["design_seeds"] = list(args.design_seeds)
    config["shuffle_seeds"] = list(args.shuffle_seeds)
    designs = []
    tunings = []
    for design_seed in args.design_seeds:
        for mode in MODES:
            design_dir = output_dir / "trajectories" / f"{mode}_seed{design_seed}"
            designs.append(synthesize(args, mode, design_seed, design_dir))
            for shuffle_seed in args.shuffle_seeds:
                tunings.append(
                    tune(
                        args,
                        mode,
                        design_seed,
                        shuffle_seed,
                        design_dir,
                        output_dir / "runs" / f"{mode}_design{design_seed}_shuffle{shuffle_seed}.json",
                    )
                )
    add_common_evaluation(args, tunings)
    summary = summarize(config, designs, tunings)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["by_mode"], indent=2), flush=True)


if __name__ == "__main__":
    main()
