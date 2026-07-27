"""Modular stages of the iterative active-learning loop.

Each stage reads its inputs and writes its outputs as files inside the
iteration folder (see experiment.py for the layout), so the loop survives the
interruptions needed to collect robot data from the SD card. Stage order:

    init -> plan-id-trajectory -> [run robot, copy SD logs into data/]
         -> decode-logs -> identify -> train-residual
         -> plan-tuning-trajectories -> tune-gains -> finalize (next iteration)

Heavy imports (jax, the pipelines) happen inside the stage functions so the
CLI stays responsive for bookkeeping commands like status.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from wmr_simulator.active_learning.experiment import (
    ITERATION_PREFIX,
    Experiment,
    IterationPaths,
    collect_plots,
    load_yaml,
    robot_config_from_problem,
    save_yaml,
    write_iteration_problem,
)

REQUIRED_STAGE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("plan-id-trajectory", "identification trajectory (.pkl + .JSN)"),
    ("decode-logs", "decoded logs (.csv) in data/"),
    ("identify", "results/identification.yaml"),
    ("train-residual", "results/residual_model.pkl"),
    ("plan-tuning-trajectories", "tuning_trajectories/*.pkl"),
    ("tune-gains", "results/gains.yaml"),
)


# ---------------------------------------------------------------------------
# init / finalize: iteration scaffolding
# ---------------------------------------------------------------------------


def stage_init(root: str | Path, overrides: dict | None = None) -> Experiment:
    """Create the experiment directory, experiment.yaml, and iteration_01."""
    experiment = Experiment.create(root, overrides)
    problem_cfg = load_yaml(experiment.config["problem"])
    robot_config = robot_config_from_problem(problem_cfg)
    _initialize_iteration(experiment, iteration=1, robot_config=robot_config)
    print(f"Initialized experiment at {experiment.root}")
    print(f"  config: {experiment.config_path}")
    print(f"  first iteration: {experiment.paths(1).root}")
    return experiment


def stage_finalize(experiment: Experiment, iteration: int) -> IterationPaths:
    """Fold this iteration's results into the next iteration's inputs."""
    paths = experiment.paths(iteration)
    if not paths.identification_result.is_file():
        raise FileNotFoundError(f"Missing {paths.identification_result}; run the identify stage first.")
    if not paths.gains_result.is_file():
        raise FileNotFoundError(f"Missing {paths.gains_result}; run the tune-gains stage first.")

    identification = load_yaml(paths.identification_result)
    gains_result = load_yaml(paths.gains_result)
    robot_config = load_yaml(paths.robot_config)
    robot_config["robot"].update(identification["estimated_params"])
    robot_config["controller"]["gains"] = [float(gain) for gain in gains_result["gains"]]
    if gains_result.get("schedule") is not None:
        robot_config["controller"].pop("gain_schedule", None)
        robot_config["controller"]["gain_parametrization"] = {
            **gains_result["schedule"],
            "enabled": bool(gains_result["schedule_enabled"]),
        }

    next_paths = _initialize_iteration(
        experiment,
        iteration + 1,
        robot_config,
        static_gains=gains_result.get("static_gains"),
    )
    print(f"Created {next_paths.root} from iteration {iteration:02d} results.")
    print(f"  firmware config for the robot: {next_paths.robotcfg_cfg}")
    return next_paths


def _initialize_iteration(
    experiment: Experiment,
    iteration: int,
    robot_config: dict,
    static_gains: list[float] | None = None,
) -> IterationPaths:
    from wmr_simulator.pololu.robot_config import export_robot_config
    from wmr_simulator.types import PhysicalParams

    paths = experiment.paths(iteration)
    paths.create_directories()
    physical_params = PhysicalParams(
        wheel_radius=robot_config["robot"]["wheel_radius"],
        base_diameter=robot_config["robot"]["base_diameter"],
        max_wheel_speed=robot_config["robot"]["max_wheel_speed"],
    )
    save_yaml(paths.robot_config, robot_config)
    write_iteration_problem(experiment.config["problem"], robot_config, paths.problem)
    export_robot_config(
        paths.robotcfg_cfg,
        physical_params=physical_params,
        controller_gains=robot_config["controller"]["gains"],
        template_path=experiment.config.get("robotcfg_template"),
    )
    gainmlp_path = _export_gain_mlp_if_configured(paths.problem, paths.gainmlp_jsn)
    if gainmlp_path is not None:
        print(f"  firmware gain-MLP for the robot: {gainmlp_path}")
    if static_gains is not None:
        _write_static_gain_config(experiment, paths, robot_config, physical_params, static_gains)
        print(f"  static-gain baseline for the robot: {paths.robotcfg_static_cfg}")
    return paths


def _write_static_gain_config(
    experiment: Experiment,
    paths: IterationPaths,
    robot_config: dict,
    physical_params,
    static_gains: list[float],
) -> None:
    """Write the gain-MLP-free baseline variant of this iteration's robot config.

    Same identified robot parameters, but the gains of the static pretune stage
    and no gain parametrization, so the two can be benchmarked against each
    other on the robot (copy ROBOTCFG_static.CFG as ROBOTCFG.CFG without a
    GAINMLP.JSN next to it).
    """
    from wmr_simulator.pololu.robot_config import export_robot_config

    static_config = copy.deepcopy(robot_config)
    static_config["controller"]["gains"] = [float(gain) for gain in static_gains]
    static_config["controller"].pop("gain_parametrization", None)
    static_config["controller"].pop("gain_schedule", None)
    save_yaml(paths.robot_config_static, static_config)
    export_robot_config(
        paths.robotcfg_static_cfg,
        physical_params=physical_params,
        controller_gains=static_config["controller"]["gains"],
        template_path=experiment.config.get("robotcfg_template"),
    )


def _export_gain_mlp_if_configured(problem_path: Path, output_path: Path) -> Path | None:
    """Export the error-MLP gain parametrization to GAINMLP.JSN for the robot.

    No-op unless the generated iteration problem enables an ``error_mlp`` gain
    parametrization; the factors are multiplicative so the file sits next to
    ROBOTCFG.CFG on the SD card and scales the firmware base gains it holds.
    """
    from wmr_simulator.gain_parametrization import error_mlp, params_from_cfg, parametrization_kind

    problem_cfg = load_yaml(problem_path)
    cfg = problem_cfg["controller"].get("gain_parametrization")
    if cfg is None or not cfg.get("enabled", False) or parametrization_kind(cfg) != error_mlp.KIND:
        return None

    from wmr_simulator.pololu.gain_mlp_exporter import export_gain_mlp

    robot_cfg = problem_cfg["robot"]
    feature_scale = [robot_cfg["v_max"], robot_cfg["omega_max"]]
    params = params_from_cfg(cfg, feature_scale)
    return export_gain_mlp(output_path, params)


# ---------------------------------------------------------------------------
# trajectory planning stages
# ---------------------------------------------------------------------------


def stage_plan_identification_trajectory(experiment: Experiment, iteration: int) -> Path:
    """Synthesize (or copy) the informative identification trajectory and
    export it as a Pololu reference JSN plus a bridged repeat variant
    (see pololu.bridge_exporter)."""
    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["identification_trajectory"]
    identification_plot_dir = paths.visualize_dir / "identification"
    identification_plot_dir.mkdir(parents=True, exist_ok=True)

    if not experiment.config["optimize_trajectories"]:
        baseline = experiment.config.get("baseline_identification_trajectory")
        if not baseline:
            raise ValueError(
                "optimize_trajectories is disabled but baseline_identification_trajectory "
                "is not set in experiment.yaml."
            )
        pickle_path = Path(shutil.copy2(baseline, paths.identification_trajectory_dir))
        print(f"Copied baseline identification trajectory: {pickle_path}")
    else:
        from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

        pipeline = TrajectoryOptimizationPipeline(
            str(paths.problem),
            time_scaling=config["time_scaling"],
            objective_mode="identification",
            fim_a_slip_max=bool(config["fim_a_slip_max"]),
        )
        pipeline.set_bezier_control_points(pipeline.initial_bezier_control_points(config["bezier_order"]))
        with collect_plots(identification_plot_dir):
            _, loss_history = pipeline.optimize_bezier_trajectory(
                order=config["bezier_order"],
                num_steps=config["opt_steps"],
                learning_rate=config["learning_rate"],
                window_length=config["window_length"],
            )
            pipeline.plot_trajectory(
                window_length=config["window_length"],
                out_prefix="identification_trajectory",
            )
            pipeline.plot_loss_history(out_prefix="identification_trajectory")
        print(f"Final identification-trajectory loss: {float(loss_history[-1]):.6e}")
        pickle_path = Path(
            pipeline.save_reference_states_pickle(
                out_dir=str(paths.identification_trajectory_dir),
                filename_prefix="identification_trajectory",
            )
        )
        print(f"Saved identification trajectory pickle: {pickle_path}")

    from wmr_simulator.pololu.reference_exporter import export_reference_trajectory, load_reference_trajectory

    jsn_path = export_reference_trajectory(
        load_reference_trajectory(pickle_path),
        paths.identification_trajectory_dir,
    )
    print(f"Exported Pololu reference JSN: {jsn_path}")

    from wmr_simulator.pololu.bridge_exporter import append_bridge_reference

    bridged_path = append_bridge_reference(
        jsn_path,
        wait_time=float(config["bridge_wait_time"]),
        bridge_time=float(config["bridge_time"]),
        plot_path=identification_plot_dir / "identification_trajectory_bridge.pdf",
    )
    print(f"Exported bridged repeat variant: {bridged_path}")
    print(f"Copy {jsn_path.name} and {paths.robotcfg_cfg.name} to the robot SD card, run the")
    print(f"experiment, then place the logs in {paths.data_dir} and run decode-logs.")
    return jsn_path


def stage_plan_tuning_trajectories(experiment: Experiment, iteration: int) -> list[Path]:
    """Synthesize (or copy) the reference trajectory set used for gain tuning."""
    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["tuning_trajectories"]
    problem_path = _identified_problem(paths)

    if not experiment.config["optimize_trajectories"]:
        baseline_dir = experiment.config.get("baseline_tuning_trajectories_dir")
        if not baseline_dir:
            raise ValueError(
                "optimize_trajectories is disabled but baseline_tuning_trajectories_dir "
                "is not set in experiment.yaml."
            )
        pickles = sorted(Path(baseline_dir).glob("*.pkl"))
        if not pickles:
            raise ValueError(f"No reference pickles found in {baseline_dir}")
        copied = [Path(shutil.copy2(path, paths.tuning_trajectories_dir)) for path in pickles]
        print(f"Copied {len(copied)} baseline tuning trajectories into {paths.tuning_trajectories_dir}")
        return copied

    import numpy as np

    from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

    pipeline = TrajectoryOptimizationPipeline(
        str(problem_path),
        time_scaling=config["time_scaling"],
        objective_mode="gain-tuning",
    )
    pipeline.set_bezier_control_points(pipeline.initial_bezier_control_points(config["bezier_order"]))
    num_trajectories = int(config["num_trajectories"])
    if num_trajectories == 1:
        control_point_batch = None
        optimized_control_points, _ = pipeline.optimize_bezier_trajectory(
            order=config["bezier_order"],
            num_steps=config["opt_steps"],
            learning_rate=config["learning_rate"],
            window_length=config["window_length"],
        )
        control_point_batch = [optimized_control_points]
    else:
        control_point_batch, _ = pipeline.optimize_bezier_trajectories(
            order=config["bezier_order"],
            num_steps=config["opt_steps"],
            learning_rate=config["learning_rate"],
            num_trajectories=num_trajectories,
            vectorized=True,
            constraint_weight_jitter=config["constraint_weight_jitter"],
            seed=int(experiment.config["seed"]),
            window_length=config["window_length"],
            verbose=False,
        )
        final_losses = np.asarray(pipeline.batch_final_losses, dtype=float)
        print(f"Optimized {len(control_point_batch)} tuning trajectories "
              f"(final losses {final_losses.min():.4e} .. {final_losses.max():.4e})")

    saved = []
    for index, control_points in enumerate(control_point_batch):
        pipeline.set_bezier_control_points(control_points)
        saved.append(
            Path(
                pipeline.save_reference_states_pickle(
                    out_dir=str(paths.tuning_trajectories_dir),
                    filename_prefix=f"tuning_trajectory_{index:02d}",
                )
            )
        )
    print(f"Saved {len(saved)} tuning trajectory pickles to {paths.tuning_trajectories_dir}")
    return saved


# ---------------------------------------------------------------------------
# data handling
# ---------------------------------------------------------------------------


def stage_decode_logs(experiment: Experiment, iteration: int) -> list[Path]:
    """Decode binary SD-card logs in data/ into csv files (skips existing) and
    render the log-loader summary plot of every decoded log into
    visualize/logs/ (skips logs that already have one)."""
    from wmr_simulator.pololu.decode_binary import decode_file

    paths = experiment.paths(iteration)
    decoded = []
    candidates = [
        path
        for path in sorted(paths.data_dir.iterdir())
        if path.is_file() and path.suffix.lower() != ".csv" and path.suffix.upper() != ".CFG"
    ] if paths.data_dir.is_dir() else []
    for path in candidates:
        output_path = path.with_name(path.name + ".csv")
        if output_path.exists():
            print(f"Already decoded, skipping: {path.name}")
            continue
        if decode_file(str(path), str(output_path)):
            decoded.append(output_path)
    if not candidates:
        print(f"No binary logs found in {paths.data_dir}")
    _plot_log_summaries(experiment, paths)
    return decoded


def _plot_log_summaries(experiment: Experiment, paths: IterationPaths) -> None:
    from wmr_simulator.pololu.log_loader import load_imu_gyro_z, load_pololu_traj_control_log
    from wmr_simulator.visualization.pololu import plot_logged_summary

    clip = experiment.config["log_loading"]["clip_after_first_trajectory"]
    plot_dir = paths.visualize_dir / "logs"
    base_gains, gain_params = _log_gain_parametrization(paths)
    for log_path in _list_log_csvs(paths):
        if (plot_dir / f"{log_path.stem}.pdf").exists():
            continue
        try:
            log = load_pololu_traj_control_log(log_path, clip_after_first_trajectory=clip)
            imu_time, imu_gyro_z = load_imu_gyro_z(log_path, clip_after_first_trajectory=clip)
        except ValueError as error:
            print(f"Log summary plot skipped for {log_path.name}: {error}")
            continue
        # Recover the applied (scheduled) gains offline; the firmware does not log them.
        gains = None
        if gain_params is not None:
            from wmr_simulator.pololu.gain_reconstruction import applied_gains_over_log

            gains = applied_gains_over_log(log, base_gains, gain_params)
        plot_path = plot_logged_summary(
            log,
            out_prefix=log_path.stem,
            out_dir=plot_dir,
            imu_time_s=imu_time,
            imu_gyro_z=imu_gyro_z,
            gains=gains,
        )
        print(f"Log summary plot: {plot_path}")


def _log_gain_parametrization(paths: IterationPaths):
    """Base gains + enabled gain parametrization used to record this iteration's
    logs (``None`` params when no parametrization is enabled)."""
    problem_cfg = load_yaml(paths.problem)
    controller = problem_cfg["controller"]
    base_gains = [float(gain) for gain in controller["gains"]]
    cfg = controller.get("gain_parametrization", controller.get("gain_schedule"))
    if cfg is None or not cfg.get("enabled", False):
        return base_gains, None

    from wmr_simulator.gain_parametrization import params_from_cfg

    robot = problem_cfg["robot"]
    params = params_from_cfg(cfg, [robot["v_max"], robot["omega_max"]])
    return base_gains, params


def _list_log_csvs(paths: IterationPaths) -> list[Path]:
    from wmr_simulator.pololu.log_loader import list_pololu_log_paths

    if not paths.data_dir.is_dir():
        return []
    try:
        return [Path(path) for path in list_pololu_log_paths(paths.data_dir)]
    except ValueError:
        return []


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------


def stage_identify(
    experiment: Experiment,
    iteration: int,
    log: str | None = None,
) -> dict:
    """Identify the robot parameters from every identification log of this iteration.

    All decoded logs sitting *directly* in data/ are identification data (record
    several, diverse trajectories per iteration); subdirectories are reserved for
    baseline comparison runs and stay out of the fit. Identification is cheap, so
    the stage first runs it per log, flags logs whose parameters disagree with the
    median of the batch (identification.outlier_z_threshold, 0 disables), and only
    then fits one parameter set jointly to the remaining logs. A parameter the
    recorded motion does not excite is better held fixed than fitted, see
    identification.identify_a_slip_max.

    Writes results/identification.yaml and problem_identified.yaml (the base
    problem with the updated robot block) used by all downstream stages.
    """
    import jax.numpy as jnp
    import numpy as np

    from wmr_simulator.identification.outliers import robust_parameter_outliers
    from wmr_simulator.identification.pipeline import run_multi_log_identification
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.types import PhysicalParams, physical_params_to_array, print_physical_params
    from wmr_simulator.visualization.identification import (
        plot_identification_log_parameters,
        plot_loss_history,
        plot_system_id,
    )

    paths = experiment.paths(iteration)
    config = experiment.config["identification"]
    log_config = experiment.config["log_loading"]
    log_paths = _resolve_log_paths(paths, log)
    print(f"Identification logs ({len(log_paths)}): {', '.join(path.name for path in log_paths)}")

    robot_config = load_yaml(paths.robot_config)["robot"]
    identify_a_slip_max = bool(config["identify_a_slip_max"])
    a_slip_max = float(robot_config.get("a_slip_max", 0.0))
    if identify_a_slip_max and a_slip_max == 0.0:
        # A zero value keeps the burnout model disabled in the log-space optimizer;
        # fall back to the experiment config init to (re-)enable identification.
        # With identification off there is nothing to re-enable: a disabled limit
        # stays disabled instead of being silently switched on at the init value.
        a_slip_max = float(config["init_a_slip_max"])
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(robot_config["wheel_radius"]),
        base_diameter=jnp.asarray(robot_config["base_diameter"]),
        max_wheel_speed=jnp.asarray(robot_config["max_wheel_speed"]),
        time_constant=jnp.asarray(robot_config["time_constant"]),
        a_slip_max=jnp.asarray(a_slip_max),
    )
    if not identify_a_slip_max:
        print(f"Holding a_slip_max at {a_slip_max:.3f} m/s^2 (identification.identify_a_slip_max is off).")

    def identify(target_logs):
        return run_multi_log_identification(
            problem_path=str(paths.problem),
            initial_params=init_params,
            target_logs=target_logs,
            num_steps=int(config["steps"]),
            learning_rate=float(config["learning_rate"]),
            seed=int(experiment.config["seed"]),
            window_length=config["window_length"],
            identify_a_slip_max=identify_a_slip_max,
        )

    pololu_logs = [
        load_pololu_traj_control_log(
            log_path,
            clip_after_first_trajectory=log_config["clip_after_first_trajectory"],
        )
        for log_path in log_paths
    ]

    per_log_results = []
    for log_path, pololu_log in zip(log_paths, pololu_logs):
        print(f"Identifying on {log_path.name} ...")
        per_log_results.append(identify([pololu_log]))
    parameter_samples = np.asarray(
        [physical_params_to_array(result["estimated_params"]) for result in per_log_results], dtype=float
    )

    report = robust_parameter_outliers(parameter_samples, z_threshold=float(config["outlier_z_threshold"]))
    kept_indices = [index for index in range(len(log_paths)) if not report.is_outlier[index]]
    if not kept_indices:
        raise RuntimeError("Outlier detection excluded every log; lower identification.outlier_z_threshold.")
    _print_per_log_parameters(log_paths, parameter_samples, per_log_results, report)

    if len(kept_indices) == len(log_paths) and len(log_paths) == 1:
        # A joint fit on a single log is exactly the per-log fit; don't redo it.
        joint_result = per_log_results[0]
    else:
        print(f"Joint identification on {len(kept_indices)} log(s) ...")
        joint_result = identify([pololu_logs[index] for index in kept_indices])

    estimated_params = joint_result["estimated_params"]
    print_physical_params("Estimated parameters (joint):", estimated_params)
    print(f"Final normalized geometry loss: {float(joint_result['loss_history'][-1]):.8f}")
    print(f"Final normalized motor loss:    {float(joint_result['motor_loss_history'][-1]):.8f}")

    payload = {
        "logs": [str(log_paths[index]) for index in kept_indices],
        "excluded_logs": [str(log_paths[index]) for index in range(len(log_paths)) if report.is_outlier[index]],
        "estimated_params": _estimated_params_dict(estimated_params),
        "final_loss": float(joint_result["loss_history"][-1]),
        "final_motor_loss": float(joint_result["motor_loss_history"][-1]),
        "outlier_z_threshold": float(config["outlier_z_threshold"]),
        "outlier_detection_evaluated": bool(report.evaluated),
        "identify_a_slip_max": identify_a_slip_max,
        "per_log": [
            {
                "log": str(log_paths[index]),
                "estimated_params": _estimated_params_dict(result["estimated_params"]),
                "final_loss": float(result["loss_history"][-1]),
                "final_motor_loss": float(result["motor_loss_history"][-1]),
                "max_robust_z": float(report.max_z_scores[index]),
                "excluded": bool(report.is_outlier[index]),
            }
            for index, result in enumerate(per_log_results)
        ],
    }
    save_yaml(paths.identification_result, payload)

    identified_robot_config = load_yaml(paths.robot_config)
    identified_robot_config["robot"].update(payload["estimated_params"])
    write_iteration_problem(experiment.config["problem"], identified_robot_config, paths.problem_identified)
    print(f"Wrote {paths.identification_result} and {paths.problem_identified}")

    with collect_plots(paths.visualize_dir / "identification"):
        # Kept logs are shown under the joint parameters (what downstream stages
        # use); excluded ones under their own fit, which is what got them flagged.
        for position, index in enumerate(kept_indices):
            pipeline = joint_result["pipelines"][position]
            plot_system_id(
                pipeline=pipeline,
                init_target_log=pipeline.target_log,
                init_log=joint_result["init_replay_logs"][position],
                predicted_log=joint_result["final_replay_logs"][position],
                out_prefix=f"identification_{log_paths[index].stem}",
            )
        for index, result in enumerate(per_log_results):
            if not report.is_outlier[index]:
                continue
            plot_system_id(
                pipeline=result["pipelines"][0],
                init_target_log=result["pipelines"][0].target_log,
                init_log=result["init_replay_logs"][0],
                predicted_log=result["final_replay_logs"][0],
                out_prefix=f"identification_{log_paths[index].stem}_excluded",
            )
        plot_loss_history(
            loss_history=joint_result["loss_history"],
            motor_loss_history=joint_result["motor_loss_history"],
            out_prefix="system_id",
        )
        plot_identification_log_parameters(
            log_names=[path.stem for path in log_paths],
            parameter_samples=parameter_samples,
            joint_params=np.asarray(physical_params_to_array(estimated_params), dtype=float),
            max_z_scores=report.max_z_scores if report.evaluated else None,
            z_scores=report.z_scores if report.evaluated else None,
            is_outlier=report.is_outlier,
            z_threshold=float(config["outlier_z_threshold"]),
        )
    return payload


def _estimated_params_dict(params) -> dict:
    return {
        "wheel_radius": float(params.wheel_radius),
        "base_diameter": float(params.base_diameter),
        "max_wheel_speed": float(params.max_wheel_speed),
        "time_constant": float(params.time_constant),
        "a_slip_max": float(params.a_slip_max),
    }


def _print_per_log_parameters(log_paths, parameter_samples, per_log_results, report) -> None:
    header = (
        f"{'log':<12}{'r [mm]':>10}{'L [mm]':>10}{'u_max':>10}"
        f"{'tau [s]':>10}{'a_slip':>10}{'loss':>12}{'max z':>8}"
    )
    print("Per-log identification:")
    print(header)
    for index, log_path in enumerate(log_paths):
        values = parameter_samples[index]
        total_loss = float(per_log_results[index]["loss_history"][-1]) + float(
            per_log_results[index]["motor_loss_history"][-1]
        )
        marker = "  EXCLUDED" if report.is_outlier[index] else ""
        print(
            f"{log_path.stem:<12}{1000.0 * values[0]:>10.2f}{1000.0 * values[1]:>10.2f}{values[2]:>10.2f}"
            f"{values[3]:>10.4f}{values[4]:>10.3f}{total_loss:>12.6f}"
            f"{report.max_z_scores[index]:>8.2f}{marker}"
        )
    if not report.evaluated:
        print("  (outlier detection off: disabled by threshold, too few logs, or no parameter spread)")


def _resolve_log_paths(paths: IterationPaths, log: str | None) -> list[Path]:
    """Identification logs of an iteration: every decoded log directly in data/,
    or just the one named by ``--log``.

    Baseline comparison runs live in subdirectories of data/ and are not
    identification data, so discovery stays one level deep (list_pololu_log_paths
    does not recurse)."""
    if log is not None:
        log_path = Path(log)
        if not log_path.is_file():
            log_path = paths.data_dir / log
        if not log_path.is_file():
            raise FileNotFoundError(f"Log not found: {log}")
        return [log_path]
    csvs = _list_log_csvs(paths)
    if not csvs:
        raise FileNotFoundError(
            f"No decoded Pololu logs in {paths.data_dir}. Copy the SD-card logs there and run decode-logs."
        )
    return csvs


# ---------------------------------------------------------------------------
# residual model
# ---------------------------------------------------------------------------


def residual_training_iterations(experiment: Experiment, iteration: int) -> list[int]:
    """Iteration indices whose decoded logs feed the residual model at ``iteration``.

    With ``residual.pool_previous_iterations`` the model trains on every decoded
    log of this iteration *and all earlier ones*, so each iteration sees strictly
    more data than the last (multiple trajectories per iteration accumulate).
    Later iterations are never included, so rerunning an old one reproduces what
    it originally saw.

    Pooling is sound because one-step residual training is gain-independent: the
    descriptor is built from each log's own recorded duty, so it does not matter
    that earlier iterations ran different tuned gains. Each log's *nominal* model
    still comes from its own iteration's params (residual.robot_params_for_log),
    which is what keeps the pool consistent.
    """
    if not experiment.config["residual"].get("pool_previous_iterations", True):
        candidates = [iteration]
    else:
        candidates = [index for index in experiment.iteration_indices() if index <= iteration]
    return [index for index in candidates if _list_log_csvs(experiment.paths(index))]


def stage_train_residual(experiment: Experiment, iteration: int) -> Path:
    """Train the residual dynamics model on the pooled decoded logs (see
    residual_training_log_dirs)."""
    from wmr_simulator.residual_model.residual import train_from_logs

    paths = experiment.paths(iteration)
    config = experiment.config["residual"]
    log_config = experiment.config["log_loading"]
    problem_path = _identified_problem(paths)
    pooled_iterations = residual_training_iterations(experiment, iteration)
    if not pooled_iterations:
        raise FileNotFoundError(
            f"No decoded Pololu logs in {paths.data_dir} or any earlier iteration."
        )
    log_dirs = [experiment.paths(index).data_dir for index in pooled_iterations]
    num_logs = sum(len(_list_log_csvs(experiment.paths(index))) for index in pooled_iterations)
    print(
        f"Residual training pool: {num_logs} logs from {len(pooled_iterations)} iteration(s) "
        f"({', '.join(f'{ITERATION_PREFIX}{index:02d}' for index in pooled_iterations)})"
    )

    train_from_logs(
        problem=str(problem_path),
        log_dirs=[str(directory) for directory in log_dirs],
        out=str(paths.residual_model),
        num_experts=int(config["num_experts"]),
        hidden_sizes=tuple(int(size) for size in config["hidden_sizes"]),
        spectral_norm_cap=float(config["spectral_norm_cap"]),
        gate_bandwidth_scale=float(config["gate_bandwidth_scale"]),
        ood_sigma=float(config["ood_sigma"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        validation_split=float(config["validation_split"]),
        seed=int(experiment.config["seed"]),
        output_reg_weight=float(config["output_reg_weight"]),
        clip_after_first_trajectory=log_config["clip_after_first_trajectory"],
        resample_uniform=bool(config["resample_uniform"]),
        # log_dirs are the data/ directories themselves; no subdirectories to walk.
        recursive=False,
        out_dir=str(paths.visualize_dir / "residual model"),
    )
    return paths.residual_model


# ---------------------------------------------------------------------------
# gain tuning
# ---------------------------------------------------------------------------


def stage_tune_gains(experiment: Experiment, iteration: int) -> dict:
    """Tune controller gains in simulation on the identified model (optionally
    with the residual model) over the tuning trajectory set."""
    from wmr_simulator.gain_tuning.pipeline import (
        resolve_gain_robot_params,
        run_gain_tuning_experiment,
    )
    from wmr_simulator.types import print_controller_gains, print_physical_params
    from wmr_simulator.visualization.gain_tuning import (
        plot_controller_tuning_errors,
        plot_gain_tuning_summary,
        plot_training_trajectory_summary,
        plot_validation_trajectory_summary,
    )
    from wmr_simulator.visualization.identification import plot_loss_history

    paths = experiment.paths(iteration)
    if experiment.config["use_standalone_gain_tuning_defaults"]:
        from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS

        config = GAIN_TUNING_DEFAULTS
        print("Gain tuning: using standalone run_gain_tuning.py defaults (GAIN_TUNING_DEFAULTS).")
    else:
        config = experiment.config["gain_tuning"]
    # The cross-iteration refinement policy (presearch band + MLP warm-start) is
    # active-learning intent and always comes from the experiment's own
    # gain_tuning block, independent of the standalone hyperparameter swap above.
    refine = experiment.config["gain_tuning"]
    problem_path = _identified_problem(paths)
    if not any(paths.tuning_trajectories_dir.glob("*.pkl")):
        raise FileNotFoundError(
            f"No tuning trajectories in {paths.tuning_trajectories_dir}; run plan-tuning-trajectories first."
        )

    residual_model = None
    if experiment.config["use_residual_model"]:
        if not paths.residual_model.is_file():
            raise FileNotFoundError(
                f"use_residual_model is enabled but {paths.residual_model} is missing; "
                "run train-residual first (or disable the flag in experiment.yaml)."
            )
        from wmr_simulator.residual_model import load_residual_model

        residual_model, checkpoint = load_residual_model(paths.residual_model)
        print(f"Loaded residual model {paths.residual_model} (config: {checkpoint['config']})")

    robot_params = resolve_gain_robot_params(str(problem_path), None, None)
    print_physical_params("Robot parameters for gain tuning:", robot_params)
    result = run_gain_tuning_experiment(
        problem_path=str(problem_path),
        robot_params=robot_params,
        num_steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        num_realizations=int(config["num_realizations"]),
        seed=int(experiment.config["seed"]),
        reference_trajectories_dir=str(paths.tuning_trajectories_dir),
        validation_split=float(config["validation_split"]),
        velocity_tracking_weight=float(config["velocity_tracking_weight"]),
        input_weight=float(config["input_weight"]),
        input_delta_weight=float(config["input_delta_weight"]),
        omega_delta_weight=float(config.get("omega_delta_weight", 0.0)),
        k_min_stab=float(config["k_min_stab"]),
        k_max_stab=float(config["k_max_stab"]),
        k_max_rest=float(config["k_max_rest"]),
        num_lhs_points=int(config["num_lhs_points"]),
        num_adam_optimizations=int(config["num_adam_optimizations"]),
        schedule_enabled=config.get("gain_parametrization", config.get("gain_schedule")),
        gain_delta_weight=float(config["gain_delta_weight"]),
        static_pretune=bool(config["static_pretune"]),
        static_pretune_steps=int(config["static_pretune_steps"]),
        static_pretune_learning_rate=float(config["static_pretune_learning_rate"]),
        # Iteration 1 has no prior result to refine from: search the full
        # presearch range instead of a band around the base gains.
        presearch_relative_range=0.0 if iteration <= 1 else float(refine["presearch_relative_range"]),
        warm_start_schedule=bool(refine["warm_start_schedule"]),
        residual_model=residual_model,
    )
    pipeline = result["pipeline"]
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    if result["static_gains"] is not None:
        print_controller_gains("Static-pretune gains (benchmark baseline):", result["static_gains"])
    print(f"Final tuning loss: {float(result['loss_history'][-1]):.8f}")

    payload = {
        "gains": [float(gain) for gain in result["optimized_gains"]],
        # Gains of the static pretune stage (no parametrization); finalize
        # exports them as the iteration's ROBOTCFG_static.CFG baseline. None
        # when there was no static stage (parametrization or pretune disabled).
        "static_gains": (
            None
            if result["static_gains"] is None
            else [float(gain) for gain in result["static_gains"]]
        ),
        "schedule_enabled": bool(result["schedule_enabled"]),
        "schedule": None,
        "used_residual_model": residual_model is not None,
        "final_loss": float(result["loss_history"][-1]),
        "final_validation_loss": (
            float(result["validation_loss_history"][-1])
            if result["validation_loss_history"] is not None
            else None
        ),
    }
    schedule_params = result.get("schedule_params")
    if schedule_params is not None:
        from wmr_simulator.gain_parametrization import to_cfg as gain_parametrization_to_cfg

        payload["schedule"] = gain_parametrization_to_cfg(schedule_params)
    save_yaml(paths.gains_result, payload)
    print(f"Wrote {paths.gains_result}")

    with collect_plots(paths.visualize_dir / "gain tuning"):
        plot_gain_tuning_summary(
            pipeline,
            init_log=result["init_hidden_log"],
            tuned_log=result["final_hidden_log"],
            static_log=result.get("static_hidden_log"),
            out_prefix="summary_gain_tuning",
        )
        plot_training_trajectory_summary(
            pipeline,
            robot_params=robot_params,
            tuned_gains=result["optimized_gains"],
            schedule_params=result["schedule_params"],
            static_gains=result["static_gains"],
            max_trajectories=None,
            out_prefix="summary_training",
        )
        plot_validation_trajectory_summary(
            pipeline,
            robot_params=robot_params,
            tuned_gains=result["optimized_gains"],
            schedule_params=result["schedule_params"],
            static_gains=result["static_gains"],
            out_prefix="summary_validation",
        )
        plot_controller_tuning_errors(
            pipeline=pipeline,
            init_log=result["init_hidden_log"],
            tuned_log=result["final_hidden_log"],
            static_log=result.get("static_hidden_log"),
        )
        plot_loss_history(
            loss_history=result["loss_history"],
            validation_loss_history=result["validation_loss_history"],
            loss_component_history=result["loss_component_history"],
            validation_loss_component_history=result["validation_loss_component_history"],
            out_prefix="ctrl_tuning",
        )
    return payload


# ---------------------------------------------------------------------------
# status / orchestration
# ---------------------------------------------------------------------------


def iteration_status(experiment: Experiment, iteration: int) -> dict[str, bool]:
    paths = experiment.paths(iteration)
    has_id_trajectory = any(paths.identification_trajectory_dir.glob("*.pkl")) and any(
        paths.identification_trajectory_dir.glob("*.JSN")
    )
    return {
        "plan-id-trajectory": has_id_trajectory,
        "decode-logs": bool(_list_log_csvs(paths)),
        "identify": paths.identification_result.is_file(),
        "train-residual": paths.residual_model.is_file() or not experiment.config["use_residual_model"],
        "plan-tuning-trajectories": any(paths.tuning_trajectories_dir.glob("*.pkl")),
        "tune-gains": paths.gains_result.is_file(),
    }


def stage_status(experiment: Experiment) -> None:
    print(f"Experiment: {experiment.root}")
    print(f"  residual model:          {'enabled' if experiment.config['use_residual_model'] else 'disabled'}")
    print(f"  trajectory optimization: {'enabled' if experiment.config['optimize_trajectories'] else 'disabled (baselines)'}")
    for iteration in experiment.iteration_indices():
        print(f"iteration_{iteration:02d}:")
        status = iteration_status(experiment, iteration)
        for stage, description in REQUIRED_STAGE_OUTPUTS:
            marker = "x" if status[stage] else " "
            print(f"  [{marker}] {stage:<26} {description}")


def stage_run(
    experiment: Experiment,
    iteration: int,
    log: str | None = None,
) -> None:
    """Run every stage of the iteration that can proceed, in order.

    Stops with instructions when robot data is required, and finalizes the
    iteration (creating the next one) once gains are tuned.
    """
    paths = experiment.paths(iteration)
    status = iteration_status(experiment, iteration)

    if not status["plan-id-trajectory"]:
        stage_plan_identification_trajectory(experiment, iteration)
        status = iteration_status(experiment, iteration)

    if not status["decode-logs"]:
        stage_decode_logs(experiment, iteration)
        status = iteration_status(experiment, iteration)
    if not status["decode-logs"]:
        print()
        print("Waiting for robot data:")
        print(f"  1. Copy {paths.robotcfg_cfg} and the reference JSN from")
        print(f"     {paths.identification_trajectory_dir} to the robot SD card.")
        print("  2. Run the experiment(s) on the robot.")
        print(f"  3. Copy the SD-card logs into {paths.data_dir}.")
        print("  4. Re-run this command.")
        return

    if not status["identify"]:
        stage_identify(experiment, iteration, log=log)
    if experiment.config["use_residual_model"] and not paths.residual_model.is_file():
        stage_train_residual(experiment, iteration)
    if not status["plan-tuning-trajectories"]:
        stage_plan_tuning_trajectories(experiment, iteration)
    if not iteration_status(experiment, iteration)["tune-gains"]:
        stage_tune_gains(experiment, iteration)
    stage_finalize(experiment, iteration)


def _identified_problem(paths: IterationPaths) -> Path:
    if not paths.problem_identified.is_file():
        raise FileNotFoundError(
            f"Missing {paths.problem_identified}; run the identify stage first "
            "(downstream stages must use the identified robot model)."
        )
    return paths.problem_identified
