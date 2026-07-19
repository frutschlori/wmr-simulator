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

import shutil
from pathlib import Path

from wmr_simulator.active_learning.experiment import (
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
    if "mocap_delay" in identification:
        robot_config.setdefault("estimator", {})["mocap_delay"] = float(identification["mocap_delay"])
    robot_config["controller"]["gains"] = [float(gain) for gain in gains_result["gains"]]
    if gains_result.get("schedule") is not None:
        robot_config["controller"].pop("gain_schedule", None)
        robot_config["controller"]["gain_parametrization"] = {
            **gains_result["schedule"],
            "enabled": bool(gains_result["schedule_enabled"]),
        }

    next_paths = _initialize_iteration(experiment, iteration + 1, robot_config)
    print(f"Created {next_paths.root} from iteration {iteration:02d} results.")
    print(f"  firmware config for the robot: {next_paths.robotcfg_cfg}")
    return next_paths


def _initialize_iteration(experiment: Experiment, iteration: int, robot_config: dict) -> IterationPaths:
    from wmr_simulator.pololu.robot_config import export_robot_config
    from wmr_simulator.types import PhysicalParams

    paths = experiment.paths(iteration)
    paths.create_directories()
    save_yaml(paths.robot_config, robot_config)
    write_iteration_problem(experiment.config["problem"], robot_config, paths.problem)
    export_robot_config(
        paths.robotcfg_cfg,
        physical_params=PhysicalParams(
            wheel_radius=robot_config["robot"]["wheel_radius"],
            base_diameter=robot_config["robot"]["base_diameter"],
            max_wheel_speed=robot_config["robot"]["max_wheel_speed"],
        ),
        controller_gains=robot_config["controller"]["gains"],
        template_path=experiment.config.get("robotcfg_template"),
    )
    return paths


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
        with collect_plots(paths.visualize_dir):
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
        plot_path=paths.visualize_dir / "identification_trajectory_bridge.pdf",
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

    plot_dir = paths.visualize_dir / "tuning_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
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
        pipeline.plot_trajectory(
            window_length=config["window_length"],
            out_path=str(plot_dir / f"trajectory_{index:02d}.pdf"),
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
    for log_path in _list_log_csvs(paths):
        if (plot_dir / f"{log_path.stem}.pdf").exists():
            continue
        try:
            log = load_pololu_traj_control_log(log_path, clip_after_first_trajectory=clip)
            imu_time, imu_gyro_z = load_imu_gyro_z(log_path, clip_after_first_trajectory=clip)
        except ValueError as error:
            print(f"Log summary plot skipped for {log_path.name}: {error}")
            continue
        plot_path = plot_logged_summary(
            log,
            out_prefix=log_path.stem,
            out_dir=plot_dir,
            imu_time_s=imu_time,
            imu_gyro_z=imu_gyro_z,
        )
        print(f"Log summary plot: {plot_path}")


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
    estimate_mocap_delay: bool | None = None,
) -> dict:
    """Run parameter identification on the recorded identification log.

    Optionally estimates the mocap transport delay first (mocap yaw rate vs
    IMU gyro z cross-correlation) and shifts the mocap timestamps accordingly.
    Writes results/identification.yaml and problem_identified.yaml (the base
    problem with the updated robot block and estimator mocap_delay) used by
    all downstream stages.
    """
    import jax.numpy as jnp

    from wmr_simulator.identification.mocap_delay import (
        estimate_mocap_delay_from_log_file,
        print_delay_result,
    )
    from wmr_simulator.identification.pipeline import run_single_experiment_identification
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.types import PhysicalParams, print_physical_params
    from wmr_simulator.visualization.identification import plot_loss_history, plot_system_id

    paths = experiment.paths(iteration)
    config = experiment.config["identification"]
    log_config = experiment.config["log_loading"]
    log_path = _resolve_log_path(paths, log)
    print(f"Identification log: {log_path}")

    if estimate_mocap_delay is None:
        estimate_mocap_delay = bool(config["estimate_mocap_delay"])
    mocap_delay = float(log_config["mocap_delay"])
    if estimate_mocap_delay:
        try:
            delay_result = estimate_mocap_delay_from_log_file(
                log_path,
                max_delay_s=float(config["mocap_delay_search_range"]),
            )
            print_delay_result(delay_result)
            mocap_delay = float(delay_result["delay_s"])
        except ValueError as error:
            print(f"Mocap delay estimation skipped: {error}")
            print(f"Falling back to configured mocap_delay = {1000.0 * mocap_delay:.2f} ms")
    print(f"Mocap delay compensation: {1000.0 * mocap_delay:.2f} ms")

    robot_config = load_yaml(paths.robot_config)["robot"]
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(robot_config["wheel_radius"]),
        base_diameter=jnp.asarray(robot_config["base_diameter"]),
        max_wheel_speed=jnp.asarray(robot_config["max_wheel_speed"]),
        time_constant=jnp.asarray(robot_config["time_constant"]),
        # A zero value keeps the burnout model disabled in the log-space optimizer;
        # fall back to the experiment config init to (re-)enable identification.
        a_slip_max=jnp.asarray(robot_config.get("a_slip_max", 0.0) or config["init_a_slip_max"]),
    )

    pololu_log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=log_config["clip_after_first_trajectory"],
        mocap_delay_s=mocap_delay,
    )
    result = run_single_experiment_identification(
        problem_path=str(paths.problem),
        initial_params=init_params,
        num_steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        seed=int(experiment.config["seed"]),
        window_length=config["window_length"],
        target_log=pololu_log,
    )
    estimated_params = result["estimated_params"]
    print_physical_params("Estimated parameters:", estimated_params)
    print(f"Final normalized geometry loss: {float(result['loss_history'][-1]):.8f}")
    print(f"Final normalized motor loss:    {float(result['motor_loss_history'][-1]):.8f}")

    payload = {
        "log": str(log_path),
        "estimated_params": {
            "wheel_radius": float(estimated_params.wheel_radius),
            "base_diameter": float(estimated_params.base_diameter),
            "max_wheel_speed": float(estimated_params.max_wheel_speed),
            "time_constant": float(estimated_params.time_constant),
            "a_slip_max": float(estimated_params.a_slip_max),
        },
        "mocap_delay": mocap_delay,
        "mocap_delay_estimated": bool(estimate_mocap_delay),
        "final_loss": float(result["loss_history"][-1]),
        "final_motor_loss": float(result["motor_loss_history"][-1]),
    }
    save_yaml(paths.identification_result, payload)

    identified_robot_config = load_yaml(paths.robot_config)
    identified_robot_config["robot"].update(payload["estimated_params"])
    identified_robot_config.setdefault("estimator", {})["mocap_delay"] = mocap_delay
    write_iteration_problem(experiment.config["problem"], identified_robot_config, paths.problem_identified)
    print(f"Wrote {paths.identification_result} and {paths.problem_identified}")

    with collect_plots(paths.visualize_dir):
        plot_system_id(
            pipeline=result["pipeline"],
            init_target_log=result["init_target_log"],
            init_log=result["init_replay_log"],
            predicted_log=result["final_replay_log"],
            out_prefix=f"identification_{Path(log_path).stem}",
        )
        plot_loss_history(
            loss_history=result["loss_history"],
            motor_loss_history=result["motor_loss_history"],
            out_prefix="system_id",
        )
    return payload


def _resolve_log_path(paths: IterationPaths, log: str | None) -> Path:
    if log is not None:
        log_path = Path(log)
        if not log_path.is_file():
            log_path = paths.data_dir / log
        if not log_path.is_file():
            raise FileNotFoundError(f"Log not found: {log}")
        return log_path
    csvs = _list_log_csvs(paths)
    if not csvs:
        raise FileNotFoundError(
            f"No decoded Pololu logs in {paths.data_dir}. Copy the SD-card logs there and run decode-logs."
        )
    if len(csvs) > 1:
        names = ", ".join(path.name for path in csvs)
        print(f"Multiple logs in {paths.data_dir} ({names}); using {csvs[0].name} (pass --log to pick another).")
    return csvs[0]


# ---------------------------------------------------------------------------
# residual model
# ---------------------------------------------------------------------------


def stage_train_residual(experiment: Experiment, iteration: int) -> Path:
    """Train the residual dynamics model on all decoded logs of this iteration."""
    from wmr_simulator.residual_model.residual import train_from_logs

    paths = experiment.paths(iteration)
    config = experiment.config["residual"]
    log_config = experiment.config["log_loading"]
    problem_path = _identified_problem(paths)
    if not _list_log_csvs(paths):
        raise FileNotFoundError(f"No decoded Pololu logs in {paths.data_dir}.")

    # Use the delay settled during identification so the residual twist targets
    # align with the actions the same way the identification replay did.
    mocap_delay = float(log_config["mocap_delay"])
    if paths.identification_result.is_file():
        mocap_delay = float(load_yaml(paths.identification_result).get("mocap_delay", mocap_delay))
    print(f"Mocap delay compensation for residual dataset: {1000.0 * mocap_delay:.2f} ms")

    train_from_logs(
        problem=str(problem_path),
        log_dir=str(paths.data_dir),
        out=str(paths.residual_model),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        hidden_width=int(config["hidden_width"]),
        hidden_depth=int(config["hidden_depth"]),
        validation_split=float(config["validation_split"]),
        seed=int(experiment.config["seed"]),
        output_reg_weight=float(config["output_reg_weight"]),
        clip_after_first_trajectory=log_config["clip_after_first_trajectory"],
        mocap_delay_s=mocap_delay,
        resample_uniform=bool(config["resample_uniform"]),
        out_dir=str(paths.visualize_dir),
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
    config = experiment.config["gain_tuning"]
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
        k_min_stab=float(config["k_min_stab"]),
        k_max_stab=float(config["k_max_stab"]),
        k_max_rest=float(config["k_max_rest"]),
        num_lhs_points=int(config["num_lhs_points"]),
        num_adam_optimizations=int(config["num_adam_optimizations"]),
        schedule_enabled=config.get("gain_parametrization", config.get("gain_schedule")),
        gain_delta_weight=float(config["gain_delta_weight"]),
        residual_model=residual_model,
    )
    pipeline = result["pipeline"]
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    print(f"Final tuning loss: {float(result['loss_history'][-1]):.8f}")

    payload = {
        "gains": [float(gain) for gain in result["optimized_gains"]],
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

    with collect_plots(paths.visualize_dir):
        plot_gain_tuning_summary(
            pipeline,
            init_log=result["init_hidden_log"],
            tuned_log=result["final_hidden_log"],
            out_prefix="summary_gain_tuning",
        )
        plot_training_trajectory_summary(
            pipeline,
            robot_params=robot_params,
            tuned_gains=result["optimized_gains"],
            max_trajectories=None,
            out_prefix="summary_training",
        )
        plot_validation_trajectory_summary(
            pipeline,
            robot_params=robot_params,
            tuned_gains=result["optimized_gains"],
            out_prefix="summary_validation",
        )
        plot_controller_tuning_errors(
            pipeline=pipeline,
            init_log=result["init_hidden_log"],
            tuned_log=result["final_hidden_log"],
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
    estimate_mocap_delay: bool | None = None,
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
        stage_identify(experiment, iteration, log=log, estimate_mocap_delay=estimate_mocap_delay)
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
