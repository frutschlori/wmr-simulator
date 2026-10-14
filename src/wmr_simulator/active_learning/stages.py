"""Modular stages of the iterative active-learning loop.

Each stage reads its inputs and writes its outputs as files inside the
iteration folder (see experiment.py for the layout). Stage order:

    init -> plan-id-trajectory -> [run robot, copy SD logs into data/]
         -> benchmark -> decode-logs -> identify -> train-residual
         -> plan-tuning-trajectories -> tune-gains -> finalize (next iteration)

simulate-deployment replaces the bracketed robot step with a MuJoCo run of the
same trajectory; benchmark drives the held-out evaluation references.

Heavy imports (jax, the pipelines) happen inside the stage functions so the
CLI stays responsive for bookkeeping commands like status.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import math
import multiprocessing
import os
import pickle
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wmr_simulator.active_learning import baseline_runs
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


# ---------------------------------------------------------------------------
# init / finalize: iteration scaffolding
# ---------------------------------------------------------------------------


def stage_init(root: str | Path, overrides: dict | None = None) -> Experiment:
    """Create the experiment directory, experiment.yaml, and its first
    iteration: iteration_00 with prior tuning, else iteration_01."""
    experiment = Experiment.create(root, overrides)
    problem_cfg = load_yaml(experiment.config["problem"])
    robot_config = robot_config_from_problem(problem_cfg)
    robot_config["robot"].update(
        {name: float(value) for name, value in (experiment.config["initial_robot_params"] or {}).items()}
    )
    if not experiment.config["use_gain_parametrization"]:
        parametrization = robot_config["controller"].get("gain_parametrization")
        if parametrization is not None:
            parametrization["enabled"] = False
    first = 0 if experiment.config["prior_tuning"] else 1
    paths = _initialize_iteration(experiment, iteration=first, robot_config=robot_config)
    if first == 0:
        _initialize_prior_iteration(experiment, paths)
    print(f"Initialized experiment at {experiment.root}")
    print(f"  config: {experiment.config_path}")
    print(f"  first iteration: {paths.root}")
    return experiment


def _initialize_prior_iteration(experiment: Experiment, paths: IterationPaths) -> None:
    """Iteration 0 tunes on the prior model: no identification, no residual."""
    shutil.copy2(paths.problem, paths.problem_identified)
    if experiment.config["use_residual_model"]:
        save_yaml(paths.residual_skipped, {"reason": "iteration 0 tunes on the prior model; no logs yet"})


def stage_finalize(experiment: Experiment, iteration: int) -> IterationPaths:
    """Fold this iteration's results into the next iteration's inputs."""
    paths = experiment.paths(iteration)
    # Iteration 0 identified nothing: the next one keeps the prior model.
    if iteration > 0 and not paths.identification_result.is_file():
        raise FileNotFoundError(f"Missing {paths.identification_result}; run the identify stage first.")
    if not paths.gains_result.is_file():
        raise FileNotFoundError(f"Missing {paths.gains_result}; run the tune-gains stage first.")

    gains_result = load_yaml(paths.gains_result)
    robot_config = load_yaml(paths.robot_config)
    if iteration > 0:
        robot_config["robot"].update(load_yaml(paths.identification_result)["estimated_params"])
    robot_config["controller"]["gains"] = [float(gain) for gain in gains_result["gains"]]
    if gains_result.get("schedule") is not None:
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
    if not experiment.config["mujoco_deployment"]["enabled"]:
        _create_benchmark_directories(next_paths, experiment.config["benchmark"])
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
    calibration = mocap_calibration(experiment)
    robot_config.pop("mocap_calibration", None)
    if calibration is not None:
        robot_config["mocap_calibration"] = calibration.to_mapping()
    save_yaml(paths.robot_config, robot_config)
    write_iteration_problem(experiment.config["problem"], robot_config, paths.problem)
    export_robot_config(
        paths.robotcfg_cfg,
        physical_params=physical_params,
        controller_gains=robot_config["controller"]["gains"],
        template_path=experiment.config.get("robotcfg_template"),
        overrides=mocap_calibration_config_values(experiment),
    )
    gainmlp_path = _export_gain_mlp_if_configured(paths.problem, paths.gainmlp_jsn)
    if gainmlp_path is not None:
        print(f"  firmware gain-MLP for the robot: {gainmlp_path}")
    if static_gains is not None:
        _write_static_gain_config(experiment, paths, robot_config, physical_params, static_gains)
        print(f"  static-gain baseline for the robot: {paths.robotcfg_static_cfg}")
    return paths


def mocap_calibration(experiment: Experiment):
    """The robot's mocap calibration (experiment.yaml ``mocap_calibration``), or None."""
    path = experiment.config.get("mocap_calibration")
    if not path:
        return None
    from wmr_simulator.pololu.mocap_calibration import MocapCalibration

    return MocapCalibration.load(path)


def mocap_calibration_config_values(experiment: Experiment) -> dict[str, float] | None:
    """The ROBOTCFG.CFG ``mocap_*`` keys of the robot's calibration, or None."""
    calibration = mocap_calibration(experiment)
    return None if calibration is None else calibration.config_values()


def _write_static_gain_config(
    experiment: Experiment,
    paths: IterationPaths,
    robot_config: dict,
    physical_params,
    static_gains: list[float],
) -> None:
    """Write the static-gain variant of this iteration's robot config.

    On the robot, copy ROBOTCFG_static.CFG as ROBOTCFG.CFG without a
    GAINMLP.JSN next to it.
    """
    from wmr_simulator.pololu.robot_config import export_robot_config

    static_config = copy.deepcopy(robot_config)
    static_config["controller"]["gains"] = [float(gain) for gain in static_gains]
    static_config["controller"].pop("gain_parametrization", None)
    save_yaml(paths.robot_config_static, static_config)
    export_robot_config(
        paths.robotcfg_static_cfg,
        physical_params=physical_params,
        controller_gains=static_config["controller"]["gains"],
        template_path=experiment.config.get("robotcfg_template"),
        overrides=mocap_calibration_config_values(experiment),
    )


def _export_gain_mlp_if_configured(problem_path: Path, output_path: Path) -> Path | None:
    """Export the gain schedule to GAINMLP.JSN if the iteration problem enables it."""
    from wmr_simulator.gain_parametrization import params_from_cfg

    problem_cfg = load_yaml(problem_path)
    cfg = problem_cfg["controller"].get("gain_parametrization")
    if cfg is None or not cfg.get("enabled", False):
        return None

    from wmr_simulator.pololu.gain_mlp_exporter import export_gain_mlp

    robot_cfg = problem_cfg["robot"]
    feature_scale = [robot_cfg["v_max"], robot_cfg["omega_max"]]
    params = params_from_cfg(cfg, feature_scale)
    return export_gain_mlp(output_path, params)


# ---------------------------------------------------------------------------
# trajectory planning stages
# ---------------------------------------------------------------------------


def _residual_in_design(experiment: Experiment, stage_config: dict) -> bool:
    """Whether a trajectory-design stage rolls out on the residual plant
    (the top-level use_residual_model is the master switch)."""
    return bool(experiment.config["use_residual_model"] and stage_config["use_residual_model"])


def _load_design_residual_model(path: Path, problem_path: Path):
    """Residual checkpoint a trajectory-design stage rolls out on.

    A missing checkpoint raises unless the iteration skipped the residual; a
    checkpoint trained against a different nominal model raises as well.
    """
    skipped = path.with_name("residual_skipped.yaml")
    if not path.is_file() and skipped.is_file():
        reason = (load_yaml(skipped) or {}).get("reason", "no residual trained")
        print(f"No residual for this design ({skipped}: {reason}); designing nominal.")
        return None
    if not path.is_file():
        raise FileNotFoundError(
            f"The residual model is enabled for trajectory design but {path} is missing; "
            "run train-residual first (or disable use_residual_model in experiment.yaml)."
        )

    from wmr_simulator.residual_model import load_residual_model
    from wmr_simulator.residual_model.residual import robot_params_from_problem

    residual_model, checkpoint = load_residual_model(path, robot_params_from_problem(str(problem_path)))
    print(f"Designing on the residual-augmented plant: {path}")
    print(f"  config: {checkpoint['config']}")
    return residual_model


def _static_design_gains(paths: IterationPaths) -> list[float] | None:
    """The static gains the trajectory designers compute their FIM around.

    The problem yaml carries the parametrized run's base gains, which are not a
    controller that is driven on their own. None: the problem's gains are the
    static ones.
    """
    if not paths.robot_config_static.is_file():
        return None
    return [float(gain) for gain in load_yaml(paths.robot_config_static)["controller"]["gains"]]


def stage_plan_identification_trajectory(experiment: Experiment, iteration: int) -> Path:
    """Synthesize (or copy) the informative identification trajectory and
    export it as a Pololu reference JSN plus a bridged repeat variant
    (see pololu.bridge_exporter)."""
    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["identification_trajectory"]
    identification_plot_dir = paths.visualize_dir / "identification"
    identification_plot_dir.mkdir(parents=True, exist_ok=True)

    if not experiment.config["optimize_identification_trajectory"]:
        baseline = experiment.config.get("baseline_identification_trajectory")
        if not baseline:
            raise ValueError(
                "optimize_identification_trajectory is disabled but "
                "baseline_identification_trajectory is not set in experiment.yaml."
            )
        baseline = Path(baseline)
        if baseline.is_dir():
            # One random draw per seed.
            draws = sorted(baseline.glob("*.pkl"))
            if not draws:
                raise FileNotFoundError(f"No identification references in {baseline}.")
            baseline = draws[int(experiment.config["seed"]) % len(draws)]
        pickle_path = Path(shutil.copy2(baseline, paths.identification_trajectory_dir))
        print(f"Copied baseline identification trajectory: {pickle_path}")
    else:
        from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

        # This iteration has no residual yet, so the previous one is used.
        residual_model = None
        if _residual_in_design(experiment, config) and iteration > 1:
            residual_model = _load_design_residual_model(
                experiment.paths(iteration - 1).residual_model, _identification_problem(paths, config)
            )
        design_gains = _static_design_gains(paths)
        if design_gains is not None:
            print(f"Designing at the static-tune gains: {design_gains}")
        pipeline = TrajectoryOptimizationPipeline(
            str(_identification_problem(paths, config)),
            time_scaling=config["time_scaling"],
            objective_mode="identification",
            controller_gains=design_gains,
            motion_phases=config["phases"],
            residual_model=residual_model,
        )
        num_control_points = int(config["num_control_points"])
        pipeline.set_control_points(pipeline.initial_control_points(num_control_points))
        with collect_plots(identification_plot_dir):
            _, loss_history = pipeline.optimize_trajectory(
                num_control_points=num_control_points,
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

    from wmr_simulator.pololu.bridge_exporter import ArcReturn, append_bridge_reference
    from wmr_simulator.pololu.reference_exporter import (
        check_firmware_limits,
        export_reference_trajectory,
        load_reference_trajectory,
    )
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    # Up to append_tuning_trajectories of the previous tuning set, the last one
    # dropped while the bridged reference is over the firmware's limits.
    count = int(config.get("append_tuning_trajectories", 0))
    while True:
        appended = _append_tuning_trajectories(experiment, iteration, pickle_path, config, count)
        jsn_path = export_reference_trajectory(
            load_reference_trajectory(pickle_path),
            paths.identification_trajectory_dir,
        )
        print(f"Exported Pololu reference JSN: {jsn_path}")

        if appended:
            environment = load_yaml(_identification_problem(paths, config))["environment"]
            with pickle_path.open("rb") as file:
                pieces = pickle.load(file)["pieces"]
            arc_return = ArcReturn(
                radius=float(config["appended_turn_radius"]),
                lateral_acceleration=float(config["appended_turn_lateral_acceleration"]),
                min_piece_duration=float(config["appended_min_piece_duration"]),
                box_min=tuple(environment["min"]),
                box_max=tuple(environment["max"]),
                margin=float(config["appended_box_margin"]),
            )
            # Over the firmware limits: drive the return line faster before
            # dropping an appended trajectory.
            line_speeds = np.arange(arc_return.peak_speed, float(config["appended_return_max_line_speed"]) + 1e-9, 0.25)
            for line_speed in line_speeds:
                bridged_path = append_bridge_reference(
                    jsn_path,
                    wait_time=float(config["appended_bridge_wait_time"]),
                    plot_path=identification_plot_dir / "identification_trajectory_bridge.pdf",
                    pieces=pieces,
                    arc_return=arc_return._replace(peak_speed=float(line_speed)),
                )
                if not check_firmware_limits(
                    num_states=len(load_pololu_reference(bridged_path).states), file_bytes=bridged_path.stat().st_size,
                ):
                    break
            if line_speed > arc_return.peak_speed:
                print(f"Bridged reference over the firmware limits at a {arc_return.peak_speed} m/s return line; "
                      f"drove the line at up to {line_speed:.2f} m/s.")
        else:
            bridged_path = append_bridge_reference(
                jsn_path,
                wait_time=float(config["bridge_wait_time"]),
                bridge_time=float(config["bridge_time"]),
                plot_path=identification_plot_dir / "identification_trajectory_bridge.pdf",
            )
        problems = _reference_problems(bridged_path, load_yaml(_identification_problem(paths, config))["environment"])
        if len(appended) <= 1 or not problems:
            break
        count = len(appended) - 1
        print(f"Bridged reference with {len(appended)} appended trajectories: {'; '.join(problems)}; "
              f"trying {count}.")
    if problems:
        raise RuntimeError(
            f"{bridged_path} cannot be driven: {'; '.join(problems)}. Nothing left to drop; "
            f"fix the reference before copying it to the robot."
        )
    print(f"Exported bridged repeat variant: {bridged_path}")
    print(f"Copy {jsn_path.name} and {paths.robotcfg_cfg.name} to the robot SD card, run the")
    print(f"experiment, then place the logs in {paths.data_dir} and run decode-logs.")
    return jsn_path


def _reference_problems(jsn_path: Path, environment: dict) -> list[str]:
    """Why the robot cannot drive ``jsn_path``: over the firmware's limits, or
    leaving the environment box (the workspace; no margin). Empty when fine."""
    from wmr_simulator.pololu.reference_exporter import check_firmware_limits
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    states = np.asarray(load_pololu_reference(jsn_path).states, dtype=float)
    problems = check_firmware_limits(num_states=len(states), file_bytes=jsn_path.stat().st_size, label=jsn_path.name)
    low, high = np.asarray(environment["min"], dtype=float), np.asarray(environment["max"], dtype=float)
    outside = float(max(np.max(low - states[:, :2]), np.max(states[:, :2] - high), 0.0))
    if outside > 1e-6:
        problems.append(f"leaves the environment box [{low[0]}, {high[0]}] x [{low[1]}, {high[1]}] m by {outside:.2f} m")
    return problems


def _append_tuning_trajectories(
    experiment: Experiment, iteration: int, pickle_path: Path, config: dict, count: int
) -> list[str]:
    """Replace everything after the identified phase of the identification
    reference by trajectories of the previous iteration's tuning set
    (``trajectory_optimization.reference_extension``).

    The reference as designed is kept in ``identification_trajectory/designed/``.
    A trajectory that cannot be placed inside the environment box is excluded
    and the representatives are chosen again. Returns the appended
    trajectories' names (empty when nothing was appended).
    """
    from wmr_simulator.trajectory_optimization.reference_extension import (
        extended_payload,
        outside_box,
        representative_trajectories,
        traction_usage,
    )

    if count <= 0 or iteration <= experiment.first_iteration:
        return []
    source = config.get("append_tuning_trajectories_from")
    source_dir = Path(source) if source else experiment.paths(iteration - 1).tuning_trajectories_dir
    previous = sorted(source_dir.glob("*.pkl"))
    if not previous:
        print(f"No tuning references in {source_dir}; the identification reference keeps its own phases.")
        return []
    designed_dir = pickle_path.parent / "designed"
    designed_dir.mkdir(exist_ok=True)
    designed = designed_dir / pickle_path.name
    if not designed.exists():
        shutil.move(str(pickle_path), designed)
    identification_problem = load_yaml(_identification_problem(experiment.paths(iteration), config))
    environment = identification_problem["environment"]
    max_usage = float(config.get("appended_max_traction_usage", 0.0))
    base_diameter = float(identification_problem["robot"]["base_diameter"])
    excluded: list[Path] = []
    while True:
        chosen = representative_trajectories(
            previous, count, exclude=excluded, max_traction_usage=max_usage, base_diameter=base_diameter
        )
        if not chosen:
            print(f"No trajectory of {source_dir} fits the environment box after the identified phase; "
                  f"the identification reference keeps its own phases.")
            shutil.copy2(designed, pickle_path)
            return []
        payload = extended_payload(
            designed, chosen, box_min=environment["min"], box_max=environment["max"],
            margin=float(config["appended_box_margin"]),
            turn_radius=float(config["appended_turn_radius"]),
            turn_lateral_acceleration=float(config["appended_turn_lateral_acceleration"]),
            turn_min_duration=float(config["appended_min_piece_duration"]),
        )
        violation, culprit = outside_box(
            payload["reference_states"], payload["pieces"], environment["min"], environment["max"]
        )
        if violation == 0.0:
            break
        # A turn leaving the box is attributed to the trajectory after it.
        names = [piece.get("name") for piece in payload["pieces"]]
        name = next((name for name in names[culprit:] if name), None)
        if name is None:
            break  # the identified phase itself: nothing to exclude (the stage refuses the reference)
        excluded.append(next(path for path in chosen if path.stem == name))
        print(f"Appended {', '.join(path.stem for path in chosen)} leave the environment box by {violation:.2f} m "
              f"in every placement; excluding {name} and choosing again.")
    with pickle_path.open("wb") as file:
        pickle.dump(payload, file)
    duration = (len(payload["reference_states"]) - 1) * payload["dt"]
    usages = []
    for path in chosen:
        with path.open("rb") as file:
            states = np.asarray(pickle.load(file)["reference_states"], dtype=float)
        usages.append(f"{path.stem} {traction_usage(states, payload['dt'], base_diameter):.2f}")
    usages = ", ".join(usages)
    print(
        f"Identification reference: identified phase ({payload['identified_duration']:.1f} s) + "
        f"{len(chosen)} trajectories of {source_dir} "
        f"({', '.join(path.stem for path in chosen)}), {duration:.1f} s in all, placed {payload['placement']}; "
        f"traction usage {usages} (limit {max_usage or 'off'}); as designed: {designed}"
    )
    return [path.name for path in chosen]


def _check_fixed_tuning_references(pickles: list[Path], problem_path: Path) -> None:
    """Refuse fixed tuning references longer than the tuning horizon (the
    tuner would cut them off at sim_time)."""
    problem = load_yaml(problem_path)
    dt = float(problem["geometry_controller_dt"])
    sim_time = float(problem["sim_time"])
    too_long = []
    for path in pickles:
        with path.open("rb") as file:
            payload = pickle.load(file)
        states = payload["reference_states"] if isinstance(payload, dict) else payload
        reference_dt = float(payload.get("dt", dt)) if isinstance(payload, dict) else dt
        if not np.isclose(reference_dt, dt):
            raise ValueError(f"{path} is sampled at {reference_dt} s, the tuning problem at {dt} s.")
        duration = (len(states) - 1) * dt
        if duration > sim_time + 0.5 * dt:
            too_long.append((path.name, duration))
    if too_long:
        longest = max(duration for _, duration in too_long)
        raise ValueError(
            f"{len(too_long)} fixed tuning reference(s) run longer than the tuning horizon "
            f"({sim_time:.2f} s) and would be cut off: {too_long}. Set "
            f"tuning_trajectories.sim_time to at least {longest:.2f} in experiment.yaml."
        )


def stage_plan_tuning_trajectories(experiment: Experiment, iteration: int) -> list[Path]:
    """Synthesize (or copy) the reference trajectory set used for gain tuning."""
    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["tuning_trajectories"]
    problem_path = _tuning_problem(paths, config)

    if not experiment.config["optimize_tuning_trajectories"]:
        baseline_dir = experiment.config.get("baseline_tuning_trajectories_dir")
        if not baseline_dir:
            raise ValueError(
                "optimize_tuning_trajectories is disabled but baseline_tuning_trajectories_dir "
                "is not set in experiment.yaml."
            )
        pickles = sorted(Path(baseline_dir).glob("*.pkl"))
        if not pickles:
            raise ValueError(f"No reference pickles found in {baseline_dir}")
        _check_fixed_tuning_references(pickles, problem_path)
        copied = [Path(shutil.copy2(path, paths.tuning_trajectories_dir)) for path in pickles]
        print(f"Copied {len(copied)} baseline tuning trajectories into {paths.tuning_trajectories_dir}")
        return copied

    import numpy as np

    from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

    residual_model = None
    if _residual_in_design(experiment, config):
        residual_model = _load_design_residual_model(paths.residual_model, problem_path)
    design_gains = _static_design_gains(paths)
    if design_gains is not None:
        print(f"Designing at the static-tune gains: {design_gains}")
    motion_limits = {name: float(value) for name, value in config.get("motion_limits", {}).items()}
    if motion_limits:
        print(f"Designing inside overridden motion limits: {motion_limits}")
    pipeline = TrajectoryOptimizationPipeline(
        str(problem_path),
        objective_mode="gain-tuning",
        kimotor_fim_scale=float(config["kimotor_fim_scale"]),
        kdmotor_fim_scale=float(config["kdmotor_fim_scale"]),
        start_offset_mode=str(config["start_offset_mode"]),
        controller_gains=design_gains,
        residual_model=residual_model,
        motion_limits=motion_limits or None,
        time_scaling=config["time_scaling"],
        min_start_goal_distance=float(config["min_start_goal_distance"]),
    )
    constraint_component_weights = {
        name: float(value) for name, value in config["constraint_component_weights"].items()
    }
    constraint_smooth_max_beta = float(config["constraint_smooth_max_beta"])
    num_control_points = int(config["num_control_points"])
    pipeline.set_control_points(pipeline.initial_control_points(num_control_points))
    num_trajectories = int(config["num_trajectories"])
    if num_trajectories == 1:
        control_point_batch = None
        optimized_control_points, _ = pipeline.optimize_trajectory(
            num_control_points=num_control_points,
            num_steps=config["opt_steps"],
            learning_rate=config["learning_rate"],
            window_length=config["window_length"],
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
        )
        control_point_batch = [optimized_control_points]
    else:
        control_point_batch, _ = pipeline.optimize_trajectories(
            num_control_points=num_control_points,
            num_steps=config["opt_steps"],
            learning_rate=config["learning_rate"],
            num_trajectories=num_trajectories,
            vectorized=True,
            constraint_weight_jitter=config["constraint_weight_jitter"],
            seed=int(experiment.config["seed"]),
            window_length=config["window_length"],
            constraint_component_weights=constraint_component_weights,
            constraint_smooth_max_beta=constraint_smooth_max_beta,
            min_speed=float(config["min_speed"]),
            min_speed_fraction=float(config["min_speed_fraction"]),
            min_lateral_acceleration=float(config["min_lateral_acceleration"]),
            min_lateral_acceleration_fraction=float(config["min_lateral_acceleration_fraction"]),
            verbose=False,
            stretch_initialization=bool(config.get("stretch_initialization", True)),
        )
        final_losses = np.asarray(pipeline.batch_final_losses, dtype=float)
        print(f"Optimized {len(control_point_batch)} tuning trajectories "
              f"(final losses {final_losses.min():.4e} .. {final_losses.max():.4e})")

    clamped_batch = [pipeline.clamp_control_points(points) for points in control_point_batch]
    # Check the whole batch before writing any of it.
    stalled = {
        index: report
        for index, points in enumerate(clamped_batch)
        if (report := pipeline.tangent_diagnostics(points))["guarded_samples"]
    }
    if stalled:
        detail = "; ".join(
            f"{index:02d}: {report['guarded_samples']}/{report['num_samples']} samples, "
            f"min |dpos/ds| {report['min_tangent_norm']:.4g} vs threshold "
            f"{report['guard_threshold']:.4g}"
            for index, report in stalled.items()
        )
        raise ValueError(
            f"{len(stalled)} of {len(clamped_batch)} tuning designs stalled onto the fallback "
            f"tangent and were not exported -- {detail}. Their heading is the constant [1, 0] "
            f"over those samples, not the curve's. Raise the designer's min_tangent_fraction "
            f"and rerun plan-tuning-trajectories."
        )

    saved = []
    for index, clamped_control_points in enumerate(clamped_batch):
        saved.append(
            Path(
                pipeline.save_reference_states_pickle(
                    out_dir=str(paths.tuning_trajectories_dir),
                    filename_prefix=f"tuning_trajectory_{index:02d}",
                    reference_states=pipeline.reference_states_from_control_points(
                        clamped_control_points
                    ),
                    control_points=clamped_control_points,
                    # The gain tuner tunes on the offsets each trajectory was designed under.
                    start_offsets=(
                        ...
                        if pipeline.batch_start_offsets is None
                        else pipeline.batch_start_offsets[index]
                    ),
                )
            )
        )
    print(f"Saved {len(saved)} tuning trajectory pickles to {paths.tuning_trajectories_dir}")
    return saved


# ---------------------------------------------------------------------------
# data collection: the robot, or MuJoCo standing in for it
# ---------------------------------------------------------------------------


def stage_simulate_deployment(
    experiment: Experiment,
    iteration: int,
    num_logs: int | None = None,
) -> list[Path]:
    """Drive this iteration's bridged identification trajectory on the MuJoCo
    plant and write binary ``TRxx`` logs into ``data/``.

    The runs are chained like robot repeats: only the first is placed by hand,
    each later one starts where the previous bridge left the robot, and a run
    ending more than ``divergence_radius`` from the start is placed again. The
    ground truth is printed, never written to ``data/``.
    """
    from wmr_simulator.mujoco_sim.deploy import run_deployment

    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["mujoco_deployment"]
    trajectory = _identification_trajectory_jsn(paths)
    count = int(config["num_logs"] if num_logs is None else num_logs)

    print(
        f"MuJoCo deployment: {count} consecutive run(s) of {trajectory.name} "
        f"under {paths.robotcfg_cfg.name}"
        f"{' + ' + paths.gainmlp_jsn.name if paths.gainmlp_jsn.is_file() else ''} "
        f"-> {paths.data_dir}"
    )
    reference_start = _reference_start_pose(trajectory)
    divergence_radius = float(config["divergence_radius"])

    written: list[Path] = []
    start_pose = None  # the first run is placed by hand
    for index in range(count):
        seed = _deployment_seed(experiment, iteration, index)
        result = run_deployment(
            paths.robotcfg_cfg,
            trajectory,
            paths.data_dir,
            seed=seed,
            start_pose=start_pose,
            start_offset_radius=float(config["start_offset_radius"]),
            start_offset_angle=float(config["start_offset_angle"]),
        )
        written.append(result.log_path)
        placement = "placed by hand" if start_pose is None else "left by the bridge"
        distance_to_start = math.dist(result.final_pose[:2], reference_start[:2])
        diverged = distance_to_start > divergence_radius
        start_pose = None if diverged else result.final_pose
        offset = result.start_offset
        print(
            f"  {result.log_path.name}  seed {seed}  {placement} "
            f"{1000 * offset[0]:+.1f}/{1000 * offset[1]:+.1f} mm, {offset[2]:+.3f} rad  "
            f"| truth: RMSE {result.tracking_rmse:.3f} m, "
            f"ended {1000 * distance_to_start:.1f} mm from the start, "
            f"duty saturated {100 * result.duty_saturated_fraction:.0f}%"
            + ("  <- diverged, replacing by hand" if diverged else "")
        )
    return written


def _identification_trajectory_jsn(paths: IterationPaths) -> Path:
    """The reference the deployment drives: this iteration's bridged JSN."""
    candidates = sorted(paths.identification_trajectory_dir.glob("*_bridge.JSN"))
    if not candidates:
        raise FileNotFoundError(
            f"No bridged reference JSN (*_bridge.JSN) in {paths.identification_trajectory_dir}; "
            "run the plan-id-trajectory stage first."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Several bridged reference JSNs in {paths.identification_trajectory_dir} ({names}); "
            "leave exactly one so the deployment is unambiguous."
        )
    return candidates[0]


def _identified_duration(paths: IterationPaths) -> float | None:
    """Seconds of each identification log the fit uses (the pickle's
    ``identified_duration``), or None for all of it."""
    import pickle

    pickles = sorted(paths.identification_trajectory_dir.glob("*.pkl"))
    if not pickles:
        return None
    if len(pickles) > 1:
        raise ValueError(
            f"Expected one identification trajectory pickle in {paths.identification_trajectory_dir}, "
            f"found {len(pickles)}."
        )
    with pickles[0].open("rb") as file:
        payload = pickle.load(file)
    duration = payload.get("identified_duration") if isinstance(payload, dict) else None
    return None if duration is None else float(duration)


def _deployment_seed(experiment: Experiment, iteration: int, index: int) -> int:
    """Distinct per (iteration, run) and unchanged across reruns, so a repeated
    deployment reproduces the same placements and sensor noise."""
    return 1_000_000 * int(experiment.config["seed"]) + 1_000 * int(iteration) + int(index)


# A reference ending this close to its start is chained without a bridge path.
SELF_CLOSING_POSITION_TOLERANCE = 0.05  # m
SELF_CLOSING_HEADING_TOLERANCE = 0.15  # rad


def stage_run_benchmark(
    experiment: Experiment,
    iteration: int,
    num_runs: int | None = None,
) -> list[Path]:
    """Drive the benchmark references under every controller variant this
    iteration ships.

    Each reference is driven ``num_runs`` times in a chain (as in
    ``stage_simulate_deployment``); open references are bridged back to their
    start. Variants share their seeds, so placements and sensor noise are
    paired. Logs go to ``data/benchmark[_static]/<shape>/``.
    """
    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["benchmark"]
    count = int(config["num_runs"] if num_runs is None else num_runs)

    sources = _benchmark_sources(config)
    # Ground truth, kept out of data/.
    report = load_yaml(paths.benchmark_result) if paths.benchmark_result.is_file() else {}
    report["divergence_radius"] = float(config["divergence_radius"])
    report.setdefault("trajectories", {})

    if len(sources) > 1:
        print(f"Benchmark set: {len(sources)} held-out reference(s) -- {', '.join(p.stem for p in sources)}")

    # Set up every chain first, then drive them (in parallel).
    written: list[Path] = []
    jobs: list[dict] = []
    job_targets: list[tuple[dict, str, Path, Path]] = []
    for source in sources:
        shape = source.stem
        trajectory = _benchmark_reference(paths, config, source)
        reference_start = _reference_start_pose(trajectory)
        # Scored over the reference alone, without the bridge.
        score_duration = _reference_duration(source)
        entry = report["trajectories"].setdefault(shape, {})
        entry["trajectory"] = str(trajectory)
        entry.setdefault("variants", {})
        for variant, robot_config_source, data_dir in _benchmark_variant_specs(paths, shape, len(sources)):
            existing = _run_log_paths(data_dir)
            if existing:
                named = f" {shape}" if len(sources) > 1 else ""
                print(
                    f"Benchmark{named} ({variant}) already recorded "
                    f"({len(existing)} run(s) in {data_dir}); skipping."
                )
                written.extend(existing)
                continue
            robot_config = _staged_benchmark_config(paths, variant, robot_config_source)
            jobs.append(
                _benchmark_chain_job(
                    experiment, iteration, variant, robot_config, trajectory, data_dir, count,
                    config, reference_start, shape=shape if len(sources) > 1 else None,
                    score_duration=score_duration,
                )
            )
            job_targets.append((entry, variant, robot_config_source, data_dir))

    results = _run_benchmark_chains(jobs, config.get("max_workers", 0)) if jobs else []
    for (entry, variant, robot_config_source, data_dir), (chain_written, runs, lines) in zip(job_targets, results):
        for line in lines:
            print(line)
        written.extend(chain_written)
        entry["variants"][variant] = {
            "robot_config": robot_config_source.name,
            "data": str(data_dir.relative_to(paths.root)),
            "num_diverged": sum(1 for run in runs if run["diverged"]),
            "mean_tracking_rmse": (
                float(sum(run["tracking_rmse"] for run in runs) / len(runs)) if runs else None
            ),
            "median_tracking_rmse": (
                float(statistics.median(run["tracking_rmse"] for run in runs)) if runs else None
            ),
            "runs": runs,
        }
    recorded_any = bool(jobs)

    if recorded_any:
        save_yaml(paths.benchmark_result, report)
        print(f"Benchmark ground truth: {paths.benchmark_result}")
    return written


def _benchmark_chain_job(
    experiment: Experiment,
    iteration: int,
    variant: str,
    robot_config: Path,
    trajectory: Path,
    data_dir: Path,
    count: int,
    config: dict,
    reference_start: tuple[float, float, float],
    shape: str | None = None,
    score_duration: float | None = None,
) -> dict:
    """Everything one chain needs, as plain (picklable) data.

    ``score_duration`` limits scoring to the unbridged reference (None: whole runs).
    """
    # Seeds differ per shape but are shared across variants.
    shape_offset = _benchmark_shape_offset(shape)
    return {
        "variant": variant,
        "robot_config": Path(robot_config),
        "trajectory": Path(trajectory),
        "data_dir": Path(data_dir),
        "reference_start": tuple(float(value) for value in reference_start),
        "divergence_radius": float(config["divergence_radius"]),
        "start_offset_radius": float(config["start_offset_radius"]),
        "start_offset_angle": float(config["start_offset_angle"]),
        "score_duration": None if score_duration is None else float(score_duration),
        "seeds": [
            _benchmark_seed(experiment, iteration, index) + shape_offset for index in range(count)
        ],
    }


def _drive_benchmark_chain(job: dict) -> tuple[list[Path], list[dict], list[str]]:
    """Drive one (shape, variant) chain of runs; returns the console lines
    instead of printing them, so parallel chains do not interleave."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment
    from wmr_simulator.mujoco_sim.firmware import GAIN_MLP_FILENAME

    robot_config = job["robot_config"]
    trajectory = job["trajectory"]
    data_dir = job["data_dir"]
    variant = job["variant"]
    divergence_radius = job["divergence_radius"]
    reference_start = job["reference_start"]

    network = ""
    if (robot_config.parent / GAIN_MLP_FILENAME).is_file():
        identity = " (identity)" if variant == baseline_runs.STATIC_VARIANT else ""
        network = f" + {GAIN_MLP_FILENAME}{identity}"
    lines = [
        f"Benchmark ({variant}): {len(job['seeds'])} consecutive run(s) of {trajectory.name} "
        f"under {robot_config.name}{network} -> {data_dir}"
    ]
    written: list[Path] = []
    runs: list[dict] = []
    start_pose = None  # the first run is placed by hand
    for seed in job["seeds"]:
        result = run_deployment(
            robot_config,
            trajectory,
            data_dir,
            seed=seed,
            start_pose=start_pose,
            start_offset_radius=job["start_offset_radius"],
            start_offset_angle=job["start_offset_angle"],
            score_duration=job.get("score_duration"),
        )
        written.append(result.log_path)
        placement = "placed by hand" if start_pose is None else "left by the previous run"
        distance_to_start = math.dist(result.final_pose[:2], reference_start[:2])
        diverged = distance_to_start > divergence_radius
        start_pose = None if diverged else result.final_pose
        offset = result.start_offset
        lines.append(
            f"  {result.log_path.name}  seed {seed}  {placement} "
            f"{1000 * offset[0]:+.0f}/{1000 * offset[1]:+.0f} mm, {offset[2]:+.3f} rad  "
            f"| truth: RMSE {result.tracking_rmse:.3f} m, ended {distance_to_start:.3f} m from the start"
            + ("  <- diverged, replacing by hand" if diverged else "")
        )
        runs.append(
            {
                "log": result.log_path.name,
                "seed": seed,
                "placed_by_hand": placement == "placed by hand",
                "start_offset": [float(value) for value in offset],
                "tracking_rmse": float(result.tracking_rmse),
                "tracking_max": float(result.tracking_max),
                # Scores cover scored_duration; tracking_rmse_full the whole run.
                "scored_duration": float(result.scored_duration),
                "tracking_rmse_full": float(result.tracking_rmse_full),
                "distance_to_start": float(distance_to_start),
                "diverged": bool(diverged),
                "max_duty": float(result.max_duty),
                "duty_saturated_fraction": float(result.duty_saturated_fraction),
            }
        )
    return written, runs, lines


def _benchmark_worker_count(num_jobs: int, configured) -> int:
    """How many chains to drive at once. 0/None = one per job, capped by CPUs."""
    if num_jobs <= 1:
        return 1
    configured = 0 if configured is None else int(configured)
    if configured == 1:
        return 1
    available = os.cpu_count() or 1
    # Leave a core for the parent.
    auto = max(1, min(num_jobs, available - 1))
    return auto if configured <= 0 else min(configured, num_jobs)


def _run_benchmark_chains(jobs: list[dict], max_workers) -> list[tuple[list[Path], list[dict], list[str]]]:
    """Drive independent chains, in spawned worker processes when there is
    more than one (JAX does not survive a fork). Without a ``__main__`` file to
    re-import (REPL), the chains run in this process; results are identical."""
    workers = _benchmark_worker_count(len(jobs), max_workers)
    main_module = sys.modules.get("__main__")
    spawnable = getattr(main_module, "__file__", None) is not None
    if workers <= 1 or not spawnable:
        if workers > 1:
            print(
                f"Driving {len(jobs)} benchmark chain(s) in this process: the entry point is not a "
                f"file, so worker processes cannot re-import it."
            )
        return [_drive_benchmark_chain(job) for job in jobs]
    print(f"Driving {len(jobs)} benchmark chain(s) on {workers} worker process(es).")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(_drive_benchmark_chain, jobs))


def _benchmark_variant_specs(
    paths: IterationPaths, shape: str | None = None, num_shapes: int = 1
) -> list[tuple[str, Path, Path]]:
    """``(variant, robot config, data dir)`` per controller this iteration ships.

    Always the static controller (``ROBOTCFG_static.CFG``, or ``ROBOTCFG.CFG``
    when the iteration ships only one), plus the parametrized controller when
    there is a separate static one, plus any post-hoc parametrized variants.
    """
    # One subdirectory per shape when the benchmark is a set.
    def data_dir(base: Path) -> Path:
        return base if shape is None or num_shapes <= 1 else base / shape

    if not paths.robotcfg_static_cfg.is_file():
        specs = [
            (baseline_runs.STATIC_VARIANT, paths.robotcfg_cfg, data_dir(paths.benchmark_static_data_dir))
        ]
    else:
        specs = [
            (baseline_runs.STATIC_VARIANT, paths.robotcfg_static_cfg, data_dir(paths.benchmark_static_data_dir)),
            (baseline_runs.PARAMETRIZED_VARIANT, paths.robotcfg_cfg, data_dir(paths.benchmark_data_dir)),
        ]
    # Post-hoc parametrized controllers (active_learning.posthoc).
    from wmr_simulator.active_learning.posthoc import VARIANT_PREFIX

    for config in sorted((paths.root / "parametrized").glob("*/ROBOTCFG.CFG")):
        variant = VARIANT_PREFIX + config.parent.name
        specs.append((variant, config, data_dir(paths.data_dir / f"benchmark_{variant}")))
    return specs


def _staged_benchmark_config(paths: IterationPaths, variant: str, source: Path) -> Path:
    """The robot config as the benchmark driver reads it.

    The firmware picks up a ``GAINMLP.JSN`` next to the config, so the static
    variant is copied into its own directory under ``benchmark/``.
    """
    if source == paths.robotcfg_cfg or source.parent.parent == paths.root / "parametrized":
        return source
    staged_dir = paths.benchmark_dir / variant
    staged_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(source, staged_dir / paths.robotcfg_cfg.name))


def _benchmark_set_description(config: dict) -> str:
    """How the status line names the benchmark: one reference, or the set."""
    try:
        sources = _benchmark_sources(config)
    except FileNotFoundError:
        return f"{config['trajectory']} (missing)"
    if len(sources) == 1:
        return str(sources[0])
    return f"each of {len(sources)} held-out references in {config['trajectory']}"


def _benchmark_sources(config: dict) -> list[Path]:
    """The reference(s) ``benchmark.trajectory`` names: a file, or every
    ``.JSN``/``.pkl`` in a directory except ``*_bridge`` copies."""
    source = Path(config["trajectory"])
    if source.is_file():
        return [source]
    if source.is_dir():
        references = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in (".jsn", ".pkl")
            and not path.stem.endswith("_bridge")
        )
        if not references:
            raise FileNotFoundError(
                f"No .JSN or .pkl references in benchmark trajectory directory {source}."
            )
        # The stem names the shape, so it has to be unique.
        duplicates = sorted(
            {path.stem for path in references if [p.stem for p in references].count(path.stem) > 1}
        )
        if duplicates:
            raise ValueError(
                f"Benchmark trajectory directory {source} has more than one file per shape: "
                f"{duplicates}. Each reference names its own shape, so keep one file per shape."
            )
        return references
    raise FileNotFoundError(
        f"Benchmark trajectory not found: {source} (benchmark.trajectory in experiment.yaml). "
        "Point it at a reference file, or at a directory of them for a cross-validation set."
    )


def _benchmark_reference(paths: IterationPaths, config: dict, source: Path | None = None) -> Path:
    """The benchmark reference as this iteration drives it, copied into
    ``benchmark/`` and bridged when it does not close on itself."""
    source = Path(config["trajectory"]) if source is None else Path(source)
    if not source.is_file():
        raise FileNotFoundError(
            f"Benchmark trajectory not found: {source} (benchmark.trajectory in experiment.yaml)."
        )
    paths.benchmark_dir.mkdir(parents=True, exist_ok=True)
    copied = Path(shutil.copy2(source, paths.benchmark_dir / source.name))
    if _is_self_closing(copied):
        return copied

    from wmr_simulator.pololu.bridge_exporter import append_bridge_reference

    bridged = append_bridge_reference(
        copied,
        wait_time=float(config["bridge_wait_time"]),
        bridge_time=float(config["bridge_time"]),
        plot_path=paths.visualize_dir / "benchmark" / f"{copied.stem}_bridge.pdf",
    )
    print(f"Benchmark reference does not close on itself; driving the bridged variant: {bridged.name}")
    return bridged


def _reference_start_pose(trajectory: Path) -> tuple[float, float, float]:
    """``(x, y, yaw)`` of the reference's first state."""
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    start = load_pololu_reference(trajectory).states[0]
    return (float(start[0]), float(start[1]), float(start[2]))


def _reference_duration(trajectory: Path) -> float:
    """How long the firmware follows ``trajectory`` [s] (states times dt)."""
    from wmr_simulator.pololu.reference_importer import load_reference

    reference = load_reference(trajectory)
    return float(reference.states.shape[0] * reference.dt)


def _is_self_closing(trajectory: Path) -> bool:
    """Whether a reference ends where it started, within the pose tolerances."""
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    states = load_pololu_reference(trajectory).states
    start, end = states[0], states[-1]
    heading_error = abs(math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2])))
    return (
        math.dist(end[:2], start[:2]) <= SELF_CLOSING_POSITION_TOLERANCE
        and heading_error <= SELF_CLOSING_HEADING_TOLERANCE
    )


def _run_log_paths(directory: Path) -> list[Path]:
    """Recordings in one run directory (binary logs or decoded csvs)."""
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("TR*") if path.is_file())


def _create_benchmark_directories(paths: IterationPaths, config: dict) -> list[Path]:
    """Create the empty ``data/`` subdirectories the robot's benchmark logs go into."""
    try:
        sources = _benchmark_sources(config)
    except FileNotFoundError:
        return []
    directories = [
        data_dir
        for source in sources
        for _, _, data_dir in _benchmark_variant_specs(paths, source.stem, len(sources))
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _benchmark_recorded(paths: IterationPaths, config: dict) -> bool:
    """Whether every controller variant and shape of this iteration has its runs."""
    try:
        sources = _benchmark_sources(config)
    except FileNotFoundError:
        return False
    return all(
        _run_log_paths(data_dir)
        for source in sources
        for _, _, data_dir in _benchmark_variant_specs(paths, source.stem, len(sources))
    )


def _benchmark_shape_offset(shape: str | None) -> int:
    """A stable per-shape seed offset (zero for a single reference)."""
    if not shape:
        return 0
    digest = hashlib.sha256(shape.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 100_000


def _benchmark_seed(experiment: Experiment, iteration: int, index: int) -> int:
    """Distinct per (iteration, run), disjoint from the deployment seeds and
    shared by the controller variants (paired comparison)."""
    return _deployment_seed(experiment, iteration, index) + 500


def stage_decode_logs(experiment: Experiment, iteration: int) -> list[Path]:
    """Decode the binary logs in data/ into csv files and plot a summary of each."""
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
    from wmr_simulator.visualization.pololu import plot_logged_summary, plot_logged_trajectories

    clip = experiment.config["log_loading"]["clip_after_first_trajectory"]
    plot_dir = paths.visualize_dir / "logs"
    base_gains, gain_params = _log_gain_parametrization(paths)
    logs, log_labels = [], []
    for log_path in _list_log_csvs(paths):
        try:
            log = load_pololu_traj_control_log(log_path, clip_after_first_trajectory=clip)
        except ValueError as error:
            print(f"Log summary plot skipped for {log_path.name}: {error}")
            continue
        logs.append(log)
        log_labels.append(log_path.stem)
        if (plot_dir / f"{log_path.stem}.pdf").exists():
            continue
        try:
            imu_time, imu_gyro_z = load_imu_gyro_z(log_path, clip_after_first_trajectory=clip)
        except ValueError as error:
            print(f"Log summary plot skipped for {log_path.name}: {error}")
            continue
        # The firmware does not log the scheduled gains; recover them offline.
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
    if logs:
        plot_path = plot_logged_trajectories(
            logs,
            log_labels,
            out_prefix="logged_trajectories",
            out_dir=paths.visualize_dir,
        )
        print(f"Logged trajectories plot: {plot_path}")


def _log_gain_parametrization(paths: IterationPaths):
    """Base gains + enabled gain parametrization used to record this iteration's
    logs (``None`` params when no parametrization is enabled)."""
    problem_cfg = load_yaml(paths.problem)
    controller = problem_cfg["controller"]
    base_gains = [float(gain) for gain in controller["gains"]]
    cfg = controller.get("gain_parametrization")
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


def plot_run_directory_logs(experiment: Experiment) -> list[Path]:
    """Summary-plot every recording in a ``data/`` subdirectory (benchmark
    runs) of every iteration into ``visualize/logs/<subdirectory>/``; existing
    plots are kept."""
    written: list[Path] = []
    for iteration in experiment.iteration_indices():
        written.extend(_plot_run_directory_logs(experiment, experiment.paths(iteration)))
    return written


def _plot_run_directory_logs(experiment: Experiment, paths: IterationPaths) -> list[Path]:
    from wmr_simulator.pololu.log_loader import load_imu_gyro_z, load_pololu_traj_control_log
    from wmr_simulator.visualization.pololu import plot_logged_summary

    if not paths.data_dir.is_dir():
        return []
    clip = experiment.config["log_loading"]["clip_after_first_trajectory"]
    written: list[Path] = []
    for directory in _run_directories(paths.data_dir):
        relative = directory.relative_to(paths.data_dir)
        out_dir = paths.visualize_dir / "logs" / relative
        base_gains, gain_params = _run_directory_gains(paths, relative)
        # A missing histogram needs every run of the directory loaded.
        histogram_path = out_dir / "motion_histograms.pdf"
        logs = []
        with tempfile.TemporaryDirectory(prefix="run_log_plots_") as temp_name:
            for log_path in _run_log_csvs(
                directory, Path(temp_name), out_dir, needed_only=histogram_path.exists()
            ):
                try:
                    log = load_pololu_traj_control_log(log_path, clip_after_first_trajectory=clip)
                    imu_time, imu_gyro_z = load_imu_gyro_z(log_path, clip_after_first_trajectory=clip)
                except ValueError as error:
                    print(f"Log summary plot skipped for {relative / log_path.stem}: {error}")
                    continue
                logs.append(log)
                if (out_dir / f"{log_path.stem}.pdf").exists():
                    continue
                gains = None
                if gain_params is not None:
                    from wmr_simulator.pololu.gain_reconstruction import applied_gains_over_log

                    gains = applied_gains_over_log(log, base_gains, gain_params)
                plot_path = plot_logged_summary(
                    log,
                    out_prefix=log_path.stem,
                    out_dir=out_dir,
                    imu_time_s=imu_time,
                    imu_gyro_z=imu_gyro_z,
                    gains=gains,
                )
                print(f"Run log summary plot: {plot_path}")
                written.append(Path(plot_path))
        if logs and not histogram_path.exists():
            plot_path = _plot_run_motion_histograms(paths, relative, logs, out_dir)
            if plot_path is not None:
                written.append(plot_path)
    return written


def _plot_run_motion_histograms(
    paths: IterationPaths, relative: Path, logs: list, out_dir: Path
) -> Path | None:
    """Motion histogram (reference and measured) of one recording set, pooled over its runs."""
    from wmr_simulator.visualization.motion_histograms import (
        measured_motion_channels_from_log,
        plot_motion_histograms,
        pool_motion_channels,
        reference_motion_channels_from_log,
    )

    # Only used for the limit lines.
    robot_cfg = load_yaml(paths.problem).get("robot") if paths.problem.is_file() else None
    return plot_motion_histograms(
        [
            ("reference", pool_motion_channels([reference_motion_channels_from_log(log) for log in logs])),
            ("measured", pool_motion_channels([measured_motion_channels_from_log(log) for log in logs])),
        ],
        out_prefix="motion_histograms",
        out_dir=out_dir,
        title=f"Motion Distributions: {relative} ({len(logs)} runs)",
        robot_cfg=robot_cfg,
    )


def _run_directories(data_dir: Path) -> list[Path]:
    """Every subdirectory of ``data/`` holding recordings directly."""
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_dir() and any(child.is_file() for child in path.glob("TR*"))
    )


def _run_log_csvs(
    directory: Path, temporary_dir: Path, out_dir: Path, needed_only: bool = True
) -> list[Path]:
    """csv paths of the recordings in ``directory`` (by default only those
    still missing a summary plot); binary logs are decoded into ``temporary_dir``."""
    from wmr_simulator.pololu.decode_binary import decode_file

    log_paths = sorted(path for path in directory.glob("TR*") if path.is_file())
    # Prefer a decoded csv over its binary.
    decoded = {path.stem for path in log_paths if path.suffix.lower() == ".csv"}
    log_paths = [path for path in log_paths if path.suffix.lower() == ".csv" or path.name not in decoded]

    csv_paths: list[Path] = []
    for log_path in log_paths:
        if needed_only and (out_dir / f"{log_path.stem}.pdf").exists():
            continue
        if log_path.suffix.lower() == ".csv":
            csv_paths.append(log_path)
            continue
        csv_path = temporary_dir / f"{log_path.name}.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            decoded_ok = decode_file(str(log_path), str(csv_path))
        if decoded_ok:
            csv_paths.append(csv_path)
        else:
            print(f"Log summary plot skipped, unreadable log: {log_path}")
    return csv_paths


def _run_directory_gains(paths: IterationPaths, relative: Path):
    """Base gains + gain parametrization the runs in ``data/<relative>/`` were
    driven under (``None`` params for a static-gain recording)."""
    if not paths.problem.is_file():
        return None, None

    from wmr_simulator.active_learning.baseline_runs import BENCHMARK_DIRECTORY_NAMES, PARAMETRIZED_VARIANT

    if relative.parts[0] == BENCHMARK_DIRECTORY_NAMES[PARAMETRIZED_VARIANT]:
        return _log_gain_parametrization(paths)
    static_gains = _static_design_gains(paths)
    if static_gains is None:
        static_gains = [float(gain) for gain in load_yaml(paths.problem)["controller"]["gains"]]
    return static_gains, None


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------


def stage_identify(
    experiment: Experiment,
    iteration: int,
    log: str | None = None,
) -> dict:
    """Identify the robot parameters from the decoded logs directly in data/.

    Fits each log on its own, excludes logs whose parameters disagree with the
    batch (``identification.outlier_z_threshold``), then fits the rest jointly.
    Writes results/identification.yaml and problem_identified.yaml.
    """
    import jax.numpy as jnp
    import numpy as np

    from wmr_simulator.identification.outliers import robust_parameter_outliers
    from wmr_simulator.identification.pipeline import run_multi_log_identification
    from wmr_simulator.pololu.log_loader import (
        clip_log_to_duration,
        load_pololu_traj_control_log_and_encoder_lag,
    )
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
    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(robot_config["wheel_radius"]),
        base_diameter=jnp.asarray(robot_config["base_diameter"]),
        max_wheel_speed=jnp.asarray(robot_config["max_wheel_speed"]),
        time_constant=jnp.asarray(robot_config["time_constant"]),
    )

    def identify(target_logs):
        return run_multi_log_identification(
            problem_path=str(paths.problem),
            initial_params=init_params,
            target_logs=target_logs,
            num_steps=int(config["steps"]),
            learning_rate=float(config["learning_rate"]),
            seed=int(experiment.config["seed"]),
            window_length=config["window_length"],
        )

    loaded = [
        load_pololu_traj_control_log_and_encoder_lag(
            log_path,
            clip_after_first_trajectory=log_config["clip_after_first_trajectory"],
        )
        for log_path in log_paths
    ]
    pololu_logs = [log for log, _ in loaded]
    # Encoder-to-mocap lag the loader measured and removed per log.
    encoder_lags = [
        None if lag is None else {
            "lag_ms": round(float(1000.0 * lag.lag_s), 2),
            "yaw_correlation": round(float(lag.yaw_correlation), 4),
            "speed_correlation": round(float(lag.speed_correlation), 4),
        }
        for _, lag in loaded
    ]
    for log_path, lag in zip(log_paths, encoder_lags):
        print(f"  {log_path.name}: encoder lag " + ("not measurable" if lag is None else f"{lag['lag_ms']:+.1f} ms (corr yaw {lag['yaw_correlation']:.3f}, speed {lag['speed_correlation']:.3f})"))
    identified_duration = _identified_duration(paths)
    if identified_duration is not None:
        print(f"Fitting the first {identified_duration:.2f} s of each log (the trajectory's identified phases).")
        pololu_logs = [clip_log_to_duration(pololu_log, identified_duration) for pololu_log in pololu_logs]

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
        "encoder_lags": {log_path.name: lag for log_path, lag in zip(log_paths, encoder_lags)},
        "outlier_z_threshold": float(config["outlier_z_threshold"]),
        "outlier_detection_evaluated": bool(report.evaluated),
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
        # Kept logs under the joint fit, excluded ones under their own.
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

    _plot_identified_parameters(experiment, paths)
    return payload


def _plot_identified_parameters(experiment: Experiment, paths: IterationPaths) -> str | None:
    """Plot the identified parameters of every iteration so far (best effort)."""
    try:
        from wmr_simulator.active_learning.identified_parameters import collect_identified_parameters
        from wmr_simulator.visualization.identified_parameters import plot_identified_parameters

        records = collect_identified_parameters(experiment)
        plot_path = plot_identified_parameters(
            records, experiment.root, out_path=paths.visualize_dir / "identified_parameters.pdf"
        )
        print(f"Wrote {plot_path}")
        return plot_path
    except Exception as error:
        print(f"Identified parameters plot skipped ({error}).")
        return None


def _estimated_params_dict(params) -> dict:
    return {
        "wheel_radius": float(params.wheel_radius),
        "base_diameter": float(params.base_diameter),
        "max_wheel_speed": float(params.max_wheel_speed),
        "time_constant": float(params.time_constant),
    }


def _print_per_log_parameters(log_paths, parameter_samples, per_log_results, report) -> None:
    header = (
        f"{'log':<12}{'r [mm]':>10}{'L [mm]':>10}{'u_max':>10}"
        f"{'tau [s]':>10}{'loss':>12}{'max z':>8}"
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
            f"{values[3]:>10.4f}{total_loss:>12.6f}"
            f"{report.max_z_scores[index]:>8.2f}{marker}"
        )
    if not report.evaluated:
        print("  (outlier detection off: disabled by threshold, too few logs, or no parameter spread)")


def _resolve_log_paths(paths: IterationPaths, log: str | None) -> list[Path]:
    """Every decoded log directly in data/, or just the one named by ``--log``."""
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
    """Iteration indices whose decoded logs feed the residual model at
    ``iteration``: this one and, with ``residual.pool_previous_iterations``,
    every earlier one."""
    if not experiment.config["residual"].get("pool_previous_iterations", True):
        candidates = [iteration]
    else:
        candidates = [index for index in experiment.iteration_indices() if index <= iteration]
    return [index for index in candidates if _list_log_csvs(experiment.paths(index))]


def _screen_diverged_logs(
    log_paths: list[Path], max_position_error: float, clip_after_first_trajectory: bool
) -> tuple[list[Path], list[tuple[Path, float]]]:
    """``(usable, [(diverged, max error)])``: logs whose mocap pose leaves the
    reference by more than ``max_position_error`` [m] diverged (0: keep all)."""
    if max_position_error <= 0.0:
        return list(log_paths), []
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log

    usable, diverged = [], []
    for path in log_paths:
        log = load_pololu_traj_control_log(path, clip_after_first_trajectory=clip_after_first_trajectory)
        reference_time = np.asarray(log.reference.time_s, dtype=float)
        reference = np.asarray(log.reference.states, dtype=float)
        pose_time = np.asarray(log.pose.time_s, dtype=float)
        pose = np.asarray(log.pose.states, dtype=float)
        error = float(np.max(np.hypot(
            np.interp(reference_time, pose_time, pose[:, 0]) - reference[:, 0],
            np.interp(reference_time, pose_time, pose[:, 1]) - reference[:, 1],
        )))
        if error > max_position_error:
            diverged.append((path, error))
        else:
            usable.append(path)
    return usable, diverged


def stage_train_residual(experiment: Experiment, iteration: int) -> Path:
    """Train the residual dynamics model on the pooled decoded logs."""
    if not experiment.config["use_residual_model"]:
        raise ValueError(
            f"use_residual_model is disabled for {experiment.root}; this experiment "
            "never trains a residual (set it in experiment.yaml to change that)."
        )
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
    pooled_logs = [log for index in pooled_iterations for log in _list_log_csvs(experiment.paths(index))]
    usable_logs, diverged_logs = _screen_diverged_logs(
        pooled_logs, float(config.get("max_position_error", 0.0)), log_config["clip_after_first_trajectory"]
    )
    print(
        f"Residual training pool: {len(usable_logs)} of {len(pooled_logs)} logs from "
        f"{len(pooled_iterations)} iteration(s) "
        f"({', '.join(f'{ITERATION_PREFIX}{index:02d}' for index in pooled_iterations)}); "
        f"{len(diverged_logs)} diverged run(s) left out"
    )
    # Validation holds out whole logs, so training needs two.
    if len(usable_logs) < 2:
        paths.residual_model.unlink(missing_ok=True)
        save_yaml(
            paths.residual_skipped,
            {
                "reason": (
                    f"{len(usable_logs)} usable pooled identification log(s), training needs 2 "
                    "(the rest diverged, residual.max_position_error)"
                ),
                "max_position_error": float(config["max_position_error"]),
                "diverged_logs": {str(path): round(error, 4) for path, error in diverged_logs},
            },
        )
        print(f"Fewer than 2 usable logs: no residual this iteration ({paths.residual_skipped}).")
        return paths.residual_skipped
    paths.residual_skipped.unlink(missing_ok=True)

    train_from_logs(
        problem=str(problem_path),
        log_dirs=[str(directory) for directory in log_dirs],
        log_paths=[str(path) for path in usable_logs],
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
        recursive=False,
        out_dir=str(paths.visualize_dir / "residual model"),
        training=str(config["training"]),
        closed_loop_window_s=float(config["closed_loop_window_s"]),
        closed_loop_stride_s=float(config["closed_loop_stride_s"]),
        closed_loop_steps=int(config["closed_loop_steps"]),
        closed_loop_learning_rate=float(config["closed_loop_learning_rate"]),
    )
    diagnostics_path = paths.results_dir / "residual_diagnostics.yaml"
    if diagnostics_path.is_file():
        diagnostics = load_yaml(diagnostics_path)
        diagnostics["excluded_diverged_logs"] = {str(path): round(error, 4) for path, error in diverged_logs}
        save_yaml(diagnostics_path, diagnostics)
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
    from wmr_simulator.gain_tuning.checks import warnings_for as check_warnings
    from wmr_simulator.types import print_controller_gains, print_physical_params
    from wmr_simulator.visualization.gain_tuning import (
        TRAINING_KEY_NAMESPACE,
        plot_controller_tuning_errors,
        plot_gain_tuning_summary,
        plot_training_trajectory_summary,
        plot_tuning_trajectory_histograms,
        plot_validation_trajectory_summary,
        realization_keys_for_set,
        rollout_realizations,
    )
    from wmr_simulator.visualization.identification import plot_loss_history

    paths = experiment.paths(iteration)
    if experiment.config["use_standalone_gain_tuning_defaults"]:
        from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS

        config = GAIN_TUNING_DEFAULTS
        print("Gain tuning: using GAIN_TUNING_DEFAULTS.")
    else:
        config = experiment.config["gain_tuning"]
    # The cross-iteration settings always come from the experiment's gain_tuning block.
    refine = experiment.config["gain_tuning"]
    problem_path = _tuning_problem(paths, experiment.config["tuning_trajectories"])
    if not any(paths.tuning_trajectories_dir.glob("*.pkl")):
        raise FileNotFoundError(
            f"No tuning trajectories in {paths.tuning_trajectories_dir}; run plan-tuning-trajectories first."
        )

    robot_params = resolve_gain_robot_params(str(problem_path), None, None)
    print_physical_params("Robot parameters for gain tuning:", robot_params)
    residual_model = None
    if experiment.config["use_residual_model"] and not refine["use_residual_model"]:
        print("Gain tuning: residual model trained but disabled for tuning "
              "(gain_tuning.use_residual_model); rolling out the nominal plant.")
    elif experiment.config["use_residual_model"] and paths.residual_skipped.is_file() and not paths.residual_model.is_file():
        print(f"Gain tuning: no residual this iteration ({paths.residual_skipped}); rolling out the nominal plant.")
    elif experiment.config["use_residual_model"]:
        if not paths.residual_model.is_file():
            raise FileNotFoundError(
                f"use_residual_model is enabled but {paths.residual_model} is missing; "
                "run train-residual first (or disable the flag in experiment.yaml)."
            )
        from wmr_simulator.residual_model import load_residual_model

        residual_model, checkpoint = load_residual_model(paths.residual_model, robot_params)
        print(f"Loaded residual model {paths.residual_model} (config: {checkpoint['config']})")
    # The static run continues from the previous static gains.
    static_init_gains = None
    if paths.robot_config_static.is_file():
        static_init_gains = [
            float(gain) for gain in load_yaml(paths.robot_config_static)["controller"]["gains"]
        ]
        print(f"Static run starts from the previous iteration's static gains: {static_init_gains}")
    result = run_gain_tuning_experiment(
        problem_path=str(problem_path),
        robot_params=robot_params,
        num_steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        num_realizations=int(config["num_realizations"]),
        seed=int(experiment.config["seed"]),
        reference_trajectories_dir=str(paths.tuning_trajectories_dir),
        validation_split=float(config["validation_split"]),
        position_tracking_weight=float(config.get("position_tracking_weight", 1.0)),
        heading_tracking_weight=float(config.get("heading_tracking_weight", 1.0)),
        linear_velocity_tracking_weight=float(config["linear_velocity_tracking_weight"]),
        angular_velocity_tracking_weight=float(config["angular_velocity_tracking_weight"]),
        input_weight=float(config["input_weight"]),
        input_delta_weight=float(config["input_delta_weight"]),
        omega_delta_weight=float(config.get("omega_delta_weight", 0.0)),
        k_min_stab=float(config["k_min_stab"]),
        k_max_stab=float(config["k_max_stab"]),
        k_max_rest=float(config["k_max_rest"]),
        num_lhs_points=int(config["num_lhs_points"]),
        num_adam_optimizations=int(config["num_adam_optimizations"]),
        optimizer=str(config["optimizer"]),
        outlier_loss_factor=float(config["outlier_loss_factor"]),
        # None follows the problem yaml.
        schedule_enabled=(
            config.get("gain_parametrization")
            if experiment.config["use_gain_parametrization"]
            else False
        ),
        gain_delta_weight=float(config["gain_delta_weight"]),
        static_tune=bool(config["static_tune"]),
        static_tune_steps=int(config["static_tune_steps"]),
        static_tune_learning_rate=float(config["static_tune_learning_rate"]),
        static_init_gains=static_init_gains,
        # The first tuning seeds the parametrized run from the static result
        # and searches the full presearch range.
        seed_parametrization_from_static=iteration <= experiment.first_iteration,
        presearch_relative_range=(
            0.0 if iteration <= experiment.first_iteration else float(refine["presearch_relative_range"])
        ),
        warm_start_schedule=bool(refine["warm_start_schedule"]),
        init_offset_radius=float(config["init_offset_radius"]),
        init_offset_angle=float(config["init_offset_angle"]),
        residual_model=residual_model,
    )
    pipeline = result["pipeline"]
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    if result["static_gains"] is not None:
        print_controller_gains("Static-tune gains (benchmark baseline):", result["static_gains"])
    print(f"Final tuning loss: {float(result['final_loss']):.8f}")

    payload = {
        "gains": [float(gain) for gain in result["optimized_gains"]],
        # Gains of the static run (None: no separate static run).
        "static_gains": (
            None
            if result["static_gains"] is None
            else [float(gain) for gain in result["static_gains"]]
        ),
        "schedule_enabled": bool(result["schedule_enabled"]),
        "schedule": None,
        "used_residual_model": residual_model is not None,
        "final_loss": float(result["final_loss"]),
        "final_validation_loss": (
            None if result["final_validation_loss"] is None else float(result["final_validation_loss"])
        ),
        "static_final_loss": (
            None if result["static_final_loss"] is None else float(result["static_final_loss"])
        ),
        "static_final_validation_loss": (
            None
            if result["static_final_validation_loss"] is None
            else float(result["static_final_validation_loss"])
        ),
        # Stall / box-bound / divergence checks per run (gain_tuning.checks).
        "checks": result["checks"],
        "warnings": check_warnings(result["checks"]),
    }
    schedule_params = result.get("schedule_params")
    if schedule_params is not None:
        from wmr_simulator.gain_parametrization import to_cfg as gain_parametrization_to_cfg

        payload["schedule"] = gain_parametrization_to_cfg(schedule_params)
    save_yaml(paths.gains_result, payload)
    print(f"Wrote {paths.gains_result}")

    with collect_plots(paths.visualize_dir / "gain tuning"):
        # Training trajectory 0, under the noise keys the objective used.
        summary_references = pipeline.training_reference_trajectories[:1]
        summary_offsets = result["summary_start_offsets"][None, ...]
        summary_robot_keys, summary_estimator_keys = realization_keys_for_set(
            result["realizations"],
            int(pipeline.training_reference_trajectories.shape[0]),
            TRAINING_KEY_NAMESPACE,
        )
        summary_robot_keys = summary_robot_keys[:1]
        summary_estimator_keys = summary_estimator_keys[:1]
        plot_gain_tuning_summary(
            pipeline,
            init_log=result["init_hidden_log"],
            tuned_log=result["final_hidden_log"],
            static_log=result.get("static_hidden_log"),
            init_realization_poses=rollout_realizations(
                pipeline,
                robot_params,
                summary_references,
                summary_offsets,
                summary_robot_keys,
                summary_estimator_keys,
                controller_gains=result["init_gains"],
                schedule_params=result["init_schedule_params"],
            )[0],
            tuned_realization_poses=rollout_realizations(
                pipeline,
                robot_params,
                summary_references,
                summary_offsets,
                summary_robot_keys,
                summary_estimator_keys,
                controller_gains=result["optimized_gains"],
                schedule_params=result["schedule_params"],
            )[0],
            static_realization_poses=(
                None
                if result["static_gains"] is None
                else rollout_realizations(
                    pipeline,
                    robot_params,
                    summary_references,
                    summary_offsets,
                    summary_robot_keys,
                    summary_estimator_keys,
                    controller_gains=result["static_gains"],
                )[0]
            ),
            out_prefix="summary_gain_tuning",
        )
        plot_training_trajectory_summary(
            pipeline,
            robot_params=robot_params,
            tuned_gains=result["optimized_gains"],
            start_offsets=result["training_start_offsets"],
            realizations=result["realizations"],
            schedule_params=result["schedule_params"],
            static_gains=result["static_gains"],
            init_gains=result["init_gains"],
            init_schedule_params=result["init_schedule_params"],
            max_trajectories=None,
            out_prefix="summary_training",
        )
        plot_validation_trajectory_summary(
            pipeline,
            robot_params=robot_params,
            tuned_gains=result["optimized_gains"],
            start_offsets=result["validation_start_offsets"],
            realizations=result["realizations"],
            schedule_params=result["schedule_params"],
            static_gains=result["static_gains"],
            init_gains=result["init_gains"],
            init_schedule_params=result["init_schedule_params"],
            out_prefix="summary_validation",
        )
        plot_tuning_trajectory_histograms(pipeline)
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
        if result["static_loss_history"] is not None:
            plot_loss_history(
                loss_history=result["static_loss_history"],
                validation_loss_history=result["static_validation_loss_history"],
                loss_component_history=result["static_loss_component_history"],
                validation_loss_component_history=result["static_validation_loss_component_history"],
                out_prefix="ctrl_tuning_static",
            )

    # Cross-iteration progress plot (best effort).
    try:
        from wmr_simulator.active_learning.progress import evaluate_pipeline_progress
        from wmr_simulator.visualization.pipeline_progress import plot_pipeline_progress

        progress_records = evaluate_pipeline_progress(experiment)
        progress_path = plot_pipeline_progress(
            progress_records, experiment.root, out_path=paths.visualize_dir / "pipeline_progress.pdf"
        )
        print(f"Wrote {progress_path}")
    except Exception as error:
        print(f"Pipeline progress plot skipped ({error}).")

    _plot_baseline_runs(experiment, iteration)
    return payload


def plot_baseline_runs_per_iteration(experiment: Experiment) -> list[str]:
    """Rewrite every iteration's benchmark figures from the recordings on disk."""
    from wmr_simulator.active_learning.baseline_runs import collect_baseline_runs

    try:
        collected = collect_baseline_runs(experiment)
    except Exception as error:
        print(f"Baseline runs plots skipped ({error}).")
        return []
    written: list[str] = []
    for iteration in experiment.iteration_indices():
        written.extend(_plot_baseline_runs(experiment, iteration, collected=collected))
    return written


def _plot_baseline_runs(experiment: Experiment, iteration: int, collected: dict | None = None) -> list[str]:
    """One figure per benchmark reference over the runs of every iteration up
    to this one (best effort). ``collected`` reuses a ``collect_baseline_runs``
    result."""
    written: list[str] = []
    paths = experiment.paths(iteration)
    try:
        from wmr_simulator.active_learning.baseline_runs import collect_baseline_runs, variant_panels
        from wmr_simulator.visualization.baseline_runs import plot_baseline_runs

        if collected is None:
            collected = collect_baseline_runs(experiment)
        collected = {
            shape: [record for record in records if record.index <= iteration]
            for shape, records in collected.items()
        }
        collected = {shape: records for shape, records in collected.items() if records}
        if not collected:
            return written
        output_dir = paths.visualize_dir / "baseline runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        for shape, records in sorted(collected.items()):
            plot_path = plot_baseline_runs(
                records,
                variant_panels(records),
                output_dir / f"baseline_runs_{shape}.pdf",
                shape=shape,
            )
            written.append(plot_path)
            print(f"Wrote {plot_path}")
    except Exception as error:
        print(f"Baseline runs plot skipped ({error}).")
    return written


# ---------------------------------------------------------------------------
# status / orchestration
# ---------------------------------------------------------------------------


# The stages an iteration has to get through, with the output that marks each done.
REQUIRED_STAGE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("plan-id-trajectory", "identification_trajectory/*.pkl + .JSN"),
    ("decode-logs", "data/*.csv"),
    ("identify", "results/identification.yaml"),
    ("train-residual", "results/residual_model.pkl"),
    ("plan-tuning-trajectories", "tuning_trajectories/*.pkl"),
    ("tune-gains", "results/gains.yaml"),
)

# Stages an iteration is complete without.
OPTIONAL_STAGE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("benchmark", "data/benchmark[_static]/TRxx"),
)

# The stages of iteration 0 (prior tuning).
PRIOR_STAGES = ("plan-tuning-trajectories", "tune-gains")


def iteration_status(experiment: Experiment, iteration: int) -> dict[str, bool]:
    paths = experiment.paths(iteration)
    has_id_trajectory = any(paths.identification_trajectory_dir.glob("*.pkl")) and any(
        paths.identification_trajectory_dir.glob("*.JSN")
    )
    return {
        "plan-id-trajectory": has_id_trajectory,
        "decode-logs": bool(_list_log_csvs(paths)),
        "identify": paths.identification_result.is_file(),
        "train-residual": (
            paths.residual_model.is_file()
            or paths.residual_skipped.is_file()
            or not experiment.config["use_residual_model"]
        ),
        "plan-tuning-trajectories": any(paths.tuning_trajectories_dir.glob("*.pkl")),
        "tune-gains": paths.gains_result.is_file(),
        "benchmark": _benchmark_recorded(paths, experiment.config["benchmark"]),
    }


def stage_status(experiment: Experiment) -> None:
    deployment = experiment.config["mujoco_deployment"]
    print(f"Experiment: {experiment.root}")
    print(f"  residual model:          {'enabled' if experiment.config['use_residual_model'] else 'disabled'}")
    print(
        "  gain parametrization:    "
        f"{'problem yaml' if experiment.config['use_gain_parametrization'] else 'off (static gains only)'}"
    )
    for label, flag, baseline in (
        ("identification", "optimize_identification_trajectory", "baseline_identification_trajectory"),
        ("tuning", "optimize_tuning_trajectories", "baseline_tuning_trajectories_dir"),
    ):
        source = "designed" if experiment.config[flag] else f"fixed ({experiment.config[baseline]})"
        print(f"  {label + ' design:':25s}{source}")
    print(
        "  residual in design:      "
        f"identification {'on' if _residual_in_design(experiment, experiment.config['identification_trajectory']) else 'off'}, "
        f"tuning {'on' if _residual_in_design(experiment, experiment.config['tuning_trajectories']) else 'off'}"
    )
    print(
        "  residual in gain tuning: "
        f"{'on' if _residual_in_design(experiment, experiment.config['gain_tuning']) else 'off'}"
    )
    print(
        "  data collection:         "
        + (
            f"MuJoCo deployment, {int(deployment['num_logs'])} log(s) per iteration"
            if deployment["enabled"]
            else "robot (SD card)"
        )
    )
    print(
        f"  benchmark:               {int(experiment.config['benchmark']['num_runs'])} run(s) of "
        f"{_benchmark_set_description(experiment.config['benchmark'])} per iteration"
    )
    print(
        f"  iterations:              {len(experiment.iteration_indices())} of "
        f"{int(experiment.config['num_iterations'])} targeted"
    )
    for iteration in experiment.iteration_indices():
        print(f"iteration_{iteration:02d}:")
        status = iteration_status(experiment, iteration)
        for stage, description in REQUIRED_STAGE_OUTPUTS + OPTIONAL_STAGE_OUTPUTS:
            if iteration == 0 and stage not in PRIOR_STAGES:
                continue
            marker = "x" if status[stage] else " "
            print(f"  [{marker}] {stage:<26} {description}")


def stage_run(
    experiment: Experiment,
    iteration: int | None = None,
    log: str | None = None,
    simulate_deployment: bool = False,
    num_iterations: int | None = None,
) -> int:
    """Run the loop from ``iteration`` on, and return the last iteration finalized.

    ``num_iterations`` (default: the experiment's) is the total to reach. Without
    a MuJoCo deployment the loop stops at the first iteration that needs robot
    data. Naming an ``iteration`` past the target raises the target to it.
    """
    target = int(experiment.config["num_iterations"] if num_iterations is None else num_iterations)
    if iteration is not None:
        target = max(target, int(iteration))
    iteration = experiment.resolve_iteration(iteration)
    if iteration > target:
        print(f"Experiment already has its {target} iteration(s); "
              f"{experiment.paths(iteration).root} holds the final model and gains.")
        print("Raise num_iterations in experiment.yaml (or pass --iterations) to continue the loop.")
        return iteration - 1

    while True:
        if not _run_iteration(
            experiment,
            iteration,
            log=log,
            simulate_deployment=simulate_deployment,
        ):
            return iteration - 1
        log = None
        if iteration >= target:
            if target > 1:
                print()
                print(f"Reached the {target}-iteration target; "
                      f"{experiment.paths(iteration + 1).root} holds the final model and gains.")
            return iteration
        iteration += 1


@contextlib.contextmanager
def logged_stage(paths: IterationPaths, stage: str):
    """Record the stage's start before it runs and its finish after (a start
    without a finish marks possibly partial outputs)."""
    import datetime

    def now() -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def save(log: dict) -> None:
        temporary = paths.stage_log.with_suffix(".yaml.tmp")
        save_yaml(temporary, log)
        os.replace(temporary, paths.stage_log)

    log = (load_yaml(paths.stage_log) if paths.stage_log.is_file() else {}) or {}
    log[stage] = {"started": now(), "finished": None, "seconds": None}
    save(log)
    start = time.monotonic()
    yield
    log = load_yaml(paths.stage_log) or {}
    log[stage] = {**log.get(stage, {}), "finished": now(), "seconds": round(time.monotonic() - start, 1)}
    save(log)


def _run_iteration(
    experiment: Experiment,
    iteration: int,
    log: str | None = None,
    simulate_deployment: bool = False,
) -> bool:
    """Run every stage of one iteration that can proceed, in order.

    Returns whether the iteration was finalized; False when it waits for robot data.
    """
    paths = experiment.paths(iteration)
    status = iteration_status(experiment, iteration)
    if iteration == 0:
        return _run_prior_iteration(experiment, status)

    # Benchmark logs may arrive late, so every iteration is rescanned.
    plot_run_directory_logs(experiment)
    plot_baseline_runs_per_iteration(experiment)

    if not status["plan-id-trajectory"]:
        with logged_stage(paths, "plan-id-trajectory"):
            stage_plan_identification_trajectory(experiment, iteration)
        status = iteration_status(experiment, iteration)

    # Benchmark the controller this iteration deployed.
    deploys_in_simulation = simulate_deployment or experiment.config["mujoco_deployment"]["enabled"]
    benchmark_directories = (
        [] if deploys_in_simulation else _create_benchmark_directories(paths, experiment.config["benchmark"])
    )
    if deploys_in_simulation and not status["benchmark"]:
        with logged_stage(paths, "benchmark"):
            stage_run_benchmark(experiment, iteration)

    if not status["decode-logs"]:
        # Logs on disk are decoded, never topped up by another deployment.
        if deploys_in_simulation and not _run_log_paths(paths.data_dir):
            with logged_stage(paths, "simulate-deployment"):
                stage_simulate_deployment(experiment, iteration)
        with logged_stage(paths, "decode-logs"):
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
        if benchmark_directories:
            print("  Benchmark runs of this iteration's controller (any time, also after it is finalized):")
            for directory in benchmark_directories:
                print(f"     {directory}")
        print("  (or run it with --simulate-deployment to have MuJoCo stand in for the robot.)")
        return False

    if not status["identify"]:
        with logged_stage(paths, "identify"):
            stage_identify(experiment, iteration, log=log)
    if not iteration_status(experiment, iteration)["train-residual"]:
        with logged_stage(paths, "train-residual"):
            stage_train_residual(experiment, iteration)
    if not status["plan-tuning-trajectories"]:
        with logged_stage(paths, "plan-tuning-trajectories"):
            stage_plan_tuning_trajectories(experiment, iteration)
    if not iteration_status(experiment, iteration)["tune-gains"]:
        with logged_stage(paths, "tune-gains"):
            stage_tune_gains(experiment, iteration)
    with logged_stage(paths, "finalize"):
        stage_finalize(experiment, iteration)
    return True


def _run_prior_iteration(experiment: Experiment, status: dict[str, bool]) -> bool:
    """Iteration 0: design a tuning set and tune on the prior model, then hand
    the gains to iteration 1 (experiment.yaml prior_tuning)."""
    paths = experiment.paths(0)
    if not status["plan-tuning-trajectories"]:
        with logged_stage(paths, "plan-tuning-trajectories"):
            stage_plan_tuning_trajectories(experiment, 0)
    if not iteration_status(experiment, 0)["tune-gains"]:
        with logged_stage(paths, "tune-gains"):
            stage_tune_gains(experiment, 0)
    with logged_stage(paths, "finalize"):
        stage_finalize(experiment, 0)
    return True


def _problem_at_sim_time(source: Path, sim_time: float, name: str, label: str) -> Path:
    """``source`` with ``sim_time`` overridden, written next to it as ``name``
    (``source`` itself when the override is off or equal)."""
    sim_time = float(sim_time or 0.0)
    if sim_time <= 0.0:
        return source
    problem = load_yaml(source)
    own_sim_time = problem.get("sim_time")
    if float(own_sim_time or 0.0) == sim_time:
        return source
    problem["sim_time"] = sim_time
    target = source.with_name(name)
    save_yaml(target, problem)
    print(f"{label} runs at sim_time {sim_time} s (the problem yaml's own is {own_sim_time}).")
    return target


def _identification_problem(paths: IterationPaths, config: dict) -> Path:
    """The problem the identification design runs against (sim_time = the
    sum of the phase durations)."""
    sim_time = sum(float(phase["duration"]) for phase in config["phases"])
    return _problem_at_sim_time(
        paths.problem, sim_time, "problem_identification.yaml", "Identification design",
    )


def _tuning_problem(paths: IterationPaths, config: dict) -> Path:
    """The problem the tuning design and the gain tuner share
    (``tuning_trajectories.sim_time`` overrides the identified problem's)."""
    return _problem_at_sim_time(
        _identified_problem(paths), config.get("sim_time"), "problem_tuning.yaml",
        "Tuning half",
    )


def _identified_problem(paths: IterationPaths) -> Path:
    if not paths.problem_identified.is_file():
        raise FileNotFoundError(
            f"Missing {paths.problem_identified}; run the identify stage first "
            "(downstream stages must use the identified robot model)."
        )
    return paths.problem_identified
