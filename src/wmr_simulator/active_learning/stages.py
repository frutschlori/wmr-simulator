"""Modular stages of the iterative active-learning loop.

Each stage reads its inputs and writes its outputs as files inside the
iteration folder (see experiment.py for the layout), so the loop survives the
interruptions needed to collect robot data from the SD card. Stage order:

    init -> plan-id-trajectory -> [run robot, copy SD logs into data/]
         -> benchmark -> decode-logs -> identify -> train-residual
         -> plan-tuning-trajectories -> tune-gains -> finalize (next iteration)

The bracketed step is the only one that is not a stage, because on the real
robot it is a person with an SD card. simulate-deployment replaces it with a
MuJoCo run of the same trajectory (mujoco_sim), which is what lets `run` carry
an experiment through several iterations on its own. benchmark is the other
half of that stand-in: the same plant driving a *fixed* baseline reference, so
every iteration is scored on something that did not change with it.

Heavy imports (jax, the pipelines) happen inside the stage functions so the
CLI stays responsive for bookkeeping commands like status.
"""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path

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

    Same identified robot parameters, but the gains of the static tuning run
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


def _load_design_residual_model(path: Path):
    """Residual checkpoint a trajectory-design stage rolls out on, or None.

    A missing checkpoint designs on the nominal plant instead of raising: an
    experiment with use_residual_model off never trains one at all.
    """
    if not path.is_file():
        print(f"No residual model at {path}; designing on the nominal plant.")
        return None

    from wmr_simulator.residual_model import load_residual_model

    residual_model, checkpoint = load_residual_model(path)
    print(f"Designing on the residual-augmented plant: {path}")
    print(f"  config: {checkpoint['config']}")
    return residual_model


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

        # The previous iteration's residual: this stage runs before the robot
        # has driven anything for *this* iteration, so its own residual does not
        # exist yet. Iteration 1 has no previous one either and designs nominal.
        residual_model = None
        if config["use_residual_model"] and iteration > 1:
            residual_model = _load_design_residual_model(experiment.paths(iteration - 1).residual_model)
        pipeline = TrajectoryOptimizationPipeline(
            str(paths.problem),
            time_scaling=config["time_scaling"],
            objective_mode="identification",
            fim_a_slip_max=bool(config["fim_a_slip_max"]),
            # This design's own motion envelope, gentler than the one the
            # tuning designer works in: the identification trajectory is placed
            # by hand and driven on the robot, not rolled out in sim.
            motion_limits=config["motion_limits"],
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

    # kimotor's FIM column is scaled by a pinned constant rather than by the
    # design point (see tuning_trajectories.kimotor_fim_scale): the tuned gains
    # this iteration inherits usually have kimotor = 0, where a tied scale
    # leaves the design blind to it and the trajectories dull. The gains
    # themselves are untouched -- the iteration's problem.yaml, the tune-gains
    # stage and everything exported to the robot keep the tuned kimotor.
    #
    # This iteration's own residual: train-residual runs between identify and
    # this stage, so the model is fitted to the logs recorded for it.
    residual_model = None
    if config["use_residual_model"]:
        residual_model = _load_design_residual_model(paths.residual_model)
    pipeline = TrajectoryOptimizationPipeline(
        str(problem_path),
        time_scaling=config["time_scaling"],
        objective_mode="gain-tuning",
        kimotor_fim_scale=float(config["kimotor_fim_scale"]),
        start_offset_mode=str(config["start_offset_mode"]),
        residual_model=residual_model,
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
            verbose=False,
        )
        final_losses = np.asarray(pipeline.batch_final_losses, dtype=float)
        print(f"Optimized {len(control_point_batch)} tuning trajectories "
              f"(final losses {final_losses.min():.4e} .. {final_losses.max():.4e})")

    clamped_batch = [pipeline.clamp_control_points(points) for points in control_point_batch]
    # Check the whole batch before writing any of it: the per-export check
    # inside save_reference_states_pickle would abort partway and leave a
    # half-written directory, and one stalled design usually means the run's
    # settings are wrong for every design.
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
                    # The curve itself, not just its samples, so a warm start
                    # picks up the decision variables directly.
                    control_points=clamped_control_points,
                    # Each trajectory ships the offsets it was designed under,
                    # which the gain tuner then tunes on. The batch path designs
                    # them per trajectory; a single design has only the shared
                    # bundle, which is what the default already writes.
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
    """Stand in for the robot: drive this iteration's identification trajectory
    in the MuJoCo plant and write binary ``TRxx`` logs into ``data/``.

    Replaces "copy the JSN and ROBOTCFG.CFG to the SD card, run the robot, copy
    the logs back". Nothing downstream can tell the difference: the logs are in
    the firmware's own binary format, in the directory decode-logs already
    reads, and the plant's true parameters stay inside ``mujoco_sim`` -- so
    identification still has something to find.

    The runs are consecutive, as they are on the robot. What is driven is the
    **bridged** JSN -- the trajectory, a wait, then a planned path back to the
    start -- so only the first run is placed by hand and every later one begins
    wherever the previous one's bridge left the robot. That reproduces the real
    repeat procedure, including the fact that the placement of run 5 is
    whatever four bridges accumulated to. A run that ends further than
    ``mujoco_deployment.divergence_radius`` from the trajectory's start point
    did not come back, so the next one is placed by hand again instead -- the
    same rule the benchmark stage chains under, and the same thing a person
    would do standing next to the robot. The log loader clips the bridge back
    off again (``log_loading.clip_after_first_trajectory``, which keys on the
    wait's zero setpoint), so identification sees the trajectory alone.

    The ground truth the deployment does know -- tracking error against the
    reference, duty saturation, where the robot started and ended -- is printed
    and *not* written next to the log: decode-logs tries to decode every
    non-csv file in ``data/``, and it would leak the hidden truth into the
    pipeline besides.
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
        # A run the bridge brought back close enough to the start is where the
        # next one begins; one that did not is a robot standing somewhere else,
        # so it is picked up and placed again.
        start_pose = None if diverged else result.final_pose
        offset = result.start_offset
        # Tenths of a millimetre, distances in mm alongside them: a bridge that
        # ends its min-jerk creep on the start point regularly lands inside a
        # millimetre, and rounding that away reads as an implausible exact zero.
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
    """The reference the deployment drives: this iteration's *bridged* JSN.

    The bridge is what makes a repeat run possible without touching the robot,
    which is what lets the deployments be chained; the unbridged variant is
    only exported because the bridged one is derived from it.
    """
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


def _deployment_seed(experiment: Experiment, iteration: int, index: int) -> int:
    """Distinct per (iteration, run) and unchanged across reruns, so a repeated
    deployment reproduces the same placements and sensor noise."""
    return 1_000_000 * int(experiment.config["seed"]) + 1_000 * int(iteration) + int(index)


# A reference whose last pose is this close to its first one already returns
# the robot to where it started, so repeat runs can be chained without a bridge
# path. Loose enough for a hand-authored baseline that closes only to plotting
# precision, tight enough that a genuinely open path is bridged.
SELF_CLOSING_POSITION_TOLERANCE = 0.05  # m
SELF_CLOSING_HEADING_TOLERANCE = 0.15  # rad


def stage_run_benchmark(
    experiment: Experiment,
    iteration: int,
    num_runs: int | None = None,
) -> list[Path]:
    """Record this iteration's runs of the fixed baseline reference.

    Every iteration drives the *same* trajectory (``benchmark.trajectory``)
    under its own controller, so the progress plot has one thing that does not
    move: a difference between two iterations' benchmark runs is a difference
    in the controller, never in what it was asked to track. The iteration's own
    identification and tuning trajectories cannot answer that question, because
    they are redesigned every iteration.

    Both controller options an iteration ships are driven when it has two (see
    ``_benchmark_variant_specs``): the deployed one, whose gain parametrization
    the runs therefore include, into ``data/benchmark/``, and the static-gain
    baseline into ``data/benchmark_static/``. They run on the *same* seeds, so
    the hand placements and the sensor noise are paired and the difference
    between the two sets is the controller and nothing else. Iteration 1 ships
    only the stock static controller and records only that set.

    The runs are consecutive and chained exactly like the identification
    deployment, and like a repeat on the real robot: only the first is placed by
    hand, and each later one starts wherever the previous run ended. A run that
    ends further than ``benchmark.divergence_radius`` from the reference's start
    point did not come back, so the next one is placed by hand again instead --
    the same thing a person would do standing next to the robot.

    Chaining needs the reference to end where it begins. A self-closing one
    (both shipped baselines are) is driven as it is; anything else gets a wait
    plus a planned bridge path back to the start appended
    (``pololu.bridge_exporter``), which is how the identification trajectory is
    repeated too.

    The logs go into subdirectories of ``data/``: the identify and residual
    stages only look one level deep, so benchmark runs never leak into the
    identification data. They stay in the firmware's binary format the way
    SD-card benchmark recordings do; the progress evaluation and the baseline
    comparison plot decode them into a temporary directory when they need them.
    """
    paths = experiment.paths(iteration)
    paths.create_directories()
    config = experiment.config["benchmark"]
    count = int(config["num_runs"] if num_runs is None else num_runs)

    trajectory = _benchmark_reference(paths, config)
    reference_start = _reference_start_pose(trajectory)

    # Ground truth the logs do not carry, kept out of data/ so nothing
    # downstream can read the hidden plant through it (mujoco_sim.deploy). One
    # variant's recording is left alone when the other one is added later.
    report = load_yaml(paths.benchmark_result) if paths.benchmark_result.is_file() else {}
    report["trajectory"] = str(trajectory)
    report["divergence_radius"] = float(config["divergence_radius"])
    report.setdefault("variants", {})

    written: list[Path] = []
    recorded_any = False
    for variant, robot_config_source, data_dir in _benchmark_variant_specs(paths):
        existing = _run_log_paths(data_dir)
        if existing:
            print(f"Benchmark ({variant}) already recorded ({len(existing)} run(s) in {data_dir}); skipping.")
            written.extend(existing)
            continue
        robot_config = _staged_benchmark_config(paths, variant, robot_config_source)
        variant_written, runs = _record_benchmark_runs(
            experiment, iteration, variant, robot_config, trajectory, data_dir, count, config, reference_start
        )
        written.extend(variant_written)
        report["variants"][variant] = {
            "robot_config": robot_config_source.name,
            "data": str(data_dir.relative_to(paths.root)),
            "num_diverged": sum(1 for run in runs if run["diverged"]),
            "mean_tracking_rmse": (
                float(sum(run["tracking_rmse"] for run in runs) / len(runs)) if runs else None
            ),
            "runs": runs,
        }
        recorded_any = True

    if recorded_any:
        save_yaml(paths.benchmark_result, report)
        print(f"Benchmark ground truth: {paths.benchmark_result}")
    return written


def _record_benchmark_runs(
    experiment: Experiment,
    iteration: int,
    variant: str,
    robot_config: Path,
    trajectory: Path,
    data_dir: Path,
    count: int,
    config: dict,
    reference_start: tuple[float, float, float],
) -> tuple[list[Path], list[dict]]:
    """Drive ``count`` chained runs of ``trajectory`` under one controller."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment
    from wmr_simulator.mujoco_sim.firmware import GAIN_MLP_FILENAME

    divergence_radius = float(config["divergence_radius"])
    # Named rather than implied: a static-variant run *with* a network beside it
    # is iteration 1, where that network is still the identity, and the line
    # would otherwise read as a contradiction.
    network = ""
    if (robot_config.parent / GAIN_MLP_FILENAME).is_file():
        identity = " (identity)" if variant == baseline_runs.STATIC_VARIANT else ""
        network = f" + {GAIN_MLP_FILENAME}{identity}"
    print(
        f"Benchmark ({variant}): {count} consecutive run(s) of {trajectory.name} "
        f"under {robot_config.name}{network} -> {data_dir}"
    )
    written: list[Path] = []
    runs: list[dict] = []
    start_pose = None  # the first run is placed by hand
    for index in range(count):
        seed = _benchmark_seed(experiment, iteration, index)
        result = run_deployment(
            robot_config,
            trajectory,
            data_dir,
            seed=seed,
            start_pose=start_pose,
            start_offset_radius=float(config["start_offset_radius"]),
            start_offset_angle=float(config["start_offset_angle"]),
        )
        written.append(result.log_path)
        placement = "placed by hand" if start_pose is None else "left by the previous run"
        distance_to_start = math.dist(result.final_pose[:2], reference_start[:2])
        diverged = distance_to_start > divergence_radius
        # A run that came back close enough to the start is where the next one
        # begins; one that did not is a robot standing somewhere else, so it is
        # picked up and placed again.
        start_pose = None if diverged else result.final_pose
        offset = result.start_offset
        print(
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
                "distance_to_start": float(distance_to_start),
                "diverged": bool(diverged),
                "max_duty": float(result.max_duty),
                "duty_saturated_fraction": float(result.duty_saturated_fraction),
            }
        )
    return written, runs


def _benchmark_variant_specs(paths: IterationPaths) -> list[tuple[str, Path, Path]]:
    """``(variant, robot config, data dir)`` per controller this iteration ships.

    Always a static-gain set, in ``data/benchmark_static/``. **Iteration 1's
    deployed controller *is* that set**: ``problem.yaml``'s gains are the stock
    static ones and the exported parametrization is exactly the identity (its
    output layer is zero, so every factor is ``clip(1 + 0, ...)`` = 1.0), so its
    runs are driven from ``ROBOTCFG.CFG`` where it lives, network beside it and
    all. Nothing tuned exists yet to compare them against -- the first tuned
    parametrization ships in iteration 2 -- so iteration 1 records one set and
    it belongs on the static side of the comparison.

    From iteration 2 the iteration also carries ``ROBOTCFG_static.CFG``, which
    ``finalize`` writes whenever the previous tuning ran an independent static
    tune beside a parametrized one. Its presence *is* the question "was the gain
    parametrization worth it": the deployed (parametrized) controller then gets
    its own set in ``data/benchmark/`` and the static side switches to those
    static-tuned gains. An experiment with the parametrization off never gets a
    second entry, correctly -- its deployed controller is static in every
    iteration.

    The static side is always the gains some run converged to, never the
    parametrized run's base gains with the network taken away: those base gains
    are not a controller anybody tuned (the factors they are tuned against range
    over ``[MIN_FACTOR, bound]``, so the base gains absorb whatever scale the
    network leaves them), and driving them bare would benchmark an artifact
    instead of the alternative actually on offer.
    """
    if not paths.robotcfg_static_cfg.is_file():
        return [(baseline_runs.STATIC_VARIANT, paths.robotcfg_cfg, paths.benchmark_static_data_dir)]
    return [
        (baseline_runs.STATIC_VARIANT, paths.robotcfg_static_cfg, paths.benchmark_static_data_dir),
        (baseline_runs.PARAMETRIZED_VARIANT, paths.robotcfg_cfg, paths.benchmark_data_dir),
    ]


def _staged_benchmark_config(paths: IterationPaths, variant: str, source: Path) -> Path:
    """The robot config as the benchmark driver reads it off its own "SD card".

    The firmware picks up a ``GAINMLP.JSN`` sitting *next to* the config
    (``mujoco_sim.firmware.FirmwareConfig.from_file``, mirroring ``sdlog.rs``),
    so the static variant cannot be driven from the iteration root -- the
    deployed network is right there and would be applied to it. It is copied
    into its own directory under ``benchmark/`` instead, which is exactly the
    swap the real SD card needs (ROBOTCFG_static.CFG as ROBOTCFG.CFG, with no
    GAINMLP.JSN beside it). The deployed config is driven where it lives.
    """
    if source == paths.robotcfg_cfg:
        return source
    staged_dir = paths.benchmark_dir / variant
    staged_dir.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(source, staged_dir / paths.robotcfg_cfg.name))


def _benchmark_reference(paths: IterationPaths, config: dict) -> Path:
    """The baseline JSN as this iteration drives it, copied into ``benchmark/``.

    Bridged when the reference does not close on itself, since chaining a
    repeat run onto the previous one's end pose only works if that end pose is
    the start pose. The copy is per iteration so the directory records what was
    actually driven, even if ``benchmark.trajectory`` is edited later.
    """
    source = Path(config["trajectory"])
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
    """``(x, y, yaw)`` a chained reference starts (and, being chainable, ends)
    at -- what a run's final pose is measured against."""
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    start = load_pololu_reference(trajectory).states[0]
    return (float(start[0]), float(start[1]), float(start[2]))


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


def _benchmark_recorded(paths: IterationPaths) -> bool:
    """Whether every controller variant of this iteration has its runs.

    Both sets are needed for the comparison, so an iteration that recorded only
    the deployed controller is not done -- rerunning the stage adds the missing
    variant without touching the one it has.
    """
    return all(_run_log_paths(data_dir) for _, _, data_dir in _benchmark_variant_specs(paths))


def _benchmark_seed(experiment: Experiment, iteration: int, index: int) -> int:
    """Distinct per (iteration, run) and disjoint from the identification
    deployment's seeds, so the two never draw the same placement or noise.

    Deliberately *not* distinct per controller variant: the two variants are a
    paired comparison, so run ``i`` of each gets the same hand placement and the
    same sensor noise (common random numbers), leaving the controller as the only
    difference between them.
    """
    return _deployment_seed(experiment, iteration, index) + 500


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
    if logs:
        # Redrawn every time, unlike the per-log plots: it is the whole set, so a
        # newly decoded log changes it.
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
        TRAINING_KEY_NAMESPACE,
        plot_controller_tuning_errors,
        plot_gain_tuning_summary,
        plot_training_trajectory_summary,
        plot_validation_trajectory_summary,
        realization_keys_for_set,
        rollout_realizations,
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

    # Two flags, because training a residual and tuning against one are separate
    # decisions: the model is worth fitting and inspecting every iteration even
    # when it is not trusted to drive the gains (a residual fitted to logs whose
    # duty saturated carries a large unexplained yaw term, and the tuner would
    # roll that out as if it were the robot).
    residual_model = None
    if experiment.config["use_residual_model"] and not refine["use_residual_model"]:
        print("Gain tuning: residual model trained but disabled for tuning "
              "(gain_tuning.use_residual_model); rolling out the nominal plant.")
    elif experiment.config["use_residual_model"]:
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
    # The static run is an independent controller option, so it refines the
    # previous iteration's *static* gains (robot_config_static_gains.yaml,
    # written by finalize) rather than the parametrized run's base gains, which
    # are what the iteration problem carries.
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
        velocity_tracking_weight=float(config["velocity_tracking_weight"]),
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
        schedule_enabled=config.get("gain_parametrization", config.get("gain_schedule")),
        gain_delta_weight=float(config["gain_delta_weight"]),
        static_tune=bool(config["static_tune"]),
        static_tune_steps=int(config["static_tune_steps"]),
        static_tune_learning_rate=float(config["static_tune_learning_rate"]),
        static_init_gains=static_init_gains,
        # Iteration 1 has no trained parametrization to warm-start from, so both
        # presearches are the same evaluation; hand the parametrized run the
        # static run's converged gains as extra candidates instead of letting it
        # start from raw LHS winners. Later iterations keep the two lineages apart.
        seed_parametrization_from_static=iteration <= 1,
        # Iteration 1 has no prior result to refine from: search the full
        # presearch range instead of a band around the base gains.
        presearch_relative_range=0.0 if iteration <= 1 else float(refine["presearch_relative_range"]),
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
        # Gains of the independent static run (no parametrization); finalize
        # exports them as the iteration's ROBOTCFG_static.CFG baseline. None
        # when there was no static run (parametrization or static_tune off).
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
        # Same losses for the independent static run, so the two controller
        # options can be compared without rerunning the stage.
        "static_final_loss": (
            None if result["static_final_loss"] is None else float(result["static_final_loss"])
        ),
        "static_final_validation_loss": (
            None
            if result["static_final_validation_loss"] is None
            else float(result["static_final_validation_loss"])
        ),
    }
    schedule_params = result.get("schedule_params")
    if schedule_params is not None:
        from wmr_simulator.gain_parametrization import to_cfg as gain_parametrization_to_cfg

        payload["schedule"] = gain_parametrization_to_cfg(schedule_params)
    save_yaml(paths.gains_result, payload)
    print(f"Wrote {paths.gains_result}")

    with collect_plots(paths.visualize_dir / "gain tuning"):
        # Trajectory 0 of the training set, under the keys the objective scored
        # it with -- split over the whole set, then sliced.
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
        # The static run is independent, so it gets its own curve instead of
        # being prepended to the parametrized one.
        if result["static_loss_history"] is not None:
            plot_loss_history(
                loss_history=result["static_loss_history"],
                validation_loss_history=result["static_validation_loss_history"],
                loss_component_history=result["static_loss_component_history"],
                validation_loss_component_history=result["static_validation_loss_component_history"],
                out_prefix="ctrl_tuning_static",
            )

    # Cross-iteration progress plot, regenerated from every iteration up to
    # this one (see active_learning.progress.evaluate_pipeline_progress) and
    # written straight into *this* iteration's visualize dir -- not through
    # collect_plots, since it never touches the shared repo-root visualize/.
    # A missing benchmark recording or a plotting failure must never abort the
    # stage that just produced this iteration's gains, so this is best-effort.
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

    _plot_baseline_runs(experiment, paths)
    return payload


def _plot_baseline_runs(experiment: Experiment, paths: IterationPaths) -> list[str]:
    """Overlay every iteration's runs of each held-out baseline reference.

    One figure per baseline shape, drawn from all iterations up to this one and
    written into *this* iteration's visualize dir, so each iteration keeps the
    comparison as it stood when it finished. Best-effort for the same reason the
    progress plot is: a missing or unreadable recording must not abort the stage
    that just produced this iteration's gains.
    """
    written: list[str] = []
    try:
        from wmr_simulator.active_learning.baseline_runs import collect_baseline_runs, variant_panels
        from wmr_simulator.visualization.baseline_runs import plot_baseline_runs

        collected = collect_baseline_runs(experiment)
        if not collected:
            return written
        for shape, records in collected.items():
            plot_path = plot_baseline_runs(
                records,
                variant_panels(records),
                paths.visualize_dir / f"baseline_runs_{shape}.pdf",
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


# The stages an iteration has to get through, with the output that marks each
# one done. simulate-deployment is not in here: it is one of two ways to fill
# data/ (the other being the SD card), and decode-logs is what records that the
# data arrived either way.
REQUIRED_STAGE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("plan-id-trajectory", "identification_trajectory/*.pkl + .JSN"),
    ("decode-logs", "data/*.csv"),
    ("identify", "results/identification.yaml"),
    ("train-residual", "results/residual_model.pkl"),
    ("plan-tuning-trajectories", "tuning_trajectories/*.pkl"),
    ("tune-gains", "results/gains.yaml"),
)

# Stages an iteration is complete without. The benchmark only scores the
# controller this iteration deployed -- nothing downstream reads it, so a
# missing one costs the progress plot a point and nothing else. It also needs a
# plant to drive, which an experiment collecting real robot data does not have.
OPTIONAL_STAGE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("benchmark", "data/benchmark[_static]/TRxx"),
)


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
        "benchmark": _benchmark_recorded(paths),
    }


def stage_status(experiment: Experiment) -> None:
    deployment = experiment.config["mujoco_deployment"]
    print(f"Experiment: {experiment.root}")
    print(f"  residual model:          {'enabled' if experiment.config['use_residual_model'] else 'disabled'}")
    print(f"  trajectory optimization: {'enabled' if experiment.config['optimize_trajectories'] else 'disabled (baselines)'}")
    print(
        "  residual in design:      "
        f"identification {'on' if experiment.config['identification_trajectory']['use_residual_model'] else 'off'}, "
        f"tuning {'on' if experiment.config['tuning_trajectories']['use_residual_model'] else 'off'}"
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
        f"{experiment.config['benchmark']['trajectory']} per iteration"
    )
    print(
        f"  iterations:              {len(experiment.iteration_indices())} of "
        f"{int(experiment.config['num_iterations'])} targeted"
    )
    for iteration in experiment.iteration_indices():
        print(f"iteration_{iteration:02d}:")
        status = iteration_status(experiment, iteration)
        for stage, description in REQUIRED_STAGE_OUTPUTS + OPTIONAL_STAGE_OUTPUTS:
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

    One iteration at a time by default. ``num_iterations`` (or the experiment's
    own ``num_iterations``) is a *target total*: the loop keeps going until that
    iteration has been finalized, and refuses to start one past it, so rerunning
    on a finished experiment is a no-op rather than another iteration's work.
    Getting anywhere unattended needs the logs to appear without a human, i.e.
    ``simulate_deployment`` or ``mujoco_deployment.enabled``; otherwise the loop
    stops at the first iteration that needs robot data.

    Naming an ``iteration`` past the target raises the target to it: an explicit
    request is the more specific instruction.
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
            # --log names one file in one data/ directory, so it can only mean
            # the iteration the command was pointed at.
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


def _run_iteration(
    experiment: Experiment,
    iteration: int,
    log: str | None = None,
    simulate_deployment: bool = False,
) -> bool:
    """Run every stage of one iteration that can proceed, in order.

    Returns whether the iteration was finalized (and the next one created).
    Stops with instructions, returning False, when robot data is required and
    no deployment stands in for it.
    """
    paths = experiment.paths(iteration)
    status = iteration_status(experiment, iteration)

    if not status["plan-id-trajectory"]:
        stage_plan_identification_trajectory(experiment, iteration)
        status = iteration_status(experiment, iteration)

    # The benchmark scores the controller this iteration deployed, so it is
    # recorded here, before anything identifies or tunes on this iteration's
    # data. Independent of the identification logs: rerunning an iteration
    # whose data/ is already filled still records the benchmark it is missing.
    deploys_in_simulation = simulate_deployment or experiment.config["mujoco_deployment"]["enabled"]
    if deploys_in_simulation and not status["benchmark"]:
        stage_run_benchmark(experiment, iteration)

    if not status["decode-logs"]:
        if deploys_in_simulation:
            stage_simulate_deployment(experiment, iteration)
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
        print("  (or run it with --simulate-deployment to have MuJoCo stand in for the robot.)")
        return False

    if not status["identify"]:
        stage_identify(experiment, iteration, log=log)
    if experiment.config["use_residual_model"] and not paths.residual_model.is_file():
        stage_train_residual(experiment, iteration)
    if not status["plan-tuning-trajectories"]:
        stage_plan_tuning_trajectories(experiment, iteration)
    if not iteration_status(experiment, iteration)["tune-gains"]:
        stage_tune_gains(experiment, iteration)
    stage_finalize(experiment, iteration)
    return True


def _identified_problem(paths: IterationPaths) -> Path:
    if not paths.problem_identified.is_file():
        raise FileNotFoundError(
            f"Missing {paths.problem_identified}; run the identify stage first "
            "(downstream stages must use the identified robot model)."
        )
    return paths.problem_identified
