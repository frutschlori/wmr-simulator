"""Generate the fixed reference sets for the fixed-trajectory configurations.

Every set is written once and shared by every experiment that names it
(baseline_identification_trajectory / baseline_tuning_trajectories_dir in
experiment.yaml); trajectory_optimization.fixed_sets documents what each one
is. The durations, control-point count, floors and phases are read from the
experiment defaults, so the sets match what the loop would design.

    python scripts/generate_fixed_trajectory_sets.py
    python scripts/generate_fixed_trajectory_sets.py --sets random-twist --random-sizes 15 45 150 --seed 1
    python scripts/generate_fixed_trajectory_sets.py --sets identification-random --random-identification 10
    python scripts/generate_fixed_trajectory_sets.py --sets matched \\
        --compare "Pololu Data/test31 1 start gains/iteration_01/tuning_trajectories"

Writes trajectory_exports/fixed_sets/<set>/ with one pickle per reference, a
summary.yaml of motion statistics and an overview figure, plus one motion
histogram comparing all tuning sets written in this call (and --compare
directories, e.g. a designed set). Existing sets are never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import yaml

from wmr_simulator.active_learning.experiment import DEFAULT_EXPERIMENT_CONFIG
from wmr_simulator.trajectory_optimization import fixed_sets
from wmr_simulator.trajectory_optimization.pipeline import ProblemDefinition
from wmr_simulator.visualization.motion_histograms import (
    motion_channels_from_reference_states,
    plot_motion_histograms,
    pool_motion_channels,
)
from wmr_simulator.visualization.trajectories import plot_reference_set_overview

SETS = ("identification", "identification-random", "matched", "benchmark", "random-bspline", "random-twist")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", default="problems/pololu_gains.yaml")
    parser.add_argument("--out-dir", default="trajectory_exports/fixed_sets")
    parser.add_argument("--sets", nargs="+", choices=SETS, default=list(SETS))
    parser.add_argument("--random-sizes", nargs="+", type=int, default=[15, 45, 150])
    parser.add_argument(
        "--random-identification", type=int, default=10,
        help="Random identification references drawn (the F level uses draw k for seed k).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-dir", default=DEFAULT_EXPERIMENT_CONFIG["benchmark"]["trajectory"])
    parser.add_argument("--compare", nargs="*", default=[], help="Extra pickle directories for the histogram.")
    return parser


def write_set(name, references, out_dir, dt, limits, environment, control_points=None, **metadata):
    set_dir = Path(out_dir) / name
    fixed_sets.export_reference_set(references, set_dir, dt, control_points=control_points, fixed_set=name, **metadata)
    summary = fixed_sets.reference_set_summary(references, dt, limits)
    with (set_dir / "summary.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({"fixed_set": name, **metadata, **summary}, file, sort_keys=False)
    plot_reference_set_overview(
        references, dt, set_dir / f"{name}_overview.pdf", title=name, environment=environment, limits=limits
    )
    ranges = summary["ranges"]
    print(
        f"{name}: {len(references)} references, {ranges['duration'][0]:.2f}-{ranges['duration'][1]:.2f} s, "
        f"mean speed {ranges['v_mean'][0]:.2f}-{ranges['v_mean'][1]:.2f} m/s, "
        f"mean |a_lat| {ranges['a_lat_mean'][0]:.2f}-{ranges['a_lat_mean'][1]:.2f} m/s^2, "
        f"worst limit ratio {ranges['worst_limit_ratio'][1]:.2f}"
    )
    return set_dir


def random_identification_references(problem_path: str, count: int):
    """``count`` random identification references from the design's curve
    family (fixed_sets.random_identification_reference), draw k from seed k."""
    import tempfile

    from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

    config = DEFAULT_EXPERIMENT_CONFIG["identification_trajectory"]
    problem_cfg = yaml.safe_load(Path(problem_path).read_text())
    problem_cfg["sim_time"] = sum(float(phase["duration"]) for phase in config["phases"])
    with tempfile.TemporaryDirectory() as directory:
        identification_problem = Path(directory) / "problem_identification.yaml"
        identification_problem.write_text(yaml.safe_dump(problem_cfg, sort_keys=False))
        pipeline = TrajectoryOptimizationPipeline(
            str(identification_problem),
            time_scaling=config["time_scaling"],
            objective_mode="identification",
            motion_phases=config["phases"],
        )
        references, control_points = {}, {}
        for index in range(count):
            states, identified_duration, points = fixed_sets.random_identification_reference(
                pipeline, int(config["num_control_points"]), seed=index
            )
            name = f"random_identification_{index:02d}"
            references[name] = states
            control_points[name] = points
    return references, control_points, identified_duration, pipeline.phase_breaks


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tuning = DEFAULT_EXPERIMENT_CONFIG["tuning_trajectories"]
    problem = ProblemDefinition(args.problem)
    problem.sim_time = float(tuning["sim_time"]) or problem.sim_time
    dt = problem.dt
    limits = fixed_sets.motion_limits(problem.robot_cfg)
    environment = (problem.environment_min, problem.environment_max)
    written = []

    if "identification" in args.sets:
        phases = DEFAULT_EXPERIMENT_CONFIG["identification_trajectory"]["phases"]
        states, identified_duration = fixed_sets.fixed_identification_reference(
            phases, dt, problem.robot_cfg, *environment
        )
        set_dir = Path(args.out_dir) / "identification"
        fixed_sets.export_reference_set(
            {"fixed_identification": states}, set_dir, dt,
            fixed_set="identification", identified_duration=identified_duration,
            phase_breaks=[identified_duration / ((len(states) - 1) * dt)],
        )
        plot_reference_set_overview(
            {"fixed_identification": states}, dt, set_dir / "identification_overview.pdf",
            title=f"fixed identification reference (identified: first {identified_duration:.1f} s)",
            environment=environment,
        )
    if "identification-random" in args.sets:
        references, control_points, identified_duration, phase_breaks = random_identification_references(
            args.problem, args.random_identification
        )
        set_dir = Path(args.out_dir) / "identification_random"
        fixed_sets.export_reference_set(
            references, set_dir, dt, control_points=control_points,
            fixed_set="identification_random", identified_duration=identified_duration,
            phase_breaks=list(phase_breaks),
        )
        plot_reference_set_overview(
            references, dt, set_dir / "identification_random_overview.pdf",
            title=f"random identification references (identified: first {identified_duration:.1f} s)",
            environment=environment,
        )
        print(f"identification_random: {len(references)} references")
    if "matched" in args.sets:
        if len(fixed_sets.MATCHED_SET_SHAPES) != int(tuning["num_trajectories"]):
            print(f"Note: the matched set has {len(fixed_sets.MATCHED_SET_SHAPES)} shapes, the designed set "
                  f"{tuning['num_trajectories']} trajectories.")
        references = fixed_sets.matched_reference_set(problem.sim_time, dt, limits, *environment)
        written.append(write_set("matched", references, args.out_dir, dt, limits, environment))
    if "benchmark" in args.sets:
        references = fixed_sets.benchmark_reference_set(args.benchmark_dir, dt)
        written.append(write_set("benchmark", references, args.out_dir, dt, limits, environment,
                                 source=str(args.benchmark_dir)))
    for size in args.random_sizes if "random-bspline" in args.sets else []:
        references, control_points, report = fixed_sets.random_bspline_reference_set(
            problem,
            size,
            int(tuning["num_control_points"]),
            time_scaling=tuning["time_scaling"],
            seed=args.seed,
            min_speed=float(tuning["min_speed"]),
            min_speed_fraction=float(tuning["min_speed_fraction"]),
            min_lateral_acceleration=float(tuning["min_lateral_acceleration"]),
            min_lateral_acceleration_fraction=float(tuning["min_lateral_acceleration_fraction"]),
        )
        written.append(write_set(
            f"random_bspline_N{size}", references, args.out_dir, dt, limits, environment,
            control_points=control_points, seed=args.seed, acceptance_rate=float(report["acceptance_rate"]),
        ))
    for size in args.random_sizes if "random-twist" in args.sets else []:
        references = fixed_sets.random_twist_reference_set(size, problem.sim_time, dt, limits, *environment, seed=args.seed)
        written.append(write_set(f"random_twist_N{size}", references, args.out_dir, dt, limits, environment,
                                 seed=args.seed))

    datasets = [
        (
            directory.name,
            pool_motion_channels(
                motion_channels_from_reference_states(states, dt)
                for states in fixed_sets.load_reference_set(directory).values()
            ),
        )
        for directory in [*written, *map(Path, args.compare)]
    ]
    if len(datasets) > 1:
        plot_motion_histograms(datasets, out_prefix="fixed_sets_motion", out_dir=args.out_dir,
                               title="Fixed reference sets", robot_cfg=problem.robot_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
