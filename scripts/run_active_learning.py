"""Command-line interface of the active-learning loop.

An experiment is a directory with an ``experiment.yaml`` and one folder per
iteration (layout: ``wmr_simulator.active_learning.experiment``).

On the robot (``run`` stops whenever robot data is needed):
    python scripts/run_active_learning.py init   --experiment experiments/exp01
    python scripts/run_active_learning.py run    --experiment experiments/exp01
    # copy ROBOTCFG.CFG and the identification JSN to the SD card, drive the
    # robot, copy the logs into iteration_XX/data/, then run again
    python scripts/run_active_learning.py run    --experiment experiments/exp01
    python scripts/run_active_learning.py status --experiment experiments/exp01

In MuJoCo, unattended for a number of iterations:
    python scripts/run_active_learning.py init --experiment experiments/exp01 --mujoco --iterations 5
    python scripts/run_active_learning.py run  --experiment experiments/exp01

Single stages (default: the latest iteration, ``--iteration N`` for another):
    plan-id-trajectory, simulate-deployment, benchmark, decode-logs, identify,
    train-residual, plan-tuning-trajectories, tune-gains, finalize,
    tune-parametrized, plot-run-logs.

Stage hyperparameters live in experiment.yaml and may be edited between
stages. Set JAX_PLATFORMS=gpu to run the optimization stages on the GPU.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.active_learning import stages
from wmr_simulator.active_learning.experiment import Experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative identification / gain-tuning active-learning loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new experiment directory and its first iteration.")
    init_parser.add_argument("--experiment", required=True, help="Experiment directory to create.")
    init_parser.add_argument("--problem", default=None, help="Base problem yaml (default: problems/pololu_gains.yaml).")
    init_parser.add_argument("--seed", type=int, default=None)
    init_parser.add_argument(
        "--residual-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable the residual model (--no-residual-model to disable): off means "
            "no residual is ever trained, and the trajectory designs and gain tuning all run "
            "on the nominal plant, whatever their own use_residual_model flags say. "
            "Default: use_residual_model in experiment.yaml, which starts enabled."
        ),
    )
    init_parser.add_argument(
        "--gain-parametrization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable the gain parametrization (--no-gain-parametrization to disable): "
            "off means the static controller records every log and is the only one tuned. "
            "Default: use_gain_parametrization in experiment.yaml (on, following the problem yaml)."
        ),
    )
    init_parser.add_argument(
        "--prior-tuning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Tune on the prior model in an iteration 0 before the first identification run "
            "(default), or --no-prior-tuning to record iteration 1 under the problem's initial gains."
        ),
    )
    init_parser.add_argument(
        "--optimize-identification",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Design the identification trajectory every iteration (default), or "
            "--no-optimize-identification to drive the fixed baseline_identification_trajectory."
        ),
    )
    init_parser.add_argument(
        "--optimize-tuning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Design the tuning set every iteration (default), or --no-optimize-tuning to tune "
            "on the fixed set in baseline_tuning_trajectories_dir."
        ),
    )
    init_parser.add_argument(
        "--standalone-gain-tuning-defaults",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Tune gains with gain_tuning.defaults.GAIN_TUNING_DEFAULTS instead of the "
            "experiment.yaml gain_tuning block."
        ),
    )
    init_parser.add_argument(
        "--baseline-id-trajectory",
        default=None,
        help=(
            "Reference pickle driven as the identification trajectory under "
            "--no-optimize-identification (default: the generated fixed one)."
        ),
    )
    init_parser.add_argument(
        "--baseline-tuning-trajectories-dir",
        default=None,
        help=(
            "Directory of reference pickles tuned on under --no-optimize-tuning "
            "(default: trajectory_exports/fixed_sets/matched)."
        ),
    )
    init_parser.add_argument("--robotcfg-template", default=None, help="Existing ROBOTCFG.CFG used as template for firmware exports.")
    init_parser.add_argument(
        "--mujoco",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Collect the identification logs from a MuJoCo deployment instead of the robot, "
            "so `run` never stops to wait for an SD card (mujoco_deployment in experiment.yaml)."
        ),
    )
    init_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "How many iterations the experiment should end up with; `run` continues until "
            "that one is finalized. Needs --mujoco to get there unattended."
        ),
    )

    stage_commands = {
        "plan-id-trajectory": "Optimize (or copy) the identification trajectory and export the Pololu JSN.",
        "simulate-deployment": (
            "Stand in for the robot: drive the identification trajectory in the MuJoCo "
            "plant and write binary TRxx logs into data/."
        ),
        "benchmark": (
            "Drive the benchmark references (benchmark.trajectory) in the MuJoCo plant under "
            "this iteration's controllers and write the runs into data/benchmark[_static]/."
        ),
        "decode-logs": "Decode binary SD-card logs in data/ to csv.",
        "identify": (
            "Identify robot parameters from every log in data/: one run per log, "
            "outlying logs dropped, then a joint fit on the rest."
        ),
        "train-residual": (
            "Train the residual dynamics model from the decoded logs of this iteration "
            "and every earlier one (residual.pool_previous_iterations)."
        ),
        "plan-tuning-trajectories": "Optimize (or copy) the gain-tuning trajectory set.",
        "tune-gains": "Tune controller gains on the identified model.",
        "finalize": "Fold the iteration results into the next iteration folder.",
        "tune-parametrized": (
            "Tune a state-dependent gain variant post hoc on this (static) iteration and deploy it "
            "into the next one (active_learning.posthoc; the variant's config must exist)."
        ),
        "plot-run-logs": (
            "Plot every data/ subdirectory recording of every iteration into that "
            "iteration's visualize/logs/<same subdirectory>/, and rewrite each iteration's "
            "benchmark figures (also done by `run`)."
        ),
        "run": "Run all stages that can proceed; stops when robot data is needed.",
        "status": "Show per-iteration stage completion.",
    }
    for command, help_text in stage_commands.items():
        stage_parser = subparsers.add_parser(command, help=help_text)
        stage_parser.add_argument("--experiment", required=True, help="Experiment directory.")
        if command not in ("status", "plot-run-logs"):
            stage_parser.add_argument("--iteration", type=int, default=None, help="Iteration number (default: latest).")
        if command in ("identify", "run"):
            stage_parser.add_argument(
                "--log",
                default=None,
                help=(
                    "Restrict identification to a single log (path or filename inside data/); "
                    "default: every log directly in data/, fitted jointly."
                ),
            )
        if command == "simulate-deployment":
            stage_parser.add_argument(
                "--num-logs",
                type=int,
                default=None,
                help="Deployments to run (default: mujoco_deployment.num_logs in experiment.yaml).",
            )
        if command == "tune-parametrized":
            stage_parser.add_argument("--variant", required=True, help="Variant name (parametrized_variants/<name>.yaml).")
        if command == "benchmark":
            stage_parser.add_argument(
                "--num-runs",
                type=int,
                default=None,
                help="Benchmark runs to record (default: benchmark.num_runs in experiment.yaml).",
            )
        if command == "run":
            stage_parser.add_argument(
                "--simulate-deployment",
                action="store_true",
                help=(
                    "Fill data/ with a MuJoCo deployment instead of stopping to wait for robot "
                    "logs (always on when mujoco_deployment.enabled)."
                ),
            )
            stage_parser.add_argument(
                "--iterations",
                type=int,
                default=None,
                help=(
                    "Keep going until this iteration is finalized (default: num_iterations in "
                    "experiment.yaml). A target total, not a count of iterations to add."
                ),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init":
        overrides: dict = {}
        if args.problem is not None:
            overrides["problem"] = args.problem
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.residual_model is not None:
            overrides["use_residual_model"] = args.residual_model
        if args.gain_parametrization is not None:
            overrides["use_gain_parametrization"] = args.gain_parametrization
        if args.prior_tuning is not None:
            overrides["prior_tuning"] = args.prior_tuning
        if args.optimize_identification is not None:
            overrides["optimize_identification_trajectory"] = args.optimize_identification
        if args.optimize_tuning is not None:
            overrides["optimize_tuning_trajectories"] = args.optimize_tuning
        if args.standalone_gain_tuning_defaults is not None:
            overrides["use_standalone_gain_tuning_defaults"] = args.standalone_gain_tuning_defaults
        if args.baseline_id_trajectory is not None:
            overrides["baseline_identification_trajectory"] = args.baseline_id_trajectory
        if args.baseline_tuning_trajectories_dir is not None:
            overrides["baseline_tuning_trajectories_dir"] = args.baseline_tuning_trajectories_dir
        if args.robotcfg_template is not None:
            overrides["robotcfg_template"] = args.robotcfg_template
        if args.mujoco is not None:
            overrides["mujoco_deployment"] = {"enabled": args.mujoco}
        if args.iterations is not None:
            overrides["num_iterations"] = args.iterations
        stages.stage_init(args.experiment, overrides)
        return 0

    experiment = Experiment.load(args.experiment)
    if args.command == "status":
        stages.stage_status(experiment)
        return 0

    if args.command == "plot-run-logs":
        stages.plot_run_directory_logs(experiment)
        stages.plot_baseline_runs_per_iteration(experiment)
        return 0

    if args.command == "run":
        stages.stage_run(
            experiment,
            args.iteration,
            log=args.log,
            simulate_deployment=args.simulate_deployment,
            num_iterations=args.iterations,
        )
        return 0

    iteration = experiment.resolve_iteration(args.iteration)
    stage_name = args.command if args.command != "tune-parametrized" else f"{args.command}:{args.variant}"
    with stages.logged_stage(experiment.paths(iteration), stage_name):
        run_stage(args, experiment, iteration)
    return 0


def run_stage(args, experiment, iteration: int) -> None:
    if args.command == "plan-id-trajectory":
        stages.stage_plan_identification_trajectory(experiment, iteration)
    elif args.command == "simulate-deployment":
        stages.stage_simulate_deployment(experiment, iteration, num_logs=args.num_logs)
    elif args.command == "benchmark":
        stages.stage_run_benchmark(experiment, iteration, num_runs=args.num_runs)
    elif args.command == "decode-logs":
        stages.stage_decode_logs(experiment, iteration)
    elif args.command == "identify":
        stages.stage_identify(experiment, iteration, log=args.log)
    elif args.command == "train-residual":
        stages.stage_train_residual(experiment, iteration)
    elif args.command == "plan-tuning-trajectories":
        stages.stage_plan_tuning_trajectories(experiment, iteration)
    elif args.command == "tune-gains":
        stages.stage_tune_gains(experiment, iteration)
    elif args.command == "finalize":
        stages.stage_finalize(experiment, iteration)
    elif args.command == "tune-parametrized":
        from wmr_simulator.active_learning.posthoc import stage_tune_parametrized

        stage_tune_parametrized(experiment, iteration, args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
