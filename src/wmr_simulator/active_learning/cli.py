"""CLI for the iterative active-learning loop (see stages.py for the stages).

Usage (via scripts/run_active_learning.py):

    run_active_learning.py init --experiment experiments/exp01 [--problem ...]
    run_active_learning.py run --experiment experiments/exp01
    run_active_learning.py <stage> --experiment experiments/exp01 [--iteration N]

``run`` executes every stage that can proceed and stops with instructions when
robot data has to be collected from the SD card; the individual stage commands
allow re-running any part in isolation.
"""

from __future__ import annotations

import argparse

from wmr_simulator.active_learning import stages
from wmr_simulator.active_learning.experiment import Experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative identification / gain-tuning active-learning loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new experiment directory and its first iteration.")
    init_parser.add_argument("--experiment", required=True, help="Experiment directory to create.")
    init_parser.add_argument("--problem", default=None, help="Base problem yaml (default: problems/pololu_gains.yaml).")
    init_parser.add_argument("--seed", type=int, default=None)
    init_parser.add_argument("--residual-model", action=argparse.BooleanOptionalAction, default=None,
                             help="Enable/disable the residual-model stage (--no-residual-model to disable).")
    init_parser.add_argument("--trajectory-optimization", action=argparse.BooleanOptionalAction, default=None,
                             help="Disable to use static baseline trajectories instead of optimizing.")
    init_parser.add_argument("--baseline-id-trajectory", default=None,
                             help="Reference pickle used as identification trajectory when optimization is disabled.")
    init_parser.add_argument("--baseline-tuning-trajectories-dir", default=None,
                             help="Directory of reference pickles used for gain tuning when optimization is disabled.")
    init_parser.add_argument("--robotcfg-template", default=None,
                             help="Existing ROBOTCFG.CFG used as template for firmware exports.")

    stage_commands = {
        "plan-id-trajectory": "Optimize (or copy) the identification trajectory and export the Pololu JSN.",
        "decode-logs": "Decode binary SD-card logs in data/ to csv.",
        "identify": "Run parameter identification on the recorded log.",
        "train-residual": "Train the residual dynamics model from the decoded logs.",
        "plan-tuning-trajectories": "Optimize (or copy) the gain-tuning trajectory set.",
        "tune-gains": "Tune controller gains on the identified model.",
        "finalize": "Fold the iteration results into the next iteration folder.",
        "run": "Run all stages that can proceed; stops when robot data is needed.",
        "status": "Show per-iteration stage completion.",
    }
    for command, help_text in stage_commands.items():
        stage_parser = subparsers.add_parser(command, help=help_text)
        stage_parser.add_argument("--experiment", required=True, help="Experiment directory.")
        if command != "status":
            stage_parser.add_argument("--iteration", type=int, default=None,
                                      help="Iteration number (default: latest).")
        if command in ("identify", "run"):
            stage_parser.add_argument("--log", default=None,
                                      help="Identification log (path or filename inside data/); "
                                           "required if data/ holds several logs.")
            stage_parser.add_argument("--estimate-mocap-delay", action=argparse.BooleanOptionalAction,
                                      default=None,
                                      help="Estimate the mocap delay from the log via IMU gyro "
                                           "cross-correlation (default: identification."
                                           "estimate_mocap_delay in experiment.yaml).")
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
        if args.trajectory_optimization is not None:
            overrides["optimize_trajectories"] = args.trajectory_optimization
        if args.baseline_id_trajectory is not None:
            overrides["baseline_identification_trajectory"] = args.baseline_id_trajectory
        if args.baseline_tuning_trajectories_dir is not None:
            overrides["baseline_tuning_trajectories_dir"] = args.baseline_tuning_trajectories_dir
        if args.robotcfg_template is not None:
            overrides["robotcfg_template"] = args.robotcfg_template
        stages.stage_init(args.experiment, overrides)
        return 0

    experiment = Experiment.load(args.experiment)
    if args.command == "status":
        stages.stage_status(experiment)
        return 0

    iteration = experiment.resolve_iteration(args.iteration)
    if args.command == "plan-id-trajectory":
        stages.stage_plan_identification_trajectory(experiment, iteration)
    elif args.command == "decode-logs":
        stages.stage_decode_logs(experiment, iteration)
    elif args.command == "identify":
        stages.stage_identify(experiment, iteration, log=args.log,
                              estimate_mocap_delay=args.estimate_mocap_delay)
    elif args.command == "train-residual":
        stages.stage_train_residual(experiment, iteration)
    elif args.command == "plan-tuning-trajectories":
        stages.stage_plan_tuning_trajectories(experiment, iteration)
    elif args.command == "tune-gains":
        stages.stage_tune_gains(experiment, iteration)
    elif args.command == "finalize":
        stages.stage_finalize(experiment, iteration)
    elif args.command == "run":
        stages.stage_run(experiment, iteration, log=args.log,
                         estimate_mocap_delay=args.estimate_mocap_delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
