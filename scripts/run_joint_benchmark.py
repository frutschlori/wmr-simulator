import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.joint_tuning.benchmark import benchmark_run, summarize
from wmr_simulator.joint_tuning.pipeline import MODE_ALTERNATING, MODES
from wmr_simulator.trajectory_optimization.objectives import CRITERIA, DEFAULT_CRITERION
from wmr_simulator.trajectory_optimization.start_offsets import START_OFFSET_MODES


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark one joint (or sequential) gain/trajectory tuning configuration."
    )
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--mode", choices=list(MODES), default=MODE_ALTERNATING)
    parser.add_argument("--stage", type=str, default="adhoc",
                        help="Sweep stage tag; becomes the record filename prefix.")
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--warm-start-rounds", type=int, default=50)
    parser.add_argument("--trajectory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gain-learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-realizations", type=int,
                        default=int(GAIN_TUNING_DEFAULTS["num_realizations"]))
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-control-points", type=int, default=7)
    parser.add_argument("--start-offset-mode", choices=sorted(START_OFFSET_MODES), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--criterion", choices=list(CRITERIA), default=DEFAULT_CRITERION,
                        help="Design criterion the trajectory objective minimizes.")
    parser.add_argument("--wheel-lp-tau", type=float, default=None,
                        help="Override the encoder low-pass for trajectory design "
                             "(default: the problem yaml's value; 0 disables it).")
    parser.add_argument("--criterion-tune-steps", type=int, default=100)
    parser.add_argument("--no-plateau", action="store_true")
    parser.add_argument("--no-criteria", action="store_true")
    parser.add_argument("--out", type=str, default="results/joint_tuning_benchmark")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Write this run's history plus its training and held-out "
                             "trajectory figures here (skipped when unset).")
    args = parser.parse_args()

    record = benchmark_run(
        args.problem,
        mode=args.mode,
        stage=args.stage,
        out_dir=args.out,
        plot_dir=args.plot_dir,
        plateau=not args.no_plateau,
        criteria=not args.no_criteria,
        criterion_tune_steps=args.criterion_tune_steps,
        num_rounds=args.rounds,
        warm_start_rounds=args.warm_start_rounds,
        num_trajectories=args.num_trajectories,
        num_control_points=args.num_control_points,
        num_realizations=args.num_realizations,
        trajectory_learning_rate=args.trajectory_learning_rate,
        gain_learning_rate=args.gain_learning_rate,
        start_offset_mode=args.start_offset_mode,
        criterion=args.criterion,
        wheel_lp_tau=args.wheel_lp_tau,
        seed=args.seed,
    )
    print(summarize(record))
    print(f"Record: {record['meta']['path']}")


if __name__ == "__main__":
    main()
