import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from datetime import datetime
import pickle

import numpy as np
import yaml

from wmr_simulator.joint_tuning.pipeline import GAIN_NAMES, run_joint_tuning
from wmr_simulator.joint_tuning.start_offsets import START_OFFSET_MODES
from wmr_simulator.visualization.joint_tuning import (
    plot_joint_tuning_history,
    plot_joint_tuning_trajectories,
)


def main():
    parser = argparse.ArgumentParser(
        description="Alternating optimization of controller gains and tuning trajectories."
    )
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--warm-start-rounds", type=int, default=50)
    parser.add_argument("--trajectory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gain-learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-realizations", type=int, default=8)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-control-points", type=int, default=7)
    parser.add_argument("--start-offset-mode", choices=sorted(START_OFFSET_MODES), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/joint_tuning")
    args = parser.parse_args()

    result = run_joint_tuning(
        args.problem,
        num_rounds=args.rounds,
        warm_start_rounds=args.warm_start_rounds,
        num_trajectories=args.num_trajectories,
        num_control_points=args.num_control_points,
        num_realizations=args.num_realizations,
        trajectory_learning_rate=args.trajectory_learning_rate,
        gain_learning_rate=args.gain_learning_rate,
        start_offset_mode=args.start_offset_mode,
        seed=args.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gains.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "problem": args.problem,
                "gains": [float(gain) for gain in result.gains],
                "start_offset_mode": args.start_offset_mode,
                "seconds_per_round": float(result.timing["seconds_per_round"]),
                "uphill_fraction_gain": float(result.history["uphill_fraction_gain"]),
                "uphill_fraction_trajectory": float(result.history["uphill_fraction_trajectory"]),
                "uphill_fraction_joint": float(result.history["uphill_fraction_joint"]),
            },
            file,
            sort_keys=False,
        )
    for index, reference_states in enumerate(np.asarray(result.reference_states, dtype=float)):
        with open(os.path.join(out_dir, f"reference_states_{index:02d}.pkl"), "wb") as file:
            pickle.dump(
                {"reference_states": reference_states, "dt": float(result.trajectory_pipeline.problem.dt)},
                file,
            )
    print(f"Saved joint tuning result to {out_dir}")

    print("Final gains:")
    for name, value in zip(GAIN_NAMES, np.asarray(result.gains, dtype=float)):
        print(f"  {name:<9}: {value:.7g}")

    plot_joint_tuning_history(result.history)
    plot_joint_tuning_trajectories(result)


if __name__ == "__main__":
    main()
