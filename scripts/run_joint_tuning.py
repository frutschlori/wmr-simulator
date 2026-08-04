import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from datetime import datetime
import pickle

import numpy as np
import yaml

from wmr_simulator.joint_tuning.pipeline import (
    GAIN_NAMES,
    MODE_ALTERNATING,
    MODES,
    run_joint_tuning,
)
from wmr_simulator.trajectory_optimization.start_offsets import START_OFFSET_MODES
from wmr_simulator.trajectory_optimization.objectives import CRITERIA, DEFAULT_CRITERION
from wmr_simulator.trajectory_optimization.pipeline import reference_states_export_payload
from wmr_simulator.visualization.joint_tuning import (
    plot_joint_tuning_history,
    plot_joint_tuning_trajectories,
)


def main():
    parser = argparse.ArgumentParser(
        description="Alternating optimization of controller gains and tuning trajectories."
    )
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--mode", choices=list(MODES), default=MODE_ALTERNATING)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--warm-start-rounds", type=int, default=500)
    parser.add_argument("--trajectory-learning-rate", type=float, default=2e-3)
    parser.add_argument("--gain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-realizations", type=int, default=4)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-control-points", type=int, default=5)
    parser.add_argument("--start-offset-mode", choices=sorted(START_OFFSET_MODES), default="optimize")
    # A directory of designed trajectory pickles to start from, skipping the
    # warm-start rounds entirely. Without it the loop designs its own from
    # scratch, which is what --warm-start-rounds pays for.
    parser.add_argument("--warm-start-trajectories", type=str, default="trajectory_exports/gain_optimized_current")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--criterion", choices=list(CRITERIA), default=DEFAULT_CRITERION)
    parser.add_argument("--wheel-lp-tau", type=float, default=None)
    parser.add_argument("--out", type=str, default="results/joint_tuning")
    args = parser.parse_args()

    result = run_joint_tuning(
        args.problem,
        mode=args.mode,
        wheel_lp_tau=args.wheel_lp_tau,
        num_rounds=args.rounds,
        warm_start_rounds=args.warm_start_rounds,
        num_trajectories=args.num_trajectories,
        num_control_points=args.num_control_points,
        num_realizations=args.num_realizations,
        trajectory_learning_rate=args.trajectory_learning_rate,
        gain_learning_rate=args.gain_learning_rate,
        warm_start_trajectories_dir=args.warm_start_trajectories,
        start_offset_mode=args.start_offset_mode,
        criterion=args.criterion,
        seed=args.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gains.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "problem": args.problem,
                "mode": args.mode,
                "wheel_lp_tau": float(result.config["wheel_lp_tau"]),
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
    # Each trajectory ships the final start offsets it was designed under: they
    # are decision variables of the design under an optimizing start_offset_mode,
    # and the conditions the FIM was averaged over even when frozen. The gain
    # tuner reads them back so it tunes on exactly those starts -- and so does
    # --warm-start-trajectories, which is this export read the other way round.
    start_offsets = np.asarray(result.start_offsets, dtype=float)
    control_points = np.asarray(result.control_points, dtype=float)
    for index, reference_states in enumerate(np.asarray(result.reference_states, dtype=float)):
        with open(os.path.join(out_dir, f"reference_states_{index:02d}.pkl"), "wb") as file:
            pickle.dump(
                reference_states_export_payload(
                    reference_states,
                    float(result.trajectory_pipeline.problem.dt),
                    start_offsets=start_offsets[index],
                    control_points=control_points[index],
                ),
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
