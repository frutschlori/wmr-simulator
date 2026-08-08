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
    save_joint_tuning_trajectory_trace,
)


def main():
    parser = argparse.ArgumentParser(
        description="Alternating optimization of controller gains and tuning trajectories."
    )
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--mode", choices=list(MODES), default=MODE_ALTERNATING)
    parser.add_argument("--rounds", type=int, default=250)
    parser.add_argument("--warm-start-rounds", type=int, default=500)
    # The trajectory block steps slower than the gain block on purpose: it is
    # the one that runs away, and every step it takes changes the problem the
    # gain block is solving.
    parser.add_argument("--trajectory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--constraint-violation-tolerance", type=float, default=0.005)
    parser.add_argument("--trust-radius", type=float, default=2e-2,
                        help="Trust region on the trajectory block's per-round movement "
                             "(decision-variable L2). Negative disables it.")
    parser.add_argument("--trust-gain-loss-increase", type=float, default=0.02,
                        help="Relative gain-loss rise above which a trajectory step is rejected.")
    parser.add_argument("--trust-stall-rounds", type=int, default=10,
                        help="Stop after this many consecutive rejections at the minimum radius; "
                             "0 lets the loop spin instead.")
    parser.add_argument("--gain-steps-per-round", type=int, default=40,
                        help="Inner budget for the gain block: a cap on a bounded BFGS solve, "
                             "whose steps are line-search trials rather than accepted updates. "
                             "40 reaches the conditional optimum; below 15 is refused.")
    # One trajectory step per round against 40 gain steps: the ratio is the
# balance knob (raising both equally is a no-op, 5/5 reproduces 1/1
# step-for-step), and this asymmetry is what keeps the trajectory block
# from outrunning the gains. Every measured run used 40/1.
    parser.add_argument("--trajectory-steps-per-round", type=int, default=1)
    parser.add_argument("--validation-trajectories", type=str,
                        default="trajectory_exports/validation_trajectories",
                        help="Held-out trajectories the shipped gains are selected on. "
                             "Empty string falls back to the frozen training design.")
    # Off by default: this loop plateaus and then goes through a basin
    # transition at ~180 trajectory steps, so any stagnation rule quits in
    # the plateau. Set >= 0 to re-enable.
    parser.add_argument("--convergence-rel-tol", type=float, default=-1.0)
    parser.add_argument("--convergence-window", type=int, default=50)
    parser.add_argument("--num-realizations", type=int, default=4)
    parser.add_argument("--num-trajectories", type=int, default=8)
    parser.add_argument("--num-control-points", type=int, default=5)
    parser.add_argument("--start-offset-mode", choices=sorted(START_OFFSET_MODES), default="optimize")
    # A directory of designed trajectory pickles to start from, skipping the
    # warm-start rounds entirely. Without it the loop designs its own from
    # scratch, which is what --warm-start-rounds pays for.
    parser.add_argument("--warm-start-trajectories", type=str,
                        default="trajectory_exports/gain_optimized_current",
                        help="Empty string designs the trajectories from scratch instead, which "
                             "is what --warm-start-rounds pays for.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--criterion", choices=list(CRITERIA), default=DEFAULT_CRITERION)
    parser.add_argument("--save-round-GIF", action="store_true", default=False,
                        help="Animate the trajectories and their closed-loop rollouts round by "
                             "round. Costs one rollout batch per frame and keeps a per-round "
                             "snapshot, so it is off by default.")
    parser.add_argument("--round-trace-stride", type=int, default=5,
                        help="Rounds between animation frames (--save-round-GIF only).")
    parser.add_argument("--out", type=str, default="results/joint_tuning")
    args = parser.parse_args()

    result = run_joint_tuning(
        args.problem,
        mode=args.mode,
        num_rounds=args.rounds,
        warm_start_rounds=args.warm_start_rounds,
        num_trajectories=args.num_trajectories,
        num_control_points=args.num_control_points,
        num_realizations=args.num_realizations,
        trajectory_learning_rate=args.trajectory_learning_rate,
        constraint_violation_tolerance=args.constraint_violation_tolerance,
        trust_radius=args.trust_radius,
        trust_gain_loss_increase=args.trust_gain_loss_increase,
        trust_stall_rounds=args.trust_stall_rounds,
        gain_steps_per_round=args.gain_steps_per_round,
        trajectory_steps_per_round=args.trajectory_steps_per_round,
        trajectory_trace_stride=args.round_trace_stride if args.save_round_GIF else 0,
        validation_trajectories_dir=args.validation_trajectories or None,
        convergence_rel_tol=args.convergence_rel_tol,
        convergence_window=args.convergence_window,
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
                # The last iterate, kept for diagnostics: on a healthy run it is
                # close to `gains`, and a large gap means the loop was still
                # being dragged around when it stopped.
                "final_gains": [float(gain) for gain in result.final_gains],
                "best_gain_score": float(result.best_gain_score),
                "validation_trajectories": result.config["validation_trajectories_dir"],
                "num_validation_trajectories": result.config["num_validation_trajectories"],
                "best_gain_round": result.history["best_gain_round"],
                "converged_at_round": result.history["converged_at_round"],
                "converged_reason": result.history["converged_reason"],
                "rejected_rounds": int(result.history["rejected_rounds"]),
                "final_trust_radius": float(result.config["trust_radius"]),
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

    # The best iterate on the frozen scoring set, not the loop's last one -- see
    # JointTuningResult. `final_gains` is printed by run_joint_tuning itself.
    print("Best gains (shipped):")
    for name, value in zip(GAIN_NAMES, np.asarray(result.gains, dtype=float)):
        print(f"  {name:<9}: {value:.7g}")

    plot_joint_tuning_history(result.history)
    plot_joint_tuning_trajectories(result)
    if args.save_round_GIF:
        save_joint_tuning_trajectory_trace(result)


if __name__ == "__main__":
    main()
