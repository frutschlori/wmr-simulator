"""Run a MuJoCo deployment: put a trajectory on the (simulated) robot.

Replaces the manual "copy to SD card, drive, copy the logs back" step. Writes
one ``TRxx`` binary log into the target directory, in the firmware's own
format, so ``decode-logs`` and everything downstream is untouched.

A ``GAINMLP.JSN`` next to ``--config`` is picked up the way the firmware picks
it up off the card, so a scheduled iteration deploys with its gain MLP.

For the repeated runs an active-learning iteration needs, use the
``simulate-deployment`` stage of ``run_active_learning.py`` instead: it drives
the bridged trajectory and chains each run's start pose onto the last.

Examples
--------
    # one run of the iteration's identification trajectory
    uv run python scripts/run_mujoco_deployment.py \
        --config experiments/exp01/iteration_00/ROBOTCFG.CFG \
        --trajectory experiments/exp01/iteration_00/identification_trajectory/identification_trajectory.JSN \
        --output-dir experiments/exp01/iteration_00/data

    # a second run with a different hand placement of the robot
    ... --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="ROBOTCFG.CFG or robot_config.yaml")
    parser.add_argument("--trajectory", required=True, help="Pololu reference .JSN")
    parser.add_argument("--output-dir", required=True, help="Directory to write the TRxx log into")
    parser.add_argument("--seed", type=int, default=0, help="Start-pose draw and sensor noise (default 0)")
    parser.add_argument("--name", default=None, help="Log file name (default: first free TRxx)")
    args = parser.parse_args(argv)

    from wmr_simulator.mujoco_sim.deploy import run_deployment

    result = run_deployment(
        robot_config=args.config,
        trajectory=args.trajectory,
        output_dir=args.output_dir,
        seed=args.seed,
        log_name=args.name,
    )

    offset = result.start_offset
    print(f"{result.log_path}  ({result.num_records} records, {result.duration:.2f} s)")
    print(f"start offset: {1000 * offset[0]:+.0f} mm, {1000 * offset[1]:+.0f} mm, {offset[2]:+.3f} rad")
    print("--- ground truth (not in the log) ---")
    print(f"tracking RMSE {result.tracking_rmse:.4f} m, max {result.tracking_max:.4f} m")
    print(f"final pose error {result.final_pose_error:.4f} m")
    print(f"max |duty| {result.max_duty:.3f}, saturated {100 * result.duty_saturated_fraction:.1f}% of inner ticks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
