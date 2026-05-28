import argparse

import numpy as np

from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.visualization.simulation import plot, visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/figure_eight.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="simulation")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-meshcat", action="store_true")
    args = parser.parse_args()

    pipeline = SimulationPipeline(
        problem_path=args.problem,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
    )
    log = pipeline.run_closed_loop(
        pipeline.hidden_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
    )

    if not args.skip_pdf:
        plot(
            (log.robot_states, log.estimator_states),
            pipeline.estimator,
            pipeline.sim_time_grid,
            reference_states=pipeline.reference_states,
            out_prefix=args.output,
        )

    if not args.skip_meshcat:
        meshcat_poses = np.vstack([
            np.asarray(pipeline.reference_states[0, :3]),
            np.asarray(log.robot_states.pose[:-1]),
        ])
        visualize(
            pipeline.problem_path,
            meshcat_poses,
            reference_states=pipeline.reference_states,
            out_prefix=args.output,
            dt=pipeline.dt,
        )


if __name__ == "__main__":
    main()
