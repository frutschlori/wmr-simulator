import argparse

from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.visualization.pololu import plot_logged_summary
from wmr_simulator.visualization.simulation import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="simulation")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-ref-meshcat", action="store_true", default=False)
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
        plot_logged_summary(log, out_prefix=args.output)

    if not args.skip_ref_meshcat:
        visualize(
            pipeline.problem_path,
            log.pose.states,
            out_prefix=args.output,
            dt=pipeline.wheel_dt,
        )


if __name__ == "__main__":
    main()
