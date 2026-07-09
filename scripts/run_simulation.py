import argparse

from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.visualization.pololu import plot_logged_summary
from wmr_simulator.visualization.simulation import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="simulation")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-ref-meshcat", action="store_true", default=True)
    # Learned residual dynamics checkpoint (scripts/train_residual_model.py);
    # the closed loop then runs the residual-augmented dynamics.
    parser.add_argument("--residual-model", type=str, default=None)
    args = parser.parse_args()

    residual_model = None
    if args.residual_model is not None:
        from wmr_simulator.residual_model import load_residual_model

        residual_model, checkpoint = load_residual_model(args.residual_model)
        print(f"Loaded residual dynamics model: {args.residual_model}")
        print(f"  config: {checkpoint['config']}")

    pipeline = SimulationPipeline(
        problem_path=args.problem,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
        residual_model=residual_model,
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
