import argparse

from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.visualization.pololu import plot_logged_summary
from wmr_simulator.visualization.simulation import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    #parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports")
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="simulation")
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-ref-meshcat", action="store_true", default=True)
    # Learned residual dynamics checkpoint (scripts/train_residual_model.py);
    # the closed loop then runs the residual-augmented dynamics.
    # parser.add_argument("--residual-model", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--residual-model", type=str, default=None)
    # Tuning result from run_gain_tuning.py (--out): the closed loop then runs
    # its gains + gain parametrization, and the summary plot shows the applied
    # gains over time.
    # parser.add_argument("--tuned-gains", type=str, default="models/tuned_gains.yaml")
    parser.add_argument("--tuned-gains", type=str, default=None)
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

    controller_gains = pipeline.gains
    schedule_params = None
    if args.tuned_gains is not None:
        import jax.numpy as jnp
        import yaml

        from wmr_simulator.gain_parametrization import params_from_cfg

        with open(args.tuned_gains, "r", encoding="utf-8") as file:
            tuned = yaml.safe_load(file)
        controller_gains = jnp.asarray(tuned["gains"], dtype=jnp.float32)
        if tuned.get("schedule") is not None and tuned.get("schedule_enabled", True):
            schedule_params = params_from_cfg(
                tuned["schedule"], pipeline.gain_parametrization_feature_scale
            )
        print(f"Loaded tuned gains from {args.tuned_gains}"
              f" (parametrization: {tuned['schedule']['kind'] if schedule_params is not None else 'none'})")

    log = pipeline.run_closed_loop(
        pipeline.hidden_params,
        use_hidden_robot=True,
        controller_gains=controller_gains,
        schedule_params=schedule_params,
    )

    if not args.skip_pdf:
        plot_logged_summary(log, out_prefix=args.output, show_gains=args.tuned_gains is not None)

    if not args.skip_ref_meshcat:
        visualize(
            pipeline.problem_path,
            log.pose.states,
            out_prefix=args.output,
            dt=pipeline.wheel_dt,
        )


if __name__ == "__main__":
    main()
