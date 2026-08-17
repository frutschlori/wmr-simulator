import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.optimizers import OPTIMIZERS, describe_optimizer
from wmr_simulator.gain_tuning.pipeline import (resolve_gain_robot_params, run_gain_tuning_experiment)
from wmr_simulator.types import print_controller_gains, print_physical_params
from wmr_simulator.visualization.gain_tuning import (
    TRAINING_KEY_NAMESPACE,
    plot_controller_tuning_errors,
    plot_gain_tuning_summary,
    plot_training_trajectory_summary,
    plot_validation_trajectory_summary,
    realization_keys_for_set,
    rollout_realizations,
)
from wmr_simulator.visualization.identification import plot_loss_history


def print_loss_breakdown(label: str, component_history: dict[str, list[float]] | None) -> None:
    if component_history is None:
        return
    final_terms = {name: float(values[-1]) for name, values in component_history.items()}
    total = sum(final_terms.values())
    print(label)
    for name in ("tracking", "velocity_tracking", "input", "input_delta", "gain_delta"):
        if name not in final_terms:
            continue
        value = final_terms[name]
        share = 100.0 * value / total if total > 0.0 else 0.0
        print(f"  {name:<16}: {value:.8f} ({share:.2f}%)")


def save_tuning_result(out_path: str, problem_path: str, result: dict) -> None:
    import yaml

    from wmr_simulator.gain_parametrization import to_cfg

    schedule_params = result.get("schedule_params")
    payload = {
        "problem": problem_path,
        "gains": [float(gain) for gain in result["optimized_gains"]],
        "static_gains": (
            None
            if result.get("static_gains") is None
            else [float(gain) for gain in result["static_gains"]]
        ),
        "schedule_enabled": bool(result["schedule_enabled"]),
        "schedule": None if schedule_params is None else to_cfg(schedule_params),
        "final_loss": float(result["final_loss"]),
        "final_validation_loss": (
            None if result["final_validation_loss"] is None else float(result["final_validation_loss"])
        ),
        "static_final_loss": (
            None if result["static_final_loss"] is None else float(result["static_final_loss"])
        ),
        "static_final_validation_loss": (
            None
            if result["static_final_validation_loss"] is None
            else float(result["static_final_validation_loss"])
        ),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    print(f"Saved tuning result to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports/gain_optimized_test")
    parser.add_argument("--validation-split", type=float, default=GAIN_TUNING_DEFAULTS["validation_split"])
    # Optimization hyper-parameters
    parser.add_argument("--num-lhs-points", type=int, default=GAIN_TUNING_DEFAULTS["num_lhs_points"]) # points on initial search grid, 0 to disable
    parser.add_argument("--num-adam-optimizations", type=int, default=GAIN_TUNING_DEFAULTS["num_adam_optimizations"]) # number of best candidates to refine
    parser.add_argument("--steps", type=int, default=GAIN_TUNING_DEFAULTS["steps"])                 # adam steps / bfgs inner budget
    parser.add_argument("--learning-rate", type=float, default=GAIN_TUNING_DEFAULTS["learning_rate"])      # adam learning rate (unused by bfgs)
    # Refinement optimizer for the selected LHS starts. "bfgs" ignores
    # --learning-rate (its line search sets the step length) and reads --steps
    # as a total inner budget split into restarts of 40.
    parser.add_argument("--optimizer", choices=list(OPTIMIZERS), default=GAIN_TUNING_DEFAULTS["optimizer"])
    parser.add_argument("--num-realizations", type=int, default=GAIN_TUNING_DEFAULTS["num_realizations"]) # noise realizations over 1 trajectory
    # Randomized start pose per realization (0 disables): the robot is placed off
    # the reference start, so the tracking gains have real error to act on.
    parser.add_argument("--init-offset-radius", type=float, default=GAIN_TUNING_DEFAULTS["init_offset_radius"])
    parser.add_argument("--init-offset-angle", type=float, default=GAIN_TUNING_DEFAULTS["init_offset_angle"])
    # Drop training rollouts whose loss at the initial gains is this many times
    # the median rollout's; one diverging rollout otherwise stalls the BFGS line
    # search and the tuner silently returns the stock gains. 0 disables.
    parser.add_argument("--outlier-loss-factor", type=float, default=GAIN_TUNING_DEFAULTS["outlier_loss_factor"])
    # Matches the active-learning experiment default so the standalone script and
    # the tune-gains stage produce the same result on the same inputs.
    parser.add_argument("--seed", type=int, default=0)
    # Loss weights
    parser.add_argument("--position-tracking-weight", type=float, default=GAIN_TUNING_DEFAULTS["position_tracking_weight"])
    parser.add_argument("--heading-tracking-weight", type=float, default=GAIN_TUNING_DEFAULTS["heading_tracking_weight"])
    parser.add_argument("--velocity-tracking-weight", type=float, default=GAIN_TUNING_DEFAULTS["velocity_tracking_weight"])
    parser.add_argument("--input-weight", type=float, default=GAIN_TUNING_DEFAULTS["input_weight"])
    parser.add_argument("--input-delta-weight", type=float, default=GAIN_TUNING_DEFAULTS["input_delta_weight"])
    # Penalty on step-to-step yaw-rate change (normalized by omega_max); discourages
    # gains that oscillate omega on the real robot. 0 disables.
    parser.add_argument("--omega-delta-weight", type=float, default=GAIN_TUNING_DEFAULTS["omega_delta_weight"])
    # Gain bounds
    parser.add_argument("--k-min-stab", type=float, default=GAIN_TUNING_DEFAULTS["k_min_stab"])
    parser.add_argument("--k-max-stab", type=float, default=GAIN_TUNING_DEFAULTS["k_max_stab"])
    parser.add_argument("--k-max-rest", type=float, default=GAIN_TUNING_DEFAULTS["k_max_rest"])
    # Optional overwrite of robot model parameters
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)
    parser.add_argument("--num-summary-training-trajectories", type=int, default=None)
    # Gain schedule: jointly tune base gains + outer-gain schedule (W), default follows problem yaml, --no-gain-schedule forces W=0 (static)
    parser.add_argument("--gain-schedule", action=argparse.BooleanOptionalAction, default=GAIN_TUNING_DEFAULTS["gain_parametrization"])
    parser.add_argument("--gain-delta-weight", type=float, default=GAIN_TUNING_DEFAULTS["gain_delta_weight"])
    # Also run the static routine (LHS + multistart Adam on the base gains only)
    # as an independent controller option next to the parametrized one; it takes
    # its own step count / learning rate (defaults to --steps / --learning-rate),
    # while the parametrized run uses --steps and --learning-rate, which
    # typically wants a lower rate than the static search.
    parser.add_argument("--static-tune", action=argparse.BooleanOptionalAction, default=GAIN_TUNING_DEFAULTS["static_tune"])
    parser.add_argument("--static-tune-steps", type=int, default=GAIN_TUNING_DEFAULTS["static_tune_steps"])
    parser.add_argument("--static-tune-learning-rate", type=float, default=GAIN_TUNING_DEFAULTS["static_tune_learning_rate"])
    # The standalone script has no prior iteration to refine from -- like
    # active-learning iteration 1, seed the parametrization run's presearch
    # with the static run's converged gains as extra candidates, since both
    # presearches are otherwise the same evaluation (identity parametrization).
    # Disable to test the two runs fully independently.
    parser.add_argument(
        "--seed-parametrization-from-static", action=argparse.BooleanOptionalAction, default=True
    )
    # Refine from a prior result: narrow each run's LHS presearch to a +/- band
    # around its init gains, and/or warm-start the parametrization from the
    # problem's gain_parametrization (theta) instead of the identity mapping.
    parser.add_argument("--presearch-relative-range", type=float, default=GAIN_TUNING_DEFAULTS["presearch_relative_range"])
    parser.add_argument("--warm-start-schedule", action=argparse.BooleanOptionalAction, default=GAIN_TUNING_DEFAULTS["warm_start_schedule"])
    # Learned residual dynamics checkpoint (scripts/train_residual_model.py); tuning
    # then rolls out the residual-augmented dynamics (model params stay fixed).
    # parser.add_argument("--residual-model", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--residual-model", type=str, default=None)
    # Tuning result (gains + trained gain parametrization) is saved here as YAML;
    # the parametrization block drops into the problem yaml's controller section.
    parser.add_argument("--out", type=str, default="models/tuned_gains.yaml")
    args = parser.parse_args()

    residual_model = None
    if args.residual_model is not None:
        from wmr_simulator.residual_model import load_residual_model

        residual_model, checkpoint = load_residual_model(args.residual_model)
        print(f"Loaded residual dynamics model: {args.residual_model}")
        print(f"  config: {checkpoint['config']}")

    robot_params = resolve_gain_robot_params(args.problem, args.fixed_wheel_radius, args.fixed_base_diameter)
    result = run_gain_tuning_experiment(
        problem_path=args.problem,
        robot_params=robot_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
        validation_split=args.validation_split,
        position_tracking_weight=args.position_tracking_weight,
        heading_tracking_weight=args.heading_tracking_weight,
        velocity_tracking_weight=args.velocity_tracking_weight,
        input_weight=args.input_weight,
        input_delta_weight=args.input_delta_weight,
        omega_delta_weight=args.omega_delta_weight,
        k_min_stab=args.k_min_stab,
        k_max_stab=args.k_max_stab,
        k_max_rest=args.k_max_rest,
        num_lhs_points=args.num_lhs_points,
        num_adam_optimizations=args.num_adam_optimizations,
        optimizer=args.optimizer,
        schedule_enabled=args.gain_schedule,
        gain_delta_weight=args.gain_delta_weight,
        residual_model=residual_model,
        static_tune=args.static_tune,
        static_tune_steps=args.static_tune_steps,
        static_tune_learning_rate=args.static_tune_learning_rate,
        seed_parametrization_from_static=args.seed_parametrization_from_static,
        presearch_relative_range=args.presearch_relative_range,
        warm_start_schedule=args.warm_start_schedule,
        init_offset_radius=args.init_offset_radius,
        init_offset_angle=args.init_offset_angle,
        outlier_loss_factor=args.outlier_loss_factor,
    )
    pipeline = result["pipeline"]
    print_physical_params("Robot parameters used for gain tuning:", robot_params)
    print_controller_gains("Initial gains:", pipeline.gains)
    if result["static_gains"] is not None:
        print_controller_gains("Static tune gains:", result["static_gains"])
    print_controller_gains("Optimized gains:", result["optimized_gains"])
    print(f"Position tracking weight: {args.position_tracking_weight:.8g}")
    print(f"Heading tracking weight: {args.heading_tracking_weight:.8g}")
    print(f"Velocity tracking weight: {args.velocity_tracking_weight:.8g}")
    print(f"Input regularization weight: {args.input_weight:.8g}")
    print(f"Input delta regularization weight: {args.input_delta_weight:.8g}")
    print(f"Omega delta regularization weight: {args.omega_delta_weight:.8g}")
    print(f"Training trajectories: {pipeline.training_reference_trajectories.shape[0]}")
    print(f"Validation trajectories: {pipeline.validation_reference_trajectories.shape[0]}")
    print(f"Stable gain search range: [{args.k_min_stab:.8g}, {args.k_max_stab:.8g}]")
    print(f"I/D motor gain search max: {args.k_max_rest:.8g}")
    print(f"LHS points: {args.num_lhs_points}")
    print(f"Refinement starts: {args.num_adam_optimizations}")
    print(f"Refinement: {describe_optimizer(args.optimizer, args.steps, args.learning_rate)}")
    print(f"Final loss: {result['final_loss']:.8f}")
    print_loss_breakdown("Final training loss components:", result["loss_component_history"])
    if result["final_validation_loss"] is not None:
        print(f"Final validation loss: {result['final_validation_loss']:.8f}")
        print_loss_breakdown("Final validation loss components:", result["validation_loss_component_history"])

    print(f"Gain schedule enabled: {result['schedule_enabled']}")
    if result.get("schedule_params") is not None:
        import numpy as _np

        from wmr_simulator.gain_parametrization import BoundedReferenceParams, ErrorMlpParams, num_params

        schedule_params = result["schedule_params"]
        print(f"Gain delta weight: {args.gain_delta_weight:.8g}")
        if isinstance(schedule_params, BoundedReferenceParams):
            print("Scheduled indices:", list(map(int, schedule_params.scheduled_indices)))
            print("Feature scale [v_max, omega_max]:", list(map(float, pipeline.gain_schedule_feature_scale)))
            print("rho:", _np.array2string(_np.asarray(schedule_params.rho), precision=5))
            print("W (rows = scheduled gains, cols = [v_d, |omega_d|]):")
            print(_np.array2string(_np.asarray(schedule_params.W), precision=5))
        elif isinstance(schedule_params, ErrorMlpParams):
            from wmr_simulator.gain_parametrization.error_mlp import hidden_sizes

            print(f"Error-MLP parametrization: hidden sizes {list(hidden_sizes(schedule_params))}, "
                  f"{num_params(schedule_params)} trainable parameters")
            print("Scheduled indices:", list(map(int, schedule_params.scheduled_indices)))
            print(f"Factor bound: {float(schedule_params.bound):.5g} "
                  f"({'learned' if schedule_params.learn_bound else 'fixed'})")
            print(f"Spectral norm cap: {float(schedule_params.spectral_norm_cap):.5g} (0 = disabled)")
            print("Feature scale:", _np.array2string(_np.asarray(schedule_params.feature_scale), precision=5))

    save_tuning_result(args.out, args.problem, result)

    # The summary panels show all realizations of the summary trajectory, each
    # from its own start offset and under its own noise draw -- the conditions
    # the loss was averaged over. The keys are split over the whole training set
    # and then sliced, since that is how the objective derived them.
    summary_offsets = result["summary_start_offsets"][None, ...]
    summary_references = pipeline.training_reference_trajectories[:1]
    summary_robot_keys, summary_estimator_keys = realization_keys_for_set(
        result["realizations"],
        int(pipeline.training_reference_trajectories.shape[0]),
        TRAINING_KEY_NAMESPACE,
    )
    summary_robot_keys = summary_robot_keys[:1]
    summary_estimator_keys = summary_estimator_keys[:1]
    summary_realization_poses = {
        name: rollout_realizations(
            pipeline,
            robot_params,
            summary_references,
            summary_offsets,
            summary_robot_keys,
            summary_estimator_keys,
            controller_gains=gains,
            schedule_params=schedule,
        )[0]
        for name, gains, schedule in (
            ("init", None, None),
            ("tuned", result["optimized_gains"], result["schedule_params"]),
            ("static", result["static_gains"], None),
        )
        if not (name == "static" and result["static_gains"] is None)
    }
    plot_gain_tuning_summary(
        pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
        static_log=result.get("static_hidden_log"),
        init_realization_poses=summary_realization_poses["init"],
        tuned_realization_poses=summary_realization_poses["tuned"],
        static_realization_poses=summary_realization_poses.get("static"),
        out_prefix="summary_gain_tuning",
    )
    plot_training_trajectory_summary(
        pipeline,
        robot_params=robot_params,
        tuned_gains=result["optimized_gains"],
        start_offsets=result["training_start_offsets"],
        realizations=result["realizations"],
        schedule_params=result["schedule_params"],
        static_gains=result["static_gains"],
        max_trajectories=args.num_summary_training_trajectories,
        out_prefix="summary_training",
    )
    plot_validation_trajectory_summary(
        pipeline,
        robot_params=robot_params,
        tuned_gains=result["optimized_gains"],
        start_offsets=result["validation_start_offsets"],
        realizations=result["realizations"],
        schedule_params=result["schedule_params"],
        static_gains=result["static_gains"],
        out_prefix="summary_validation",
    )
    plot_controller_tuning_errors(
        pipeline=pipeline,
        init_log=result["init_hidden_log"],
        tuned_log=result["final_hidden_log"],
        static_log=result.get("static_hidden_log"),
    )
    plot_loss_history(
        loss_history=result["loss_history"],
        validation_loss_history=result["validation_loss_history"],
        loss_component_history=result["loss_component_history"],
        validation_loss_component_history=result["validation_loss_component_history"],
        out_prefix="ctrl_tuning",
    )
    if result["static_loss_history"] is not None:
        plot_loss_history(
            loss_history=result["static_loss_history"],
            validation_loss_history=result["static_validation_loss_history"],
            loss_component_history=result["static_loss_component_history"],
            validation_loss_component_history=result["static_validation_loss_component_history"],
            out_prefix="ctrl_tuning_static",
        )


if __name__ == "__main__":
    main()
