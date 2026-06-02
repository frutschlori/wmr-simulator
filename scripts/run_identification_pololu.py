import argparse

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.pololu import load_pololu_traj_control_log
from wmr_simulator.types import (
    PhysicalParams,
    physical_params_to_array,
    physical_params_from_array,
)
from wmr_simulator.visualization.identification import (
    plot_loss_history,
    plot_system_id,
    plot_windowed_replay_trajectory,
)


def print_param_block(label: str, params: PhysicalParams, signed: bool = False, show_mean: bool = False):
    values_mm = 1000.0 * np.asarray(physical_params_to_array(params), dtype=float)
    value_format = "+.2f" if signed else ".2f"
    print(label)
    print(f"  wheel_radius   = {values_mm[0]:{value_format}} mm")
    print(f"  base_diameter  = {values_mm[1]:{value_format}} mm")
    if show_mean:
        print(f"  mean           = {np.mean(np.abs(values_mm)):.2f} mm")


def main():
    parser = argparse.ArgumentParser()
    # Problem configuration (contains real robot parameters, noise, optionally reference traj)
    parser.add_argument("--problem", type=str, default="problems/problem_hidden.yaml")
    # Optimization hyper-parameters
    parser.add_argument("--window-length", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    # Initial guess robot parameters
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.2)
    # Noise settings
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--stochastic-replay", action="store_true", default=False)
    parser.add_argument("--num-realizations", type=int, default=8)
    parser.add_argument("--replay-wheel-speeds", choices=("true", "noisy"), default="true",
                        help="Use true wheel speeds or noisy estimator wheel speeds as replay input.")
    # Optionally load target trajectory from disk
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    # Bootstrap parameters (for identification performance eval), set 1 to disable
    parser.add_argument("--bootstrap-samples", type=int, default=1)

    # Arguments for running pipeline on real experiment log
    parser.add_argument("--pololu-log", type=str, default="Pololu Data/Logs/TR06")
    parser.add_argument("--pololu-start-time", type=float, default=None)
    parser.add_argument("--pololu-stop-time", type=float, default=12.0)
    parser.add_argument("--pololu-wheel-radius", type=float, default=0.016)
    parser.add_argument("--pololu-base-diameter", type=float, default=0.0842)
    parser.add_argument("--pololu-no-trim-stationary", action="store_true")
    args = parser.parse_args()

    is_real_experiment = args.pololu_log is not None
    pololu_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.pololu_wheel_radius),
        base_diameter=jnp.asarray(args.pololu_base_diameter),
    )
    init_params = pololu_params if is_real_experiment else PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
    )

    target_log = None
    target_time_s = None
    target_reference_states = None
    if is_real_experiment:
        pololu_log = load_pololu_traj_control_log(
            args.pololu_log,
            trim_stationary=not args.pololu_no_trim_stationary,
            start_time=args.pololu_start_time,
            stop_time=args.pololu_stop_time,
        )
        target_log = pololu_log.to_target_log(
            wheel_radius=float(pololu_params.wheel_radius),
            base_diameter=float(pololu_params.base_diameter),
        )
        target_time_s = pololu_log.time_s
        target_reference_states = pololu_log.target_reference_states()

    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=None if is_real_experiment else args.reference_trajectories_dir,
        window_length=args.window_length,
        bootstrap_samples=args.bootstrap_samples,
        deterministic_replay=not args.stochastic_replay,
        target_log=target_log,
        target_time_s=target_time_s,
        target_reference_states=target_reference_states,
        replay_wheel_speed_source=args.replay_wheel_speeds,
    )
    pipeline = result["pipeline"]
    if is_real_experiment:
        print_param_block("Used Pololu parameters:", pololu_params)
    else:
        print_param_block("Initial guess:", init_params)
        print()
        print_param_block("Hidden parameters:", pipeline.hidden_params)
    if result["bootstrap"] is not None:
        bootstrap = result["bootstrap"]
        mean_params = physical_params_from_array(bootstrap["parameter_mean"])
        bias = physical_params_from_array(
            bootstrap["parameter_mean"] - physical_params_to_array(pipeline.hidden_params)
        )
        covariance_mm2 = 1e6 * np.asarray(bootstrap["parameter_covariance"])
        std_params = PhysicalParams(
            wheel_radius=jnp.asarray(np.sqrt(covariance_mm2[0, 0]) / 1000.0),
            base_diameter=jnp.asarray(np.sqrt(covariance_mm2[1, 1]) / 1000.0),
        )
        print()
        print(f"Bootstrap samples: {args.bootstrap_samples}")
        print(f"Mean loss: {result['loss_history'][-1]:.8f}")
        print()
        print_param_block("Mean estimated parameters:", mean_params)
        print_param_block("Bias of mean estimate:", bias, signed=True, show_mean=True)
        print_param_block("Standard deviation:", std_params, show_mean=True)
        print("Parameter covariance [mm²]:")
        print(np.array2string(covariance_mm2, precision=4, suppress_small=False))
    else:
        print()
        print_param_block("Estimated parameters:", result["estimated_params"])
        print()
        print(f"Final loss: {result['loss_history'][-1]:.8f}")
        if not is_real_experiment:
            print(f"Final parameter RMSE: {np.sqrt(result['parameter_mse_history'][-1]):.2f} mm")

    plot_system_id(
        pipeline=pipeline,
        init_target_log=result["init_target_log"],
        init_log=result["init_replay_log"],
        predicted_log=result["final_replay_log"],
    )
    plot_windowed_replay_trajectory(
        pipeline=pipeline,
        closed_loop_log=pipeline.target_log,
        replay_log=result["final_replay_log"],
        window_length=args.window_length,
        out_prefix="system_identification_trajectory",
    )
    plot_loss_history(
        loss_history=result["loss_history"],
        parameter_error_history=None if is_real_experiment else result["parameter_mse_history"],
    )


if __name__ == "__main__":
    main()
