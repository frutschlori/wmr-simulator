import argparse

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.types import (
    PhysicalParams,
    physical_params_to_array,
    physical_params_from_array,
)
from wmr_simulator.visualization.identification import (
    plot_loss_history,
    plot_system_id,
    plot_velocity_difference,
)


def print_param_block(label: str, params: PhysicalParams, signed: bool = False, show_mean: bool = False):
    values_mm = 1000.0 * np.asarray(physical_params_to_array(params), dtype=float)
    value_format = "+.2f" if signed else ".2f"
    print(label)
    print(f"  wheel_radius   = {values_mm[0]:{value_format}} mm")
    print(f"  base_diameter  = {values_mm[1]:{value_format}} mm")
    if show_mean:
        print(f"  average geometry error = {np.mean(np.abs(values_mm[:2])):.2f} mm")

    print(f"  max_wheel_speed = {float(params.max_wheel_speed):{value_format}} rad/s")
    print(f"  time_constant  = {float(params.time_constant):{value_format}} s")


def main():
    parser = argparse.ArgumentParser()
    # Problem configuration (contains hidden robot parameters, noise, optionally reference traj)
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    # Optimization hyper-parameters
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--window-length", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--motor-learning-rate", type=float, default=1e-3)
    # Ignore slippy measurements where mocap and odometry velocities mismatch too much
    parser.add_argument("--max-linear-velocity-difference", type=float, default=None)
    parser.add_argument("--max-angular-velocity-difference", type=float, default=None)
    # Initial guess robot parameters
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    parser.add_argument("--init-max-wheel-speed", type=float, default=300.0)
    parser.add_argument("--init-time-constant", type=float, default=0.2)
    # Noise settings
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--stochastic-replay", action="store_true")
    parser.add_argument("--num-realizations", type=int, default=1)
    parser.add_argument("--replay-wheel-speeds", choices=("true", "noisy"), default="noisy")
    # Optionally load target trajectory from disk
    parser.add_argument("--reference-trajectories-dir", type=str, default="trajectory_exports")
    # parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--show-markers", action="store_true")
    parser.add_argument("--hide-velocity-difference-plot", action="store_true", default=False)

    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
        max_wheel_speed=jnp.asarray(args.init_max_wheel_speed),
        time_constant=jnp.asarray(args.init_time_constant),
    )

    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        motor_learning_rate=args.motor_learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=args.reference_trajectories_dir,
        window_length=args.window_length,
        bootstrap_samples=args.bootstrap_samples,
        deterministic_replay=not args.stochastic_replay,
        replay_wheel_speed_source=args.replay_wheel_speeds,
        max_linear_velocity_difference=args.max_linear_velocity_difference,
        max_angular_velocity_difference=args.max_angular_velocity_difference,
    )
    pipeline = result["pipeline"]
    print_param_block("Initial guess:", init_params)
    print()
    print_param_block("Hidden parameters:", pipeline.hidden_params)
    if result["bootstrap"] is not None:
        bootstrap = result["bootstrap"]
        mean_params = physical_params_from_array(bootstrap["parameter_mean"])
        bias = physical_params_from_array(
            bootstrap["parameter_mean"] - physical_params_to_array(pipeline.hidden_params)
        )
        covariance = np.asarray(bootstrap["parameter_covariance"], dtype=float)
        covariance_scales = np.asarray([1000.0, 1000.0, 1.0, 1.0], dtype=float)
        covariance_scaled = covariance * np.outer(covariance_scales, covariance_scales)
        std_params = PhysicalParams(
            wheel_radius=jnp.asarray(np.sqrt(covariance_scaled[0, 0]) / 1000.0),
            base_diameter=jnp.asarray(np.sqrt(covariance_scaled[1, 1]) / 1000.0),
            max_wheel_speed=jnp.asarray(np.sqrt(covariance_scaled[2, 2])),
            time_constant=jnp.asarray(np.sqrt(covariance_scaled[3, 3])),
        )
        print()
        print(f"Bootstrap samples: {args.bootstrap_samples}")
        print(f"Mean normalized geometry loss: {result['loss_history'][-1]:.8f}")
        print(f"Mean normalized motor loss:    {result['motor_loss_history'][-1]:.8f}")
        print()
        print_param_block("Mean estimated parameters:", mean_params)
        print_param_block("Bias of mean estimate:", bias, signed=True, show_mean=True)
        print_param_block("Standard deviation:", std_params, show_mean=True)
        # print("Parameter covariance [mm, mm, rad/s, s]:")
        # print(np.array2string(covariance_scaled, precision=4, suppress_small=False))
    else:
        print()
        print_param_block("Estimated parameters:", result["estimated_params"])
        print()
        print(f"Final normalized geometry loss: {result['loss_history'][-1]:.8f}")
        print(f"Final normalized motor loss:    {result['motor_loss_history'][-1]:.8f}")
        print(f"Final mixed-unit parameter RMSE: {np.sqrt(result['parameter_mse_history'][-1]):.4f}")

    plot_system_id(
        pipeline=pipeline,
        init_target_log=result["init_target_log"],
        init_log=result["init_replay_log"],
        predicted_log=result["final_replay_log"],
        show_markers=args.show_markers,
        out_prefix="identification_sim",
    )
    if not args.hide_velocity_difference_plot:
        plot_velocity_difference(
            pipeline=pipeline,
            target_log=result["init_target_log"],
            out_prefix="identification_sim",
            wheel_speed_source=args.replay_wheel_speeds,
            show_markers=args.show_markers,
        )
    plot_loss_history(
        loss_history=result["loss_history"],
        motor_loss_history=result["motor_loss_history"],
        parameter_error_history=result["parameter_mse_history"],
    )


if __name__ == "__main__":
    main()
