import argparse

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.pololu import load_pololu_traj_control_log
from wmr_simulator.types import (
    PhysicalParams,
    physical_params_to_array,
)
from wmr_simulator.visualization.identification import (
    plot_loss_history,
    plot_system_id,
)
from wmr_simulator.visualization.pololu import plot_velocity_difference


def print_param_block(label: str, params: PhysicalParams):
    values_mm = 1000.0 * np.asarray(physical_params_to_array(params), dtype=float)
    print(label)
    print(f"  wheel_radius   = {values_mm[0]:.2f} mm")
    print(f"  base_diameter  = {values_mm[1]:.2f} mm")


def main():
    parser = argparse.ArgumentParser()
    # Problem configuration (contains real robot parameters, noise, optionally reference traj)
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    # Optimization hyper-parameters
    parser.add_argument("--window-length", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-linear-velocity-difference", type=float, default=None)
    parser.add_argument("--max-angular-velocity-difference", type=float, default=None)
    # Initial guess robot parameters
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    # Noise settings
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--stochastic-replay", action="store_true", default=False)
    parser.add_argument("--num-realizations", type=int, default=8)
    parser.add_argument("--replay-wheel-speeds", choices=("true", "noisy"), default="true")

    # Arguments for running pipeline on real experiment log
    parser.add_argument("--pololu-log", type=str, default="Pololu Data/Logs/20cp_constrained_scurve/TR02")
    parser.add_argument("--show-markers", action="store_true")
    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
    )

    pololu_log = load_pololu_traj_control_log(args.pololu_log)
    target_log = pololu_log.to_target_log()
    target_time_s = pololu_log.time_s
    target_reference_states = pololu_log.target_reference_states()

    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        num_realizations=args.num_realizations,
        seed=args.seed,
        reference_trajectories_dir=None,
        window_length=args.window_length,
        deterministic_replay=not args.stochastic_replay,
        target_log=target_log,
        target_time_s=target_time_s,
        target_reference_states=target_reference_states,
        target_odometry_vel_omega=pololu_log.wheel_odometry_vel_omega(),
        replay_wheel_speed_source=args.replay_wheel_speeds,
        max_linear_velocity_difference=args.max_linear_velocity_difference,
        max_angular_velocity_difference=args.max_angular_velocity_difference,
    )
    pipeline = result["pipeline"]
    print_param_block("Initial guess:", init_params)
    print()
    print_param_block("Estimated parameters:", result["estimated_params"])
    print()
    print(f"Final loss: {result['loss_history'][-1]:.8f}")

    out_prefix = "identification_log_{name}".format(name=args.pololu_log[-4:])
    rejection_weights = pipeline.target_loss_weights(pipeline.target_log)
    plot_system_id(
        pipeline=pipeline,
        init_target_log=result["init_target_log"],
        init_log=result["init_replay_log"],
        predicted_log=result["final_replay_log"],
        show_markers=args.show_markers,
        out_prefix=out_prefix,
    )
    plot_velocity_difference(
        pololu_log,
        out_prefix=out_prefix,
        show_markers=args.show_markers,
        rejection_weights=None if rejection_weights is None else np.asarray(rejection_weights, dtype=float),
    )
    plot_loss_history(
        loss_history=result["loss_history"],
        parameter_error_history=None,
    )


if __name__ == "__main__":
    main()
