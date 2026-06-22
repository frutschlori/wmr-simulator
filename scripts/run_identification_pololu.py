import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

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


def print_param_block(label: str, params: PhysicalParams):
    values = np.asarray(physical_params_to_array(params), dtype=float)
    print(label)
    print(f"  wheel_radius    = {1000.0 * values[0]:.2f} mm")
    print(f"  base_diameter   = {1000.0 * values[1]:.2f} mm")
    print(f"  max_wheel_speed = {values[2]:.2f} rad/s")
    print(f"  time_constant   = {values[3]:.4f} s")


def main():
    parser = argparse.ArgumentParser()
    # Problem configuration (contains robot configuration and optimization defaults)
    parser.add_argument("--problem", type=str, default="problems/pololu.yaml")
    # Optimization hyper-parameters
    parser.add_argument("--window-length", type=int, default=40)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # Initial guess robot parameters
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    parser.add_argument("--init-max-wheel-speed", type=float, default=300.0)
    parser.add_argument("--init-time-constant", type=float, default=0.3)

    # Path to real experiment log
    parser.add_argument("--pololu-log", type=str, default="Pololu Data/Logs/2026_06_22/optimized/50ms_turbo/TR00")
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
        max_wheel_speed=jnp.asarray(args.init_max_wheel_speed),
        time_constant=jnp.asarray(args.init_time_constant),
    )

    pololu_log = load_pololu_traj_control_log(
        args.pololu_log,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
    )

    result = run_single_experiment_identification(
        problem_path=args.problem,
        initial_params=init_params,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        seed=0,
        reference_trajectories_dir=None,
        window_length=args.window_length,
        target_log=pololu_log,
    )
    pipeline = result["pipeline"]
    print_param_block("Initial guess:", init_params)
    print()
    print_param_block("Estimated parameters:", result["estimated_params"])
    print()
    print(f"Final normalized geometry loss: {result['loss_history'][-1]:.8f}")
    print(f"Final normalized motor loss:    {result['motor_loss_history'][-1]:.8f}")

    out_prefix = "identification_log_{name}".format(name=args.pololu_log[-4:])
    plot_system_id(
        pipeline=pipeline,
        init_target_log=result["init_target_log"],
        init_log=result["init_replay_log"],
        predicted_log=result["final_replay_log"],
        out_prefix=out_prefix,
    )
    plot_loss_history(
        loss_history=result["loss_history"],
        motor_loss_history=result["motor_loss_history"],
        parameter_error_history=None,
    )


if __name__ == "__main__":
    main()
