import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.identification.mocap_delay import (
    estimate_mocap_delay_from_log_file,
    print_delay_result,
)
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
    # base_diameter is the *effective* wheelbase: tire scrub in turns is absorbed
    # into it (Borenstein & Feng 1996, E_b), so it may differ from the geometric one.
    print(f"  base_diameter   = {1000.0 * values[1]:.2f} mm (effective wheelbase)")
    print(f"  max_wheel_speed = {values[2]:.2f} rad/s")
    print(f"  time_constant   = {values[3]:.4f} s")


def main():
    parser = argparse.ArgumentParser()
    # Problem configuration (contains robot configuration and optimization defaults)
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    # Optimization hyper-parameters
    parser.add_argument("--window-length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # Initial guess robot parameters
    parser.add_argument("--init-wheel-radius", type=float, default=0.02)
    parser.add_argument("--init-base-diameter", type=float, default=0.1)
    parser.add_argument("--init-max-wheel-speed", type=float, default=300.0)
    parser.add_argument("--init-time-constant", type=float, default=0.3)

    # Path to real experiment log
    parser.add_argument("--pololu-log", type=str,
                        # default="Pololu Data/Experiments/2026_07_06/04_New_Mocap_timestamps/decoded/TR05.csv")
                        default="Pololu Data/Experiments/2026_07_01/TR03.csv")
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    # Zero-phase moving-average window (seconds) applied to the mocap positions in the
    # log loader before the spline fit; 0 disables.
    parser.add_argument("--mocap-filter-window", type=float, default=0.0)
    # Smoothing-spline parameters for the mocap poses/twists (see
    # pololu.pose_smoothing.fit_pose_splines and the log loader).
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--spline-noise-std-xy", type=float, default=1e-3)
    parser.add_argument("--spline-noise-std-yaw", type=float, default=5e-3)
    parser.add_argument("--spline-smoothing-factor", type=float, default=1.0)
    # Mocap transport latency (seconds); mocap timestamps are shifted back by this
    # before identification. Default: estimator.mocap_delay from the problem yaml.
    parser.add_argument("--mocap-delay", type=float, default=None)
    # Estimate the delay from the log first (mocap yaw rate vs IMU gyro z
    # cross-correlation, see identification/mocap_delay.py) and use the estimate.
    parser.add_argument("--estimate-mocap-delay", action="store_true", default=False)
    parser.add_argument("--mocap-delay-search-range", type=float, default=0.2)
    args = parser.parse_args()

    init_params = PhysicalParams(
        wheel_radius=jnp.asarray(args.init_wheel_radius),
        base_diameter=jnp.asarray(args.init_base_diameter),
        max_wheel_speed=jnp.asarray(args.init_max_wheel_speed),
        time_constant=jnp.asarray(args.init_time_constant),
    )

    mocap_delay = args.mocap_delay
    mocap_delay_source = "--mocap-delay"
    if mocap_delay is None:
        with open(args.problem, "r", encoding="utf-8") as file:
            mocap_delay = float(yaml.safe_load(file).get("estimator", {}).get("mocap_delay", 0.0))
        mocap_delay_source = "problem yaml"
    if args.estimate_mocap_delay:
        try:
            delay_result = estimate_mocap_delay_from_log_file(
                args.pololu_log,
                max_delay_s=args.mocap_delay_search_range,
                mocap_filter_window_s=args.mocap_filter_window,
            )
            print_delay_result(delay_result)
            mocap_delay = delay_result["delay_s"]
            mocap_delay_source = "estimated from log"
        except ValueError as error:
            print(f"Mocap delay estimation skipped: {error}")
    print(f"Mocap delay compensation: {1000.0 * mocap_delay:.2f} ms ({mocap_delay_source})")

    pololu_log = load_pololu_traj_control_log(
        args.pololu_log,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
        mocap_filter_window_s=args.mocap_filter_window,
        mocap_delay_s=mocap_delay,
        spline_order=args.spline_order,
        spline_noise_std_xy=args.spline_noise_std_xy,
        spline_noise_std_yaw=args.spline_noise_std_yaw,
        spline_smoothing_factor=args.spline_smoothing_factor,
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
