import argparse

from wmr_simulator.gain_tuning.analysis import create_gain_surface_gif
from wmr_simulator.gain_tuning.pipeline import resolve_gain_robot_params


def main():
    parser = argparse.ArgumentParser(
        description="Render a GIF of closed-loop loss surfaces over trajectory windows and controller gains."
    )
    parser.add_argument("--problem", type=str, default="problems/figure_eight.yaml")
    parser.add_argument("--reference-trajectories-dir", type=str, default=None)
    parser.add_argument("--num-realizations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-wheel-radius", type=float, default=None)
    parser.add_argument("--fixed-base-diameter", type=float, default=None)
    parser.add_argument("--kx-min", type=float, default=0.0)
    parser.add_argument("--kx-max", type=float, default=20.0)
    parser.add_argument("--kx-points", type=int, default=100)
    parser.add_argument("--ky-min", type=float, default=0.0)
    parser.add_argument("--ky-max", type=float, default=20.0)
    parser.add_argument("--ky-points", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--z-poly-order", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="gain_tracking_error_surface_frames")
    parser.add_argument("--gif-name", type=str, default="gain_surface_animation.gif")
    parser.add_argument("--frame-time", type=float, default=0.02)
    parser.add_argument("--stop-time", type=float, default=1.0)
    args = parser.parse_args()

    robot_params = resolve_gain_robot_params(
        args.problem,
        args.fixed_wheel_radius,
        args.fixed_base_diameter,
    )
    gif_path = create_gain_surface_gif(
        problem_path=args.problem,
        robot_params=robot_params,
        output_dir_name=args.output_dir,
        gif_name=args.gif_name,
        frame_time=args.frame_time,
        stop_time=args.stop_time,
        kx_min=args.kx_min,
        kx_max=args.kx_max,
        kx_points=args.kx_points,
        ky_min=args.ky_min,
        ky_max=args.ky_max,
        ky_points=args.ky_points,
        num_realizations=args.num_realizations,
        seed=args.seed,
        window_size=args.window_size,
        window_stride=args.window_stride,
        z_poly_order=args.z_poly_order,
        reference_trajectories_dir=args.reference_trajectories_dir,
    )
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
