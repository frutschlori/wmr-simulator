import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import yaml


TRAJECTORY_CHOICES = ("figure_eight", "circle", "spiral", "square")


def ensure_matplotlib_cache_dir() -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "wmr_simulator_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.fspath(matplotlib_cache))


def time_grid_from_problem(problem: dict) -> np.ndarray:
    dt = float(problem["geometry_controller_dt"])
    sim_time = float(problem["sim_time"])
    steps = int(np.round(sim_time / dt))
    return np.linspace(0.0, steps * dt, steps + 1)


def reference_from_derivatives(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
) -> np.ndarray:
    theta = np.unwrap(np.arctan2(vy, vx))
    speed_sq = vx**2 + vy**2
    omega = (vx * ay - vy * ax) / np.maximum(speed_sq, 1e-10)
    return np.column_stack([x, y, theta, vx, vy, omega, ax, ay])


def finite_difference_reference(x: np.ndarray, y: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    dt = float(time_s[1] - time_s[0])
    edge_order = 2 if len(time_s) > 2 else 1
    vx = np.gradient(x, dt, edge_order=edge_order)
    vy = np.gradient(y, dt, edge_order=edge_order)
    ax = np.gradient(vx, dt, edge_order=edge_order)
    ay = np.gradient(vy, dt, edge_order=edge_order)
    return reference_from_derivatives(x, y, vx, vy, ax, ay)


def make_circle(time_s: np.ndarray, radius: float, center: tuple[float, float], turns: float) -> np.ndarray:
    total_time = float(time_s[-1])
    phi_dot = 2.0 * np.pi * turns / total_time
    phi = phi_dot * time_s
    cx, cy = center
    x = cx + radius * np.cos(phi)
    y = cy + radius * np.sin(phi)
    vx = -radius * phi_dot * np.sin(phi)
    vy = radius * phi_dot * np.cos(phi)
    ax = -radius * phi_dot**2 * np.cos(phi)
    ay = -radius * phi_dot**2 * np.sin(phi)
    return reference_from_derivatives(x, y, vx, vy, ax, ay)


def make_figure_eight(time_s: np.ndarray, radius: float, center: tuple[float, float], turns: float) -> np.ndarray:
    total_time = float(time_s[-1])
    phi_dot = 2.0 * np.pi * turns / total_time
    phi = phi_dot * time_s
    cx, cy = center
    x = cx + radius * np.sin(phi)
    y = cy + 0.5 * radius * np.sin(2.0 * phi)
    vx = radius * phi_dot * np.cos(phi)
    vy = radius * phi_dot * np.cos(2.0 * phi)
    ax = -radius * phi_dot**2 * np.sin(phi)
    ay = -2.0 * radius * phi_dot**2 * np.sin(2.0 * phi)
    return reference_from_derivatives(x, y, vx, vy, ax, ay)


def make_spiral(
    time_s: np.ndarray,
    start_radius: float,
    spiral_coefficient: float,
    center: tuple[float, float],
    turns: float,
) -> np.ndarray:
    total_time = float(time_s[-1])
    s = time_s / total_time
    radius = start_radius + spiral_coefficient * s
    if np.any(radius <= 0.0):
        raise ValueError("spiral radius must stay positive over the configured sim_time.")

    radius_dot = spiral_coefficient / total_time
    phi_dot = 2.0 * np.pi * turns / total_time
    phi = phi_dot * time_s
    cx, cy = center
    x = cx + radius * np.cos(phi)
    y = cy + radius * np.sin(phi)
    vx = radius_dot * np.cos(phi) - radius * phi_dot * np.sin(phi)
    vy = radius_dot * np.sin(phi) + radius * phi_dot * np.cos(phi)
    ax = -2.0 * radius_dot * phi_dot * np.sin(phi) - radius * phi_dot**2 * np.cos(phi)
    ay = 2.0 * radius_dot * phi_dot * np.cos(phi) - radius * phi_dot**2 * np.sin(phi)
    return reference_from_derivatives(x, y, vx, vy, ax, ay)


def make_square(time_s: np.ndarray, side_length: float, center: tuple[float, float]) -> np.ndarray:
    cx, cy = center
    half_side = 0.5 * side_length
    segment = 4.0 * (time_s / float(time_s[-1]))
    segment = np.where(segment >= 4.0, 0.0, segment)

    x = np.empty_like(time_s, dtype=float)
    y = np.empty_like(time_s, dtype=float)
    first = segment < 1.0
    second = (segment >= 1.0) & (segment < 2.0)
    third = (segment >= 2.0) & (segment < 3.0)
    fourth = ~(first | second | third)

    x[first] = cx - half_side + side_length * segment[first]
    y[first] = cy - half_side
    x[second] = cx + half_side
    y[second] = cy - half_side + side_length * (segment[second] - 1.0)
    x[third] = cx + half_side - side_length * (segment[third] - 2.0)
    y[third] = cy + half_side
    x[fourth] = cx - half_side
    y[fourth] = cy + half_side - side_length * (segment[fourth] - 3.0)
    return finite_difference_reference(x, y, time_s)


def generate_reference_states(args, time_s: np.ndarray) -> np.ndarray:
    center = (args.center_x, args.center_y)
    if args.trajectory == "figure_eight":
        return make_figure_eight(time_s, args.radius, center, args.turns)
    if args.trajectory == "circle":
        return make_circle(time_s, args.radius, center, args.turns)
    if args.trajectory == "spiral":
        return make_spiral(time_s, args.start_radius, args.spiral_coefficient, center, args.turns)
    if args.trajectory == "square":
        return make_square(time_s, args.side_length, center)
    raise ValueError(f"Unsupported trajectory type: {args.trajectory}")


def save_reference_pickle(
    reference_states: np.ndarray,
    out_dir: Path,
    trajectory_type: str,
    dt: float,
    metadata: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trajectory_type}.pkl"
    payload = {
        "reference_states": np.asarray(reference_states),
        "dt": float(dt),
        **metadata,
    }
    with open(out_path, "wb") as file:
        pickle.dump(payload, file)
    return out_path


def run_closed_loop(problem_path: str, reference_states: np.ndarray, seed: int):
    import jax.numpy as jnp

    from wmr_simulator.simulation import SimulationPipeline

    pipeline = SimulationPipeline(problem_path=problem_path, seed=seed, reference_trajectories_dir=None)
    return pipeline.run_closed_loop(
        pipeline.hidden_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
        wheel_speed_log_source="estimated",
        reference_states=jnp.asarray(reference_states, dtype=jnp.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate baseline reference trajectories and closed-loop PDFs.")
    parser.add_argument("--problem", default="problems/pololu.yaml")
    parser.add_argument("--trajectory", choices=("figure_eight", "circle", "spiral", "square"),
                        default="square")
    parser.add_argument("--out-dir", default="trajectory_exports/baselines", help="Directory for the reference pickle.")
    parser.add_argument("--pdf-dir", default="visualize/baseline_references", help="Directory for the closed-loop PDF.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--radius", type=float, default=0.5, help="Circle radius, or figure-eight lobe scale [m].")
    parser.add_argument("--start-radius", type=float, default=0.2, help="Spiral radius at t=0 [m].")
    parser.add_argument("--spiral-coefficient", type=float, default=0.3, help="Spiral radius change over sim_time [m].")
    parser.add_argument("--side-length", type=float, default=1.0, help="Square side length [m].")
    parser.add_argument("--turns", type=float, default=1.0, help="Revolutions over sim_time for circle, figure eight, and spiral.")
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_matplotlib_cache_dir()
    problem_path = os.fspath(args.problem)
    with open(problem_path, "r", encoding="utf-8") as file:
        problem = yaml.safe_load(file)

    time_s = time_grid_from_problem(problem)
    dt = float(problem["geometry_controller_dt"])
    reference_states = generate_reference_states(args, time_s)
    metadata = {
        "trajectory_type": args.trajectory,
        "problem_path": problem_path,
        "sim_time": float(problem["sim_time"]),
        "center": (float(args.center_x), float(args.center_y)),
        "radius": float(args.radius),
        "start_radius": float(args.start_radius),
        "spiral_coefficient": float(args.spiral_coefficient),
        "side_length": float(args.side_length),
        "turns": float(args.turns),
    }

    pickle_path = save_reference_pickle(
        reference_states=reference_states,
        out_dir=Path(args.out_dir),
        trajectory_type=args.trajectory,
        dt=dt,
        metadata=metadata,
    )

    log = run_closed_loop(problem_path, reference_states, seed=args.seed)
    from wmr_simulator.visualization.pololu import plot_logged_summary

    pdf_path = plot_logged_summary(
        log,
        out_prefix=f"{args.trajectory}_closed_loop",
        out_dir=Path(args.pdf_dir),
    )

    print(f"Saved reference pickle: {pickle_path}")
    print(f"Saved closed-loop PDF: {pdf_path}")


if __name__ == "__main__":
    main()
