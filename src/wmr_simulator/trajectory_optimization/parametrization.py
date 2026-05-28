import numpy as np
import jax.numpy as jnp

from wmr_simulator.planner import compute_reference_trajectory


def sigma(t: jnp.ndarray, total_time: jnp.ndarray) -> jnp.ndarray:
    """5th-order S-curve on [0, T] -> [0, 1]."""
    tau = t / total_time
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def sigma_dot(t: jnp.ndarray, total_time: jnp.ndarray) -> jnp.ndarray:
    tau = t / total_time
    return (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / total_time


def sigma_ddot(t: jnp.ndarray, total_time: jnp.ndarray) -> jnp.ndarray:
    tau = t / total_time
    return (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (total_time**2)


def normalize_time_scaling(time_scaling: str | None) -> str:
    if time_scaling is None:
        return "s_curve"
    normalized = time_scaling.lower().replace("-", "_")
    if normalized in {"s_curve", "scurve"}:
        return "s_curve"
    if normalized == "linear":
        return "linear"
    raise ValueError(f"Unsupported time scaling '{time_scaling}'. Expected 's-curve' or 'linear'.")


def time_scaling_derivatives(
    t: jnp.ndarray,
    total_time: jnp.ndarray,
    time_scaling: str | None = "s_curve",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    selected_time_scaling = normalize_time_scaling(time_scaling)
    if selected_time_scaling == "linear":
        return t / total_time, jnp.ones_like(t) / total_time, jnp.zeros_like(t)
    return sigma(t, total_time), sigma_dot(t, total_time), sigma_ddot(t, total_time)


class Trajectory:
    def __init__(self, time: np.ndarray, poses: np.ndarray, speeds: np.ndarray):
        self.time = np.asarray(time, dtype=float)
        self.poses = np.asarray(poses, dtype=float)
        self.speeds = np.asarray(speeds, dtype=float)

    def linear_speed(self) -> np.ndarray:
        theta = self.poses[:, 2]
        vx = self.speeds[:, 0]
        vy = self.speeds[:, 1]
        return vx * np.cos(theta) + vy * np.sin(theta)


class TrajectoryGenerator:
    def reference_states(self, problem) -> jnp.ndarray:
        raise NotImplementedError("TrajectoryGenerator subclasses must implement reference_states().")

    def generate(self, problem) -> Trajectory:
        raise NotImplementedError("TrajectoryGenerator subclasses must implement generate().")


class PlannerTrajectoryGenerator(TrajectoryGenerator):
    def reference_states(self, problem) -> jnp.ndarray:
        time_grid = problem.planner_time_grid()
        reference_states, _ = compute_reference_trajectory(
            start=problem.start,
            goal=problem.goal,
            intermediate_waypoints=problem.planner_waypoints,
            time=time_grid,
        )
        return jnp.asarray(reference_states, dtype=jnp.float32)

    def generate(self, problem) -> Trajectory:
        reference_states = np.asarray(self.reference_states(problem), dtype=float)
        time_grid = problem.planner_time_grid()
        return Trajectory(time_grid, reference_states[:, :3], reference_states[:, 3:6])


def build_trajectory_generator(
    problem,
    generator_type: str | None = None,
    time_scaling: str | None = None,
) -> TrajectoryGenerator:
    planner_generator_cfg = problem.planner_cfg.get("trajectory_generator", {})
    selected_generator = planner_generator_cfg.get("type", "planner") if generator_type is None else generator_type
    selected_time_scaling = (
        planner_generator_cfg.get("time_scaling", "s_curve")
        if time_scaling is None
        else time_scaling
    )

    if selected_generator == "planner":
        return PlannerTrajectoryGenerator()
    if selected_generator == "bezier":
        from wmr_simulator.trajectory_optimization.bezier import BezierTrajectoryGenerator

        return BezierTrajectoryGenerator(time_scaling=selected_time_scaling)

    raise ValueError(f"Unsupported trajectory generator '{selected_generator}'.")
