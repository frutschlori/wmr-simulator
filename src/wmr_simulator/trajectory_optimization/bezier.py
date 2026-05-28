from math import comb

import numpy as np
import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.parametrization import (
    Trajectory,
    TrajectoryGenerator,
    normalize_time_scaling,
    time_scaling_derivatives,
)


def bezier_basis(num_control_points: int, s: jnp.ndarray) -> jnp.ndarray:
    degree = num_control_points - 1
    return jnp.stack(
        [
            comb(degree, i) * ((1.0 - s) ** (degree - i)) * (s ** i)
            for i in range(num_control_points)
        ],
        axis=1,
    )


def evaluate_bezier(control_points: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    basis = bezier_basis(control_points.shape[0], s)
    return basis @ control_points


class BezierTrajectoryGenerator(TrajectoryGenerator):
    """
    Simple Bezier reference generator.

    The control points define a single Bezier curve in x/y over the planner time
    horizon. The resulting reference state layout matches existing planner module:
      [x, y, theta, vx, vy, omega, ax, ay]
    """

    def __init__(self, control_points=None, time_scaling: str | None = "s_curve"):
        self.control_points = None if control_points is None else np.asarray(control_points, dtype=float)
        self.time_scaling = normalize_time_scaling(time_scaling)

    def resolve_control_points(self, problem) -> jnp.ndarray:
        if self.control_points is None:
            control_points = np.asarray([problem.start[:2], problem.goal[:2]], dtype=float)
        else:
            control_points = self.control_points
        return jnp.asarray(control_points, dtype=jnp.float32)

    def reference_states_from_control_points(
        self,
        problem,
        control_points: jnp.ndarray,
    ) -> jnp.ndarray:
        control_points = jnp.asarray(control_points, dtype=jnp.float32)
        time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
        total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
        s, s_dot, s_ddot = time_scaling_derivatives(
            time_grid,
            total_time,
            time_scaling=self.time_scaling,
        )
        s = jnp.clip(s, 0.0, 1.0)

        degree = control_points.shape[0] - 1
        if degree < 1:
            raise ValueError("Bezier trajectory requires at least two control points.")

        position = evaluate_bezier(control_points, s)

        first_diff = degree * (control_points[1:] - control_points[:-1])
        dpos_ds = evaluate_bezier(first_diff, s)

        if degree >= 2:
            second_diff = (degree - 1) * (first_diff[1:] - first_diff[:-1])
            d2pos_ds2 = evaluate_bezier(second_diff, s)
        else:
            d2pos_ds2 = jnp.zeros_like(position)

        velocity = dpos_ds * s_dot[:, None]
        acceleration = dpos_ds * s_ddot[:, None] + d2pos_ds2 * (s_dot[:, None] ** 2)

        tangent_heading = jnp.arctan2(dpos_ds[:, 1], dpos_ds[:, 0])
        theta = tangent_heading

        speed_sq = jnp.sum(velocity ** 2, axis=1)
        omega = (
            velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
        ) / (speed_sq + 1e-8)

        return jnp.column_stack(
            [
                position[:, 0],
                position[:, 1],
                theta,
                velocity[:, 0],
                velocity[:, 1],
                omega,
                acceleration[:, 0],
                acceleration[:, 1],
            ]
        )

    def reference_states(self, problem) -> jnp.ndarray:
        control_points = self.resolve_control_points(problem)
        return self.reference_states_from_control_points(problem, control_points)

    def generate(self, problem) -> Trajectory:
        reference_states = np.asarray(self.reference_states(problem), dtype=float)
        time_grid = problem.sim_time_grid()
        return Trajectory(time_grid, reference_states[:, :3], reference_states[:, 3:6])
