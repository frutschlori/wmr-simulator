from math import comb

import numpy as np
import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.parametrization import (
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


def initial_bezier_control_points(problem, order: int) -> jnp.ndarray:
    num_control_points = order + 1
    line_samples = np.linspace(0.0, 1.0, num_control_points)[:, None]
    start = problem.start[:2][None, :]
    goal = problem.goal[:2][None, :]
    control_points = start + line_samples * (goal - start)
    return jnp.asarray(control_points, dtype=jnp.float32)


def start_heading_unit(problem) -> jnp.ndarray:
    start_theta = jnp.asarray(problem.start[2], dtype=jnp.float32)
    return jnp.stack([jnp.cos(start_theta), jnp.sin(start_theta)])


def max_start_heading_distance(problem) -> jnp.ndarray:
    start = jnp.asarray(problem.start[:2], dtype=jnp.float32)
    heading = start_heading_unit(problem)
    env_min = jnp.asarray(problem.environment_min, dtype=jnp.float32)
    env_max = jnp.asarray(problem.environment_max, dtype=jnp.float32)
    safe_heading = jnp.where(jnp.abs(heading) > 1e-6, heading, 1.0)
    positive_limits = jnp.where(heading > 1e-6, (env_max - start) / safe_heading, jnp.inf)
    negative_limits = jnp.where(heading < -1e-6, (env_min - start) / safe_heading, jnp.inf)
    return jnp.maximum(jnp.min(jnp.minimum(positive_limits, negative_limits)), 0.0)


def start_heading_control_point(problem, control_point: jnp.ndarray) -> jnp.ndarray:
    start = jnp.asarray(problem.start[:2], dtype=jnp.float32)
    heading = start_heading_unit(problem)
    raw_distance = jnp.dot(jnp.asarray(control_point, dtype=jnp.float32) - start, heading)
    distance = jnp.clip(raw_distance, 0.0, max_start_heading_distance(problem))
    return start + distance * heading


def clamp_control_points(problem, control_points: jnp.ndarray) -> jnp.ndarray:
    env_min = jnp.asarray(problem.environment_min, dtype=jnp.float32)
    env_max = jnp.asarray(problem.environment_max, dtype=jnp.float32)
    clamped = jnp.clip(control_points, env_min, env_max)
    clamped = clamped.at[0].set(jnp.asarray(problem.start[:2], dtype=jnp.float32))
    clamped = clamped.at[1].set(start_heading_control_point(problem, control_points[1]))
    return clamped.at[-1].set(jnp.asarray(problem.goal[:2], dtype=jnp.float32))


def decision_variables_from_control_points(problem, control_points: jnp.ndarray) -> jnp.ndarray:
    control_points = clamp_control_points(problem, control_points)
    start = jnp.asarray(problem.start[:2], dtype=jnp.float32)
    heading = start_heading_unit(problem)
    distance = jnp.dot(control_points[1] - start, heading)
    return jnp.concatenate([distance[None], jnp.ravel(control_points[2:])], axis=0)


def control_points_from_decision_variables(problem, decision_variables: jnp.ndarray) -> jnp.ndarray:
    decision_variables = jnp.ravel(decision_variables)
    start_point = jnp.asarray(problem.start[:2], dtype=jnp.float32)[None, :]
    heading = start_heading_unit(problem)
    distance = jnp.clip(decision_variables[0], 0.0, max_start_heading_distance(problem))
    second_control_point = (start_point[0] + distance * heading)[None, :]
    remaining_control_points = jnp.reshape(decision_variables[1:], (-1, 2))
    control_points = jnp.concatenate([start_point, second_control_point, remaining_control_points], axis=0)
    return clamp_control_points(problem, control_points)


class BezierCurve:
    def __init__(self, control_points):
        self.control_points = jnp.asarray(control_points, dtype=jnp.float32)
        self.degree = self.control_points.shape[0] - 1
        self.first_diff = self.degree * (self.control_points[1:] - self.control_points[:-1])
        self.second_diff = (self.degree - 1) * (self.first_diff[1:] - self.first_diff[:-1])

    def eval(self, s: jnp.ndarray) -> jnp.ndarray:
        return evaluate_bezier(self.control_points, s)

    def evald(self, s: jnp.ndarray) -> jnp.ndarray:
        return evaluate_bezier(self.first_diff, s)

    def evaldd(self, s: jnp.ndarray) -> jnp.ndarray:
        return evaluate_bezier(self.second_diff, s)


def compute_bezier_reference(
    problem,
    control_points: jnp.ndarray,
    time_scaling: str | None = "s_curve",
) -> jnp.ndarray:
    curve = BezierCurve(control_points)

    time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
    total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
    s, s_dot, s_ddot = time_scaling_derivatives(
        time_grid,
        total_time,
        time_scaling=normalize_time_scaling(time_scaling),
    )
    s = jnp.clip(s, 0.0, 1.0)

    position = curve.eval(s)
    dpos_ds = curve.evald(s)
    d2pos_ds2 = curve.evaldd(s)

    velocity = dpos_ds * s_dot[:, None]
    acceleration = dpos_ds * s_ddot[:, None] + d2pos_ds2 * (s_dot[:, None] ** 2)
    theta = jnp.arctan2(dpos_ds[:, 1], dpos_ds[:, 0])

    tangent_norm_sq = jnp.sum(dpos_ds**2, axis=1)
    dtheta_ds = (
        dpos_ds[:, 0] * d2pos_ds2[:, 1] - dpos_ds[:, 1] * d2pos_ds2[:, 0]
    ) / (tangent_norm_sq + 1e-8)
    omega = dtheta_ds * s_dot

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


def tangent_floor_loss(
    problem,
    control_points: jnp.ndarray,
    min_tangent_norm: float = 1e-2,
) -> jnp.ndarray:
    curve = BezierCurve(control_points)
    time_grid = jnp.asarray(problem.sim_time_grid(), dtype=jnp.float32)
    total_time = jnp.asarray(problem.sim_time, dtype=jnp.float32)
    s = jnp.clip(time_grid / total_time, 0.0, 1.0)
    tangent_norm = jnp.sqrt(jnp.sum(curve.evald(s) ** 2, axis=1) + 1e-8)
    violation = jnp.maximum(jnp.asarray(min_tangent_norm, dtype=jnp.float32) - tangent_norm, 0.0)
    return jnp.mean((violation / min_tangent_norm) ** 2)
