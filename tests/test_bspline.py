"""Clamped uniform B-spline parametrization used by the FIM trajectory
optimizer.

The load-bearing claims are that the basis is a constant linear map from
control points to sampled curve, and that the *approximating* properties the
optimizer relies on hold (partition of unity, convex hull, clamped ends,
locality).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline
from wmr_simulator.trajectory_optimization.bspline import (
    BSplinePlan,
    DEFAULT_SPLINE_DEGREE,
    bspline_basis_matrices,
    clamped_uniform_knots,
)

PROBLEM = "problems/pololu_gains.yaml"
NUM_CONTROL_POINTS = 8


@pytest.fixture(scope="module")
def identification_pipeline():
    return TrajectoryOptimizationPipeline(
        problem_path=PROBLEM, objective_mode="identification"
    )


@pytest.fixture(scope="module")
def gain_tuning_pipeline():
    return TrajectoryOptimizationPipeline(
        problem_path=PROBLEM, objective_mode="gain-tuning"
    )


def test_basis_is_a_partition_of_unity():
    """Rows of B0 sum to 1 and are non-negative -- this is what makes the curve
    an affine combination of its control points, hence bounded by their convex
    hull, hence actually clipped by clipping them into the environment box."""
    s = np.linspace(0.0, 1.0, 257)
    B0, _, _ = bspline_basis_matrices(s, NUM_CONTROL_POINTS)
    np.testing.assert_allclose(B0.sum(axis=1), 1.0, atol=1e-12)
    assert B0.min() >= -1e-12


def test_basis_is_clamped_at_both_ends():
    """A clamped spline starts at the first control point and ends at the last;
    the start/goal pin and the start-heading projection both rely on it."""
    B0, B1, _ = bspline_basis_matrices(np.array([0.0, 1.0]), NUM_CONTROL_POINTS)
    np.testing.assert_allclose(B0[0], np.eye(NUM_CONTROL_POINTS)[0], atol=1e-12)
    np.testing.assert_allclose(B0[-1], np.eye(NUM_CONTROL_POINTS)[-1], atol=1e-12)
    # ... and the start tangent points along P1 - P0 only.
    assert abs(B1[0, 0] + B1[0, 1]) < 1e-9
    assert np.max(np.abs(B1[0, 2:])) < 1e-9


def test_basis_has_local_support():
    """Each basis function is non-zero on at most degree+1 knot spans. This is
    the property the global Bernstein basis lacks and why it conditions badly at
    high degree."""
    s = np.linspace(0.0, 1.0, 1001)
    B0, _, _ = bspline_basis_matrices(s, NUM_CONTROL_POINTS)
    knots = clamped_uniform_knots(NUM_CONTROL_POINTS, DEFAULT_SPLINE_DEGREE)
    span = knots[DEFAULT_SPLINE_DEGREE + 1] - knots[DEFAULT_SPLINE_DEGREE]
    for column in range(NUM_CONTROL_POINTS):
        support = s[B0[:, column] > 1e-9]
        width = support.max() - support.min()
        assert width <= (DEFAULT_SPLINE_DEGREE + 1) * span + 1e-6


def test_a_straight_control_polygon_gives_a_straight_line(gain_tuning_pipeline):
    """The straight-line initialization must have exactly zero curvature, or the
    optimizer starts off already spending its acceleration budget."""
    points = gain_tuning_pipeline.initial_control_points(NUM_CONTROL_POINTS)
    states = np.asarray(
        gain_tuning_pipeline.reference_states_from_control_points(points), dtype=float
    )
    assert np.max(np.abs(states[:, 5])) < 1e-4  # omega


def test_curve_is_linear_in_its_control_points(gain_tuning_pipeline):
    """The whole hot path is one constant matmul, so the map must be exactly
    linear: any nonlinearity would mean a solve had leaked into it."""
    plan = BSplinePlan(gain_tuning_pipeline.problem, NUM_CONTROL_POINTS, "s_curve")
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=(NUM_CONTROL_POINTS, 2)), dtype=jnp.float32)
    b = jnp.asarray(rng.normal(size=(NUM_CONTROL_POINTS, 2)), dtype=jnp.float32)
    residual = plan.evaluate(a + b)[0] - plan.evaluate(a)[0] - plan.evaluate(b)[0]
    assert float(jnp.max(jnp.abs(residual))) < 1e-4


def test_curve_stays_in_the_convex_hull_of_its_control_points(gain_tuning_pipeline):
    plan = BSplinePlan(gain_tuning_pipeline.problem, NUM_CONTROL_POINTS, "s_curve")
    rng = np.random.default_rng(1)
    points = rng.uniform(-1.0, 1.0, size=(NUM_CONTROL_POINTS, 2))
    position = np.asarray(plan.evaluate(jnp.asarray(points, jnp.float32))[0], dtype=float)
    assert position.min(axis=0).min() >= points.min(axis=0).min() - 1e-5
    assert position.max(axis=0).max() <= points.max(axis=0).max() + 1e-5


def test_reference_states_are_finite_and_start_at_the_problem_start(identification_pipeline):
    points = identification_pipeline.initial_control_points(NUM_CONTROL_POINTS)
    reference_states = identification_pipeline.reference_states_from_control_points(points)
    assert reference_states.shape[1] == 8
    assert bool(jnp.all(jnp.isfinite(reference_states)))
    np.testing.assert_allclose(
        np.asarray(reference_states[0, :2], dtype=float),
        np.asarray(identification_pipeline.problem.start[:2], dtype=float),
        atol=1e-4,
    )


def test_reference_states_are_differentiable_wrt_control_points(gain_tuning_pipeline):
    points = gain_tuning_pipeline.initial_control_points(NUM_CONTROL_POINTS)

    def scalar(control_points):
        return jnp.sum(gain_tuning_pipeline.reference_states_from_control_points(control_points) ** 2)

    gradient = jax.grad(scalar)(points)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_start_heading_is_pinned_in_identification_mode(identification_pipeline):
    """The start heading is how the robot is physically placed for a run, so the
    curve must leave the start along it whatever the control points say."""
    assert identification_pipeline.pin_start_heading()
    rng = np.random.default_rng(4)
    points = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(NUM_CONTROL_POINTS, 2)), dtype=jnp.float32
    )
    states = identification_pipeline.reference_states_from_control_points(
        identification_pipeline.clamp_control_points(points)
    )
    expected = float(identification_pipeline.problem.start[2])
    delta = abs((float(states[0, 2]) - expected + np.pi) % (2 * np.pi) - np.pi)
    assert delta < 1e-3, f"start heading {states[0, 2]} != {expected}"


def test_gain_tuning_pins_nothing(gain_tuning_pipeline):
    assert gain_tuning_pipeline.pinned_positions() == {}
    assert not gain_tuning_pipeline.pin_start_heading()


def test_pinned_control_points_receive_no_gradient(identification_pipeline):
    """Identification pins the *start* control point to the problem's start
    pose, so the optimizer must not be able to move it however the loss is
    shaped. Only the start: it is the one pose that has to be set by hand on the
    real robot, and the goal control point is a free decision variable."""
    points = identification_pipeline.initial_control_points(NUM_CONTROL_POINTS)
    assert tuple(sorted(identification_pipeline.pinned_positions())) == (0,)

    def scalar(control_points):
        states = identification_pipeline.reference_states_from_control_points(
            identification_pipeline.clamp_control_points(control_points)
        )
        return jnp.sum(states**2)

    gradient = jax.grad(scalar)(points)
    assert float(jnp.max(jnp.abs(gradient[0]))) == 0.0
    # ...while every free control point, the goal included, does move.
    assert float(jnp.max(jnp.abs(gradient[2:]))) > 0.0
    assert float(jnp.max(jnp.abs(gradient[-1]))) > 0.0


def test_decision_variables_round_trip(gain_tuning_pipeline, identification_pipeline):
    for pipeline in (gain_tuning_pipeline, identification_pipeline):
        points = pipeline.clamp_control_points(
            pipeline.initial_control_points(NUM_CONTROL_POINTS)
        )
        restored = pipeline.control_points_from_decision_variables(
            pipeline.decision_variables_from_control_points(points)
        )
        np.testing.assert_allclose(
            np.asarray(points, dtype=float), np.asarray(restored, dtype=float), atol=1e-5
        )


def test_too_few_control_points_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        clamped_uniform_knots(3, degree=3)


def test_vanishing_tangent_keeps_the_reference_differentiable(gain_tuning_pipeline):
    """A stalled curve must not poison the gradient.

    Heading and yaw rate both divide by the tangent, and gain-tuning mode leaves
    every control point free, so the optimizer can walk into a configuration
    where the tangent is numerically zero -- coincident control points are the
    extreme case, and the environment clip produces them by collapsing
    neighbours onto the same box edge.
    """
    stalled = {
        "coincident start": jnp.asarray(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.5, 1.5],
             [1.0, 2.0], [1.0, 2.5], [1.0, 3.0]], dtype=jnp.float32
        ),
        "fully degenerate": jnp.zeros((7, 2), dtype=jnp.float32),
        "clipped onto corners": jnp.asarray(
            [[-1.5, -2.5], [-1.5, -2.5], [-1.5, -2.5], [0.0, 0.0],
             [1.5, 2.5], [1.5, 2.5], [1.5, 2.5]], dtype=jnp.float32
        ),
    }

    def scalar(control_points):
        return jnp.sum(
            gain_tuning_pipeline.reference_states_from_control_points(control_points) ** 2
        )

    for name, control_points in stalled.items():
        states = gain_tuning_pipeline.reference_states_from_control_points(control_points)
        assert bool(jnp.all(jnp.isfinite(states))), name
        gradient = jax.grad(scalar)(control_points)
        assert bool(jnp.all(jnp.isfinite(gradient))), name


def test_tangent_floor_leaves_ordinary_curves_alone(gain_tuning_pipeline):
    """The floor may not buy its robustness with a biased yaw rate: the
    reference is what the robot is asked to drive."""
    import wmr_simulator.trajectory_optimization.curves as curves

    angles = np.linspace(0.0, 2.0 * np.pi, 7)
    circle = jnp.asarray(
        np.stack([np.cos(angles), np.sin(angles)], axis=1), dtype=jnp.float32
    )
    floored = gain_tuning_pipeline.reference_states_from_control_points(circle)
    original = curves.TANGENT_FLOOR_FRACTION
    curves.TANGENT_FLOOR_FRACTION = 0.0
    try:
        unfloored = gain_tuning_pipeline.reference_states_from_control_points(circle)
    finally:
        curves.TANGENT_FLOOR_FRACTION = original

    omega_scale = float(jnp.max(jnp.abs(unfloored[:, 5])))
    assert float(jnp.max(jnp.abs(floored[:, 5] - unfloored[:, 5]))) < 1e-3 * omega_scale
    np.testing.assert_allclose(
        np.asarray(floored[:, 2], dtype=float), np.asarray(unfloored[:, 2], dtype=float), atol=1e-6
    )
