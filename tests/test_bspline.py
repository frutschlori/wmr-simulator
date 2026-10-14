"""Clamped uniform B-spline parametrization of the trajectory designer."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline
from wmr_simulator.trajectory_optimization.bspline import (
    BSplinePlan,
    bspline_basis_matrices,
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
    """Rows of B0 sum to 1 and are non-negative."""
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


def test_curve_stays_in_the_convex_hull_of_its_control_points(gain_tuning_pipeline):
    plan = BSplinePlan(gain_tuning_pipeline.problem, NUM_CONTROL_POINTS, "s_curve")
    rng = np.random.default_rng(1)
    points = rng.uniform(-1.0, 1.0, size=(NUM_CONTROL_POINTS, 2))
    position = np.asarray(plan.evaluate(jnp.asarray(points, jnp.float32))[0], dtype=float)
    assert position.min(axis=0).min() >= points.min(axis=0).min() - 1e-5
    assert position.max(axis=0).max() <= points.max(axis=0).max() + 1e-5


def test_reference_states_are_differentiable_wrt_control_points(gain_tuning_pipeline):
    points = gain_tuning_pipeline.initial_control_points(NUM_CONTROL_POINTS)

    def scalar(control_points):
        return jnp.sum(gain_tuning_pipeline.reference_states_from_control_points(control_points) ** 2)

    gradient = jax.grad(scalar)(points)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.max(jnp.abs(gradient))) > 0.0


def test_pinned_control_points_receive_no_gradient(identification_pipeline):
    """Identification pins the start control point; every other one moves."""
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


def _bunched_control_points():
    """A long route with four control points collapsed into a ~0.15 m cluster,
    which stalls the curve into a cusp."""
    return jnp.asarray(
        [[0.686, -0.882], [-0.306, -1.682], [-0.519, -1.840],
         [-0.378, -1.760], [-0.721, -1.870], [-1.500, -2.500]],
        dtype=jnp.float32,
    )


def test_tangent_floor_loss_fires_on_a_stalled_curve(gain_tuning_pipeline):
    """A stalled curve costs an order of magnitude more than the ~2 log units
    of FIM it gains."""
    stalled = gain_tuning_pipeline.clamp_control_points(_bunched_control_points())
    loss = gain_tuning_pipeline.stabilization_loss_from_control_points(stalled)
    assert float(loss) > 10.0


def test_a_stalled_curve_is_refused_at_export(gain_tuning_pipeline, tmp_path):
    """A stalled design must fail the run, not ship a pickle whose heading is
    the constant fallback tangent."""
    off = TrajectoryOptimizationPipeline(
        problem_path=PROBLEM, objective_mode="gain-tuning", min_tangent_fraction=0.0
    )
    coincident = jnp.asarray(
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=jnp.float32,
    )
    report = off.tangent_diagnostics(coincident)
    assert report["guarded_samples"] > 0, "fixture no longer stalls the curve"
    with pytest.raises(ValueError, match="Refusing to export a stalled curve"):
        off.save_reference_states_pickle(
            out_dir=str(tmp_path),
            filename_prefix="stalled",
            reference_states=off.reference_states_from_control_points(coincident),
            control_points=coincident,
            start_offsets=None,
        )
    assert not list(tmp_path.glob("*.pkl"))
