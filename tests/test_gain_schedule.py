"""Validation checks for the unified base * factor gain schedule."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.gain_schedule import (
    apply_gain_schedule,
    gain_schedule_params_from_cfg,
    schedule_factors,
    scheduled_outer_gains_over_refs,
    schedule_num_params,
    with_flat_W,
)
from wmr_simulator.gain_tuning.objectives import (
    closed_loop_objective_terms,
    scheduled_closed_loop_objective_terms,
)
from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline

PROBLEM = "problems/pololu_gains.yaml"
FEATURE_SCALE = [2.0, 10.0]  # matches pololu_gains robot v_max / omega_max


def _cfg(W=None, rho=None):
    cfg = {"enabled": True, "scheduled_indices": [0, 1, 2], "rho": rho or [0.5, 0.5, 0.5]}
    if W is not None:
        cfg["W"] = W
    return cfg


def _params(W=None, rho=None):
    return gain_schedule_params_from_cfg(_cfg(W=W, rho=rho), FEATURE_SCALE)


def _pipeline():
    return ControllerTuningPipeline(problem_path=PROBLEM, seed=0)


def _replay_keys(pipeline, n=2):
    return (jax.random.split(pipeline.robot_key, n), jax.random.split(pipeline.estimator_key, n))


def test_identity_factors_for_all_refs():
    params = _params()  # W defaults to zeros
    ref_states = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    for ref in ref_states:
        factors = schedule_factors(params, jnp.asarray(ref))
        np.testing.assert_allclose(np.asarray(factors), np.ones(3), atol=1e-6)


def test_feature_scale_autoderived_from_robot_limits():
    pipeline = _pipeline()
    np.testing.assert_allclose(pipeline.gain_schedule_feature_scale, FEATURE_SCALE)
    np.testing.assert_allclose(np.asarray(pipeline.gain_schedule_params.feature_scale), FEATURE_SCALE)


def test_apply_gain_schedule_shape_and_motor_passthrough():
    params = _params(W=[[1.0, -1.0], [0.5, 0.2], [-0.3, 0.4]])
    base = jnp.asarray([5.0, 5.0, 3.0, 0.4, 0.2, 0.05], dtype=jnp.float32)
    ref = jnp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0], dtype=jnp.float32)
    scheduled = apply_gain_schedule(base, params, ref)
    assert scheduled.shape == (6,)
    # Motor gains (indices 3..5) are always passed through unchanged.
    np.testing.assert_allclose(np.asarray(scheduled[3:]), np.asarray(base[3:]), atol=1e-7)


def test_scheduled_outer_gains_shape():
    params = _params()
    base = jnp.asarray([5.0, 5.0, 3.0, 0.4, 0.2, 0.0], dtype=jnp.float32)
    ref_states = jnp.asarray(np.random.default_rng(1).normal(size=(11, 8)), dtype=jnp.float32)
    outer = scheduled_outer_gains_over_refs(base, params, ref_states)
    assert outer.shape == (10, 3)


def test_positivity_under_bounded_rho():
    rng = np.random.default_rng(2)
    base = jnp.asarray([5.0, 5.0, 3.0, 0.4, 0.2, 0.0], dtype=jnp.float32)
    for _ in range(50):
        params = _params(W=rng.normal(size=(3, 2)).tolist(), rho=[0.9, 0.9, 0.9])
        ref = jnp.asarray(rng.normal(size=8), dtype=jnp.float32)
        scheduled = apply_gain_schedule(base, params, ref)
        # factor in [1-rho, 1+rho] = [0.1, 1.9] -> outer gains stay positive.
        assert np.all(np.asarray(scheduled[:3]) > 0.0)


def test_with_flat_W_roundtrip():
    template = _params()
    assert schedule_num_params(template) == 6
    theta = np.array([1.0, -2.0, 0.5, 0.25, -0.1, 0.3], dtype=np.float32)
    rebuilt = with_flat_W(theta, template)
    np.testing.assert_allclose(np.asarray(rebuilt.W).reshape(-1), theta)


def test_cfg_validation_rejects_wrong_features():
    cfg = _cfg()
    cfg["feature_names"] = ["v_d", "tracking_error"]
    with pytest.raises(ValueError):
        gain_schedule_params_from_cfg(cfg, FEATURE_SCALE)


def test_identity_schedule_matches_static_rollout():
    pipeline = _pipeline()
    params = _params()  # W = 0
    base = pipeline.gains
    static_log = pipeline.run_closed_loop(pipeline.robot_params, controller_gains=base)
    scheduled_log = pipeline.run_closed_loop(
        pipeline.robot_params, controller_gains=base, schedule_params=params
    )
    np.testing.assert_allclose(
        np.asarray(scheduled_log.pose.states), np.asarray(static_log.pose.states), atol=1e-5
    )


def test_disabled_schedule_objective_matches_static():
    pipeline = _pipeline()
    params = _params()  # W = 0 == static
    base = pipeline.gains
    robot_keys, est_keys = _replay_keys(pipeline)

    static_terms = np.asarray(
        closed_loop_objective_terms(
            pipeline, base, robot_keys, est_keys,
            velocity_tracking_weight=0.05, input_delta_weight=0.1,
        )
    )
    scheduled_terms = np.asarray(
        scheduled_closed_loop_objective_terms(
            pipeline, base, params, robot_keys, est_keys,
            velocity_tracking_weight=0.05, input_delta_weight=0.1, gain_delta_weight=1.0,
        )
    )
    np.testing.assert_allclose(scheduled_terms[:4], static_terms, atol=1e-5)
    assert scheduled_terms[4] == pytest.approx(0.0, abs=1e-7)  # gain_delta zero at W=0


def test_gradient_wrt_W_is_finite_and_nonzero():
    pipeline = _pipeline()
    template = _params()
    base = pipeline.gains
    robot_keys, est_keys = _replay_keys(pipeline)
    theta0 = jnp.zeros(schedule_num_params(template), dtype=jnp.float32)

    def loss(theta):
        params = with_flat_W(theta, template)
        return jnp.sum(
            scheduled_closed_loop_objective_terms(
                pipeline, base, params, robot_keys, est_keys,
                velocity_tracking_weight=0.05, input_delta_weight=0.1,
            )
        )

    grad = np.asarray(jax.grad(loss)(theta0))
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) > 0.0
