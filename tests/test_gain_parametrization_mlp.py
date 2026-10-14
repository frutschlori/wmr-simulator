"""Validation checks for the error-MLP gain parametrization."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.gain_parametrization import (
    num_params,
    params_from_cfg,
    to_cfg,
    with_flat_params,
)
from wmr_simulator.gain_parametrization.error_mlp import (
    factors,
    features,
)
from wmr_simulator.gain_tuning.objectives import (
    scheduled_closed_loop_objective_terms,
)
from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline

PROBLEM = "problems/pololu_gains.yaml"
FEATURE_SCALE = [2.0, 10.0]  # [v_max, omega_max]


def _cfg(**overrides):
    cfg = {"kind": "error_mlp", "enabled": True, "hidden_sizes": [8], "seed": 0}
    cfg.update(overrides)
    return cfg


def _params(**overrides):
    return params_from_cfg(_cfg(**overrides), FEATURE_SCALE)


def _pipeline():
    return ControllerTuningPipeline(problem_path=PROBLEM, seed=0)


def _random_inputs(rng):
    ref = jnp.asarray(rng.normal(size=8), dtype=jnp.float32)
    pose = jnp.asarray(rng.normal(size=3), dtype=jnp.float32)
    twist = jnp.asarray(rng.normal(size=2), dtype=jnp.float32)
    return ref, pose, twist


def test_identity_factors_at_zero_theta():
    params = _params()  # theta defaults to zeros -> zero output layer
    rng = np.random.default_rng(0)
    for _ in range(20):
        ref, pose, twist = _random_inputs(rng)
        np.testing.assert_allclose(np.asarray(factors(params, ref, pose, twist)), np.ones(5), atol=1e-6)


def test_factors_respect_bounds():
    template = _params(bound=1.5)
    rng = np.random.default_rng(1)
    theta = jnp.asarray(rng.normal(scale=5.0, size=num_params(template)), dtype=jnp.float32)
    params = with_flat_params(theta, template)
    for _ in range(50):
        ref, pose, twist = _random_inputs(rng)
        values = np.asarray(factors(params, ref, pose, twist))
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.5 + 1e-6)


def test_to_cfg_roundtrip():
    template = _params(learn_bound=True)
    rng = np.random.default_rng(2)
    theta = jnp.asarray(rng.normal(size=num_params(template)), dtype=jnp.float32)
    params = with_flat_params(theta, template)
    rebuilt = params_from_cfg({**to_cfg(params), "enabled": True}, FEATURE_SCALE)
    ref, pose, twist = _random_inputs(rng)
    np.testing.assert_allclose(
        np.asarray(factors(rebuilt, ref, pose, twist)),
        np.asarray(factors(params, ref, pose, twist)),
        atol=1e-6,
    )


def test_spectral_norm_bounds_factor_lipschitz():
    # With cap c per matrix and 1-Lipschitz tanh, |f(z1) - f(z2)| <= c^2 |z1 - z2|
    # in normalized feature space (2-norm), before the output clip.
    cap = 1.0
    template = _params(spectral_norm_cap=cap, bound=100.0)  # bound large: isolate the net
    rng = np.random.default_rng(7)
    theta = jnp.asarray(rng.normal(scale=10.0, size=num_params(template)), dtype=jnp.float32)
    params = with_flat_params(theta, template)
    for _ in range(25):
        ref, pose1, twist = _random_inputs(rng)
        pose2 = pose1 + jnp.asarray(rng.normal(scale=0.05, size=3), dtype=jnp.float32)
        z1 = features(ref, pose1, twist, params.feature_scale)
        z2 = features(ref, pose2, twist, params.feature_scale)
        df = np.asarray(factors(params, ref, pose2, twist)) - np.asarray(factors(params, ref, pose1, twist))
        dz = np.linalg.norm(np.asarray(z2 - z1))
        assert np.linalg.norm(df) <= cap**2 * dz + 1e-5


def test_identity_parametrization_matches_static_rollout():
    pipeline = _pipeline()
    params = _params()
    base = pipeline.gains
    static_log = pipeline.run_closed_loop(pipeline.robot_params, controller_gains=base)
    parametrized_log = pipeline.run_closed_loop(
        pipeline.robot_params, controller_gains=base, schedule_params=params
    )
    np.testing.assert_allclose(
        np.asarray(parametrized_log.pose.states), np.asarray(static_log.pose.states), atol=1e-5
    )


def test_gradient_wrt_theta_is_finite_and_nonzero_at_identity():
    # At theta = 0 the hidden activations are nonzero, so the gradient must not vanish.
    pipeline = _pipeline()
    template = _params()
    base = pipeline.gains
    robot_keys = jax.random.split(pipeline.robot_key, 2)
    est_keys = jax.random.split(pipeline.estimator_key, 2)
    theta0 = jnp.zeros(num_params(template), dtype=jnp.float32)

    def loss(theta):
        params = with_flat_params(theta, template)
        return jnp.sum(
            scheduled_closed_loop_objective_terms(
                pipeline, base, params, robot_keys, est_keys,
                linear_velocity_tracking_weight=0.05, angular_velocity_tracking_weight=0.05, input_delta_weight=0.1,
            )
        )

    grad = np.asarray(jax.grad(loss)(theta0))
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) > 0.0
