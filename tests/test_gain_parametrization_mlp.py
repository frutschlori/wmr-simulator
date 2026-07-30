"""Validation checks for the error-MLP gain parametrization."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.gain_parametrization import (
    ErrorMlpParams,
    apply,
    num_params,
    params_from_cfg,
    outer_gains_over_refs,
    to_cfg,
    with_flat_params,
    zero_params,
)
from wmr_simulator.gain_parametrization.error_mlp import NUM_FEATURES, effective_layers, factors, features
from wmr_simulator.gain_tuning.objectives import (
    closed_loop_objective_terms,
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


def test_cfg_selects_error_mlp_kind():
    params = _params()
    assert isinstance(params, ErrorMlpParams)
    assert params.feature_scale.shape == (NUM_FEATURES,)


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


def test_learn_bound_adds_one_param_and_is_applied():
    fixed = _params(learn_bound=False)
    learnable = _params(learn_bound=True)
    assert num_params(learnable) == num_params(fixed) + 1
    theta = jnp.zeros(num_params(learnable), dtype=jnp.float32).at[-1].set(0.75)
    rebuilt = with_flat_params(theta, learnable)
    assert float(rebuilt.bound) == pytest.approx(2.75)


def test_flat_params_zero_matches_zero_params():
    template = _params()
    theta = jnp.zeros(num_params(template), dtype=jnp.float32)
    rebuilt = with_flat_params(theta, template)
    identity = zero_params(template)
    for layer, expected in zip(rebuilt.layers, identity.layers):
        np.testing.assert_allclose(np.asarray(layer), np.asarray(expected), atol=1e-7)


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


def test_spectral_norm_caps_effective_weights():
    template = _params(spectral_norm_cap=1.0)
    rng = np.random.default_rng(4)
    theta = jnp.asarray(rng.normal(scale=10.0, size=num_params(template)), dtype=jnp.float32)
    params = with_flat_params(theta, template)
    for weight in effective_layers(params)[0::2]:
        sigma = np.linalg.svd(np.asarray(weight), compute_uv=False)[0]
        assert sigma <= 1.0 + 1e-4
    # Raw weights are far above the cap, so normalization must actually bite.
    raw_sigma = np.linalg.svd(np.asarray(params.layers[0]), compute_uv=False)[0]
    assert raw_sigma > 1.5


def test_spectral_norm_zero_cap_disables():
    template = _params(spectral_norm_cap=0.0)
    rng = np.random.default_rng(5)
    theta = jnp.asarray(rng.normal(scale=10.0, size=num_params(template)), dtype=jnp.float32)
    params = with_flat_params(theta, template)
    for effective, raw in zip(effective_layers(params), params.layers):
        np.testing.assert_array_equal(np.asarray(effective), np.asarray(raw))


def test_spectral_norm_preserves_identity_at_zero_theta():
    params = _params(spectral_norm_cap=0.5)  # cap far below the random init's sigma
    rng = np.random.default_rng(6)
    for _ in range(10):
        ref, pose, twist = _random_inputs(rng)
        np.testing.assert_allclose(np.asarray(factors(params, ref, pose, twist)), np.ones(5), atol=1e-6)


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


def test_scheduled_indices_keep_motor_gains_static():
    template = _params(scheduled_indices=[0, 1, 2])
    rng = np.random.default_rng(8)
    theta = jnp.asarray(rng.normal(scale=5.0, size=num_params(template)), dtype=jnp.float32)
    params = with_flat_params(theta, template)
    base = jnp.asarray([5.0, 5.0, 3.0, 0.4, 0.2], dtype=jnp.float32)
    scheduled_any = False
    for _ in range(20):
        ref, pose, twist = _random_inputs(rng)
        assert factors(params, ref, pose, twist).shape == (3,)
        gains = np.asarray(apply(base, params, ref, pose_est=pose, twist_est=twist))
        np.testing.assert_array_equal(gains[3:], np.asarray(base[3:]))
        scheduled_any = scheduled_any or not np.allclose(gains[:3], np.asarray(base[:3]))
    assert scheduled_any  # the outer gains are actually being scheduled
    # Fewer outputs -> fewer trainable parameters than the all-gains network.
    assert num_params(template) < num_params(_params())


def test_scheduled_indices_roundtrip_and_validation():
    template = _params(scheduled_indices=[0, 1, 2])
    rebuilt = params_from_cfg({**to_cfg(template), "enabled": True}, FEATURE_SCALE)
    np.testing.assert_array_equal(
        np.asarray(rebuilt.scheduled_indices), np.asarray(template.scheduled_indices)
    )
    with pytest.raises(ValueError):
        _params(scheduled_indices=[0, 0, 1])
    with pytest.raises(ValueError):
        _params(scheduled_indices=[5])


def test_on_reference_gains_shape():
    params = _params()
    base = jnp.asarray([5.0, 5.0, 3.0, 0.4, 0.2], dtype=jnp.float32)
    ref_states = jnp.asarray(np.random.default_rng(3).normal(size=(11, 8)), dtype=jnp.float32)
    gains = outer_gains_over_refs(base, params, ref_states)
    assert gains.shape == (10, 5)
    np.testing.assert_allclose(np.asarray(gains), np.tile(np.asarray(base), (10, 1)), atol=1e-6)


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


def test_identity_objective_matches_static():
    pipeline = _pipeline()
    params = _params()
    base = pipeline.gains
    robot_keys = jax.random.split(pipeline.robot_key, 2)
    est_keys = jax.random.split(pipeline.estimator_key, 2)

    static_terms = np.asarray(
        closed_loop_objective_terms(
            pipeline, base, robot_keys, est_keys,
            velocity_tracking_weight=0.05, input_delta_weight=0.1,
        )
    )
    parametrized_terms = np.asarray(
        scheduled_closed_loop_objective_terms(
            pipeline, base, params, robot_keys, est_keys,
            velocity_tracking_weight=0.05, input_delta_weight=0.1, gain_delta_weight=1.0,
        )
    )
    np.testing.assert_allclose(parametrized_terms[:5], static_terms, atol=1e-5)
    assert parametrized_terms[5] == pytest.approx(0.0, abs=1e-7)


def test_static_tune_runs_independently_of_parametrized_run():
    from wmr_simulator.gain_tuning.pipeline import resolve_gain_robot_params, run_gain_tuning_experiment

    robot_params = resolve_gain_robot_params(PROBLEM, None, None)
    # num_steps=0 keeps both runs at pure evaluation. With an identity
    # parametrization and the same init gains they must score the same loss (up
    # to the reparametrization round-trip), which pins that the parametrized
    # run's presearch is evaluated with the parametrization in the loop.
    result = run_gain_tuning_experiment(
        problem_path=PROBLEM,
        robot_params=robot_params,
        num_steps=0,
        learning_rate=1e-3,
        num_realizations=1,
        num_lhs_points=0,
        num_adam_optimizations=1,
        schedule_enabled=True,
        static_tune=True,
    )
    assert result["static_tune"] is True
    assert result["static_gains"] is not None
    assert isinstance(result["schedule_params"], ErrorMlpParams)
    # Separate histories: one initial evaluation each, not concatenated.
    assert len(result["loss_history"]) == 1
    assert len(result["static_loss_history"]) == 1
    for values in result["loss_component_history"].values():
        assert len(values) == 1
    # The closed-loop loss is stiff under the in-loop encoder low-pass +
    # trapezoidal encoder lag (see memory gain-fim-wheel-lp-stiffness), which
    # amplifies the ~1e-6 reparametrization round-trip difference into a
    # sub-percent loss gap; the tolerance reflects that, while still catching any
    # non-identity parametrization (which would move the loss by O(10%)+).
    assert result["loss_history"][0] == pytest.approx(result["static_loss_history"][0], rel=2e-2)


def test_static_tune_independent_budget_and_init_gains():
    from wmr_simulator.gain_tuning.pipeline import resolve_gain_robot_params, run_gain_tuning_experiment

    robot_params = resolve_gain_robot_params(PROBLEM, None, None)
    result = run_gain_tuning_experiment(
        problem_path=PROBLEM,
        robot_params=robot_params,
        num_steps=1,
        learning_rate=1e-4,
        num_realizations=1,
        num_lhs_points=2,
        num_adam_optimizations=2,
        schedule_enabled=True,
        static_tune=True,
        static_tune_steps=2,
        static_tune_learning_rate=1e-3,
        static_init_gains=[0.5, 0.5, 0.5, 1.0, 0.0],
    )
    # Each run keeps its own budget: static initial eval + 2 steps, parametrized
    # initial eval + 1 step.
    assert len(result["static_loss_history"]) == 3
    assert len(result["loss_history"]) == 2
    for values in result["loss_component_history"].values():
        assert len(values) == 2
    for values in result["static_loss_component_history"].values():
        assert len(values) == 3
    assert isinstance(result["schedule_params"], ErrorMlpParams)


def test_seed_parametrization_from_static_uses_static_results_as_candidates():
    # First-iteration policy: the parametrized run's presearch gets the static
    # run's converged gains as extra candidates. With the LHS off, they are the
    # only candidates besides the problem's gains, so the winning start must be
    # one of them (the static Adam result beats the untuned problem gains).
    from wmr_simulator.gain_tuning.pipeline import resolve_gain_robot_params, run_gain_tuning_experiment

    robot_params = resolve_gain_robot_params(PROBLEM, None, None)
    result = run_gain_tuning_experiment(
        problem_path=PROBLEM,
        robot_params=robot_params,
        num_steps=0,
        learning_rate=1e-3,
        num_realizations=1,
        num_lhs_points=0,
        num_adam_optimizations=1,
        schedule_enabled=True,
        static_tune=True,
        static_tune_steps=3,
        static_tune_learning_rate=1e-2,
        seed_parametrization_from_static=True,
    )
    static_gains = np.asarray(result["static_gains"], dtype=float)
    optimized_gains = np.asarray(result["optimized_gains"], dtype=float)
    np.testing.assert_allclose(optimized_gains, static_gains, rtol=1e-4)
    # Still two independent runs: separate histories, static run kept its budget.
    assert len(result["static_loss_history"]) == 4
    assert len(result["loss_history"]) == 1


def test_gradient_wrt_theta_is_finite_and_nonzero_at_identity():
    # The critical check for the delta-on-random-init construction: at theta = 0
    # the hidden activations are nonzero, so the gradient must not vanish (an
    # all-zero MLP would be a dead saddle for the presearch warm start).
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
                velocity_tracking_weight=0.05, input_delta_weight=0.1,
            )
        )

    grad = np.asarray(jax.grad(loss)(theta0))
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) > 0.0
