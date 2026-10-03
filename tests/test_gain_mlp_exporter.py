"""Firmware export of the error-MLP gain parametrization.

The numpy ``reference_forward`` mirrors the math of the firmware crate
``firmware/libs/gain_mlp`` (pololu-rs); matching it against the JAX
implementation validates that the exported payload (with spectral
normalization baked into the weights) reproduces ``error_mlp.factors``.

The comparison is at the *factor* level, which is the whole firmware contract:
the factors are unitless, so they apply unchanged to the firmware's inner motor
gains even though those are in duty/(rad/s) rather than the simulator's
wheel-speed units.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.gain_parametrization import params_from_cfg
from wmr_simulator.gain_parametrization.error_mlp import (
    NUM_SCHEDULABLE_GAINS,
    apply,
    factors as jax_factors,
    with_flat_params,
)
from wmr_simulator.pololu.gain_mlp_exporter import (
    firmware_base_gains,
    gain_mlp_payload,
    golden_payload,
    reference_forward,
)

FEATURE_SCALE = [2.5, 10.0]


def _random_params(cfg: dict, seed: int = 7, scale: float = 0.3):
    template = params_from_cfg(cfg, FEATURE_SCALE)
    rng = np.random.default_rng(seed)
    num_weights = sum(int(np.asarray(layer).size) for layer in template.init_layers)
    theta = scale * rng.standard_normal(num_weights).astype(np.float32)
    if template.learn_bound:
        theta = np.concatenate([theta, np.zeros(1, dtype=np.float32)])
    return with_flat_params(jnp.asarray(theta), template)


@pytest.mark.parametrize("scheduled", [[0, 1, 2], [0, 1, 2, 3, 4], [3, 4]])
def test_reference_forward_matches_jax(scheduled):
    cfg = {"kind": "error_mlp", "hidden_sizes": [16], "scheduled_indices": scheduled,
           "bound": 5.0, "spectral_norm_cap": 1.0}
    params = _random_params(cfg)
    payload = gain_mlp_payload(params)

    rng = np.random.default_rng(3)
    base_gains = np.asarray([4.5, 6.0, 12.0, 2.5, 5.0], dtype=np.float32)
    for _ in range(20):
        ref = [*rng.uniform(-1.0, 1.0, size=2), rng.uniform(-np.pi, np.pi),
               rng.uniform(0.0, 1.0), rng.uniform(-3.0, 3.0)]
        pose = [ref[0] + rng.normal(0, 0.2), ref[1] + rng.normal(0, 0.2), ref[2] + rng.normal(0, 0.5)]
        twist = [ref[3] + rng.normal(0, 0.2), ref[4] + rng.normal(0, 0.5)]

        factors = reference_forward(payload, ref, pose, twist)
        assert factors.shape == (NUM_SCHEDULABLE_GAINS,)
        unscheduled = [index for index in range(NUM_SCHEDULABLE_GAINS) if index not in scheduled]
        assert np.all(factors[unscheduled] == 1.0)

        ref_state = jnp.asarray([ref[0], ref[1], ref[2], ref[3], 0.0, ref[4], 0.0, 0.0], dtype=jnp.float32)
        pose_j = jnp.asarray(pose, dtype=jnp.float32)
        twist_j = jnp.asarray(twist, dtype=jnp.float32)

        expected_factors = np.ones(NUM_SCHEDULABLE_GAINS, dtype=np.float32)
        expected_factors[scheduled] = np.asarray(jax_factors(params, ref_state, pose_j, twist_j))
        np.testing.assert_allclose(factors, expected_factors, rtol=1e-5, atol=1e-6)

        expected = np.asarray(apply(jnp.asarray(base_gains), params, ref_state, pose_j, twist_j))
        np.testing.assert_allclose(base_gains * factors, expected, rtol=1e-5, atol=1e-6)


def test_golden_payload_consistent_with_reference_forward():
    cfg = {"kind": "error_mlp", "hidden_sizes": [16], "scheduled_indices": [0, 1, 2],
           "bound": 5.0, "spectral_norm_cap": 1.0}
    params = _random_params(cfg)
    base_gains = firmware_base_gains([4.5, 6.0, 12.0, 2.5, 5.0], 240.0)
    payload = golden_payload(params, base_gains, num_cases=8, seed=1)

    assert len(payload["cases"]) == 8
    for case in payload["cases"]:
        factors = reference_forward(payload["network"], case["ref"], case["pose"], case["twist"])
        np.testing.assert_allclose(factors, case["expected_factors"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(case["base_gains"], dtype=np.float32) * factors,
            case["expected_gains"], rtol=1e-5, atol=1e-6,
        )
    # Untrained parametrization must be exercised: factors should not all be 1.
    all_factors = np.asarray([case["expected_factors"] for case in payload["cases"]])
    assert np.any(np.abs(all_factors - 1.0) > 1e-3)


def test_identity_parametrization_exports_unit_factors():
    cfg = {"kind": "error_mlp", "hidden_sizes": [16], "scheduled_indices": [0, 1, 2], "bound": 5.0}
    params = params_from_cfg(cfg, FEATURE_SCALE)  # theta = 0 -> identity
    payload = gain_mlp_payload(params)
    factors = reference_forward(payload, [0.3, -0.2, 0.5, 0.4, 1.0], [0.1, 0.0, 0.2], [0.3, 0.8])
    np.testing.assert_allclose(factors, np.ones(NUM_SCHEDULABLE_GAINS), atol=1e-7)


def _gated_params(seed: int = 7):
    from wmr_simulator.gain_parametrization.error_mlp import with_gate

    cfg = {"kind": "error_mlp", "hidden_sizes": [16], "bound": 5.0, "spectral_norm_cap": 1.0, "gate_num_centers": 4}
    params = _random_params(cfg, seed=seed, scale=0.5)
    # Training-like features: small errors, moderate speeds.
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.normal(0, 0.05, 400), rng.normal(0, 0.05, 400), rng.normal(0, 0.1, 400),
        rng.normal(0, 0.03, 400), rng.normal(0, 0.05, 400), rng.uniform(0.1, 0.5, 400), rng.uniform(0, 0.2, 400),
    ])
    return with_gate(params, samples)


def test_gate_keeps_the_network_on_its_data_and_the_static_gains_off_it():
    from wmr_simulator.gain_parametrization.error_mlp import _forward, effective_layers, features, gate

    params = _gated_params()
    on = (jnp.asarray([0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0]), jnp.zeros(3), jnp.asarray([0.75, 1.0]))
    off = (jnp.asarray([0.0, 0.0, 0.0, 2.5, 0.0, 9.0, 0.0, 0.0]), jnp.asarray([-0.4, 0.3, 0.8]), jnp.asarray([1.0, 3.0]))
    z_on = features(*on, params.feature_scale)
    assert float(gate(params, z_on)) > 0.9
    ungated = np.clip(1.0 + np.asarray(_forward(effective_layers(params), z_on)), 0.0, 5.0)
    np.testing.assert_allclose(np.asarray(jax_factors(params, *on)), ungated, atol=0.05)
    z_off = features(*off, params.feature_scale)
    assert float(gate(params, z_off)) < 1e-3
    np.testing.assert_allclose(np.asarray(jax_factors(params, *off)), 1.0, atol=1e-3)


def test_gate_round_trips_through_the_config_and_the_firmware_payload():
    from wmr_simulator.gain_parametrization import to_cfg

    params = _gated_params()
    restored = params_from_cfg(to_cfg(params), FEATURE_SCALE)
    np.testing.assert_allclose(np.asarray(restored.gate_centers), np.asarray(params.gate_centers))
    payload = gain_mlp_payload(params)
    assert "gate" in payload
    rng = np.random.default_rng(5)
    for _ in range(20):
        ref = [*rng.uniform(-1.0, 1.0, size=2), rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 2.5), rng.uniform(-6.0, 6.0)]
        pose = [ref[0] + rng.normal(0, 0.1), ref[1] + rng.normal(0, 0.1), ref[2] + rng.normal(0, 0.2)]
        twist = [ref[3] + rng.normal(0, 0.1), ref[4] + rng.normal(0, 0.3)]
        ref_state = jnp.asarray([ref[0], ref[1], ref[2], ref[3], 0.0, ref[4], 0.0, 0.0], dtype=jnp.float32)
        expected = np.asarray(jax_factors(restored, ref_state, jnp.asarray(pose), jnp.asarray(twist)))
        np.testing.assert_allclose(reference_forward(payload, ref, pose, twist), expected, rtol=1e-4, atol=1e-5)


def test_no_gate_is_the_ungated_network():
    cfg = {"kind": "error_mlp", "hidden_sizes": [16], "bound": 5.0, "spectral_norm_cap": 1.0}
    params = _random_params(cfg)
    assert params.gate_centers.shape[0] == 0
    assert "gate" not in gain_mlp_payload(params)
