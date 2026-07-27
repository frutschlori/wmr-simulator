import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.residual_model import (
    apply_residual_model,
    gate_weights,
    init_residual_model,
    load_residual_model,
    residual_corrected_twist,
    residual_features,
    save_residual_model,
)
from wmr_simulator.residual_model.residual import (
    RESIDUAL_INPUT_DIM,
    RESIDUAL_OUTPUT_DIM,
    ResidualEnsemble,
    split_datasets_by_file,
    train_residual_ensemble,
)
from wmr_simulator.robot import DiffDrive, body_velocities

ROBOT_CFG = {
    "wheel_radius": 0.0164,
    "base_diameter": 0.0853,
    "max_wheel_speed": 248.0,
    "time_constant": 0.22,
}


def small_model(num_experts=3, hidden_sizes=(16,), seed=0, spectral_norm_cap=2.0, gate_scale=8.0):
    """Untrained ensemble with a non-trivial gate and non-zero output layers.

    ``init_residual_model`` zero-inits the last layer (so a fresh ensemble is
    exactly the nominal model); tests that need a *non-zero* residual perturb
    the final weights and give the gate real centers/scales.

    The centers sit on the diagonal of [-1, 1] but ``gate_scale`` is wide, because
    the descriptor's action half (the commanded twist) is an order of magnitude
    larger than its state half at these duties -- a trained model normalizes that
    away, this fixture does not. Wide enough to cover what ``DiffDrive.step``
    feeds in, still narrow enough that the far-field OOD probes gate off.
    """
    model = init_residual_model(
        jax.random.PRNGKey(seed),
        num_experts=num_experts,
        hidden_sizes=hidden_sizes,
        spectral_norm_cap=spectral_norm_cap,
    )
    key = jax.random.PRNGKey(seed + 1)
    last_weight = 0.5 * jax.random.normal(key, model.layers[-2].shape)
    centers = jnp.asarray(np.linspace(-1.0, 1.0, num_experts)[:, None] * np.ones((1, RESIDUAL_INPUT_DIM)))
    return ResidualEnsemble(
        layers=(*model.layers[:-2], last_weight, model.layers[-1]),
        centers=centers.astype(jnp.float32),
        scales=jnp.full(num_experts, gate_scale, dtype=jnp.float32),
        ood_sigma=model.ood_sigma,
        input_mean=model.input_mean,
        input_std=model.input_std,
        target_mean=model.target_mean,
        target_std=jnp.ones(RESIDUAL_OUTPUT_DIM, dtype=jnp.float32),
        spectral_norm_cap=spectral_norm_cap,
    )


def test_apply_residual_model_shapes():
    model = small_model()
    single = apply_residual_model(model, jnp.zeros(RESIDUAL_INPUT_DIM))
    assert single.shape == (RESIDUAL_OUTPUT_DIM,)
    batch = apply_residual_model(model, jnp.zeros((7, RESIDUAL_INPUT_DIM)))
    assert batch.shape == (7, RESIDUAL_OUTPUT_DIM)


def test_residual_features_layout():
    features = residual_features(0.5, 1.5, 0.9, 2.0)
    assert features.shape == (RESIDUAL_INPUT_DIM,)
    np.testing.assert_allclose(np.asarray(features), [0.5, 1.5, 0.9, 2.0])


def test_residual_corrected_twist():
    model = small_model()
    features = residual_features(0.5, 1.0, 0.5, 1.0)
    delta = apply_residual_model(model, features)
    v_x, v_y, omega = residual_corrected_twist(model, features, 0.5, 1.0)
    assert np.isclose(float(v_x), 0.5 + float(delta[0]))
    assert float(v_y) == 0.0
    assert np.isclose(float(omega), 1.0 + float(delta[1]))


def test_untrained_ensemble_is_zero_residual():
    """A fresh ensemble (zero final layer) is exactly the nominal model."""
    model = init_residual_model(jax.random.PRNGKey(0), num_experts=4, hidden_sizes=(16, 16))
    delta = apply_residual_model(model, residual_features(0.7, 0.3, 0.7, 0.3))
    np.testing.assert_allclose(np.asarray(delta), 0.0, atol=1e-7)


def test_gate_zeroes_residual_out_of_distribution():
    """Far outside every cluster the gate collapses the residual to zero."""
    model = small_model()
    # In-distribution but off the origin (zero biases make the origin output
    # exactly zero regardless of weights, so it is not a useful probe).
    in_dist = apply_residual_model(model, residual_features(0.4, 0.4, 0.4, 0.4))
    far = apply_residual_model(model, residual_features(1000.0, 1000.0, 1000.0, 1000.0))
    assert np.linalg.norm(np.asarray(far)) < 1e-3
    assert np.linalg.norm(np.asarray(in_dist)) > 0.0
    # Gate weights vanish far away; near a center they dominate the null expert.
    assert float(gate_weights(model, residual_features(1000.0, 1000.0, 1000.0, 1000.0)).sum()) < 1e-3
    assert float(gate_weights(model, residual_features(0.4, 0.4, 0.4, 0.4)).sum()) > 0.5


def test_spectral_norm_bounds_expert_lipschitz():
    """The spectral-norm cap limits how fast the residual changes with the twist."""
    model = small_model(spectral_norm_cap=1.0)
    rng = np.random.default_rng(0)
    base = jnp.asarray(rng.normal(size=(64, RESIDUAL_INPUT_DIM)).astype(np.float32))
    step = 1e-3 * jnp.asarray(rng.normal(size=base.shape).astype(np.float32))
    delta0 = apply_residual_model(model, base)
    delta1 = apply_residual_model(model, base + step)
    output_change = jnp.linalg.norm(delta1 - delta0, axis=1)
    input_change = jnp.linalg.norm(step, axis=1)
    # Lipschitz constant of the map (physical, target_std = 1 here) is bounded by
    # cap^depth; with cap=1 and identity normalization the ratio stays modest.
    assert float(jnp.max(output_change / input_change)) < 5.0


def test_save_load_roundtrip(tmp_path):
    model = small_model(num_experts=3, hidden_sizes=(16,))
    config = {
        "input_dim": RESIDUAL_INPUT_DIM,
        "output_dim": RESIDUAL_OUTPUT_DIM,
        "num_experts": 3,
        "hidden_sizes": [16],
        "spectral_norm_cap": 2.0,
    }
    path = tmp_path / "model.pkl"
    save_residual_model(path, model, config, {"note": "test"})
    loaded, checkpoint = load_residual_model(path)
    assert checkpoint["metadata"]["note"] == "test"
    features = jax.random.normal(jax.random.PRNGKey(1), (5, RESIDUAL_INPUT_DIM))
    np.testing.assert_allclose(
        np.asarray(apply_residual_model(model, features)),
        np.asarray(apply_residual_model(loaded, features)),
        rtol=1e-6,
    )


def test_load_rejects_mismatched_config(tmp_path):
    model = small_model(num_experts=3, hidden_sizes=(16,))
    config = {
        "input_dim": RESIDUAL_INPUT_DIM,
        "output_dim": RESIDUAL_OUTPUT_DIM,
        "num_experts": 3,
        "hidden_sizes": [32, 32],  # extra layer -> leaf-count mismatch on load
        "spectral_norm_cap": 2.0,
    }
    path = tmp_path / "model.pkl"
    save_residual_model(path, model, config)
    with pytest.raises(Exception):
        load_residual_model(path)


def test_step_without_residual_unchanged():
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    state = robot.get_init_state(jax.random.PRNGKey(0))
    next_state = robot.step(state, jnp.asarray([0.5, 0.3]))
    baseline = robot.step(state, jnp.asarray([0.5, 0.3]), residual_model=None)
    np.testing.assert_allclose(np.asarray(next_state.pose), np.asarray(baseline.pose))
    assert next_state.vel_omega.shape == (2,)
    assert float(next_state.vel_lateral) == 0.0


def test_step_with_residual_changes_pose_and_keeps_shapes():
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    model = small_model()
    state = robot.get_init_state(jax.random.PRNGKey(0))
    duty = jnp.asarray([0.5, 0.3])
    for _ in range(5):
        state = robot.step(state, duty)
    without = robot.step(state, duty)
    with_residual = robot.step(state, duty, residual_model=model)
    assert with_residual.pose.shape == (3,)
    assert with_residual.vel_omega.shape == (2,)
    assert not np.allclose(np.asarray(without.pose), np.asarray(with_residual.pose))


def test_step_residual_matches_manual_correction():
    """DiffDrive.step adds exactly the residual delta to the nominal twist."""
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    model = small_model()
    state = robot.get_init_state(jax.random.PRNGKey(0))
    duty = jnp.asarray([0.6, 0.2])
    nominal = robot.step(state, duty)  # residual off -> nominal twist
    v_cmd, w_cmd = body_velocities(
        ROBOT_CFG["max_wheel_speed"] * duty, ROBOT_CFG["wheel_radius"], ROBOT_CFG["base_diameter"]
    )
    delta = apply_residual_model(
        model, residual_features(nominal.vel_omega[0], nominal.vel_omega[1], v_cmd, w_cmd)
    )
    with_residual = robot.step(state, duty, residual_model=model)
    np.testing.assert_allclose(
        np.asarray(with_residual.vel_omega),
        np.asarray(nominal.vel_omega) + np.asarray(delta),
        rtol=1e-5,
    )


def test_residual_rollout_differentiable_wrt_gains():
    """Gradient of a residual-augmented multi-step rollout w.r.t. a gain-like
    scalar that scales the duty cycle must be finite and nonzero."""
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    model = small_model()

    def rollout_loss(gain):
        state = robot.get_init_state(jax.random.PRNGKey(0))

        def body(carry, _):
            carry = robot.step(carry, gain * jnp.asarray([0.5, 0.3]), residual_model=model)
            return carry, carry.pose

        _, poses = jax.lax.scan(body, state, None, length=20)
        return jnp.sum(poses**2)

    grad = jax.grad(rollout_loss)(jnp.asarray(1.0))
    assert np.isfinite(float(grad))
    assert abs(float(grad)) > 0.0


def test_train_reduces_residual_rmse():
    """One-step training must beat the zero-residual baseline on a smooth,
    operating-point-dependent synthetic residual, and route via the gate."""
    rng = np.random.default_rng(0)
    v_nom = rng.uniform(-1.0, 1.0, 400)
    omega_nom = rng.uniform(-2.0, 2.0, 400)
    # Commanded twist: the state plus an independent step the lag is heading for.
    v_cmd = v_nom + rng.uniform(-0.5, 0.5, 400)
    omega_cmd = omega_nom + rng.uniform(-1.0, 1.0, 400)
    features = np.column_stack([v_nom, omega_nom, v_cmd, omega_cmd]).astype(np.float32)
    # Smooth residual with both a quasi-static part (a function of the state) and
    # a transient part (proportional to command minus state) -- the latter is only
    # learnable from the action half of the descriptor.
    targets = np.column_stack(
        [
            0.1 * np.sin(v_nom) + 0.05 * omega_nom,
            0.2 * np.tanh(omega_nom) + 0.1 * (omega_cmd - omega_nom),
        ]
    ).astype(np.float32)

    model, history = train_residual_ensemble(
        features[:320],
        targets[:320],
        features[320:],
        targets[320:],
        num_experts=3,
        hidden_sizes=(16, 16),
        epochs=200,
        batch_size=128,
        seed=0,
    )
    predictions = np.asarray(apply_residual_model(model, features))
    model_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    baseline_rmse = np.sqrt(np.mean(targets**2))
    assert model_rmse < 0.5 * baseline_rmse
    assert history["validation_loss"][-1] < history["validation_loss"][0]


def test_split_holds_out_whole_files():
    """Whole-file splits only -- the residual must see complete trajectories.
    Too few files for the requested fraction still holds one file out."""
    for num_files, expected_validation in ((2, 1), (3, 1), (4, 1), (5, 1), (10, 2)):
        datasets = [{"features": np.zeros((8, RESIDUAL_INPUT_DIM))} for _ in range(num_files)]
        train, validation = split_datasets_by_file(datasets, 0.2, seed=0)
        assert len(validation) == expected_validation
        assert len(train) == num_files - expected_validation
        assert set(train).isdisjoint(validation)
        assert set(train) | set(validation) == set(range(num_files))


def test_split_rejects_a_single_file():
    with pytest.raises(ValueError, match="at least 2 logs"):
        split_datasets_by_file([{"features": np.zeros((8, RESIDUAL_INPUT_DIM))}], 0.2, seed=0)


def test_gate_weights_shape():
    model = small_model(num_experts=4)
    weights = gate_weights(model, jnp.zeros((6, RESIDUAL_INPUT_DIM)))
    assert weights.shape == (6, 4)
    # Expert weights plus the null weight partition unity.
    assert np.all(np.asarray(weights).sum(axis=1) <= 1.0 + 1e-5)
