import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.residual_model import (
    apply_residual_model,
    init_residual_model,
    load_residual_model,
    residual_features,
    residual_corrected_twist,
    save_residual_model,
)
from wmr_simulator.residual_model.residual import RESIDUAL_INPUT_DIM, RESIDUAL_OUTPUT_DIM
from wmr_simulator.robot import DiffDrive

ROBOT_CFG = {
    "wheel_radius": 0.0164,
    "base_diameter": 0.0853,
    "max_wheel_speed": 248.0,
    "time_constant": 0.22,
}


def small_model():
    return init_residual_model(jax.random.PRNGKey(0), hidden_width=16, hidden_depth=2)


def test_apply_residual_model_shapes():
    model = small_model()
    single = apply_residual_model(model, jnp.zeros(RESIDUAL_INPUT_DIM))
    assert single.shape == (RESIDUAL_OUTPUT_DIM,)
    batch = apply_residual_model(model, jnp.zeros((7, RESIDUAL_INPUT_DIM)))
    assert batch.shape == (7, RESIDUAL_OUTPUT_DIM)


def test_residual_corrected_twist():
    model = small_model()
    features = jnp.ones(RESIDUAL_INPUT_DIM)
    delta = apply_residual_model(model, features)
    v_x, v_y, omega = residual_corrected_twist(model, features, 0.5, 1.0)
    assert np.isclose(float(v_x), 0.5 + float(delta[0]))
    assert np.isclose(float(v_y), float(delta[1]))
    assert np.isclose(float(omega), 1.0 + float(delta[2]))


def test_save_load_roundtrip(tmp_path):
    model = small_model()
    config = {"input_dim": RESIDUAL_INPUT_DIM, "hidden_width": 16, "hidden_depth": 2, "output_dim": 3}
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
    model = small_model()
    config = {"input_dim": RESIDUAL_INPUT_DIM, "hidden_width": 32, "hidden_depth": 3, "output_dim": 3}
    path = tmp_path / "model.pkl"
    save_residual_model(path, model, config)
    with pytest.raises(Exception):
        loaded, _ = load_residual_model(path)
        apply_residual_model(loaded, jnp.zeros(RESIDUAL_INPUT_DIM))


def test_step_without_residual_unchanged():
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    state = robot.get_init_state(jax.random.PRNGKey(0))
    next_state = robot.step(state, jnp.asarray([0.5, 0.3]))
    baseline = robot.step(state, jnp.asarray([0.5, 0.3]), residual_model=None)
    np.testing.assert_allclose(np.asarray(next_state.pose), np.asarray(baseline.pose))
    assert next_state.vel_omega.shape == (2,)


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


def test_residual_features_order():
    """Layout: [vx, vy, omega, wheel_r/l, cmd_r/l, slip_r/l, v_omega, filt_vx, filt_w]."""
    features = residual_features(
        jnp.asarray([1.0, 2.0, 3.0]),  # body twist
        jnp.asarray([4.0, 5.0]),       # nominal lag wheel speeds
        jnp.asarray([6.0, 7.0]),       # wheel-speed command
        jnp.asarray([3.0, 3.0]),       # previous traction-limited ground speeds
        jnp.asarray([8.0, 9.0]),       # low-pass-filtered [vx, omega]
    )
    assert features.shape == (RESIDUAL_INPUT_DIM,)
    # slip_proxy = wheel - prev_ground, v_omega = vx * omega
    np.testing.assert_allclose(
        np.asarray(features),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 8.0, 9.0],
    )


def test_residual_feature_names():
    from wmr_simulator.residual_model.residual import RESIDUAL_FEATURE_NAMES

    assert RESIDUAL_FEATURE_NAMES == (
        "vx_body",
        "vy_body",
        "omega",
        "wheel_speed_r",
        "wheel_speed_l",
        "wheel_cmd_r",
        "wheel_cmd_l",
        "slip_proxy_r",
        "slip_proxy_l",
        "v_omega",
        "vx_body_filt",
        "omega_filt",
    )
    assert RESIDUAL_INPUT_DIM == 12


def test_step_tracks_lateral_velocity_state():
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    model = small_model()
    state = robot.get_init_state(jax.random.PRNGKey(0))
    duty = jnp.asarray([0.6, 0.2])
    wheel_cmd = jnp.asarray([120.0, 40.0])
    without = robot.step(state, duty)
    assert float(without.vel_lateral) == 0.0
    with_residual = robot.step(state, duty, residual_model=model, wheel_speed_cmd=wheel_cmd)
    assert with_residual.vel_lateral.shape == ()
    # The residual's lateral correction must be carried in the state so the
    # next step's features can condition on it. The wheel-speed feature the model
    # sees is the nominal first-order lag prediction (there is no wheel residual);
    # the slip proxy differences against the carried traction-limited ground
    # speeds.
    alpha = np.exp(-0.01 / ROBOT_CFG["time_constant"])
    nominal_lag = alpha * state.wheel_speeds + (1.0 - alpha) * ROBOT_CFG["max_wheel_speed"] * duty
    features = jnp.concatenate(
        [
            jnp.asarray([state.vel_omega[0], state.vel_lateral, state.vel_omega[1]]),
            nominal_lag,
            wheel_cmd,
            nominal_lag - state.ground_wheel_speeds,
            jnp.asarray([state.vel_omega[0] * state.vel_omega[1]]),
            state.twist_filtered,
        ]
    )
    from wmr_simulator.residual_model import apply_residual_model as apply

    delta = apply(model, features)
    np.testing.assert_allclose(float(with_residual.vel_lateral), float(delta[1]), rtol=1e-5)
    # The wheel speed carried in the state is the pure nominal lag (no wheel residual).
    np.testing.assert_allclose(
        np.asarray(with_residual.wheel_speeds), np.asarray(nominal_lag), rtol=1e-5
    )
    # The applied command is still carried for diagnostics/logging.
    np.testing.assert_allclose(np.asarray(with_residual.wheel_speed_cmd), np.asarray(wheel_cmd))
    # The filtered-twist memory follows the corrected twist with the model's tau.
    beta = np.exp(-0.01 / float(model.feature_filter_tau))
    expected_filt = beta * np.asarray(state.twist_filtered) + (1.0 - beta) * np.asarray(
        [with_residual.vel_omega[0], with_residual.vel_omega[1]]
    )
    np.testing.assert_allclose(
        np.asarray(with_residual.twist_filtered), expected_filt, rtol=1e-5
    )
    # Without a residual model the filter state is carried unchanged.
    np.testing.assert_allclose(np.asarray(without.twist_filtered), 0.0)


def test_jacobian_regularization_reduces_sensitivity():
    """With a strong Jacobian penalty the trained MLP must be measurably
    flatter (smaller input-output Jacobian norm) than without it."""
    from wmr_simulator.residual_model.residual import train_residual_model

    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, RESIDUAL_INPUT_DIM)).astype(np.float32)
    # High-frequency target: tempts the net into a steep fit.
    y = np.sin(5.0 * x[:, :RESIDUAL_OUTPUT_DIM]).astype(np.float32)

    def jacobian_norm(model):
        z = jnp.asarray((x - np.asarray(model.input_mean)) / np.asarray(model.input_std))
        jac = jax.vmap(jax.jacrev(model.mlp))(z)
        return float(jnp.mean(jnp.sum(jac**2, axis=(1, 2))))

    common = dict(seed=0, hidden_width=16, hidden_depth=2, epochs=50, batch_size=256)
    plain, _ = train_residual_model(x, y, x[:0], y[:0], jacobian_reg_weight=0.0, **common)
    damped, _ = train_residual_model(x, y, x[:0], y[:0], jacobian_reg_weight=1.0, **common)
    assert jacobian_norm(damped) < 0.5 * jacobian_norm(plain)


def test_apply_residual_model_clamps_out_of_distribution_inputs():
    """Normalized inputs saturate at +-RESIDUAL_INPUT_CLIP_SIGMA: far outside
    the training range the model must stop extrapolating."""
    from wmr_simulator.residual_model.residual import RESIDUAL_INPUT_CLIP_SIGMA

    model = small_model()  # identity normalization: features are already sigmas
    at_clip = apply_residual_model(model, jnp.full(RESIDUAL_INPUT_DIM, RESIDUAL_INPUT_CLIP_SIGMA))
    far_out = apply_residual_model(model, jnp.full(RESIDUAL_INPUT_DIM, 100.0))
    np.testing.assert_allclose(np.asarray(far_out), np.asarray(at_clip), rtol=1e-6)
    inside = apply_residual_model(model, jnp.full(RESIDUAL_INPUT_DIM, 1.0))
    assert not np.allclose(np.asarray(inside), np.asarray(at_clip))


def test_synthetic_start_sequences_shapes_and_stacking():
    """Synthetic from-rest windows: zero initial state, pose_weight 0, and they
    stack with (longer) data-like windows via padding + mask."""
    from wmr_simulator.residual_model.residual import (
        SYNTHETIC_START_DUTY_PROFILES,
        build_synthetic_start_sequences,
        stack_sequences,
    )

    synthetic = build_synthetic_start_sequences(0.01, 248.0, num_steps=10)
    n = len(SYNTHETIC_START_DUTY_PROFILES)
    assert synthetic["duty"].shape == (n, 10, 2)
    assert synthetic["wheel_cmd"].shape == (n, 10, 2)
    assert synthetic["poses"].shape == (n, 11, 3)
    np.testing.assert_array_equal(synthetic["init_twist"], 0.0)
    np.testing.assert_array_equal(synthetic["init_wheel"], 0.0)
    np.testing.assert_array_equal(synthetic["pose_weight"], 0.0)
    np.testing.assert_allclose(synthetic["wheel_cmd"], synthetic["duty"] * 248.0)

    data_like = {
        "init_twist": np.zeros((2, 3), np.float32),
        "init_wheel": np.zeros((2, 2), np.float32),
        "init_filt": np.zeros((2, 2), np.float32),
        "duty": np.zeros((2, 25, 2), np.float32),
        "wheel_cmd": np.zeros((2, 25, 2), np.float32),
        "poses": np.zeros((2, 26, 3), np.float32),
        "dt": np.full((2, 25), 0.01, np.float32),
        "pose_weight": np.ones(2, np.float32),
    }
    stacked = stack_sequences([data_like, synthetic])
    assert stacked["dt"].shape == (2 + n, 25)
    np.testing.assert_array_equal(stacked["pose_weight"], [1.0, 1.0] + [0.0] * n)
    # Synthetic windows are shorter: their padded steps are masked out.
    np.testing.assert_array_equal(stacked["mask"][2:, 10:], 0.0)
    np.testing.assert_array_equal(stacked["mask"][:, :10], 1.0)
