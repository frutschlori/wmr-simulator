import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.models import (
    apply_residual_model,
    init_residual_model,
    load_residual_model,
    residual_features,
    residual_corrected_twist,
    save_residual_model,
)
from wmr_simulator.models.residual import RESIDUAL_INPUT_DIM, RESIDUAL_OUTPUT_DIM
from wmr_simulator.models.robot import DiffDrive

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
    """State-action layout: [vx, vy, omega, wheel_r, wheel_l, duty_r, duty_l]."""
    features = residual_features(
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([4.0, 5.0]),
        jnp.asarray([6.0, 7.0]),
    )
    assert features.shape == (RESIDUAL_INPUT_DIM,)
    np.testing.assert_allclose(np.asarray(features), np.arange(1.0, 8.0, dtype=np.float32))


def test_state_action_features_exclude_cmd_and_nominal():
    from wmr_simulator.models.residual import RESIDUAL_FEATURE_NAMES

    assert RESIDUAL_FEATURE_NAMES == (
        "vx_body",
        "vy_body",
        "omega",
        "wheel_speed_r",
        "wheel_speed_l",
        "duty_r",
        "duty_l",
    )


def test_step_tracks_lateral_velocity_state():
    robot = DiffDrive(ROBOT_CFG, dt=0.01)
    model = small_model()
    state = robot.get_init_state(jax.random.PRNGKey(0))
    duty = jnp.asarray([0.6, 0.2])
    without = robot.step(state, duty)
    assert float(without.vel_lateral) == 0.0
    with_residual = robot.step(state, duty, residual_model=model)
    assert with_residual.vel_lateral.shape == ()
    # The residual's lateral correction must be carried in the state so the
    # next step's features can condition on it.
    features = jnp.concatenate(
        [
            jnp.asarray([state.vel_omega[0], state.vel_lateral, state.vel_omega[1]]),
            with_residual.wheel_speeds,
            duty,
        ]
    )
    from wmr_simulator.models import apply_residual_model as apply

    delta = apply(model, features)
    np.testing.assert_allclose(float(with_residual.vel_lateral), float(delta[1]), rtol=1e-5)
