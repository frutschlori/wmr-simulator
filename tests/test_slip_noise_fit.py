"""Tests for the AR(1) slip-noise fit from logs and the zero-phase mocap filter."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.slip_noise import estimate_slip_noise, estimate_slip_noise_from_log
from wmr_simulator.pololu.log_loader import zero_phase_moving_average
from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.models.slip import ar1_slip_update, integrate_planar_pose, slip_body_velocities
from wmr_simulator.types import PhysicalParams


def test_zero_phase_filter_preserves_ramp_interior():
    ramp = np.linspace(0.0, 10.0, 200)
    filtered = zero_phase_moving_average(ramp, 9)
    # A symmetric average leaves a linear ramp unchanged away from the edges.
    np.testing.assert_allclose(filtered[10:-10], ramp[10:-10], atol=1e-9)


def test_zero_phase_filter_no_phase_shift():
    x = np.zeros(201)
    x[100] = 1.0  # impulse at the center
    filtered = zero_phase_moving_average(x, 11)
    # A boxcar spreads the impulse into a plateau; zero phase delay means the
    # response centroid stays exactly on the impulse and the response is symmetric.
    centroid = float(np.sum(np.arange(len(filtered)) * filtered) / np.sum(filtered))
    np.testing.assert_allclose(centroid, 100.0, atol=1e-9)
    response = filtered[95:106]
    np.testing.assert_allclose(response, response[::-1], atol=1e-12)


def _synthetic_slip_rollout(sigma, tau, dt, n, ur=120.0, ul=80.0):
    """Kinematic rollout with known AR(1) slip; returns pose series + encoder speeds."""
    params = PhysicalParams(
        wheel_radius=jnp.asarray(0.0164),
        base_diameter=jnp.asarray(0.0853),
        max_wheel_speed=jnp.asarray(248.0),
        time_constant=jnp.asarray(0.0),
    )
    r = float(params.wheel_radius)
    L = float(params.base_diameter)
    u_enc = jnp.asarray([ur, ul])

    def step(carry, key):
        pose, eta = carry
        eta = ar1_slip_update(eta, key, sigma, tau, dt)
        u_eff = u_enc * (1.0 - eta)
        v, w = slip_body_velocities(u_eff, r, L)
        pose = integrate_planar_pose(pose, v, w, dt)
        return (pose, eta), pose

    keys = jax.random.split(jax.random.PRNGKey(3), n)
    _, poses = jax.lax.scan(step, (jnp.zeros(3), jnp.zeros(2)), keys)
    pose_time = dt * np.arange(1, n + 1)
    wheel_time = pose_time
    wheel_speeds = np.tile([ur, ul], (n, 1))
    return params, pose_time, np.asarray(poses), wheel_time, wheel_speeds


def test_slip_noise_fit_recovers_ar1_parameters():
    sigma, tau, dt = 0.05, 0.3, 0.01
    params, pose_t, poses, wheel_t, wheel_u = _synthetic_slip_rollout(sigma, tau, dt, n=40_000)
    fit = estimate_slip_noise(pose_t, poses, wheel_t, wheel_u, params)
    assert abs(fit["sigma"] - sigma) / sigma < 0.1, fit
    assert abs(fit["tau"] - tau) / tau < 0.25, fit


def test_slip_noise_fit_near_zero_on_clean_rollout():
    params, pose_t, poses, wheel_t, wheel_u = _synthetic_slip_rollout(0.0, 0.0, 0.01, n=5_000)
    fit = estimate_slip_noise(pose_t, poses, wheel_t, wheel_u, params)
    assert fit["sigma"] < 1e-4, fit


def test_slip_noise_fit_from_simulation_log_runs():
    # Wiring smoke test on a full closed-loop log (true poses/wheel speeds).
    pipeline = SimulationPipeline(problem_path="problems/pololu_gains.yaml", seed=0)
    log = pipeline.run_closed_loop(
        pipeline.hidden_params, use_hidden_robot=True, wheel_speed_log_source="true"
    )
    log = log._replace(pose=log.pose._replace(states=log.pose.true_states))
    fit = estimate_slip_noise_from_log(log, pipeline.hidden_params)
    assert np.isfinite(fit["sigma"]) and fit["sigma"] > 0.0
    assert fit["num_samples_r"] > 100
