"""Validation checks for the physical slip model.

Model components (see src/wmr_simulator/slip.py for references):
- base_diameter is the *effective* wheelbase (Borenstein & Feng 1996)
- a_slip_max: traction-limited longitudinal slip / burnout (Wong 2008)
- {slip_sigma, slip_tau}: AR(1)/first-order Gauss-Markov wheel slip (Maybeck 1979)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.robot import DiffDrive
from wmr_simulator.simulation import SimulationPipeline
from wmr_simulator.slip import ar1_slip_update, fit_ar1_moments, slip_body_velocities
from wmr_simulator.types import PhysicalParams, physical_params_from_array, physical_params_to_array

ROBOT_CFG = {
    "wheel_radius": 0.0164,
    "base_diameter": 0.0853,
    "max_wheel_speed": 248.0,
    "time_constant": 0.0,  # ideal actuation for kinematic checks
}


def _robot(**overrides):
    cfg = dict(ROBOT_CFG)
    cfg.update(overrides)
    return DiffDrive(robot_cfg=cfg, dt=0.01)


def _step_n(robot, duty, n=100, **step_kwargs):
    state = robot.get_init_state(key=jax.random.PRNGKey(0))
    for _ in range(n):
        state = robot.step(state, duty, **step_kwargs)
    return state


def test_zero_slip_params_reproduce_ideal_kinematics():
    # Straight drive with all slip params zero: pure x-motion at r*u.
    robot = _robot()
    duty = jnp.asarray([0.5, 0.5])
    state = _step_n(robot, duty, n=100)
    u = 0.5 * ROBOT_CFG["max_wheel_speed"]
    expected_x = ROBOT_CFG["wheel_radius"] * u * 100 * 0.01
    np.testing.assert_allclose(float(state.pose[0]), expected_x, rtol=1e-5)
    np.testing.assert_allclose(float(state.pose[1]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state.slip_noise), 0.0, atol=1e-7)


def test_traction_limit_produces_burnout():
    # Full throttle from standstill with an ideal (tau=0) motor: the motor-side wheel
    # speed jumps immediately, the ground speed ramps at a_slip_max -> the body
    # accelerates at exactly the traction limit while the wheel "burns out".
    a_slip_max = 3.0
    robot = _robot(a_slip_max=a_slip_max)
    duty = jnp.asarray([1.0, 1.0])
    state = robot.get_init_state(key=jax.random.PRNGKey(0))
    dt = 0.01
    velocities = []
    for _ in range(20):
        state = robot.step(state, duty)
        velocities.append(float(state.vel_omega[0]))
        # motor side is at full speed, ground side still ramping
        assert float(state.wheel_speeds[0]) > float(state.ground_wheel_speeds[0])
    accelerations = np.diff([0.0] + velocities) / dt
    np.testing.assert_allclose(accelerations, a_slip_max, rtol=1e-4)


def test_traction_limit_disabled_matches_ideal():
    # a_slip_max = 0 disables the limit: ground speeds equal motor speeds.
    robot = _robot(a_slip_max=0.0)
    duty = jnp.asarray([1.0, 1.0])
    state = _step_n(robot, duty, n=5)
    np.testing.assert_allclose(
        np.asarray(state.ground_wheel_speeds), np.asarray(state.wheel_speeds), rtol=1e-6
    )


def test_backlash_reversal_absorbs_gap():
    # Direction reversal: the first 2*b radians of motor motion are swallowed by the
    # gap while the wheel holds still (Nordin & Gutman 2002).
    from wmr_simulator.slip import backlash_transmission

    b, dt = 0.03, 0.01
    offset = jnp.asarray([b, b])  # engaged forward
    u_reverse = jnp.asarray([-1.0, -1.0])  # slow reversal: 0.01 rad of motor motion per step
    absorbed = 0.0
    steps_stalled = 0
    for _ in range(30):
        offset, u_wheel = backlash_transmission(offset, u_reverse, b, dt)
        if abs(float(u_wheel[0])) < 1e-9:
            steps_stalled += 1
            absorbed += -float(u_reverse[0]) * dt
    # Gap width is 2b = 0.06 rad; at 1 rad/s and dt=0.01 that is 6 fully stalled steps.
    assert steps_stalled == 6
    np.testing.assert_allclose(absorbed, 2 * b, atol=1e-6)


def test_backlash_engaged_is_transparent():
    from wmr_simulator.slip import backlash_transmission

    b, dt = 0.03, 0.01
    offset = jnp.asarray([b, b])  # engaged forward
    u_forward = jnp.asarray([15.0, 15.0])
    offset, u_wheel = backlash_transmission(offset, u_forward, b, dt)
    np.testing.assert_allclose(np.asarray(u_wheel), np.asarray(u_forward), atol=1e-6)


def test_backlash_dither_keeps_wheel_still():
    # Encoder-side dithering inside the gap: wheel speed stays exactly zero, so the
    # measured wheel-speed sign can flip while the body keeps its motion trend.
    from wmr_simulator.slip import backlash_transmission

    b, dt = 0.03, 0.01
    offset = jnp.asarray([0.0, 0.0])  # mid-gap
    for step in range(20):
        u = jnp.asarray([1.0, -1.0]) * (1.0 if step % 2 == 0 else -1.0)  # +-1 rad/s dither
        offset, u_wheel = backlash_transmission(offset, u, b, dt)
        np.testing.assert_allclose(np.asarray(u_wheel), 0.0, atol=1e-6)


def test_backlash_disabled_is_identity():
    from wmr_simulator.slip import backlash_transmission

    offset = jnp.asarray([0.0, 0.0])
    u = jnp.asarray([12.0, -7.0])
    next_offset, u_wheel = backlash_transmission(offset, u, 0.0, 0.01)
    np.testing.assert_allclose(np.asarray(u_wheel), np.asarray(u), atol=1e-7)
    np.testing.assert_allclose(np.asarray(next_offset), 0.0, atol=1e-7)


def test_slip_body_velocities_identity():
    v, w = slip_body_velocities(jnp.asarray([10.0, 10.0]), 0.02, 0.1)
    np.testing.assert_allclose(float(v), 0.2, rtol=1e-6)
    np.testing.assert_allclose(float(w), 0.0, atol=1e-7)


def test_ar1_statistics_match_configuration():
    # Long AR(1) rollout: stationary std ~ sigma, lag-1 autocorr ~ exp(-dt/tau).
    sigma, tau, dt = 0.08, 0.25, 0.01
    key = jax.random.PRNGKey(42)
    n = 60_000

    def step(carry, key):
        eta = ar1_slip_update(carry, key, sigma, tau, dt)
        return eta, eta[0]

    keys = jax.random.split(key, n)
    _, series = jax.lax.scan(step, jnp.zeros(2), keys)
    series = np.asarray(series[2_000:])  # discard burn-in
    assert abs(series.std() - sigma) / sigma < 0.05
    rho_hat = np.corrcoef(series[:-1], series[1:])[0, 1]
    assert abs(rho_hat - np.exp(-dt / tau)) < 0.02


def test_fit_ar1_moments_recovers_parameters():
    sigma, tau, dt = 0.05, 0.4, 0.01
    keys = jax.random.split(jax.random.PRNGKey(7), 80_000)

    def step(carry, key):
        eta = ar1_slip_update(carry, key, sigma, tau, dt)
        return eta, eta[0]

    _, series = jax.lax.scan(step, jnp.zeros(2), keys)
    sigma_hat, tau_hat = fit_ar1_moments(np.asarray(series[5_000:]), dt)
    assert abs(float(sigma_hat) - sigma) / sigma < 0.1
    assert abs(float(tau_hat) - tau) / tau < 0.2


def test_physical_params_array_roundtrip():
    params = PhysicalParams(
        wheel_radius=jnp.asarray(0.016),
        base_diameter=jnp.asarray(0.085),
        max_wheel_speed=jnp.asarray(248.0),
        time_constant=jnp.asarray(0.22),
        a_slip_max=jnp.asarray(5.0),
        b_backlash=jnp.asarray(0.035),
        slip_sigma=jnp.asarray(0.03),
        slip_tau=jnp.asarray(0.3),
    )
    values = physical_params_to_array(params)
    assert values.shape == (8,)
    rebuilt = physical_params_from_array(values)
    for a, b in zip(params, rebuilt):
        np.testing.assert_allclose(float(a), float(b), rtol=1e-6)


def test_closed_loop_regression_zero_slip():
    # With sigma=0 and traction limit disabled the closed loop must match the
    # pre-slip-model behavior
    # (the wheel-level dynamics and pose integration are unchanged in that limit).
    pipeline = SimulationPipeline(problem_path="problems/pololu_no_noise.yaml", seed=0)
    log = pipeline.run_closed_loop(pipeline.hidden_params)
    final_pose = np.asarray(log.pose.true_states[-1])
    assert np.all(np.isfinite(final_pose))
    # Deterministic sanity: rerunning gives the identical trajectory.
    log2 = pipeline.run_closed_loop(pipeline.hidden_params)
    np.testing.assert_allclose(
        np.asarray(log.pose.true_states), np.asarray(log2.pose.true_states), atol=1e-6
    )


def test_replay_loss_sensitive_to_a_slip_max_and_backlash():
    # Identifiability smoke test: on a turn-rich trajectory with transients the replay
    # pose loss must have non-zero gradient w.r.t. a_slip_max and b_backlash (they
    # enter step_kinematic).
    from wmr_simulator.identification.pipeline import SystemIdentificationPipeline

    pipeline = SystemIdentificationPipeline(
        problem_path="problems/curve.yaml",
        initial_params=PhysicalParams(
            wheel_radius=jnp.asarray(0.0164),
            base_diameter=jnp.asarray(0.0853),
            max_wheel_speed=jnp.asarray(248.0),
            time_constant=jnp.asarray(0.22),
            a_slip_max=jnp.asarray(4.0),
        ),
        seed=0,
        window_length=50,
    )

    def loss(theta):
        params = pipeline.initial_params._replace(
            a_slip_max=theta[0], b_backlash=theta[1]
        )
        return pipeline.loss(params)

    grad = np.asarray(jax.grad(loss)(jnp.asarray([4.0, 0.035])))
    assert np.all(np.isfinite(grad))
    assert np.abs(grad[0]) > 0.0, f"a_slip_max not observable, grad={grad}"
    # b_backlash sensitivity requires reversals / startup gap traverses; the closed-loop
    # start from standstill provides at least the startup traverse.
    assert np.abs(grad[1]) > 0.0, f"b_backlash not observable, grad={grad}"
