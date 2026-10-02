import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.residual_model import closed_loop
from wmr_simulator.residual_model.residual import init_residual_model, shift_simulation_log_time
from wmr_simulator.simulation import SimulationPipeline

PROBLEM = str(Path(__file__).resolve().parents[1] / "problems" / "pololu_gains.yaml")


def test_sim_time_override_sets_the_horizon():
    pipeline = SimulationPipeline(problem_path=PROBLEM, sim_time=1.0)
    assert pipeline.sim_time == 1.0
    assert len(pipeline.reference_time_grid) == int(round(1.0 / pipeline.geometry_dt)) + 1


def test_rollout_can_start_with_turning_wheels():
    pipeline = SimulationPipeline(problem_path=PROBLEM, sim_time=0.5)
    wheels = jnp.asarray([100.0, 80.0])
    log = pipeline.run_closed_loop(pipeline.hidden_params, use_hidden_robot=True, initial_wheel_speeds=wheels)
    at_rest = pipeline.run_closed_loop(pipeline.hidden_params, use_hidden_robot=True)
    # Already moving: the first wheel step covers ground the resting start does not.
    moved = np.linalg.norm(np.asarray(log.pose.true_states[1, :2] - log.pose.true_states[0, :2]))
    rested = np.linalg.norm(np.asarray(at_rest.pose.true_states[1, :2] - at_rest.pose.true_states[0, :2]))
    assert moved > 10 * rested


def test_shift_moves_every_time_axis():
    pipeline = SimulationPipeline(problem_path=PROBLEM, sim_time=0.5)
    log = pipeline.run_closed_loop(pipeline.hidden_params, use_hidden_robot=True)
    shifted = shift_simulation_log_time(log, 0.04)
    for before, after in ((log.pose.time_s, shifted.pose.time_s), (log.reference.time_s, shifted.reference.time_s),
                          (log.wheel.time_s, shifted.wheel.time_s)):
        np.testing.assert_allclose(np.asarray(after) - np.asarray(before), 0.04, atol=1e-6)


def test_windows_of_a_simulated_run_replay_it_and_train_finitely(tmp_path):
    pipeline = SimulationPipeline(problem_path=PROBLEM, sim_time=2.0)
    recorded = pipeline.run_closed_loop(pipeline.hidden_params, use_hidden_robot=True)
    # A recorded log carries the mocap pose in ``states``; use the true track.
    recorded = recorded._replace(pose=recorded.pose._replace(states=recorded.pose.true_states))
    windows = closed_loop.build_windows(
        [recorded], [PROBLEM], window_s=0.5, stride_s=0.5,
        geometry_dt=pipeline.geometry_dt, wheel_dt=pipeline.wheel_dt,
    )
    assert windows is not None and len(windows) >= 2
    plant = yaml.safe_load(open(PROBLEM))["robot"]
    loss = closed_loop.window_loss_fn(closed_loop.window_pipeline(PROBLEM, plant, 0.5), windows)
    model = init_residual_model(jax.random.PRNGKey(0), num_experts=2, hidden_sizes=(8,))
    nominal_loss, _ = loss(closed_loop.zero_residual(model))
    # The nominal plant re-simulating its own run from mid-run states is close
    # (noise draws differ), far below a metre-scale error.
    assert float(nominal_loss) < 1e3
    trained, history = closed_loop.train_closed_loop(model, loss, steps=2, learning_rate=1e-3, log_every=0)
    assert np.all(np.isfinite(history))
