import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax
import numpy as np
import yaml

from wmr_simulator.residual_model import closed_loop
from wmr_simulator.residual_model.residual import init_residual_model
from wmr_simulator.simulation import SimulationPipeline

PROBLEM = str(Path(__file__).resolve().parents[1] / "problems" / "pololu_gains.yaml")


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
