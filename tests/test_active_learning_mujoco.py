"""MuJoCo standing in for the robot inside the active-learning loop.

The deployment itself is covered by test_mujoco_deploy; what is tested here is
the wiring: the logs land in the iteration the pipeline is waiting on, in the
form decode-logs expects, and the loop knows when it is finished.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from wmr_simulator.active_learning.stages import (
    _deployment_seed,
    _identification_trajectory_jsn,
    stage_decode_logs,
    stage_finalize,
    stage_init,
    stage_run,
    stage_simulate_deployment,
)

PROBLEM = "problems/pololu_gains.yaml"


def write_reference_jsn(path, duration=2.0, dt=0.05):
    """A gentle arc as a Pololu reference, standing in for a designed trajectory."""
    num = int(round(duration / dt)) + 1
    time = np.arange(num) * dt
    speed, yaw_rate = 0.5, 0.4
    yaw = yaw_rate * time
    radius = speed / yaw_rate
    states = np.column_stack([radius * np.sin(yaw), radius * (1.0 - np.cos(yaw)), yaw])
    path.write_text(
        json.dumps(
            {
                "result": [
                    {
                        "dt": dt,
                        "num_states": num,
                        "num_actions": num - 1,
                        "states": states.tolist(),
                        "actions": np.column_stack(
                            [np.full(num - 1, speed), np.full(num - 1, yaw_rate)]
                        ).tolist(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def experiment_with_trajectory(tmp_path):
    """An initialized experiment whose iteration 1 has a bridged JSN ready.

    Stands in for the plan-id-trajectory stage, which is a full trajectory
    optimization and far too slow for a test. The arc is written as the bridged
    variant because that is the one a deployment drives; it does not actually
    return to the start, which only matters to how far the chained runs drift.
    """
    experiment = stage_init(
        tmp_path / "exp",
        {"problem": PROBLEM, "use_residual_model": False, "mujoco_deployment": {"num_logs": 2}},
    )
    paths = experiment.paths(1)
    write_reference_jsn(
        paths.identification_trajectory_dir / "identification_trajectory_bridge.JSN"
    )
    return experiment


def test_deployment_fills_the_iterations_data_dir(experiment_with_trajectory):
    experiment = experiment_with_trajectory
    paths = experiment.paths(1)

    written = stage_simulate_deployment(experiment, 1)

    assert [path.name for path in written] == ["TR00", "TR01"]
    # Nothing but the logs: decode-logs tries to decode every non-csv file it
    # finds here, and a ground-truth sidecar would leak the hidden plant too.
    assert sorted(path.name for path in paths.data_dir.iterdir()) == ["TR00", "TR01"]

    stage_decode_logs(experiment, 1)
    assert sorted(path.name for path in paths.data_dir.glob("*.csv")) == ["TR00.csv", "TR01.csv"]


def test_a_second_deployment_appends_instead_of_overwriting(experiment_with_trajectory):
    experiment = experiment_with_trajectory
    stage_simulate_deployment(experiment, 1, num_logs=1)
    written = stage_simulate_deployment(experiment, 1, num_logs=1)
    assert [path.name for path in written] == ["TR01"]


def test_runs_are_chained_and_reproducible(experiment_with_trajectory, capsys):
    """Only the first run is placed by hand; the rest start where the last ended."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment

    experiment = experiment_with_trajectory
    paths = experiment.paths(1)
    first, second = stage_simulate_deployment(experiment, 1, num_logs=2)
    assert first.read_bytes() != second.read_bytes()

    report = capsys.readouterr().out
    assert "placed by hand" in report and "left by the bridge" in report

    # The second run really did start from the first one's final pose.
    trajectory = _identification_trajectory_jsn(paths)
    replay = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.data_dir,
        seed=_deployment_seed(experiment, 1, 0), log_name="REPLAY",
    )
    repeat = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.data_dir,
        seed=_deployment_seed(experiment, 1, 1), start_pose=replay.final_pose, log_name="REPEAT",
    )
    assert repeat.log_path.read_bytes() == second.read_bytes()


def test_deployment_seeds_are_distinct_per_run_and_stable(experiment_with_trajectory):
    experiment = experiment_with_trajectory
    seeds = {_deployment_seed(experiment, iteration, index)
             for iteration in (1, 2) for index in range(4)}
    assert len(seeds) == 8
    assert _deployment_seed(experiment, 1, 0) == _deployment_seed(experiment, 1, 0)


def test_the_bridged_variant_is_what_gets_deployed(experiment_with_trajectory):
    """The bridge drives the robot back to the start, which is what makes the
    chained repeat runs possible without touching it."""
    paths = experiment_with_trajectory.paths(1)
    write_reference_jsn(paths.identification_trajectory_dir / "identification_trajectory.JSN")
    assert _identification_trajectory_jsn(paths).name == "identification_trajectory_bridge.JSN"


def test_a_missing_trajectory_names_the_stage_that_makes_one(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM})
    with pytest.raises(FileNotFoundError, match="plan-id-trajectory"):
        _identification_trajectory_jsn(experiment.paths(1))


def test_run_stops_once_the_iteration_target_is_reached(tmp_path, capsys):
    """A finished experiment is a no-op, not another iteration's work."""
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "num_iterations": 1})
    paths = experiment.paths(1)
    with paths.identification_result.open("w") as file:
        yaml.safe_dump({"estimated_params": {}}, file)
    with paths.gains_result.open("w") as file:
        yaml.safe_dump({"gains": [1.0, 2.0, 3.0, 4.0, 0.0], "schedule_enabled": False}, file)
    stage_finalize(experiment, 1)
    assert experiment.latest_iteration() == 2

    stage_run(experiment)

    assert "already has its 1 iteration" in capsys.readouterr().out
    # Untouched: no trajectory was planned for the final iteration.
    assert not any(experiment.paths(2).identification_trajectory_dir.iterdir())
