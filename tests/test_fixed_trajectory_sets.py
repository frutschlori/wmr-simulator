import pickle
from pathlib import Path

import numpy as np
import pytest

from wmr_simulator.active_learning.experiment import DEFAULT_EXPERIMENT_CONFIG
from wmr_simulator.trajectory_optimization import fixed_sets
from wmr_simulator.trajectory_optimization.baselines import circle_reference, summarize_motion
from wmr_simulator.trajectory_optimization.pipeline import ProblemDefinition

PROBLEM = "problems/pololu_gains.yaml"
DT = 0.05


@pytest.fixture(scope="module")
def problem():
    problem = ProblemDefinition(PROBLEM)
    problem.sim_time = float(DEFAULT_EXPERIMENT_CONFIG["tuning_trajectories"]["sim_time"])
    return problem


def forward_speed(states):
    return states[:, 3] * np.cos(states[:, 2]) + states[:, 4] * np.sin(states[:, 2])


def assert_inside(states, problem):
    assert np.all(states[:, :2] >= problem.environment_min - 1e-9)
    assert np.all(states[:, :2] <= problem.environment_max + 1e-9)


def test_the_benchmark_set_converts_every_reference_whole():
    references = fixed_sets.benchmark_reference_set("trajectory_exports/benchmark_set", DT)
    assert len(references) == len(list(Path("trajectory_exports/benchmark_set").glob("*.JSN")))
    assert max(summarize_motion(states, DT)["duration"] for states in references.values()) > 5.0


def test_random_twist_references_are_feasible_placed_and_reproducible(problem):
    limits = fixed_sets.motion_limits(problem.robot_cfg)
    make = lambda num: fixed_sets.random_twist_reference_set(
        num, problem.sim_time, DT, limits, problem.environment_min, problem.environment_max, seed=3
    )
    references = make(6)
    for states in references.values():
        assert fixed_sets.is_within_limits(states, DT, limits)
        assert forward_speed(states).min() >= -1e-9
        assert np.linalg.norm(states[0, 3:5]) < 1e-9 and np.linalg.norm(states[-1, 3:5]) < 1e-9
        assert_inside(states, problem)
    # Same seed, larger set: the smaller one is its prefix.
    larger = make(8)
    for name, states in references.items():
        np.testing.assert_array_equal(states, larger[name])


def test_random_bspline_references_are_feasible_and_keep_their_speed(problem):
    tuning = DEFAULT_EXPERIMENT_CONFIG["tuning_trajectories"]
    references, control_points, report = fixed_sets.random_bspline_reference_set(
        problem,
        6,
        int(tuning["num_control_points"]),
        seed=1,
        min_speed=float(tuning["min_speed"]),
        min_speed_fraction=float(tuning["min_speed_fraction"]),
    )
    limits = fixed_sets.motion_limits(problem.robot_cfg)
    assert set(references) == set(control_points)
    for name, states in references.items():
        assert fixed_sets.is_within_limits(states, DT, limits)
        assert_inside(states, problem)
        assert control_points[name].shape == (int(tuning["num_control_points"]), 2)
        # The projection must not shrink the curves (the lower end of the draw is 0.4 m/s).
        assert summarize_motion(states, DT)["v_mean"] > 0.3
    assert 0.0 < report["acceptance_rate"] <= 1.0


def test_an_existing_set_is_never_overwritten(tmp_path):
    references = {"line": circle_reference(0.5, 4.0, DT)}
    fixed_sets.export_reference_set(references, tmp_path / "set", DT, fixed_set="test")
    with open(tmp_path / "set" / "line.pkl", "rb") as file:
        assert pickle.load(file)["fixed_set"] == "test"
    with pytest.raises(FileExistsError):
        fixed_sets.export_reference_set(references, tmp_path / "set", DT)


def test_the_designed_and_fixed_switches_are_independent(tmp_path):
    from wmr_simulator.active_learning.stages import _identified_duration, stage_init, stage_plan_identification_trajectory

    identification = tmp_path / "identification"
    phases = DEFAULT_EXPERIMENT_CONFIG["identification_trajectory"]["phases"]
    problem = ProblemDefinition(PROBLEM)
    states, identified_duration = fixed_sets.fixed_identification_reference(
        phases, DT, problem.robot_cfg, problem.environment_min, problem.environment_max
    )
    fixed_sets.export_reference_set(
        {"fixed_identification": states}, identification, DT, identified_duration=identified_duration
    )
    experiment = stage_init(
        tmp_path / "exp",
        {
            "problem": PROBLEM,
            "prior_tuning": False,
            "use_residual_model": False,
            "optimize_identification_trajectory": False,
            "baseline_identification_trajectory": str(identification / "fixed_identification.pkl"),
        },
    )
    assert experiment.config["optimize_tuning_trajectories"] is True
    jsn = stage_plan_identification_trajectory(experiment, 1)
    assert jsn.is_file()
    assert _identified_duration(experiment.paths(1)) == pytest.approx(identified_duration)
