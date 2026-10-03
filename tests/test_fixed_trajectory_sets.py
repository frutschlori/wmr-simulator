import pickle
from pathlib import Path

import numpy as np
import pytest

from wmr_simulator.active_learning.experiment import DEFAULT_EXPERIMENT_CONFIG
from wmr_simulator.trajectory_optimization import fixed_sets
from wmr_simulator.trajectory_optimization.baselines import chain_references, circle_reference, summarize_motion
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


def test_a_clockwise_circle_is_driven_forwards():
    states = circle_reference(0.5, 4.0, DT, clockwise=True)
    assert forward_speed(states).min() >= -1e-9
    assert states[1:-1, 5].max() < 0.0


def test_chained_references_continue_the_pose_through_a_stop():
    first = circle_reference(0.5, 2.0, DT, sweep=np.pi)
    second = circle_reference(0.8, 3.0, DT, sweep=2.0, clockwise=True)
    chained = chain_references(first, second)
    assert len(chained) == len(first) + len(second) - 1
    seam = len(first) - 1
    # At rest on both sides of the stop, so neighbouring samples barely move.
    assert np.max(np.linalg.norm(np.diff(chained[seam - 1 : seam + 2, :2], axis=0), axis=1)) < 0.01
    assert abs(chained[seam + 1, 2] - chained[seam - 1, 2]) < 0.05
    assert forward_speed(chained).min() >= -1e-9


def test_the_matched_set_matches_the_designed_set_and_its_limits(problem):
    limits = fixed_sets.motion_limits(problem.robot_cfg)
    references = fixed_sets.matched_reference_set(
        problem.sim_time, DT, limits, problem.environment_min, problem.environment_max
    )
    assert len(references) == DEFAULT_EXPERIMENT_CONFIG["tuning_trajectories"]["num_trajectories"]
    for states in references.values():
        assert summarize_motion(states, DT)["duration"] == pytest.approx(problem.sim_time)
        assert fixed_sets.is_within_limits(states, DT, limits)
        assert forward_speed(states).min() >= -1e-9
        assert_inside(states, problem)


def test_the_fixed_identification_reference_keeps_each_phase_in_its_limits(problem):
    phases = DEFAULT_EXPERIMENT_CONFIG["identification_trajectory"]["phases"]
    states, identified_duration = fixed_sets.fixed_identification_reference(
        phases, DT, problem.robot_cfg, problem.environment_min, problem.environment_max
    )
    assert identified_duration == pytest.approx(phases[0]["duration"])
    assert (len(states) - 1) * DT == pytest.approx(sum(phase["duration"] for phase in phases))
    seam = int(round(identified_duration / DT))
    for part, phase in ((states[: seam + 1], phases[0]), (states[seam:], phases[1])):
        assert fixed_sets.is_within_limits(part, DT, fixed_sets.motion_limits(problem.robot_cfg, phase["motion_limits"]))
    assert np.linalg.norm(states[seam, 3:5]) < 1e-9
    # The identified phase turns both ways.
    assert states[:seam, 5].max() > 0.5 and states[:seam, 5].min() < -0.5
    assert_inside(states, problem)


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


def _export(directory, total_time):
    fixed_sets.export_reference_set({"circle": circle_reference(0.5, total_time, DT)}, directory, DT)
    return directory


def test_a_fixed_tuning_set_is_copied_and_one_longer_than_the_horizon_is_refused(tmp_path):
    from wmr_simulator.active_learning.stages import stage_init, stage_plan_tuning_trajectories

    fitting = _export(tmp_path / "fits", 4.0)
    experiment = stage_init(
        tmp_path / "exp",
        {
            "problem": PROBLEM,
            "prior_tuning": False,
            "use_residual_model": False,
            "optimize_tuning_trajectories": False,
            "baseline_tuning_trajectories_dir": str(fitting),
        },
    )
    paths = experiment.paths(1)
    # Stands in for the identify stage, which the tuning stages build on.
    paths.problem_identified.write_text(paths.problem.read_text())
    copied = stage_plan_tuning_trajectories(experiment, 1)
    assert [path.name for path in copied] == ["circle.pkl"]

    too_long = _export(tmp_path / "long", 6.0)
    experiment.config["baseline_tuning_trajectories_dir"] = str(too_long)
    for path in experiment.paths(1).tuning_trajectories_dir.glob("*.pkl"):
        path.unlink()
    with pytest.raises(ValueError, match="tuning_trajectories.sim_time to at least 6.00"):
        stage_plan_tuning_trajectories(experiment, 1)


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
