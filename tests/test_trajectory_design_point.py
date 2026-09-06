"""The gains the trajectory designer computes its FIM around.

Every rollout in TrajectoryOptimizationPipeline drives the *static* controller
-- none of them passes ``schedule_params`` -- so the design point has to be the
static gains. A parametrized run's base gains are not a controller anybody
runs, and designing at them while driving the static controller designs for a
robot that does not exist.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from wmr_simulator.active_learning.experiment import load_yaml, save_yaml
from wmr_simulator.active_learning.stages import _static_design_gains, stage_finalize, stage_init
from wmr_simulator.gain_parametrization import num_params, params_from_cfg, to_cfg, with_flat_params
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

PROBLEM = "problems/pololu_gains.yaml"
STATIC_GAINS = [2.4, 5.1, 6.3, 0.9, 0.0]


@pytest.fixture(scope="module")
def default_pipeline():
    return TrajectoryOptimizationPipeline(problem_path=PROBLEM, objective_mode="gain-tuning")


@pytest.fixture(scope="module")
def static_gain_pipeline():
    return TrajectoryOptimizationPipeline(
        problem_path=PROBLEM, objective_mode="gain-tuning", controller_gains=STATIC_GAINS
    )


def test_design_point_defaults_to_the_problem_gains(default_pipeline):
    problem_gains = load_yaml(PROBLEM)["controller"]["gains"]
    np.testing.assert_allclose(np.asarray(default_pipeline.controller_gains), problem_gains, rtol=1e-6)


def test_explicit_gains_are_the_design_point_and_the_driven_gains(
    default_pipeline, static_gain_pipeline
):
    """In gain-tuning mode the FIM's design parameters *are* these gains, and
    they are also what the closed loop is rolled out at -- so overriding them
    has to move both."""
    np.testing.assert_allclose(np.asarray(static_gain_pipeline.controller_gains), STATIC_GAINS, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(static_gain_pipeline.nominal_parameters()), STATIC_GAINS, rtol=1e-6
    )
    # Same initial curve, different controller: the rollout must differ.
    assert not np.allclose(
        np.asarray(static_gain_pipeline.closed_loop_poses),
        np.asarray(default_pipeline.closed_loop_poses),
    )


def test_a_gain_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="controller_gains must have shape"):
        TrajectoryOptimizationPipeline(problem_path=PROBLEM, controller_gains=[1.0, 2.0, 3.0])


def _trained_problem(tmp_path) -> Path:
    """The stock problem with a non-identity gain parametrization baked in."""
    problem_cfg = load_yaml(PROBLEM)
    template = params_from_cfg(
        problem_cfg["controller"]["gain_parametrization"],
        [problem_cfg["robot"]["v_max"], problem_cfg["robot"]["omega_max"]],
    )
    rng = np.random.default_rng(0)
    trained = with_flat_params(0.5 * rng.normal(size=num_params(template)), template)
    problem_cfg["controller"]["gain_parametrization"] = {**to_cfg(trained), "enabled": True}
    return Path(save_yaml(tmp_path / "problem_trained_parametrization.yaml", problem_cfg))


def test_a_trained_gain_parametrization_does_not_reach_the_design(tmp_path, default_pipeline):
    """The designer drives the static controller, so a trained network in the
    problem yaml changes nothing about the design -- which is exactly why the
    base gains it was trained against are the wrong design point."""
    trained_pipeline = TrajectoryOptimizationPipeline(
        problem_path=str(_trained_problem(tmp_path)), objective_mode="gain-tuning"
    )
    assert trained_pipeline.simulation.gain_parametrization_enabled
    np.testing.assert_array_equal(
        np.asarray(trained_pipeline.closed_loop_poses),
        np.asarray(default_pipeline.closed_loop_poses),
    )


def test_static_design_gains_prefer_the_static_baseline(tmp_path):
    """The active-learning designers read their design point out of
    robot_config_static_gains.yaml -- the same file the tune-gains stage warm
    starts its static run from -- not out of the iteration's problem yaml."""
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM})
    paths = experiment.paths(1)

    # Iteration 1 has no separate static controller: its own gains are static.
    assert _static_design_gains(paths) is None

    with paths.identification_result.open("w") as file:
        yaml.safe_dump({"estimated_params": {}}, file)
    with paths.gains_result.open("w") as file:
        yaml.safe_dump(
            {
                "gains": [9.1, 8.2, 41.0, 21.5, 0.0],
                "static_gains": STATIC_GAINS,
                "schedule_enabled": True,
                "schedule": {"scheduled_indices": [0, 1, 2], "rho": [0.5] * 3, "W": [[0.1, 0.0]] * 3},
            },
            file,
        )
    next_paths = stage_finalize(experiment, 1)

    # The iteration problem carries the parametrized run's (runaway) base gains;
    # the designers must not use those.
    assert load_yaml(next_paths.problem)["controller"]["gains"][2] == 41.0
    assert _static_design_gains(next_paths) == STATIC_GAINS


def test_a_trained_parametrization_without_explicit_gains_is_flagged(tmp_path, capsys):
    """Designing at a trained network's base gains is the flaw ``controller_gains``
    exists for, and nothing downstream can tell it happened -- so say so. The
    stock yamls ship an *enabled* parametrization at theta = 0, which is the
    static controller exactly, so enablement alone must not trip it."""
    TrajectoryOptimizationPipeline(problem_path=PROBLEM)
    assert "WARNING" not in capsys.readouterr().out

    TrajectoryOptimizationPipeline(problem_path=str(_trained_problem(tmp_path)))
    assert "gain parametrization is trained" in capsys.readouterr().out

    TrajectoryOptimizationPipeline(
        problem_path=str(_trained_problem(tmp_path)), controller_gains=STATIC_GAINS
    )
    assert "WARNING" not in capsys.readouterr().out
