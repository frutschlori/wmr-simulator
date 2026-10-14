import pytest

from wmr_simulator.active_learning.experiment import Experiment, load_yaml, save_yaml
from wmr_simulator.active_learning.stages import logged_stage, stage_init
from wmr_simulator.active_learning.study import (
    clean_interrupted_stages,
    configuration_overrides,
    interrupted_stages,
)

PROBLEM = "problems/pololu_gains.yaml"


def test_configuration_tags_set_the_three_switches():
    assert configuration_overrides("S-A-N") == {
        "use_gain_parametrization": False,
        "optimize_identification_trajectory": True,
        "optimize_tuning_trajectories": True,
        "use_residual_model": False,
    }
    assert configuration_overrides("D-F-R") == {
        "use_gain_parametrization": True,
        "optimize_identification_trajectory": False,
        "optimize_tuning_trajectories": False,
        "use_residual_model": True,
    }
    extra = configuration_overrides("S-AF-N")
    assert extra["optimize_identification_trajectory"] and not extra["optimize_tuning_trajectories"]
    with pytest.raises(ValueError):
        configuration_overrides("S-X-N")


def test_gain_parametrization_switch_disables_the_network(tmp_path):
    experiment = stage_init(
        tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False, "use_residual_model": False, "use_gain_parametrization": False}
    )
    paths = experiment.paths(1)
    assert load_yaml(paths.problem)["controller"]["gain_parametrization"]["enabled"] is False
    assert not paths.gainmlp_jsn.exists()


def test_interrupted_stage_outputs_are_removed(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False, "use_residual_model": False})
    paths = experiment.paths(1)
    with logged_stage(paths, "identify"):
        save_yaml(paths.identification_result, {"estimated_params": {}})
    (paths.tuning_trajectories_dir / "tuning_trajectory_00.pkl").write_bytes(b"partial")
    with pytest.raises(RuntimeError):
        with logged_stage(paths, "plan-tuning-trajectories"):
            raise RuntimeError("killed")

    experiment = Experiment.load(experiment.root)
    assert interrupted_stages(experiment) == [(1, "plan-tuning-trajectories")]
    clean_interrupted_stages(experiment)
    assert not any(paths.tuning_trajectories_dir.iterdir())
    assert paths.identification_result.is_file()
    assert interrupted_stages(experiment) == []
    assert load_yaml(paths.stage_log)["identify"]["seconds"] is not None
