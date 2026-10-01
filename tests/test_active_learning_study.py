import numpy as np
import pytest

from wmr_simulator.active_learning.experiment import Experiment, load_yaml, save_yaml
from wmr_simulator.active_learning.stages import logged_stage, stage_init
from wmr_simulator.active_learning.study import (
    REPO_ROOT,
    clean_interrupted_stages,
    configuration_overrides,
    interrupted_stages,
    run_overrides,
    study_runs,
)
from wmr_simulator.gain_tuning.checks import bound_hits, gains_unchanged, stalled, warnings_for

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


def test_run_overrides_resolve_paths_and_let_the_tag_win():
    spec = {
        "phase": "p",
        "iterations": 3,
        "seeds": [0, 1],
        "overrides": {"use_residual_model": True, "gain_tuning": {"steps": 10}},
        "configurations": {"a": {"tag": "S-A-N", "overrides": {"gain_tuning": {"optimizer": "adam"}}}},
    }
    overrides = run_overrides(spec, "a", 1)
    assert overrides["use_residual_model"] is False
    assert overrides["gain_tuning"] == {"steps": 10, "optimizer": "adam"}
    assert overrides["seed"] == 1 and overrides["num_iterations"] == 3
    assert overrides["use_standalone_gain_tuning_defaults"] is False
    assert overrides["problem"] == str(REPO_ROOT / PROBLEM)
    assert overrides["benchmark"]["trajectory"].startswith(str(REPO_ROOT))
    assert study_runs(spec) == [("a", 0), ("a", 1)]


def test_gain_parametrization_switch_disables_the_network(tmp_path):
    experiment = stage_init(
        tmp_path / "exp", {"problem": PROBLEM, "use_residual_model": False, "use_gain_parametrization": False}
    )
    paths = experiment.paths(1)
    assert load_yaml(paths.problem)["controller"]["gain_parametrization"]["enabled"] is False
    assert not paths.gainmlp_jsn.exists()


def test_interrupted_stage_outputs_are_removed(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "use_residual_model": False})
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


def test_bound_hits_and_stall_checks():
    hits = bound_hits([20.0, 5.0, 1e-3, 3.0, 0.0, 0.5], k_min_stab=1e-3, k_max_stab=20.0, k_max_rest=20.0)
    assert [(hit["gain"], hit["bound"]) for hit in hits] == [
        ("kx", "k_max_stab"),
        ("kth", "k_min_stab"),
        ("kimotor", "zero"),
    ]
    assert gains_unchanged([1.0] * 6, [1.0] * 6, 1e-3, 20.0, 20.0)
    assert not gains_unchanged([1.0] * 6, [1.0, 1.0, 1.0, 1.0, 1.0, 1.1], 1e-3, 20.0, 20.0)

    initial = np.array([[0.1, 0.2], [0.3, 0.4]])
    moved = {"best_start_index": 1, "initial_values_per_start": initial, "final_values_per_start": initial + [[0, 0], [0.1, 0]]}
    parked = {"best_start_index": 0, "initial_values_per_start": initial, "final_values_per_start": initial + [[0, 0], [0.1, 0]]}
    assert not stalled(moved)
    assert stalled(parked)

    lines = warnings_for(
        {"static": {"stalled": True, "unchanged_from_init": False, "bound_hits": hits,
                    "divergence": {"tuned": {"rollouts": 8, "diverged": 2}}}}
    )
    assert len(lines) == 1 + len(hits) + 1


def test_interrupted_decode_removes_the_binary_logs_too(tmp_path):
    # `run` redeploys whenever no decoded log exists; binaries left behind would
    # make it add more logs next to them instead of reproducing them.
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "use_residual_model": False})
    paths = experiment.paths(1)
    with logged_stage(paths, "simulate-deployment"):
        for name in ("TR00", "TR01"):
            (paths.data_dir / name).write_bytes(b"log")
    (paths.data_dir / "TR00.csv").write_text("partial")
    with pytest.raises(RuntimeError):
        with logged_stage(paths, "decode-logs"):
            raise RuntimeError("killed")
    clean_interrupted_stages(Experiment.load(experiment.root))
    assert not list(paths.data_dir.glob("TR*"))


def test_a_skipped_residual_counts_as_done_and_designs_run_nominal(tmp_path):
    # An iteration whose pooled logs all diverged trains no residual
    # (residual.max_position_error); the loop must move on, nominal.
    from wmr_simulator.active_learning.stages import _load_design_residual_model, iteration_status

    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "use_residual_model": True})
    paths = experiment.paths(1)
    assert not iteration_status(experiment, 1)["train-residual"]
    save_yaml(paths.residual_skipped, {"reason": "every pooled identification log diverged"})
    assert iteration_status(experiment, 1)["train-residual"]
    assert _load_design_residual_model(paths.residual_model, paths.problem) is None
