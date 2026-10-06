import json

import numpy as np
import yaml

from wmr_simulator.active_learning.experiment import Experiment, load_yaml
from wmr_simulator.active_learning.stages import iteration_status, stage_finalize, stage_init
from wmr_simulator.pololu.gain_mlp_exporter import reference_forward
from wmr_simulator.pololu.robot_config import load_robot_config_file

PROBLEM = "problems/pololu_gains.yaml"


def test_init_creates_iteration_scaffolding(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False, "use_residual_model": False})
    paths = experiment.paths(1)

    assert experiment.config_path.is_file()
    for directory in (
        paths.identification_trajectory_dir,
        paths.tuning_trajectories_dir,
        paths.data_dir,
        paths.results_dir,
        paths.visualize_dir,
    ):
        assert directory.is_dir()

    problem_cfg = load_yaml(PROBLEM)
    initial = experiment.config["initial_robot_params"]
    robot_config = load_yaml(paths.robot_config)
    # The loop starts from the configured initial model, not the problem's robot.
    assert robot_config["robot"] == {**{k: float(problem_cfg["robot"][k]) for k in initial}, **initial}
    assert robot_config["controller"]["gains"] == problem_cfg["controller"]["gains"]

    # Generated problem carries the current robot params and estimator geometry.
    iteration_problem = load_yaml(paths.problem)
    assert iteration_problem["robot"]["wheel_radius"] == initial["wheel_radius"]
    assert iteration_problem["estimator"]["wheel_radius"] == initial["wheel_radius"]

    # Firmware export reflects params and gain conversion.
    firmware = load_robot_config_file(paths.robotcfg_cfg)
    assert firmware["wheel_radius"] == initial["wheel_radius"]
    assert firmware["kx_traj"] == problem_cfg["controller"]["gains"][0]
    expected_kp_inner = problem_cfg["controller"]["gains"][3] / initial["max_wheel_speed"]
    assert abs(firmware["kp_inner"] - expected_kp_inner) < 1e-9

    # The firmware network is exported next to ROBOTCFG.CFG (identity network
    # before any tuning) exactly when the base problem enables a gain
    # parametrization -- init follows the yaml, it does not force one on.
    parametrization = problem_cfg["controller"].get("gain_parametrization") or {}
    if parametrization.get("enabled", False):
        network = json.loads(paths.gainmlp_jsn.read_text())
        assert network["kind"] == parametrization["kind"]
    else:
        assert not paths.gainmlp_jsn.exists()

    status = iteration_status(experiment, 1)
    assert not status["identify"]
    assert status["train-residual"]  # disabled flag counts as satisfied


def test_finalize_rolls_results_into_next_iteration(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False})
    paths = experiment.paths(1)

    identification = {
        "estimated_params": {
            "wheel_radius": 0.0171,
            "base_diameter": 0.0912,
            "max_wheel_speed": 240.0,
            "time_constant": 0.21,
        },
    }
    gains = {
        "gains": [9.1, 8.2, 6.3, 7.0, 11.0],
        "static_gains": [4.1, 3.2, 2.3, 5.0, 9.0],
        "schedule_enabled": True,
        "schedule": {"kind": "error_mlp", "hidden_sizes": [8], "seed": 3},
    }
    with paths.identification_result.open("w") as file:
        yaml.safe_dump(identification, file)
    with paths.gains_result.open("w") as file:
        yaml.safe_dump(gains, file)

    next_paths = stage_finalize(experiment, 1)
    assert experiment.latest_iteration() == 2

    next_robot_config = load_yaml(next_paths.robot_config)
    assert next_robot_config["robot"]["wheel_radius"] == 0.0171
    assert next_robot_config["controller"]["gains"] == gains["gains"]
    assert next_robot_config["controller"]["gain_parametrization"]["hidden_sizes"] == [8]
    assert next_robot_config["controller"]["gain_parametrization"]["seed"] == 3

    next_problem = load_yaml(next_paths.problem)
    assert next_problem["robot"]["base_diameter"] == 0.0912
    assert next_problem["estimator"]["base_diameter"] == 0.0912
    assert next_problem["controller"]["gains"] == gains["gains"]

    firmware = load_robot_config_file(next_paths.robotcfg_cfg)
    assert firmware["wheel_base"] == 0.0912
    assert abs(firmware["kp_inner"] - 7.0 / 240.0) < 1e-9

    experiment_reloaded = Experiment.load(experiment.root)
    assert experiment_reloaded.resolve_iteration(None) == 2


def test_finalize_writes_static_gain_baseline(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False})
    paths = experiment.paths(1)

    identification = {"estimated_params": {"max_wheel_speed": 240.0}}
    gains = {
        "gains": [9.1, 8.2, 6.3, 7.0, 11.0],
        "static_gains": [4.1, 3.2, 2.3, 5.0, 9.0],
        "schedule_enabled": True,
        "schedule": {"kind": "error_mlp", "hidden_sizes": [8], "seed": 3},
    }
    with paths.identification_result.open("w") as file:
        yaml.safe_dump(identification, file)
    with paths.gains_result.open("w") as file:
        yaml.safe_dump(gains, file)

    next_paths = stage_finalize(experiment, 1)

    # Same identified robot params, static gains, and no gain parametrization.
    static_config = load_yaml(next_paths.robot_config_static)
    tuned_config = load_yaml(next_paths.robot_config)
    assert static_config["robot"] == tuned_config["robot"]
    assert static_config["controller"]["gains"] == gains["static_gains"]
    assert "gain_parametrization" not in static_config["controller"]
    assert tuned_config["controller"]["gains"] == gains["gains"]

    static_firmware = load_robot_config_file(next_paths.robotcfg_static_cfg)
    assert static_firmware["kx_traj"] == 4.1
    assert abs(static_firmware["kp_inner"] - 5.0 / 240.0) < 1e-9
    assert load_robot_config_file(next_paths.robotcfg_cfg)["kx_traj"] == 9.1


def test_finalize_without_static_gains_writes_no_baseline(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False})
    paths = experiment.paths(1)

    with paths.identification_result.open("w") as file:
        yaml.safe_dump({"estimated_params": {}}, file)
    with paths.gains_result.open("w") as file:
        yaml.safe_dump({"gains": [9.1, 8.2, 6.3, 7.0, 11.0], "schedule_enabled": False}, file)

    next_paths = stage_finalize(experiment, 1)
    assert not next_paths.robot_config_static.exists()
    assert not next_paths.robotcfg_static_cfg.exists()


def test_finalize_exports_trained_gain_mlp(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False})
    paths = experiment.paths(1)

    tuned = load_yaml("models/tuned_gains.yaml")
    identification = {"estimated_params": load_yaml(paths.robot_config)["robot"]}
    gains = {
        "gains": [float(gain) for gain in tuned["gains"]],
        "schedule_enabled": True,
        "schedule": tuned["schedule"],
    }
    with paths.identification_result.open("w") as file:
        yaml.safe_dump(identification, file)
    with paths.gains_result.open("w") as file:
        yaml.safe_dump(gains, file)

    next_paths = stage_finalize(experiment, 1)

    assert next_paths.gainmlp_jsn.is_file()
    network = json.loads(next_paths.gainmlp_jsn.read_text())
    assert network["kind"] == "error_mlp"
    # A trained schedule must produce non-identity factors on the robot.
    factors = reference_forward(
        network, ref=[0.3, -0.2, 0.5, 0.4, 1.0], pose=[0.1, 0.0, 0.2], twist=[0.3, 0.8]
    )
    assert np.any(np.abs(factors - 1.0) > 1e-3)


def test_residual_flag_off_disables_training_and_every_residual_rollout(tmp_path):
    import pytest

    from wmr_simulator.active_learning.stages import _residual_in_design, stage_train_residual

    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False, "use_residual_model": False})
    assert load_yaml(experiment.config_path)["use_residual_model"] is False

    # The stage flags default on for the tuning design and gain tuning; the
    # top-level switch overrides them, so a stray checkpoint is never loaded.
    for stage in ("identification_trajectory", "tuning_trajectories", "gain_tuning"):
        assert not _residual_in_design(experiment, experiment.config[stage])
    with pytest.raises(ValueError, match="use_residual_model is disabled"):
        stage_train_residual(experiment, 1)


def test_an_enabled_residual_without_a_checkpoint_refuses_to_design_nominal(tmp_path):
    import pytest

    from wmr_simulator.active_learning.stages import _load_design_residual_model, _residual_in_design

    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False})
    assert _residual_in_design(experiment, experiment.config["tuning_trajectories"])
    paths = experiment.paths(1)
    with pytest.raises(FileNotFoundError, match="train-residual"):
        _load_design_residual_model(paths.residual_model, paths.problem)


def test_prior_tuning_starts_at_iteration_zero_on_the_prior_model(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM})
    assert experiment.iteration_indices() == [0]
    assert experiment.first_iteration == 0
    paths = experiment.paths(0)
    # The tuning half reads the prior model, and there is no residual to load.
    assert load_yaml(paths.problem_identified) == load_yaml(paths.problem)
    assert paths.residual_skipped.is_file()
    status = iteration_status(experiment, 0)
    assert status["train-residual"] and not status["tune-gains"]


def test_finalizing_iteration_zero_hands_its_gains_to_iteration_one(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "use_gain_parametrization": False})
    paths = experiment.paths(0)
    tuned = [2.1, 5.9, 7.1, 2.7, 0.0, 0.0]
    with paths.gains_result.open("w") as file:
        yaml.safe_dump({"gains": tuned, "static_gains": None, "schedule_enabled": False, "schedule": None}, file)

    next_paths = stage_finalize(experiment, 0)
    assert next_paths == experiment.paths(1)
    robot_config = load_yaml(next_paths.robot_config)
    # Nothing was identified: iteration 1 keeps the prior model.
    assert robot_config["robot"] == load_yaml(paths.robot_config)["robot"]
    assert robot_config["controller"]["gains"] == tuned
    assert load_robot_config_file(next_paths.robotcfg_cfg)["kx_traj"] == 2.1
    assert Experiment.load(experiment.root).first_iteration == 0


def test_finalize_on_the_robot_creates_the_benchmark_directories(tmp_path):
    """Hand-recorded benchmark runs go where the MuJoCo benchmark stage would
    write them: one empty directory per shape, under benchmark_static for a
    static experiment, and none of it counts as recorded yet."""
    benchmark_set = tmp_path / "set"
    benchmark_set.mkdir()
    for shape in ("circle_fast", "turbo_drift"):
        (benchmark_set / f"{shape}.JSN").write_text("{}")
    (benchmark_set / "turbo_drift_bridge.JSN").write_text("{}")
    experiment = stage_init(tmp_path / "exp", {
        "problem": PROBLEM, "use_gain_parametrization": False, "benchmark": {"trajectory": str(benchmark_set)},
    })
    with experiment.paths(0).gains_result.open("w") as file:
        yaml.safe_dump({"gains": [2.1, 5.9, 7.1, 2.7, 0.0, 0.0], "static_gains": None,
                        "schedule_enabled": False, "schedule": None}, file)

    next_paths = stage_finalize(experiment, 0)
    benchmark_dir = next_paths.data_dir / "benchmark_static"
    assert sorted(path.name for path in benchmark_dir.iterdir()) == ["circle_fast", "turbo_drift"]
    assert not iteration_status(experiment, 1)["benchmark"]

    # In MuJoCo the benchmark stage records them; finalize leaves data/ alone.
    mujoco = stage_init(tmp_path / "mujoco", {
        "problem": PROBLEM, "use_gain_parametrization": False, "benchmark": {"trajectory": str(benchmark_set)},
        "mujoco_deployment": {"enabled": True},
    })
    with mujoco.paths(0).gains_result.open("w") as file:
        yaml.safe_dump({"gains": [2.1, 5.9, 7.1, 2.7, 0.0, 0.0], "static_gains": None,
                        "schedule_enabled": False, "schedule": None}, file)
    assert not (stage_finalize(mujoco, 0).data_dir / "benchmark_static").exists()


def test_an_experiment_from_before_prior_tuning_still_starts_at_one(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False})
    config = load_yaml(experiment.config_path)
    del config["prior_tuning"]
    with experiment.config_path.open("w") as file:
        yaml.safe_dump(config, file)
    assert Experiment.load(experiment.root).first_iteration == 1


def test_without_an_initial_model_the_loop_starts_from_the_problem_robot(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "prior_tuning": False, "initial_robot_params": None})
    problem_robot = load_yaml(PROBLEM)["robot"]
    robot = load_yaml(experiment.paths(1).robot_config)["robot"]
    assert robot == {key: float(problem_robot[key]) for key in robot}
