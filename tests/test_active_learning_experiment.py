import json

import numpy as np
import yaml

from wmr_simulator.active_learning.experiment import Experiment, load_yaml
from wmr_simulator.active_learning.stages import iteration_status, stage_finalize, stage_init
from wmr_simulator.pololu.gain_mlp_exporter import reference_forward
from wmr_simulator.pololu.robot_config import load_robot_config_file

PROBLEM = "problems/pololu_gains.yaml"


def test_init_creates_iteration_scaffolding(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM, "use_residual_model": False})
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
    robot_config = load_yaml(paths.robot_config)
    assert robot_config["robot"]["wheel_radius"] == problem_cfg["robot"]["wheel_radius"]
    assert robot_config["controller"]["gains"] == problem_cfg["controller"]["gains"]

    # Generated problem carries the current robot params and estimator geometry.
    iteration_problem = load_yaml(paths.problem)
    assert iteration_problem["robot"]["wheel_radius"] == problem_cfg["robot"]["wheel_radius"]
    assert iteration_problem["estimator"]["wheel_radius"] == problem_cfg["robot"]["wheel_radius"]

    # Firmware export reflects params and gain conversion.
    firmware = load_robot_config_file(paths.robotcfg_cfg)
    assert firmware["wheel_radius"] == problem_cfg["robot"]["wheel_radius"]
    assert firmware["kx_traj"] == problem_cfg["controller"]["gains"][0]
    expected_kp_inner = problem_cfg["controller"]["gains"][3] / problem_cfg["robot"]["max_wheel_speed"]
    assert abs(firmware["kp_inner"] - expected_kp_inner) < 1e-9

    # The base problem enables the error-MLP schedule, so its firmware network
    # is exported next to ROBOTCFG.CFG (identity network before any tuning).
    assert paths.gainmlp_jsn.is_file()
    network = json.loads(paths.gainmlp_jsn.read_text())
    assert network["kind"] == "error_mlp"

    status = iteration_status(experiment, 1)
    assert not status["identify"]
    assert status["train-residual"]  # disabled flag counts as satisfied


def test_finalize_rolls_results_into_next_iteration(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM})
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
        "schedule_enabled": True,
        "schedule": {"scheduled_indices": [0, 1, 2], "rho": [0.5, 0.5, 0.5], "W": [[0.1, 0.0]] * 3},
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
    assert next_robot_config["controller"]["gain_parametrization"]["W"] == [[0.1, 0.0]] * 3

    next_problem = load_yaml(next_paths.problem)
    assert next_problem["robot"]["base_diameter"] == 0.0912
    assert next_problem["estimator"]["base_diameter"] == 0.0912
    assert next_problem["controller"]["gains"] == gains["gains"]

    firmware = load_robot_config_file(next_paths.robotcfg_cfg)
    assert firmware["wheel_base"] == 0.0912
    assert abs(firmware["kp_inner"] - 7.0 / 240.0) < 1e-9

    experiment_reloaded = Experiment.load(experiment.root)
    assert experiment_reloaded.resolve_iteration(None) == 2


def test_finalize_exports_trained_gain_mlp(tmp_path):
    experiment = stage_init(tmp_path / "exp", {"problem": PROBLEM})
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
