"""The benchmark stage: chaining, bridging, scoring and paired variants."""

from __future__ import annotations

import json

import numpy as np
import pytest

from wmr_simulator.active_learning.stages import (
    _benchmark_seed,
    _write_static_gain_config,
    stage_init,
    stage_run_benchmark,
)

PROBLEM = "problems/pololu_gains.yaml"


def write_circle_jsn(path, radius=0.5, speed=0.5, dt=0.05):
    """A full circle as a Pololu reference: it ends exactly where it started,
    so repeat runs can be chained without a bridge path."""
    yaw_rate = speed / radius
    num = int(round(2.0 * np.pi / yaw_rate / dt)) + 1
    yaw = yaw_rate * np.arange(num) * dt
    states = np.column_stack(
        [radius * np.cos(yaw), radius * np.sin(yaw), yaw + 0.5 * np.pi]
    )
    return _write_jsn(path, states, speed, yaw_rate, dt)


def write_arc_jsn(path, duration=2.0, speed=0.5, yaw_rate=0.4, dt=0.05):
    """A gentle arc: it does *not* return to its start, so the benchmark has to
    bridge it before it can be repeated."""
    num = int(round(duration / dt)) + 1
    yaw = yaw_rate * np.arange(num) * dt
    radius = speed / yaw_rate
    states = np.column_stack([radius * np.sin(yaw), radius * (1.0 - np.cos(yaw)), yaw])
    return _write_jsn(path, states, speed, yaw_rate, dt)


def _write_jsn(path, states, speed, yaw_rate, dt):
    num = len(states)
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
def experiment(tmp_path):
    """An initialized experiment whose benchmark reference is a small, slow circle."""
    trajectory = write_circle_jsn(tmp_path / "benchmark_circle.JSN")

    def build(**benchmark_overrides):
        return stage_init(
            tmp_path / f"exp{len(list(tmp_path.iterdir()))}",
            {
                "problem": PROBLEM,
                "prior_tuning": False,
                "use_residual_model": False,
                "benchmark": {"trajectory": str(trajectory), "num_runs": 2, **benchmark_overrides},
            },
        )

    return build


def _benchmark_placement(exp) -> dict:
    """The hand-placement spread the benchmark stage draws under."""
    config = exp.config["benchmark"]
    return {
        "start_offset_radius": float(config["start_offset_radius"]),
        "start_offset_angle": float(config["start_offset_angle"]),
    }


def test_a_run_that_came_back_is_where_the_next_one_starts(experiment):
    """The chained repeat: nobody touches the robot between runs."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment

    exp = experiment(divergence_radius=100.0)  # nothing counts as diverged
    paths = exp.paths(1)
    trajectory = paths.benchmark_dir / "benchmark_circle.JSN"

    first, second = stage_run_benchmark(exp, 1)

    # The hand placement is drawn from the experiment's offsets.
    offsets = _benchmark_placement(exp)
    replay = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.benchmark_static_data_dir,
        seed=_benchmark_seed(exp, 1, 0), log_name="REPLAY", **offsets,
    )
    repeat = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.benchmark_static_data_dir,
        seed=_benchmark_seed(exp, 1, 1), start_pose=replay.final_pose, log_name="REPEAT",
        **offsets,
    )
    assert replay.log_path.read_bytes() == first.read_bytes()
    assert repeat.log_path.read_bytes() == second.read_bytes()


def test_a_run_that_diverged_is_placed_by_hand_again(experiment, capsys):
    """A robot that did not come back is somewhere else entirely; carrying the
    next run's start pose over from it would benchmark that, not the controller."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment

    exp = experiment(divergence_radius=0.0)  # everything counts as diverged
    paths = exp.paths(1)
    trajectory = paths.benchmark_dir / "benchmark_circle.JSN"

    _, second = stage_run_benchmark(exp, 1)

    report = capsys.readouterr().out
    assert report.count("placed by hand") == 2
    assert "diverged" in report

    hand_placed = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.benchmark_static_data_dir,
        seed=_benchmark_seed(exp, 1, 1), log_name="REPLACED", **_benchmark_placement(exp),
    )
    assert hand_placed.log_path.read_bytes() == second.read_bytes()


def test_a_reference_that_does_not_close_is_bridged(experiment, tmp_path):
    """Chaining onto the previous run's end pose only works if that end pose is
    the start pose, which is what the bridge path arranges."""
    arc = write_arc_jsn(tmp_path / "benchmark_arc.JSN")
    exp = experiment()
    exp.config["benchmark"]["trajectory"] = str(arc)
    exp.config["benchmark"]["num_runs"] = 1
    paths = exp.paths(1)

    stage_run_benchmark(exp, 1)

    assert (paths.benchmark_dir / "benchmark_arc_bridge.JSN").is_file()


def test_a_bridged_run_is_scored_over_the_reference_alone(experiment, tmp_path):
    """The wait at the goal and the slow return path are driven but not scored,
    so they cannot dilute the position RMSE of an open reference."""
    import yaml

    arc = write_arc_jsn(tmp_path / "benchmark_arc.JSN", duration=2.0)
    exp = experiment()
    exp.config["benchmark"]["trajectory"] = str(arc)
    exp.config["benchmark"]["num_runs"] = 1
    paths = exp.paths(1)

    stage_run_benchmark(exp, 1)

    report = yaml.safe_load(paths.benchmark_result.read_text())
    run = report["trajectories"]["benchmark_arc"]["variants"]["static"]["runs"][0]
    assert run["scored_duration"] == pytest.approx(41 * 0.05)
    assert run["tracking_rmse_full"] > 0.0
    assert run["tracking_rmse"] != run["tracking_rmse_full"]


# ---------------------------------------------------------------------------
# the static-gain baseline variant
# ---------------------------------------------------------------------------


def _add_static_gain_baseline(exp, iteration=1, static_gains=(1.0, 2.0, 3.0, 2.0, 0.0)):
    """Give an iteration the ROBOTCFG_static.CFG that finalize would write."""
    from wmr_simulator.active_learning.experiment import load_yaml
    from wmr_simulator.types import PhysicalParams

    paths = exp.paths(iteration)
    robot_config = load_yaml(paths.robot_config)
    robot = robot_config["robot"]
    _write_static_gain_config(
        exp,
        paths,
        robot_config,
        PhysicalParams(
            wheel_radius=robot["wheel_radius"],
            base_diameter=robot["base_diameter"],
            max_wheel_speed=robot["max_wheel_speed"],
        ),
        list(static_gains),
    )
    return paths


def test_the_two_variants_share_their_seeds(experiment):
    """A paired comparison: run i of each controller gets the same hand
    placement and the same sensor noise, so the controller is the only
    difference between the two sets."""
    exp = experiment()
    paths = _add_static_gain_baseline(exp)

    stage_run_benchmark(exp, 1)

    import yaml

    report = yaml.safe_load(paths.benchmark_result.read_text())
    seeds = {
        variant: [run["seed"] for run in payload["runs"]]
        for variant, payload in report["trajectories"]["benchmark_circle"]["variants"].items()
    }
    assert seeds["static"] == seeds["parametrized"]
    assert (
        report["trajectories"]["benchmark_circle"]["variants"]["static"]["runs"][0]["start_offset"]
        == report["trajectories"]["benchmark_circle"]["variants"]["parametrized"]["runs"][0]["start_offset"]
    )


def test_a_directory_of_references_is_driven_as_a_cross_validation_set(experiment, tmp_path):
    """``benchmark.trajectory`` pointing at a directory drives every reference in
    it, each under both controller variants, into its own data subdirectory."""
    reference_dir = tmp_path / "benchmark_set"
    reference_dir.mkdir()
    write_circle_jsn(reference_dir / "small_circle.JSN")
    write_circle_jsn(reference_dir / "other_circle.JSN")

    exp = experiment(trajectory=str(reference_dir))
    paths = exp.paths(1)
    stage_run_benchmark(exp, 1)

    for shape in ("small_circle", "other_circle"):
        runs = sorted(path.name for path in (paths.benchmark_static_data_dir / shape).iterdir())
        assert runs == ["TR00", "TR01"], f"{shape} recorded {runs}"

    import yaml

    report = yaml.safe_load(paths.benchmark_result.read_text())
    assert set(report["trajectories"]) == {"small_circle", "other_circle"}
    for shape, entry in report["trajectories"].items():
        assert len(entry["variants"]["static"]["runs"]) == 2, shape
        assert entry["variants"]["static"]["median_tracking_rmse"] is not None
