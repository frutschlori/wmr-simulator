"""The fixed-baseline benchmark inside the active-learning loop.

Every iteration drives the *same* reference under its own controller, which is
the one thing in the loop that does not change with the iteration -- so a
difference between two iterations' benchmark runs is a difference in the
controller. What is tested here is that wiring: where the runs land, how they
are chained, and what happens to a run that does not come back.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wmr_simulator.active_learning.stages import (
    _benchmark_seed,
    _benchmark_variant_specs,
    _deployment_seed,
    _is_self_closing,
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
    """An initialized experiment whose benchmark reference is a small circle.

    Small and slow on purpose: the shipped default (circle_fast) is a
    trajectory the stock gains cannot track, which is the point of it as a
    benchmark but useless for testing the chaining rule.
    """
    trajectory = write_circle_jsn(tmp_path / "benchmark_circle.JSN")

    def build(**benchmark_overrides):
        return stage_init(
            tmp_path / f"exp{len(list(tmp_path.iterdir()))}",
            {
                "problem": PROBLEM,
                "use_residual_model": False,
                "benchmark": {"trajectory": str(trajectory), "num_runs": 2, **benchmark_overrides},
            },
        )

    return build


def test_runs_land_beside_the_identification_data_not_in_it(experiment):
    """The identify and residual stages only look one level deep, so benchmark
    runs in a data/ subdirectory can never reach the identification fit."""
    exp = experiment()
    paths = exp.paths(1)

    written = stage_run_benchmark(exp, 1)

    assert [path.name for path in written] == ["TR00", "TR01"]
    assert all(path.parent == paths.benchmark_static_data_dir for path in written)
    # data/ itself holds nothing but the benchmark subdirectory.
    assert [entry.name for entry in paths.data_dir.iterdir()] == ["benchmark_static"]


def test_ground_truth_stays_out_of_the_data_directory(experiment):
    """decode-logs decodes every non-csv file in data/, and the truth would
    leak the hidden plant into the pipeline besides -- so it goes to results/."""
    exp = experiment()
    paths = exp.paths(1)

    stage_run_benchmark(exp, 1)

    assert sorted(path.name for path in paths.benchmark_static_data_dir.iterdir()) == ["TR00", "TR01"]
    assert paths.benchmark_result.is_file()

    import yaml

    report = yaml.safe_load(paths.benchmark_result.read_text())
    static = report["variants"]["static"]
    assert len(static["runs"]) == 2
    assert {"tracking_rmse", "diverged", "distance_to_start"} <= set(static["runs"][0])


def test_a_run_that_came_back_is_where_the_next_one_starts(experiment):
    """The chained repeat: nobody touches the robot between runs."""
    from wmr_simulator.mujoco_sim.deploy import run_deployment

    exp = experiment(divergence_radius=100.0)  # nothing counts as diverged
    paths = exp.paths(1)
    trajectory = paths.benchmark_dir / "benchmark_circle.JSN"

    first, second = stage_run_benchmark(exp, 1)

    replay = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.benchmark_static_data_dir,
        seed=_benchmark_seed(exp, 1, 0), log_name="REPLAY",
    )
    repeat = run_deployment(
        paths.robotcfg_cfg, trajectory, paths.benchmark_static_data_dir,
        seed=_benchmark_seed(exp, 1, 1), start_pose=replay.final_pose, log_name="REPEAT",
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
        seed=_benchmark_seed(exp, 1, 1), log_name="REPLACED",
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


def test_a_self_closing_reference_is_driven_as_it_is(experiment):
    exp = experiment(num_runs=1)
    paths = exp.paths(1)

    stage_run_benchmark(exp, 1)

    assert [path.name for path in sorted(paths.benchmark_dir.iterdir())] == ["benchmark_circle.JSN"]


def test_self_closing_reads_the_whole_pose(tmp_path):
    circle = write_circle_jsn(tmp_path / "circle.JSN")
    arc = write_arc_jsn(tmp_path / "arc.JSN")
    assert _is_self_closing(circle)
    assert not _is_self_closing(arc)


def test_rerunning_the_stage_keeps_the_recorded_runs(experiment, capsys):
    """Benchmark runs are chained, so re-recording half of them would produce a
    set that never happened; the stage is a no-op once it has one."""
    exp = experiment()
    first = stage_run_benchmark(exp, 1)
    capsys.readouterr()

    again = stage_run_benchmark(exp, 1)

    assert again == first
    assert "skipping" in capsys.readouterr().out


def test_benchmark_seeds_never_collide_with_the_deployment_seeds(experiment):
    """The two run in the same iteration; sharing a seed would give them the
    same hand placement and the same sensor noise."""
    exp = experiment()
    benchmark = {_benchmark_seed(exp, iteration, index)
                 for iteration in (1, 2) for index in range(8)}
    deployment = {_deployment_seed(exp, iteration, index)
                  for iteration in (1, 2) for index in range(8)}
    assert len(benchmark) == 16
    assert not benchmark & deployment


def test_a_missing_benchmark_trajectory_names_the_setting(experiment):
    exp = experiment()
    exp.config["benchmark"]["trajectory"] = "no/such/reference.JSN"
    with pytest.raises(FileNotFoundError, match="benchmark.trajectory"):
        stage_run_benchmark(exp, 1)


# ---------------------------------------------------------------------------
# the static-gain baseline variant
# ---------------------------------------------------------------------------


def _add_static_gain_baseline(exp, iteration=1, static_gains=(1.0, 2.0, 3.0, 2.0, 0.0)):
    """Give an iteration the ROBOTCFG_static.CFG that finalize would write.

    Its presence is what makes an iteration a two-controller comparison, so it
    is what the benchmark keys the second run set off.
    """
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


def test_iteration_one_is_recorded_as_the_static_baseline(experiment):
    """Iteration 1 deploys the stock controller: its gains are the static ones
    and its parametrization is exactly the identity, so its runs *are* the
    static-gain baseline. Nothing tuned exists to compare them with yet -- the
    first tuned parametrization ships in iteration 2 -- so it records one set,
    and it belongs on the static side.
    """
    exp = experiment()
    paths = exp.paths(1)

    stage_run_benchmark(exp, 1)

    assert [entry.name for entry in paths.data_dir.iterdir()] == ["benchmark_static"]
    assert _benchmark_variant_specs(paths) == [
        ("static", paths.robotcfg_cfg, paths.benchmark_static_data_dir)
    ]


def test_iteration_ones_parametrization_is_the_identity(experiment):
    """What makes the run above a static-gain recording rather than an
    approximation of one: the exported network's output layer is zero, so every
    gain factor is exactly 1.0 and the deployed controller is its base gains."""
    import numpy as np

    from wmr_simulator.mujoco_sim.firmware import FirmwareConfig
    from wmr_simulator.pololu.gain_mlp_exporter import reference_forward

    exp = experiment()
    paths = exp.paths(1)
    network = FirmwareConfig.from_file(paths.robotcfg_cfg).gain_mlp
    assert network is not None

    rng = np.random.default_rng(0)
    for _ in range(8):
        factors = reference_forward(
            network,
            rng.normal(size=5),  # setpoint [x_d, y_d, theta_d, v_d, omega_d]
            rng.normal(size=3),  # estimated pose
            rng.normal(size=2),  # encoder body twist
        )
        assert np.array_equal(factors, np.ones(5, dtype=np.float32))


def test_both_controllers_are_driven_when_the_iteration_ships_two(experiment):
    """The comparison the plot draws: the deployed (parametrized) controller and
    the static-gain baseline over the same reference, in separate data dirs."""
    exp = experiment()
    paths = _add_static_gain_baseline(exp)

    stage_run_benchmark(exp, 1)

    assert sorted(path.name for path in paths.benchmark_data_dir.iterdir()) == ["TR00", "TR01"]
    assert sorted(path.name for path in paths.benchmark_static_data_dir.iterdir()) == ["TR00", "TR01"]
    # Both sets stay a level below data/, out of reach of the identification fit.
    assert sorted(entry.name for entry in paths.data_dir.iterdir()) == ["benchmark", "benchmark_static"]

    import yaml

    report = yaml.safe_load(paths.benchmark_result.read_text())
    assert set(report["variants"]) == {"parametrized", "static"}
    assert report["variants"]["static"]["robot_config"] == "ROBOTCFG_static.CFG"


def test_the_static_variant_is_driven_without_the_deployed_gain_network(experiment):
    """The firmware reads GAINMLP.JSN from beside the config, so a static run
    started in the iteration root would silently carry the network it is the
    baseline for."""
    from wmr_simulator.mujoco_sim.firmware import GAIN_MLP_FILENAME, FirmwareConfig
    from wmr_simulator.active_learning.stages import _staged_benchmark_config

    exp = experiment()
    paths = _add_static_gain_baseline(exp)
    assert paths.gainmlp_jsn.is_file()  # the deployed controller does have one

    specs = {variant: source for variant, source, _ in _benchmark_variant_specs(paths)}
    staged = _staged_benchmark_config(paths, "static", specs["static"])

    assert not (staged.parent / GAIN_MLP_FILENAME).exists()
    assert FirmwareConfig.from_file(staged).gain_mlp is None
    assert FirmwareConfig.from_file(paths.robotcfg_cfg).gain_mlp is not None


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
        for variant, payload in report["variants"].items()
    }
    assert seeds["static"] == seeds["parametrized"]
    assert (
        report["variants"]["static"]["runs"][0]["start_offset"]
        == report["variants"]["parametrized"]["runs"][0]["start_offset"]
    )


def test_a_missing_variant_is_added_without_rerecording_the_other(experiment, capsys):
    """Benchmark runs are chained, so re-recording a set that exists would
    produce one that never happened -- an interrupted stage has to resume on the
    variant it did not get to."""
    import shutil

    exp = experiment()
    paths = _add_static_gain_baseline(exp)
    stage_run_benchmark(exp, 1)
    static = {path.name: path.read_bytes() for path in paths.benchmark_static_data_dir.iterdir()}
    # What an interruption between the two variants leaves behind: the static
    # set is recorded first, so the deployed one is the half that is missing.
    shutil.rmtree(paths.benchmark_data_dir)
    capsys.readouterr()

    stage_run_benchmark(exp, 1)

    report = capsys.readouterr().out
    assert "Benchmark (static) already recorded" in report
    assert {path.name: path.read_bytes() for path in paths.benchmark_static_data_dir.iterdir()} == static
    assert sorted(path.name for path in paths.benchmark_data_dir.iterdir()) == ["TR00", "TR01"]


def test_an_iteration_is_not_benchmarked_until_both_variants_are_recorded(experiment):
    """`run` skips the benchmark once its status says done, so a status that
    ignores the missing static set would leave the comparison half-recorded."""
    from wmr_simulator.active_learning.stages import iteration_status

    exp = experiment()
    paths = _add_static_gain_baseline(exp)
    stage_run_benchmark(exp, 1)
    assert iteration_status(exp, 1)["benchmark"]

    import shutil

    shutil.rmtree(paths.benchmark_data_dir)
    assert not iteration_status(exp, 1)["benchmark"]

    stage_run_benchmark(exp, 1)
    assert iteration_status(exp, 1)["benchmark"]
