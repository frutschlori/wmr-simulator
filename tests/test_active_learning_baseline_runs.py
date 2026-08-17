"""Collecting an experiment's held-out baseline runs across its iterations.

A baseline run is a recording of a reference the iteration neither identified on
nor tuned against, so it is the one thing that stays fixed while the controller
moves. What is tested here is the bookkeeping that makes the comparison figure
honest: which directories count as baseline runs, which controller each one
belongs to, and that identification data never gets in.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmr_simulator.active_learning.baseline_runs import (
    MAX_RUNS_PER_ITERATION,
    PARAMETRIZED_VARIANT,
    STATIC_VARIANT,
    baseline_run_directories,
    collect_baseline_runs,
    load_run_logs,
    run_tracking_rmse,
    variant_panels,
)
from wmr_simulator.active_learning.experiment import Experiment
from wmr_simulator.pololu.log_loader import POLOLU_TRAJ_CONTROL_COLUMNS

CIRCLE_RADIUS = 0.5
CIRCLE_RATE = 1.0


def write_run_csv(path, *, radial_offset=0.0, duration=6.5, dt=0.02):
    """A traj-control recording of a circle, tracked with a radial offset.

    The reference is the exact circle and the mocap track the same circle at
    ``radius + radial_offset``, so the run's position RMSE against its reference
    is ``|radial_offset|`` by construction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in np.arange(0.0, duration, dt):
        yaw = CIRCLE_RATE * t
        row = {name: "" for name in POLOLU_TRAJ_CONTROL_COLUMNS}
        row["ts"] = f"{1000.0 * t:.0f}"
        measured = CIRCLE_RADIUS + radial_offset
        row.update(
            x_raw=f"{measured * np.cos(yaw):.6f}",
            y_raw=f"{measured * np.sin(yaw):.6f}",
            yaw_raw=f"{np.arctan2(np.sin(yaw + 0.5 * np.pi), np.cos(yaw + 0.5 * np.pi)):.6f}",
        )
        row.update(
            x_des=f"{CIRCLE_RADIUS * np.cos(yaw):.6f}",
            y_des=f"{CIRCLE_RADIUS * np.sin(yaw):.6f}",
            yaw_des=f"{np.arctan2(np.sin(yaw + 0.5 * np.pi), np.cos(yaw + 0.5 * np.pi)):.6f}",
            v_ff=f"{CIRCLE_RADIUS * CIRCLE_RATE:.6f}",
            w_ff=f"{CIRCLE_RATE:.6f}",
        )
        row.update(omega_l_meas="10.0", omega_r_meas="12.0", omega_l_cmd="10.0", omega_r_cmd="12.0")
        row.update(v_actual="0.5", w_actual="1.0", duty_l="0.3", duty_r="0.35")
        rows.append(row)

    lines = [",".join(POLOLU_TRAJ_CONTROL_COLUMNS)]
    lines += [",".join(row[name] for name in POLOLU_TRAJ_CONTROL_COLUMNS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def experiment(tmp_path):
    """An experiment with empty iteration folders, ready to have runs dropped in.

    Only the directory layout matters here, so the iterations are created
    directly rather than by running the loop.
    """

    def build(num_iterations=2):
        exp = Experiment.create(tmp_path / "exp", {"num_iterations": num_iterations})
        for index in range(1, num_iterations + 1):
            exp.paths(index).create_directories()
        return exp

    return build


def _write_runs(directory, count, radial_offset=0.0):
    for index in range(count):
        write_run_csv(directory / f"TR{index:02d}.csv", radial_offset=radial_offset)
    return directory


def test_the_benchmark_pair_is_one_shape_recorded_under_two_controllers(experiment):
    """Both benchmark directories hold runs of benchmark.trajectory, so they are
    one shape compared across controllers -- not two unrelated shapes."""
    exp = experiment(1)
    paths = exp.paths(1)
    _write_runs(paths.benchmark_data_dir, 2)
    _write_runs(paths.benchmark_static_data_dir, 2)

    directories = baseline_run_directories(paths)

    assert list(directories) == ["benchmark"]
    assert directories["benchmark"][PARAMETRIZED_VARIANT] == paths.benchmark_data_dir
    assert directories["benchmark"][STATIC_VARIANT] == paths.benchmark_static_data_dir


def test_identification_logs_are_not_baseline_runs(experiment):
    """Runs directly in data/ are what the iteration was fitted on, so they are
    not held out and cannot serve as the fixed reference point."""
    exp = experiment(1)
    paths = exp.paths(1)
    _write_runs(paths.data_dir, 3)

    assert baseline_run_directories(paths) == {}


def test_hand_recorded_robot_runs_split_by_controller(experiment):
    """On the robot a shape sits directly in data/ when it was driven on static
    gains and under the gain-MLP directory when the network was on the card;
    the recorded experiments carry both spellings of that directory."""
    exp = experiment(2)
    first, second = exp.paths(1), exp.paths(2)
    _write_runs(first.data_dir / "circle", 2)
    _write_runs(first.data_dir / "with_gain_MLP" / "circle", 2)
    _write_runs(second.data_dir / "lemniscate", 2)
    _write_runs(second.data_dir / "with gain MLP" / "lemniscate", 2)

    assert baseline_run_directories(first) == {
        "circle": {
            STATIC_VARIANT: first.data_dir / "circle",
            PARAMETRIZED_VARIANT: first.data_dir / "with_gain_MLP" / "circle",
        }
    }
    assert baseline_run_directories(second) == {
        "lemniscate": {
            STATIC_VARIANT: second.data_dir / "lemniscate",
            PARAMETRIZED_VARIANT: second.data_dir / "with gain MLP" / "lemniscate",
        }
    }


def test_the_gain_mlp_directory_is_not_itself_a_shape(experiment):
    """It is a controller, not a reference: the shapes are its subdirectories."""
    exp = experiment(1)
    paths = exp.paths(1)
    _write_runs(paths.data_dir / "with gain MLP" / "circle", 2)

    assert list(baseline_run_directories(paths)) == ["circle"]


def test_only_five_runs_of_a_group_are_kept(experiment):
    """Repeats of one reference under one controller: past a handful the extra
    lines land on top of each other and only thicken the same band."""
    exp = experiment(1)
    paths = exp.paths(1)
    _write_runs(paths.benchmark_data_dir, MAX_RUNS_PER_ITERATION + 3)

    logs = load_run_logs(paths.benchmark_data_dir, max_runs=MAX_RUNS_PER_ITERATION)

    assert [name for name, _ in logs] == [f"TR{index:02d}" for index in range(MAX_RUNS_PER_ITERATION)]


def test_a_decoded_log_is_read_once_not_twice(experiment):
    """decode-logs leaves TRxx.csv next to the binary TRxx, so both spellings of
    one run are in the glob; reading both would double every line in the plot."""
    exp = experiment(1)
    paths = exp.paths(1)
    write_run_csv(paths.benchmark_data_dir / "TR00.csv")
    (paths.benchmark_data_dir / "TR00").write_bytes(b"binary source of TR00.csv")

    assert [name for name, _ in load_run_logs(paths.benchmark_data_dir)] == ["TR00"]


def test_the_tracking_error_is_measured_against_the_reference_the_run_tracked(experiment):
    """The number the legend ranks iterations by: a run that held a constant
    0.04 m offset from its reference scores exactly that."""
    exp = experiment(1)
    paths = exp.paths(1)
    write_run_csv(paths.benchmark_data_dir / "TR00.csv", radial_offset=0.04)

    (_, log), = load_run_logs(paths.benchmark_data_dir)

    assert run_tracking_rmse(log) == pytest.approx(0.04, abs=2e-3)


def test_the_loops_own_shape_is_collected_as_the_pipeline_records_it(experiment):
    """The layout a MuJoCo experiment produces: iteration 1 records only the
    stock static controller, and every later iteration records both its deployed
    (parametrized) controller and its static-gain baseline."""
    exp = experiment(3)
    _write_runs(exp.paths(1).benchmark_static_data_dir, 2, radial_offset=0.05)
    _write_runs(exp.paths(2).benchmark_static_data_dir, 2, radial_offset=0.03)
    _write_runs(exp.paths(2).benchmark_data_dir, 2, radial_offset=0.02)
    _write_runs(exp.paths(3).benchmark_static_data_dir, 2, radial_offset=0.03)
    _write_runs(exp.paths(3).benchmark_data_dir, 2, radial_offset=0.01)

    collected = collect_baseline_runs(exp)

    assert list(collected) == ["benchmark"]
    records = collected["benchmark"]
    assert [record.index for record in records] == [1, 2, 3]
    assert list(records[0].runs) == [STATIC_VARIANT]
    assert set(records[1].runs) == {PARAMETRIZED_VARIANT, STATIC_VARIANT}
    assert variant_panels(records) == (
        (STATIC_VARIANT, "Static gains"),
        (PARAMETRIZED_VARIANT, "Gain parametrization"),
    )
    # Both variants recorded the same reference, so the figure has one to draw.
    assert records[1].reference.shape[1] == 3


def test_an_experiment_that_started_benchmarking_late_plots_what_it_has(experiment):
    """A missing iteration is a missing colour, not a missing figure."""
    exp = experiment(3)
    _write_runs(exp.paths(3).benchmark_static_data_dir, 2)

    records = collect_baseline_runs(exp)["benchmark"]

    assert [record.index for record in records] == [3]


def test_a_single_controller_gets_a_single_panel(experiment):
    """Iteration 1 deploys the stock static controller and has nothing tuned to
    compare it against, so a second panel would be an empty one."""
    exp = experiment(1)
    _write_runs(exp.paths(1).benchmark_static_data_dir, 2)

    records = collect_baseline_runs(exp)["benchmark"]

    assert variant_panels(records) == ((STATIC_VARIANT, "Static gains"),)


def test_the_static_panel_comes_first(experiment):
    """The comparison reads left to right as "before the parametrization, after
    it", so the panel order is fixed rather than dictionary order."""
    exp = experiment(1)
    paths = exp.paths(1)
    _write_runs(paths.benchmark_data_dir, 2)
    _write_runs(paths.benchmark_static_data_dir, 2)

    records = collect_baseline_runs(exp)["benchmark"]

    assert [variant for variant, _ in variant_panels(records)] == [STATIC_VARIANT, PARAMETRIZED_VARIANT]


def test_the_figure_is_written_for_every_shape(experiment):
    """The stage hook: one figure per baseline shape in the iteration's own
    visualize dir, so each iteration keeps the comparison as it stood."""
    from wmr_simulator.active_learning.stages import _plot_baseline_runs

    exp = experiment(2)
    _write_runs(exp.paths(1).data_dir / "circle", 2)
    _write_runs(exp.paths(2).data_dir / "circle", 2, radial_offset=0.02)
    _write_runs(exp.paths(2).benchmark_data_dir, 2, radial_offset=0.01)
    paths = exp.paths(2)

    written = _plot_baseline_runs(exp, paths)

    assert sorted(path.rsplit("/", 1)[-1] for path in written) == [
        "baseline_runs_benchmark.pdf",
        "baseline_runs_circle.pdf",
    ]
    assert all((paths.visualize_dir / name.rsplit("/", 1)[-1]).is_file() for name in written)


def test_an_experiment_without_baseline_runs_plots_nothing(experiment):
    """Most iterations of a real-robot experiment have no baseline recordings at
    all; that is not a failure worth a traceback in the tune-gains stage."""
    from wmr_simulator.active_learning.stages import _plot_baseline_runs

    exp = experiment(1)

    assert _plot_baseline_runs(exp, exp.paths(1)) == []
