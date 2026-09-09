"""Summary plots of the recordings sitting in an iteration's data/ subdirectories.

Those subdirectories are the benchmark stage's two controller variants and the
hand-recorded baseline shapes. Nothing in the loop plotted them, and they arrive
after the iteration that recorded them has been left behind -- a person copies
them off the SD card later -- so the whole experiment is rescanned whenever the
loop proceeds. What is tested here is that bookkeeping: where a plot lands, that
the identification logs are left to the decode-logs stage, and that a rescan
does not redraw what is already there.
"""

from __future__ import annotations

import pytest

from wmr_simulator.active_learning import stages
from wmr_simulator.active_learning.experiment import Experiment, save_yaml

from test_active_learning_baseline_runs import write_run_csv


@pytest.fixture
def experiment(tmp_path):
    """An experiment with empty iteration folders, ready to have runs dropped in.

    Only the directory layout and each iteration's problem.yaml matter here, so
    the iterations are created directly rather than by running the loop.
    """

    def build(num_iterations=2):
        exp = Experiment.create(tmp_path / "exp", {"num_iterations": num_iterations})
        for index in range(1, num_iterations + 1):
            paths = exp.paths(index)
            paths.create_directories()
            save_yaml(paths.problem, {"controller": {"gains": [4.5, 6.0, 12.0, 2.5, 5.0]}})
        return exp

    return build


def test_every_data_subdirectory_is_mirrored_into_visualize(experiment):
    """A recording is plotted under visualize/logs/ + the directory name it
    had in data/, beside the identification logs' own plots."""
    exp = experiment()
    paths = exp.paths(1)
    write_run_csv(paths.data_dir / "benchmark_static" / "TR00.csv")
    write_run_csv(paths.data_dir / "with gain MLP" / "circle" / "TR00.csv")

    stages.plot_run_directory_logs(exp)

    assert (paths.visualize_dir / "logs" / "benchmark_static" / "TR00.pdf").is_file()
    assert (paths.visualize_dir / "logs" / "with gain MLP" / "circle" / "TR00.pdf").is_file()


def test_identification_logs_are_left_to_the_decode_stage(experiment):
    """Recordings directly in data/ are identification data, plotted by
    decode-logs into visualize/logs/; this scan must not duplicate them."""
    exp = experiment()
    paths = exp.paths(1)
    write_run_csv(paths.data_dir / "TR00.csv")

    assert stages.plot_run_directory_logs(exp) == []
    assert not (paths.visualize_dir / "logs" / "TR00.pdf").exists()


def test_earlier_iterations_are_rescanned(experiment):
    """Benchmark logs of an iteration the loop already finalized still get their
    plots, which is the whole reason the scan is experiment-wide."""
    exp = experiment(num_iterations=3)
    write_run_csv(exp.paths(1).data_dir / "benchmark" / "TR00.csv")
    write_run_csv(exp.paths(3).data_dir / "benchmark" / "TR00.csv")

    written = stages.plot_run_directory_logs(exp)

    assert len(written) == 2
    assert (exp.paths(1).visualize_dir / "logs" / "benchmark" / "TR00.pdf").is_file()
    assert (exp.paths(3).visualize_dir / "logs" / "benchmark" / "TR00.pdf").is_file()


def test_existing_plots_are_kept(experiment):
    """A rescan of a finished experiment costs nothing and overwrites nothing."""
    exp = experiment()
    write_run_csv(exp.paths(1).data_dir / "benchmark" / "TR00.csv")
    stages.plot_run_directory_logs(exp)
    plot_path = exp.paths(1).visualize_dir / "logs" / "benchmark" / "TR00.pdf"
    stamp = plot_path.stat().st_mtime_ns

    assert stages.plot_run_directory_logs(exp) == []
    assert plot_path.stat().st_mtime_ns == stamp
