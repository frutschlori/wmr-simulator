"""Collecting the identified robot parameters across an experiment's iterations.

The figure this feeds is the loop's answer to "are the identified parameters
settling?", so what matters here is the bookkeeping behind it: which iterations
contribute a point at all, which parameters are carried, and that logs the
outlier screen dropped are carried as such rather than silently mixed into the
spread.
"""

from __future__ import annotations

import pytest

from wmr_simulator.active_learning.experiment import Experiment, save_yaml
from wmr_simulator.active_learning.identified_parameters import (
    PARAMETER_NAMES,
    collect_identified_parameters,
)
from wmr_simulator.visualization.identified_parameters import plot_identified_parameters

BASE_PARAMS = {
    "wheel_radius": 0.0161,
    "base_diameter": 0.0843,
    "max_wheel_speed": 228.0,
    "time_constant": 0.17,
    "a_slip_max": 3.0,
}


@pytest.fixture
def experiment(tmp_path):
    """An experiment with empty iteration folders, ready to have identification
    results dropped in; only the directory layout matters here."""

    def build(num_iterations=2):
        exp = Experiment.create(tmp_path / "exp", {"num_iterations": num_iterations})
        for index in range(1, num_iterations + 1):
            exp.paths(index).create_directories()
        return exp

    return build


def _write_identification(paths, *, params=None, per_log=()):
    payload = {
        "estimated_params": {**BASE_PARAMS, **(params or {})},
        "per_log": [
            {
                "log": f"{paths.data_dir}/{name}.csv",
                "estimated_params": {**BASE_PARAMS, **overrides},
                "excluded": excluded,
            }
            for name, overrides, excluded in per_log
        ],
    }
    save_yaml(paths.identification_result, payload)


def test_only_identified_iterations_contribute_a_point(experiment):
    """An iteration still waiting for data has no parameters to plot, and must
    not stop the ones before it from plotting."""
    exp = experiment(3)
    _write_identification(exp.paths(1))
    _write_identification(exp.paths(2))

    records = collect_identified_parameters(exp)

    assert [record["index"] for record in records] == [1, 2]


def test_only_the_four_plotted_parameters_are_carried(experiment):
    """a_slip_max is stored beside the four but is not plotted -- it is held
    fixed on most runs (identification.identify_a_slip_max), so a panel for it
    would be a flat line about a parameter nobody fitted."""
    exp = experiment(1)
    _write_identification(exp.paths(1), params={"wheel_radius": 0.0161})

    (record,) = collect_identified_parameters(exp)

    assert set(record["params"]) == set(PARAMETER_NAMES)
    assert record["params"]["wheel_radius"] == pytest.approx(0.0161)


def test_excluded_logs_are_kept_flagged_not_dropped(experiment):
    """The collector reports every per-log fit with its screening verdict; what
    to draw is the figure's decision, not the collector's."""
    exp = experiment(1)
    _write_identification(
        exp.paths(1),
        per_log=[
            ("TR00", {"wheel_radius": 0.0122}, True),
            ("TR01", {"wheel_radius": 0.0164}, False),
        ],
    )

    (record,) = collect_identified_parameters(exp)

    assert [(entry["log"], entry["excluded"]) for entry in record["per_log"]] == [
        ("TR00", True),
        ("TR01", False),
    ]


def test_plotting_an_experiment_without_results_raises(experiment):
    """A figure with no points is a silent lie about a pipeline that has not
    identified anything yet."""
    exp = experiment(1)

    with pytest.raises(ValueError):
        plot_identified_parameters(collect_identified_parameters(exp), exp.root)


def test_plot_writes_one_figure_per_experiment(experiment, tmp_path):
    exp = experiment(2)
    _write_identification(exp.paths(1), per_log=[("TR00", {}, True), ("TR01", {}, False)])
    _write_identification(exp.paths(2), params={"time_constant": 0.175})

    out_path = plot_identified_parameters(
        collect_identified_parameters(exp), exp.root, out_path=tmp_path / "params.pdf"
    )

    assert (tmp_path / "params.pdf").is_file()
    assert out_path == str(tmp_path / "params.pdf")
