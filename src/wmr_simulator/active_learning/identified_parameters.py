"""Cross-iteration history of the identified robot parameters.

Every iteration's ``identify`` stage writes one joint parameter fit into
``results/identification.yaml``, together with the per-log fits behind it and
which of those the outlier screen dropped. This module gathers that history
across a whole experiment so the plot can show how each physical parameter
settles (or fails to) as the loop feeds it more informative data;
``visualization.identified_parameters`` only renders what is returned here, the
same split ``progress.py`` / ``baseline_runs.py`` use.

Nothing here reads the hidden MuJoCo ground truth: the pipeline never gets to
see the true parameters, so neither does its own progress plot.
"""

from pathlib import Path

from wmr_simulator.active_learning.experiment import load_yaml

# Parameter key, axis label, and the factor taking the stored SI value to the
# plotted unit.
PARAMETER_SPECS: tuple[tuple[str, str, float], ...] = (
    ("wheel_radius", "wheel radius [mm]", 1000.0),
    ("base_diameter", "wheelbase [mm]", 1000.0),
    ("max_wheel_speed", "max wheel speed [rad/s]", 1.0),
    ("time_constant", "motor time constant [s]", 1.0),
)

PARAMETER_NAMES: tuple[str, ...] = tuple(name for name, _, _ in PARAMETER_SPECS)


def _plotted_params(estimated: dict) -> dict | None:
    """The four plotted parameters of one stored ``estimated_params`` block.

    A fit missing any of the four is skipped rather than plotted with holes in
    it.
    """
    if not estimated:
        return None
    values = {name: estimated.get(name) for name in PARAMETER_NAMES}
    if any(value is None for value in values.values()):
        return None
    return {name: float(value) for name, value in values.items()}


def collect_identified_parameters(experiment) -> list[dict]:
    """Identified parameters of every iteration that has run ``identify``.

    Returns a list (sorted by iteration) of dicts with keys ``index``,
    ``params`` (the joint fit downstream stages actually use) and ``per_log``
    (one ``{"log", "params", "excluded"}`` entry per identification log, the
    spread behind the joint fit). Iterations without a readable
    ``results/identification.yaml`` are left out entirely, so an experiment
    whose last iteration is still waiting for data still plots.
    """
    records = []
    for index in experiment.iteration_indices():
        result_path = experiment.paths(index).identification_result
        if not result_path.is_file():
            continue
        payload = load_yaml(result_path) or {}
        params = _plotted_params(payload.get("estimated_params") or {})
        if params is None:
            continue

        per_log = []
        for entry in payload.get("per_log") or []:
            log_params = _plotted_params(entry.get("estimated_params") or {})
            if log_params is None:
                continue
            per_log.append(
                {
                    "log": Path(str(entry.get("log", ""))).stem,
                    "params": log_params,
                    "excluded": bool(entry.get("excluded", False)),
                }
            )

        records.append({"index": index, "params": params, "per_log": per_log})
    return records
