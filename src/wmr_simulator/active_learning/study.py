"""Multi-configuration, multi-seed studies of the active-learning loop.

A *study* runs a list of loop configurations over several seeds, one MuJoCo
experiment at a time, and collects their benchmark results into one table. It
is how the thesis evaluation (Chapter 5) is produced, so every run carries a
manifest of what exactly ran.

A study is described by a spec yaml (``studies/``)::

    phase: phase1_tuner            # output directory under the study root
    iterations: 3                  # tuning passes per run
    seeds: [0, 1, 2]
    overrides: {...}               # experiment.yaml overrides for every run
    configurations:
      bfgs_current:                # run directory prefix
        tag: S-A-N                 # the three switches, see configuration_overrides
        overrides: {...}           # on top of the common ones

and every (configuration, seed) pair becomes ``<root>/<phase>/<name>_seed<k>/``,
an ordinary experiment directory plus ``study_manifest.yaml``,
``git_diff_session<n>.patch`` and ``logs/session<n>.log``.

Runs are resumable. A rerun of the launcher skips finished runs, and inside an
unfinished one it first deletes the outputs of any stage that started but never
finished (``stage_log.yaml``), since `run` would otherwise take a half-written
data/ or tuning set for a finished one.

After the loop, the launcher benchmarks the last tuned controller as well:
iteration ``N + 1`` exists once iteration ``N`` is finalized, but `run` stops
before recording it.
"""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from wmr_simulator.active_learning.experiment import (
    DEFAULT_EXPERIMENT_CONFIG,
    Experiment,
    load_yaml,
    merge_config,
    save_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_SCRIPT = REPO_ROOT / "scripts" / "run_active_learning.py"
MANIFEST_NAME = "study_manifest.yaml"

# Experiment keys holding paths the loop opens relative to its cwd. Runs start
# from their own directory, so these are made absolute against the repo.
_PATH_KEYS = (
    ("problem",),
    ("baseline_identification_trajectory",),
    ("baseline_tuning_trajectories_dir",),
    ("benchmark", "trajectory"),
)

# What each stage writes, relative to its iteration directory, so a stage that
# was interrupted can be undone. ``finalize`` writes the *next* iteration.
_STAGE_OUTPUTS = {
    "plan-id-trajectory": ("identification_trajectory/*", "problem_identification.yaml"),
    "benchmark": ("data/benchmark", "data/benchmark_static", "benchmark", "results/benchmark.yaml"),
    "simulate-deployment": ("data/TR*",),
    "decode-logs": ("data/*.csv",),
    "identify": ("results/identification.yaml", "problem_identified.yaml"),
    "train-residual": ("results/residual_model.pkl",),
    "plan-tuning-trajectories": ("tuning_trajectories/*", "problem_tuning.yaml"),
    "tune-gains": ("results/gains.yaml",),
}


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------


def configuration_overrides(tag: str) -> dict:
    """Experiment overrides for a configuration tag ``<gains>-<trajectories>-<plant>``.

    * gains: ``S`` static (gain parametrization off), ``D`` state-dependent (on);
    * trajectories: ``A`` designed identification and tuning trajectories,
      ``F`` both fixed, ``AF`` designed identification with the fixed tuning
      set (separates the two design problems);
    * plant for the designs and the tuning: ``N`` nominal (no residual model
      is trained), ``R`` nominal + residual.
    """
    try:
        gains, trajectories, plant = tag.split("-")
    except ValueError:
        raise ValueError(f"Configuration tag {tag!r} is not <gains>-<trajectories>-<plant>.") from None
    if gains not in ("S", "D") or trajectories not in ("A", "F", "AF") or plant not in ("N", "R"):
        raise ValueError(
            f"Configuration tag {tag!r}: gains S|D, trajectories A|F|AF, plant N|R."
        )
    return {
        "use_gain_parametrization": gains == "D",
        "optimize_identification_trajectory": trajectories in ("A", "AF"),
        "optimize_tuning_trajectories": trajectories == "A",
        "use_residual_model": plant == "R",
    }


def run_overrides(spec: dict, name: str, seed: int) -> dict:
    """The full experiment override set for one (configuration, seed) run."""
    configuration = spec["configurations"][name]
    overrides = merge_config(spec.get("overrides") or {}, configuration.get("overrides") or {})
    overrides = merge_config(overrides, configuration_overrides(configuration["tag"]))
    overrides = merge_config(
        overrides,
        {
            "seed": int(seed),
            "num_iterations": int(spec["iterations"]),
            "mujoco_deployment": {"enabled": True},
            # The gain_tuning block of experiment.yaml is what runs, so the
            # manifest's copy of it is the complete tuner configuration.
            "use_standalone_gain_tuning_defaults": False,
        },
    )
    # The loop opens these relative to its cwd; runs start in their own
    # directory, so resolve them against the repo.
    full = merge_config(DEFAULT_EXPERIMENT_CONFIG, overrides)
    for key in _PATH_KEYS:
        section = overrides if len(key) == 1 else overrides.setdefault(key[0], {})
        value = full[key[0]] if len(key) == 1 else full[key[0]][key[1]]
        if value and not Path(value).is_absolute():
            section[key[-1]] = str((REPO_ROOT / value).resolve())
    return overrides


def run_directory(study_root: Path, spec: dict, name: str, seed: int) -> Path:
    return Path(study_root) / spec["phase"] / f"{name}_seed{int(seed)}"


def study_runs(spec: dict) -> list[tuple[str, int]]:
    """Every (configuration, seed) pair, seed-major, so that an interrupted
    study has every configuration on the first seeds rather than every seed of
    the first configurations."""
    return [(name, int(seed)) for seed in spec["seeds"] for name in spec["configurations"]]


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout


def git_state() -> tuple[str, str]:
    """``(HEAD hash, dirty diff)``: tracked changes against HEAD plus every
    untracked file under src/, scripts/, problems/ and studies/ as a new-file
    diff, so the patch reproduces the tree the run used."""
    head = _git("rev-parse", "HEAD").strip()
    diff = _git("diff", "HEAD")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts", "problems", "studies")
    for path in untracked.splitlines():
        diff += _git("diff", "--no-index", "--", "/dev/null", path)
    return head, diff


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / MANIFEST_NAME
    return (load_yaml(path) or {}) if path.is_file() else {}


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------


def interrupted_stages(experiment: Experiment) -> list[tuple[int, str]]:
    """``(iteration, stage)`` for every stage that started and never finished."""
    found = []
    for iteration in experiment.iteration_indices():
        log_path = experiment.paths(iteration).stage_log
        log = (load_yaml(log_path) or {}) if log_path.is_file() else {}
        found.extend((iteration, stage) for stage, entry in log.items() if entry and not entry.get("finished"))
    return found


def clean_interrupted_stages(experiment: Experiment) -> list[str]:
    """Delete the outputs of every interrupted stage so `run` redoes it."""
    removed = []
    for iteration, stage in interrupted_stages(experiment):
        paths = experiment.paths(iteration)
        if stage == "finalize":
            targets = [experiment.paths(iteration + 1).root]
        else:
            targets = [match for pattern in _STAGE_OUTPUTS[stage] for match in paths.root.glob(pattern)]
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            removed.append(str(target))
        log = load_yaml(paths.stage_log) or {}
        log.pop(stage, None)
        save_yaml(paths.stage_log, log)
        print(f"  iteration {iteration:02d}: {stage} was interrupted; removed {len(targets)} output(s).", flush=True)
    return removed


def run_finished(experiment: Experiment) -> bool:
    """The loop reached its target and the last tuned controller is benchmarked."""
    from wmr_simulator.active_learning.stages import iteration_status

    final = int(experiment.config["num_iterations"]) + 1
    if final not in experiment.iteration_indices() or interrupted_stages(experiment):
        return False
    return iteration_status(experiment, final)["benchmark"]


# ---------------------------------------------------------------------------
# launching
# ---------------------------------------------------------------------------


def _cli(run_dir: Path, log_file, *args: str) -> int:
    """One run_active_learning.py command in its own process, from the run
    directory (plots land in a cwd-relative visualize/), output appended to the
    session log."""
    command = [sys.executable, str(CLI_SCRIPT), *args, "--experiment", str(run_dir)]
    log_file.write(f"\n$ {' '.join(command)}\n")
    log_file.flush()
    return subprocess.run(command, cwd=run_dir, stdout=log_file, stderr=subprocess.STDOUT, check=False).returncode


def launch_run(study_root: Path, spec: dict, spec_path: Path, name: str, seed: int) -> dict:
    """Create or resume one (configuration, seed) run and drive it to the end.

    Returns the session record it appended to the manifest.
    """
    from wmr_simulator.active_learning import stages

    run_dir = run_directory(study_root, spec, name, seed)
    tag = spec["configurations"][name]["tag"]
    manifest = _load_manifest(run_dir)
    if not (run_dir / "experiment.yaml").is_file():
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} holds files but no experiment.yaml; move them away first.")
        run_dir.mkdir(parents=True, exist_ok=True)
        overrides = run_overrides(spec, name, seed)
        experiment = stages.stage_init(run_dir, overrides)
        manifest = {
            "phase": spec["phase"],
            "configuration": name,
            "tag": tag,
            "seed": int(seed),
            "iterations": int(spec["iterations"]),
            "spec": str(spec_path),
            "created": _now(),
            "overrides": overrides,
            "sessions": [],
        }
    experiment = Experiment.load(run_dir)
    if run_finished(experiment):
        print(f"{run_dir.name}: finished, skipping.", flush=True)
        return {}

    session_index = len(manifest.get("sessions", []))
    head, diff = git_state()
    diff_name = f"git_diff_session{session_index}.patch"
    (run_dir / diff_name).write_text(diff)
    session = {
        "started": _now(),
        "finished": None,
        "wall_clock_s": None,
        "git_head": head,
        "git_dirty": bool(diff.strip()),
        "git_diff": diff_name,
        "host": platform.node(),
        "python": sys.executable,
        "cleaned": clean_interrupted_stages(experiment),
        "exit_codes": {},
    }
    manifest.setdefault("sessions", []).append(session)
    # The configuration the session ran with, in full (defaults included).
    manifest["experiment_config"] = load_yaml(experiment.config_path)
    save_yaml(run_dir / MANIFEST_NAME, manifest)

    (run_dir / "logs").mkdir(exist_ok=True)
    start = time.monotonic()
    print(f"{run_dir.name}: session {session_index} ({tag}, seed {seed}) -> {run_dir / 'logs'}", flush=True)
    with (run_dir / "logs" / f"session{session_index}.log").open("a", encoding="utf-8") as log_file:
        code = _cli(run_dir, log_file, "run")
        session["exit_codes"]["run"] = code
        final = int(experiment.config["num_iterations"]) + 1
        if code == 0 and final in Experiment.load(run_dir).iteration_indices():
            session["exit_codes"]["benchmark"] = _cli(
                run_dir, log_file, "benchmark", "--iteration", str(final)
            )
    session["finished"] = _now()
    session["wall_clock_s"] = round(time.monotonic() - start, 1)
    session["tuning_warnings"] = collect_tuning_warnings(Experiment.load(run_dir))
    save_yaml(run_dir / MANIFEST_NAME, manifest)
    status = "ok" if all(code == 0 for code in session["exit_codes"].values()) else "FAILED"
    print(f"{run_dir.name}: {status} after {session['wall_clock_s']:.0f} s, exit codes {session['exit_codes']}", flush=True)
    return session


def launch_study(spec_path: str | Path, study_root: str | Path, only: list[str] | None = None) -> None:
    """Run every (configuration, seed) of a spec, one after the other."""
    spec_path = Path(spec_path).resolve()
    spec = load_yaml(spec_path)
    phase_dir = Path(study_root) / spec["phase"]
    phase_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec_path, phase_dir / "study_spec.yaml")
    for name, seed in study_runs(spec):
        if only and name not in only:
            continue
        launch_run(Path(study_root), spec, spec_path, name, seed)


def collect_tuning_warnings(experiment: Experiment) -> dict:
    """``{iteration: [warning, ...]}`` from every gains.yaml of the run."""
    warnings = {}
    for iteration in experiment.iteration_indices():
        path = experiment.paths(iteration).gains_result
        if path.is_file():
            lines = (load_yaml(path) or {}).get("warnings") or []
            if lines:
                warnings[int(iteration)] = list(lines)
    return warnings


# ---------------------------------------------------------------------------
# summarizing
# ---------------------------------------------------------------------------

GAIN_NAMES = ("kx", "ky", "kth", "kpmotor", "kimotor", "kdmotor")
IDENTIFIED_NAMES = ("wheel_radius", "base_diameter", "max_wheel_speed", "time_constant")


def _median(values):
    values = [value for value in values if value is not None]
    return float(statistics.median(values)) if values else None


def _yaw_ringing_by_log(data_dir: Path) -> dict[str, float | None]:
    from wmr_simulator.active_learning.baseline_runs import load_run_logs, run_yaw_ringing

    return {run.name: run_yaw_ringing(run) for run in load_run_logs(data_dir)}


def _controller_gains(paths, variant: str) -> list[float] | None:
    source = paths.robot_config_static if variant == "static" and paths.robot_config_static.is_file() else paths.robot_config
    if not source.is_file():
        return None
    gains = [float(gain) for gain in load_yaml(source)["controller"]["gains"]]
    return gains + [0.0] * (len(GAIN_NAMES) - len(gains))


def summarize_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    """``(rows, run_rows)`` of one run.

    One row per (controller generation, variant): generation ``g`` is the
    controller tuned in iteration ``g`` and benchmarked in iteration ``g + 1``
    (generation 0 is the initial controller, benchmarked in iteration 1). Each
    row carries that controller's benchmark, its gains, the checks of the
    tuning that produced it, the parameters it was tuned on and the stage times
    of iteration ``g``. ``run_rows`` is the per-benchmark-run long format.
    """
    run_dir = Path(run_dir)
    experiment = Experiment.load(run_dir)
    manifest = _load_manifest(run_dir)
    base = {
        "phase": manifest.get("phase"),
        "configuration": manifest.get("configuration", run_dir.name),
        "tag": manifest.get("tag"),
        "seed": experiment.config["seed"],
    }
    rows, run_rows = [], []
    for benchmark_iteration in experiment.iteration_indices():
        paths = experiment.paths(benchmark_iteration)
        if not paths.benchmark_result.is_file():
            continue
        generation = benchmark_iteration - 1
        tuned = experiment.paths(generation) if generation >= 1 else None
        gains_result = load_yaml(tuned.gains_result) if tuned is not None and tuned.gains_result.is_file() else {}
        identification = (
            load_yaml(tuned.identification_result)
            if tuned is not None and tuned.identification_result.is_file()
            else {}
        )
        stage_log = (load_yaml(tuned.stage_log) or {}) if tuned is not None and tuned.stage_log.is_file() else {}
        benchmark = load_yaml(paths.benchmark_result) or {}
        variants = sorted(
            {variant for entry in benchmark.get("trajectories", {}).values() for variant in entry.get("variants", {})}
        )
        for variant in variants:
            row = {**base, "iteration": generation, "benchmark_iteration": benchmark_iteration, "variant": variant}
            shape_rmse, shape_ringing, shape_saturation = [], [], []
            diverged_total = 0
            for shape, entry in sorted(benchmark.get("trajectories", {}).items()):
                record = entry.get("variants", {}).get(variant)
                if record is None:
                    continue
                ringing = _yaw_ringing_by_log(paths.root / record["data"])
                runs = record.get("runs", [])
                rmse = _median(run["tracking_rmse"] for run in runs)
                ring = _median(ringing.get(Path(run["log"]).stem) for run in runs)
                saturation = _median(run.get("duty_saturated_fraction") for run in runs)
                diverged = sum(1 for run in runs if run["diverged"])
                row[f"rmse_{shape}"] = rmse
                row[f"diverged_{shape}"] = diverged
                row[f"ringing_{shape}"] = ring
                row[f"saturation_{shape}"] = saturation
                shape_rmse.append(rmse)
                shape_ringing.append(ring)
                shape_saturation.append(saturation)
                diverged_total += diverged
                for run in runs:
                    run_rows.append(
                        {
                            **base,
                            "iteration": generation,
                            "benchmark_iteration": benchmark_iteration,
                            "variant": variant,
                            "shape": shape,
                            "log": run["log"],
                            "benchmark_seed": run["seed"],
                            "placed_by_hand": run["placed_by_hand"],
                            "position_rmse": run["tracking_rmse"],
                            "position_max": run.get("tracking_max"),
                            "diverged": run["diverged"],
                            "yaw_ringing": ringing.get(Path(run["log"]).stem),
                            "duty_saturation_fraction": run.get("duty_saturated_fraction"),
                            "max_duty": run.get("max_duty"),
                        }
                    )
            row["rmse_median"] = _median(shape_rmse)
            row["diverged_total"] = diverged_total
            row["ringing_median"] = _median(shape_ringing)
            row["saturation_median"] = _median(shape_saturation)
            gains = _controller_gains(paths, variant)
            for name, value in zip(GAIN_NAMES, gains or [None] * len(GAIN_NAMES)):
                row[name] = value
            checks = (gains_result.get("checks") or {}).get(variant, {})
            row["stalled"] = checks.get("stalled")
            row["unchanged_from_init"] = checks.get("unchanged_from_init")
            hits = checks.get("bound_hits", [])
            row["bound_hits"] = ";".join(f"{hit['gain']}@{hit['bound']}" for hit in hits)
            row["box_bound_hits"] = sum(1 for hit in hits if hit["bound"] != "zero")
            divergence = checks.get("divergence", {})
            for label in ("tuned", "bare_base_gains"):
                if label in divergence:
                    row[f"sim_diverged_{label}"] = divergence[label]["diverged"]
                    row[f"sim_rollouts_{label}"] = divergence[label]["rollouts"]
            loss_key = "final_validation_loss" if variant != "static" or gains_result.get("static_gains") is None else "static_final_validation_loss"
            row["tuning_validation_loss"] = gains_result.get(loss_key)
            for name in IDENTIFIED_NAMES:
                row[name] = identification.get("estimated_params", {}).get(name)
            for stage, entry in stage_log.items():
                row[f"seconds_{stage}"] = (entry or {}).get("seconds")
            rows.append(row)
    return rows, run_rows


def summarize_study(phase_dir: str | Path) -> tuple[Path, Path, Path]:
    """Write ``summary.csv``, ``benchmark_runs.csv`` and ``summary.md`` for
    every run under ``phase_dir``."""
    import csv

    phase_dir = Path(phase_dir)
    rows, run_rows = [], []
    for run_dir in sorted(path for path in phase_dir.iterdir() if (path / "experiment.yaml").is_file()):
        run_summary, runs = summarize_run(run_dir)
        rows.extend(run_summary)
        run_rows.extend(runs)

    def write_csv(path: Path, records: list[dict]) -> Path:
        columns = list(dict.fromkeys(key for record in records for key in record))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        return path

    summary_csv = write_csv(phase_dir / "summary.csv", rows)
    runs_csv = write_csv(phase_dir / "benchmark_runs.csv", run_rows)
    markdown = phase_dir / "summary.md"
    markdown.write_text(summary_markdown(rows))
    return summary_csv, runs_csv, markdown


def _format(value, digits=3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_markdown(rows: list[dict]) -> str:
    """Per (configuration, iteration, variant): medians over seeds of the
    per-run medians, and summed counts."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["configuration"], row["iteration"], row["variant"]), []).append(row)
    header = (
        "| configuration | tag | it | variant | seeds | RMSE [m] | div. circle_fast | div. lemniscates | "
        "div. total | ringing [rad/s] | saturation | box-bound hits | zero hits | stalls | kdmotor | tune [s] |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for (configuration, iteration, variant), group in sorted(groups.items()):
        def summed(key):
            values = [row.get(key) for row in group if row.get(key) is not None]
            return sum(values) if values else None

        lemniscates = sum(
            value for row in group for key, value in row.items()
            if key.startswith("diverged_lemniscate") and value is not None
        )
        zero_hits = sum(row["bound_hits"].count("@zero") for row in group if row.get("bound_hits"))
        stalls = sum(1 for row in group if row.get("stalled"))
        lines.append(
            "| " + " | ".join(
                [
                    configuration,
                    str(group[0]["tag"]),
                    str(iteration),
                    variant,
                    str(len(group)),
                    _format(_median(row["rmse_median"] for row in group), 4),
                    _format(summed("diverged_circle_fast")),
                    str(lemniscates),
                    _format(summed("diverged_total")),
                    _format(_median(row["ringing_median"] for row in group)),
                    _format(_median(row["saturation_median"] for row in group)),
                    _format(summed("box_bound_hits")),
                    str(zero_hits) if iteration > 0 else "-",
                    str(stalls) if iteration > 0 else "-",
                    _format(_median(row.get("kdmotor") for row in group), 4),
                    _format(_median(row.get("seconds_tune-gains") for row in group), 0),
                ]
            ) + " |"
        )
    note = (
        "\nRMSE: median over seeds of each run's median over shapes of the per-shape median "
        "position RMSE. Divergence and bound counts are summed over seeds. Iteration g is the "
        "controller tuned in iteration g (0 = initial), benchmarked in iteration g + 1.\n"
    )
    return header + "\n".join(lines) + "\n" + note
