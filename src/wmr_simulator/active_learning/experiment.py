"""Experiment workspace for the iterative active-learning loop.

An experiment lives in its own directory and holds one folder per iteration:

    experiments/<name>/
      experiment.yaml               flags + stage hyperparameters (editable)
      iteration_01/
        robot_config.yaml           robot params + gains of this iteration
        ROBOTCFG.CFG                firmware export of robot_config.yaml (SD card)
        GAINMLP.JSN                 firmware export of the gain schedule, if enabled
        robot_config_static_gains.yaml  same params with the static gains
        ROBOTCFG_static.CFG         firmware export of the static-gain controller
        problem.yaml                base problem + this iteration's params/gains
        problem_identified.yaml     written by the identify stage
        identification_trajectory/  identification reference (.pkl + .JSN)
        tuning_trajectories/        tuning reference set (.pkl)
        data/                       identification logs (SD card or MuJoCo);
                                    subdirectories hold benchmark runs
        data/benchmark[_static]/    benchmark runs under the parametrized /
                                    static controller
        benchmark/                  benchmark references as driven (bridged)
        results/                    identification.yaml, gains.yaml, residual_model.pkl
        visualize/                  plots
      iteration_02/                 created by the finalize stage

All stage results are plain files, so the loop can be interrupted at any point
(e.g. to collect robot data from the SD card) and resumed stage by stage.
"""

from __future__ import annotations

import copy
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS

ITERATION_PREFIX = "iteration_"

DEFAULT_EXPERIMENT_CONFIG: dict = {
    # Base problem yaml; robot block and controller gains are overridden per iteration.
    "problem": "problems/pololu_gains.yaml",
    "seed": 0,
    # Total number of iterations `run` continues to (unattended only with
    # mujoco_deployment.enabled).
    "num_iterations": 5,
    "use_residual_model": True,
    # Master switch for the state-dependent gain schedule. Off: only the static
    # controller is tuned, exported and driven.
    "use_gain_parametrization": True,
    # Iteration 0: design a tuning set and tune on the prior model before the
    # first experiment, so iteration 1 records its logs under a tuned controller.
    "prior_tuning": True,
    # Initial model, overriding the problem yaml's robot block (None: use it).
    "initial_robot_params": {
        "wheel_radius": 0.020,
        "base_diameter": 0.090,
        "max_wheel_speed": 250.0,
        "time_constant": 0.25,
    },
    # Design the identification trajectory / tuning set (FIM) or use the fixed
    # references below.
    "optimize_identification_trajectory": True,
    "optimize_tuning_trajectories": True,
    # Use gain_tuning.defaults.GAIN_TUNING_DEFAULTS instead of the gain_tuning block.
    "use_standalone_gain_tuning_defaults": True,
    # Fixed references (scripts/generate_fixed_trajectory_sets.py), relative to
    # the cwd. A directory of identification references gives one per seed.
    "baseline_identification_trajectory": "trajectory_exports/fixed_sets/identification_random",
    "baseline_tuning_trajectories_dir": "trajectory_exports/fixed_sets/matched",
    # Firmware export template (None -> built-in defaults from robot_config.py).
    "robotcfg_template": None,
    # Mocap rigid-body calibration yaml (scripts/mocap_calibration.py), exported
    # into every ROBOTCFG.CFG. None: frames go uncorrected.
    "mocap_calibration": None,
    "log_loading": {
        "clip_after_first_trajectory": True,
    },
    # Drive the identification trajectory on the hidden MuJoCo plant instead of
    # the robot; the logs are written to data/ in the firmware's format.
    "mujoco_deployment": {
        "enabled": False,
        # Chained runs of the bridged trajectory per iteration.
        "num_logs": 5,
        # A run ending further than this [m] from the start is placed by hand again.
        "divergence_radius": 0.25,
        # Hand-placement spread of a placed run's start pose [m, rad].
        "start_offset_radius": 0.15,
        "start_offset_angle": 0.2,
    },
    # Held-out evaluation references, driven every iteration under both
    # controller variants (needs a MuJoCo plant; on the robot the runs are
    # recorded by hand into data/benchmark[_static]/<shape>/).
    "benchmark": {
        # A single reference file, or a directory of .JSN/.pkl references.
        "trajectory": "trajectory_exports/benchmark_set",
        # Chained runs per reference.
        "num_runs": 5,
        "divergence_radius": 0.25,
        "start_offset_radius": 0.15,
        "start_offset_angle": 0.2,
        # Open references get a wait and a bridge path back to their start.
        "bridge_wait_time": 1.5,
        "bridge_time": 8.5,
        # Worker processes (0: one per chain, at most CPUs - 1; 1: serial).
        "max_workers": 0,
    },
    "identification_trajectory": {
        # Design on the previous iteration's residual plant.
        "use_residual_model": False,
        "opt_steps": 750,
        "learning_rate": 5e-3,
        "num_control_points": 12,
        "time_scaling": "s-curve",
        # Replay window of the design FIM (as identification.window_length).
        "window_length": 10,
        # Consecutive motion phases with their own limits and floors; the
        # durations add up to the design's sim_time. Only ``identify`` phases
        # enter the FIM and the fit, so they come first.
        "phases": [
            {
                "duration": 3.0,
                "identify": True,
                "motion_limits": {"v_max": 1.2, "a_max": 2.0, "a_max_lateral": 2.0, "omega_max": 6.0, "alpha_max": 10.0},
                "min_speed": 0.6,
            },
            {
                "duration": 4.0,
                "identify": False,
                "motion_limits": {"v_max": 2.5, "a_max": 4.5, "a_max_lateral": 5.0, "omega_max": 10.0, "alpha_max": 15.0},
                "min_speed": 1.3,
                "min_lateral_acceleration": 2.0,
            },
        ],
        # Wait and bridge path back to the start of <name>_bridge.JSN.
        "bridge_wait_time": 1.5,
        "bridge_time": 8.5,
        # Number of trajectories of the previous tuning set appended after the
        # identified phase (trajectory_optimization.reference_extension),
        # joined by forward arc turns and returned arc - line - arc. 0: keep
        # the designed phases.
        "append_tuning_trajectories": 2,
        # Take the appended trajectories from this directory instead of the
        # previous tuning set (None: the tuning set).
        "append_tuning_trajectories_from": None,
        # Skip appended trajectories above this friction-ellipse usage (0: off).
        "appended_max_traction_usage": 1.1,
        "appended_bridge_wait_time": 0.25,
        "appended_turn_radius": 0.25,
        "appended_turn_lateral_acceleration": 3.0,
        "appended_min_piece_duration": 0.75,
        "appended_box_margin": 0.15,
        # Upper speed of the return line while the JSN exceeds the firmware limits.
        "appended_return_max_line_speed": 2.5,
    },
    "identification": {
        "steps": 600,
        "learning_rate": 1e-3,
        # Replay window [samples]; each window restarts from the measured pose.
        "window_length": 10,
        # Modified z-score above which a log is left out of the joint fit (0: off).
        "outlier_z_threshold": 3.5,
    },
    "residual": {
        "num_experts": 4,
        "hidden_sizes": [16, 16],
        # Per-matrix spectral-norm cap of the experts (0: off).
        "spectral_norm_cap": 1.5,
        # Gate bandwidth = intra-cluster RMS distance * this.
        "gate_bandwidth_scale": 1.0,
        # Distance [bandwidths] of the zero expert; beyond it the residual -> 0.
        "ood_sigma": 3.0,
        "epochs": 1500,
        "batch_size": 16384,
        "learning_rate": 2e-4,
        "validation_split": 0.2,
        "output_reg_weight": 1e-2,
        "resample_uniform": True,
        # "closed_loop" (residual_model.closed_loop) or "one_step"; epochs only
        # apply to one_step.
        "training": "closed_loop",
        "closed_loop_window_s": 1.0,
        "closed_loop_stride_s": 1.0,
        "closed_loop_steps": 400,
        "closed_loop_learning_rate": 1e-3,
        # Train on the logs of every iteration up to this one.
        "pool_previous_iterations": True,
        # Logs leaving their reference by more than this [m] are excluded (0: off).
        "max_position_error": 0.25,
    },
    "tuning_trajectories": {
        # Design on this iteration's residual plant (the plant the tuner uses).
        "use_residual_model": True,
        "num_trajectories": 15,
        "opt_steps": 250,
        "learning_rate": 1e-2,
        "num_control_points": 7,
        # Initialize floored designs at the arc length their speed floor needs
        # instead of on random lines.
        "stretch_initialization": False,
        # "s-curve" | "linear"
        "time_scaling": "s-curve",
        "constraint_weight_jitter": 0.4,
        # Duration of a tuning trajectory [s] (0: the problem's sim_time).
        "sim_time": 4.0,
        # Minimum distance between the start and goal of an initial line [m].
        "min_start_goal_distance": 1.0,
        # Mean-speed floor [m/s] and the share of the batch that carries one.
        "min_speed": 1.1,
        "min_speed_fraction": 0.5,
        # Mean |v * omega| floor [m/s^2] and the share of the batch that carries one.
        "min_lateral_acceleration": 2.0,
        "min_lateral_acceleration_fraction": 0.25,
        # Motion limits of the constraint term, overriding the problem's robot
        # block (empty: the problem's limits).
        "motion_limits": {},
        "window_length": 50,
        # Fixed FIM column scales of the gains that may be tuned to exactly 0.
        "kimotor_fim_scale": 5.0,
        "kdmotor_fim_scale": 0.01,
        "start_offset_mode": "optimize",
        "constraint_component_weights": {
            "v": 1.0,
            "a": 1.0,
            "lateral": 1.0,
            "omega": 1.0,
            "alpha": 0.5,
            "v_min": 0.5,
            "a_lat_min": 0.5,
        },
        # Sharpness of the smooth max in the constraint penalty.
        "constraint_smooth_max_beta": 10.0,
    },
    "gain_tuning": {
        **GAIN_TUNING_DEFAULTS,
        "presearch_relative_range": 0.3,
        # Tune on the residual plant (also requires use_residual_model).
        "use_residual_model": True,
    },
}


@dataclass(frozen=True)
class IterationPaths:
    root: Path

    @property
    def problem(self) -> Path:
        return self.root / "problem.yaml"

    @property
    def problem_identified(self) -> Path:
        return self.root / "problem_identified.yaml"

    @property
    def robot_config(self) -> Path:
        return self.root / "robot_config.yaml"

    @property
    def robot_config_static(self) -> Path:
        return self.root / "robot_config_static_gains.yaml"

    @property
    def robotcfg_cfg(self) -> Path:
        return self.root / "ROBOTCFG.CFG"

    @property
    def robotcfg_static_cfg(self) -> Path:
        return self.root / "ROBOTCFG_static.CFG"

    @property
    def gainmlp_jsn(self) -> Path:
        return self.root / "GAINMLP.JSN"

    @property
    def identification_trajectory_dir(self) -> Path:
        return self.root / "identification_trajectory"

    @property
    def tuning_trajectories_dir(self) -> Path:
        return self.root / "tuning_trajectories"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def benchmark_dir(self) -> Path:
        """The baseline reference as this iteration drove it (bridged or not)."""
        return self.root / "benchmark"

    @property
    def benchmark_data_dir(self) -> Path:
        return self.data_dir / "benchmark"

    @property
    def benchmark_static_data_dir(self) -> Path:
        return self.data_dir / "benchmark_static"

    @property
    def benchmark_result(self) -> Path:
        return self.results_dir / "benchmark.yaml"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def identification_result(self) -> Path:
        return self.results_dir / "identification.yaml"

    @property
    def residual_skipped(self) -> Path:
        """Written instead of residual_model.pkl when no log is usable."""
        return self.results_dir / "residual_skipped.yaml"

    @property
    def gains_result(self) -> Path:
        return self.results_dir / "gains.yaml"

    @property
    def residual_model(self) -> Path:
        return self.results_dir / "residual_model.pkl"

    @property
    def visualize_dir(self) -> Path:
        return self.root / "visualize"

    @property
    def stage_log(self) -> Path:
        """Start/finish times of every stage; a start without a finish was interrupted."""
        return self.root / "stage_log.yaml"

    def create_directories(self) -> None:
        for directory in (
            self.root,
            self.identification_trajectory_dir,
            self.tuning_trajectories_dir,
            self.data_dir,
            self.results_dir,
            self.visualize_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


class Experiment:
    def __init__(self, root: str | Path, config: dict):
        self.root = Path(root)
        self.config = config

    @property
    def config_path(self) -> Path:
        return self.root / "experiment.yaml"

    @classmethod
    def load(cls, root: str | Path) -> "Experiment":
        root = Path(root)
        config_path = root / "experiment.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"No experiment.yaml in {root}. Create the experiment first with the 'init' stage."
            )
        with config_path.open("r", encoding="utf-8") as file:
            user_config = yaml.safe_load(file) or {}
        return cls(root, merge_config(DEFAULT_EXPERIMENT_CONFIG, user_config))

    @classmethod
    def create(cls, root: str | Path, overrides: dict | None = None) -> "Experiment":
        root = Path(root)
        if (root / "experiment.yaml").exists():
            raise FileExistsError(f"Experiment already initialized: {root / 'experiment.yaml'}")
        root.mkdir(parents=True, exist_ok=True)
        config = merge_config(DEFAULT_EXPERIMENT_CONFIG, overrides or {})
        experiment = cls(root, config)
        experiment.save_config()
        return experiment

    def save_config(self) -> None:
        with self.config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, sort_keys=False)

    def iteration_indices(self) -> list[int]:
        if not self.root.is_dir():
            return []
        indices = []
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name.startswith(ITERATION_PREFIX):
                try:
                    indices.append(int(entry.name[len(ITERATION_PREFIX):]))
                except ValueError:
                    continue
        return sorted(indices)

    @property
    def first_iteration(self) -> int:
        """0 with prior tuning, else 1 (read off the directories once they exist)."""
        indices = self.iteration_indices()
        if indices:
            return indices[0]
        return 0 if self.config["prior_tuning"] else 1

    def latest_iteration(self) -> int:
        indices = self.iteration_indices()
        if not indices:
            raise FileNotFoundError(f"No iterations found in {self.root}.")
        return indices[-1]

    def paths(self, iteration: int) -> IterationPaths:
        return IterationPaths(self.root / f"{ITERATION_PREFIX}{iteration:02d}")

    def resolve_iteration(self, iteration: int | None) -> int:
        return self.latest_iteration() if iteration is None else int(iteration)


def merge_config(defaults: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return path


def robot_config_from_problem(problem_cfg: dict) -> dict:
    """Initial robot_config.yaml payload from a problem yaml dict."""
    robot_cfg = problem_cfg["robot"]
    payload = {
        "robot": {
            key: float(robot_cfg[key])
            for key in (
                "wheel_radius",
                "base_diameter",
                "max_wheel_speed",
                "time_constant",
            )
        },
        "controller": {
            "gains": [float(gain) for gain in problem_cfg["controller"]["gains"]],
        },
    }
    gain_parametrization = problem_cfg["controller"].get("gain_parametrization")
    if gain_parametrization is not None:
        payload["controller"]["gain_parametrization"] = copy.deepcopy(gain_parametrization)
    return payload


def write_iteration_problem(
    base_problem_path: str | Path,
    robot_config: dict,
    output_path: str | Path,
    update_estimator_geometry: bool = True,
) -> Path:
    """Generate a problem yaml with the iteration's robot params and gains.

    The base problem provides everything else (sim timing, environment,
    planner, estimator noise, ...). The estimator geometry follows the current
    best wheel_radius/base_diameter so the simulated estimator matches what
    the identified model believes.
    """
    problem_cfg = load_yaml(base_problem_path)
    # Pad a missing start heading with the direction toward the goal.
    start = [float(value) for value in problem_cfg["start"]]
    if len(start) == 2:
        import math

        goal = [float(value) for value in problem_cfg["goal"][:2]]
        start.append(math.atan2(goal[1] - start[1], goal[0] - start[0]))
        problem_cfg["start"] = start
    problem_cfg["robot"].update({key: float(value) for key, value in robot_config["robot"].items()})
    problem_cfg.setdefault("controller", {})["gains"] = [
        float(gain) for gain in robot_config["controller"]["gains"]
    ]
    gain_parametrization = robot_config["controller"].get("gain_parametrization")
    if gain_parametrization is not None:
        problem_cfg["controller"]["gain_parametrization"] = copy.deepcopy(gain_parametrization)
    if update_estimator_geometry and "estimator" in problem_cfg:
        problem_cfg["estimator"]["wheel_radius"] = float(robot_config["robot"]["wheel_radius"])
        problem_cfg["estimator"]["base_diameter"] = float(robot_config["robot"]["base_diameter"])
    return save_yaml(output_path, problem_cfg)


@contextmanager
def collect_plots(target_dir: str | Path, source_dir: str | Path = "visualize"):
    """Move plots written to ``visualize/`` during the block into ``target_dir``."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    def snapshot() -> dict[Path, float]:
        if not source_dir.is_dir():
            return {}
        return {path: path.stat().st_mtime for path in source_dir.rglob("*") if path.is_file()}

    before = snapshot()
    yield
    for path, mtime in snapshot().items():
        if path in before and before[path] == mtime:
            continue
        destination = target_dir / path.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
