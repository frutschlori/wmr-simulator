"""State-dependent gains tuned post hoc on a finished static run.

The thesis' static (S) runs keep the gain parametrization off, so their data,
identified models, residuals and tuning sets never depend on a parametrized
controller. The state-dependent (D) controller is tuned afterwards on exactly
those artefacts, iteration by iteration, which makes S vs D a comparison of the
parametrization alone (decided 2026-10-01; trajectory designs are made at
static gains anyway).

A *variant* is one way of doing that (a hypothesis of Phase 3, e.g. a trust
region or a speed-only schedule): a name plus overrides of the run's
``gain_tuning`` block and of the problem's ``gain_parametrization``, stored in
``<run>/parametrized_variants/<name>.yaml``. Per iteration ``g``:

- ``iteration_g/results/parametrized/<name>/`` holds the tuning problem and
  ``gains.yaml`` (base gains, schedule, checks), tuned on iteration g's
  identified model, residual (if the run has one) and tuning set;
- ``iteration_{g+1}/parametrized/<name>/`` holds the deployable controller
  (``robot_config.yaml``, ``problem.yaml``, ``ROBOTCFG.CFG``, ``GAINMLP.JSN``)
  with iteration g's identified parameters, as ``finalize`` would write it;
- the ``benchmark`` stage of iteration g+1 discovers that folder and drives it
  as variant ``parametrized_<name>`` on the iteration's paired seeds, into
  ``data/benchmark_parametrized_<name>/``.

Lineage: iteration 1 starts from the static run's own result of iteration 1
with an identity network (the loop's ``seed_parametrization_from_static``);
later iterations warm-start from the variant's previous base gains and network.
"""

from __future__ import annotations

from pathlib import Path

from wmr_simulator.active_learning.experiment import (
    Experiment,
    load_yaml,
    merge_config,
    save_yaml,
    write_iteration_problem,
)

VARIANTS_DIR = "parametrized_variants"
VARIANT_PREFIX = "parametrized_"


def variant_config_path(experiment: Experiment, variant: str) -> Path:
    return experiment.root / VARIANTS_DIR / f"{variant}.yaml"


def save_variant_config(experiment: Experiment, variant: str, overrides: dict) -> Path:
    path = variant_config_path(experiment, variant)
    if path.is_file() and (load_yaml(path) or {}) != (overrides or {}):
        raise ValueError(f"{path} already holds a different configuration for variant {variant!r}.")
    return save_yaml(path, overrides or {})


def tuning_dir(experiment: Experiment, iteration: int, variant: str) -> Path:
    return experiment.paths(iteration).results_dir / "parametrized" / variant


def deploy_dir(experiment: Experiment, iteration: int, variant: str) -> Path:
    """Where the controller tuned in ``iteration - 1`` is deployed from."""
    return experiment.paths(iteration).root / "parametrized" / variant


def result_path(experiment: Experiment, iteration: int, variant: str) -> Path:
    return tuning_dir(experiment, iteration, variant) / "gains.yaml"


def stage_tune_parametrized(experiment: Experiment, iteration: int, variant: str) -> dict:
    """Tune the variant's parametrized controller on iteration ``iteration``
    and deploy it into iteration ``iteration + 1``."""
    from wmr_simulator.active_learning import stages
    from wmr_simulator.gain_parametrization import to_cfg as parametrization_to_cfg
    from wmr_simulator.gain_tuning.checks import warnings_for
    from wmr_simulator.gain_tuning.pipeline import resolve_gain_robot_params, run_gain_tuning_experiment
    from wmr_simulator.pololu.robot_config import export_robot_config
    from wmr_simulator.types import PhysicalParams, print_controller_gains

    overrides = load_yaml(variant_config_path(experiment, variant)) or {}
    config = merge_config(experiment.config["gain_tuning"], overrides.get("gain_tuning") or {})
    paths = experiment.paths(iteration)
    next_paths = experiment.paths(iteration + 1)
    if not paths.gains_result.is_file() or not next_paths.robot_config.is_file():
        raise FileNotFoundError(f"Iteration {iteration} of {experiment.root} is not finalized.")
    out_dir = tuning_dir(experiment, iteration, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Base gains and network to start from.
    parametrization = merge_config(
        load_yaml(experiment.config["problem"])["controller"]["gain_parametrization"],
        overrides.get("gain_parametrization") or {},
    )
    # ``base_from_static`` (H3): the base gains are this iteration's static
    # result every time and are held there; only the network is trained (and
    # warm-started), so the controller is the static one times a schedule.
    base_from_static = bool(overrides.get("base_from_static", False))
    previous = result_path(experiment, iteration - 1, variant) if iteration > 1 else None
    if previous is not None and previous.is_file():
        previous_result = load_yaml(previous)
        base_gains = load_yaml(paths.gains_result)["gains"] if base_from_static else previous_result["gains"]
        parametrization = {**parametrization, **previous_result["schedule"]}
        warm_start = True
        source = (
            f"static result of iteration {iteration} + variant network of iteration {iteration - 1}"
            if base_from_static
            else f"variant result of iteration {iteration - 1}"
        )
    else:
        base_gains = load_yaml(paths.gains_result)["gains"]
        warm_start = False
        source = f"static result of iteration {iteration}, identity network"
    parametrization["enabled"] = True
    print(f"Parametrized variant {variant!r}, iteration {iteration}: starting from the {source}: {base_gains}")

    problem = load_yaml(stages._tuning_problem(paths, experiment.config["tuning_trajectories"]))
    problem["controller"]["gains"] = [float(gain) for gain in base_gains]
    problem["controller"]["gain_parametrization"] = parametrization
    problem_path = save_yaml(out_dir / "problem_tuning.yaml", problem)

    robot_params = resolve_gain_robot_params(str(problem_path), None, None)
    residual_model = None
    if (
        experiment.config["use_residual_model"]
        and config.get("use_residual_model", True)
        and paths.residual_model.is_file()
    ):
        from wmr_simulator.residual_model import load_residual_model

        residual_model, _ = load_residual_model(paths.residual_model, robot_params)
        print(f"Tuning on the residual-augmented plant: {paths.residual_model}")

    result = run_gain_tuning_experiment(
        problem_path=str(problem_path),
        robot_params=robot_params,
        num_steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]),
        num_realizations=int(config["num_realizations"]),
        seed=int(experiment.config["seed"]),
        reference_trajectories_dir=str(paths.tuning_trajectories_dir),
        validation_split=float(config["validation_split"]),
        position_tracking_weight=float(config["position_tracking_weight"]),
        heading_tracking_weight=float(config["heading_tracking_weight"]),
        linear_velocity_tracking_weight=float(config["linear_velocity_tracking_weight"]),
        angular_velocity_tracking_weight=float(config["angular_velocity_tracking_weight"]),
        input_weight=float(config["input_weight"]),
        input_delta_weight=float(config["input_delta_weight"]),
        omega_delta_weight=float(config.get("omega_delta_weight", 0.0)),
        k_min_stab=float(config["k_min_stab"]),
        k_max_stab=float(config["k_max_stab"]),
        k_max_rest=float(config["k_max_rest"]),
        num_lhs_points=int(config["num_lhs_points"]),
        num_adam_optimizations=int(config["num_adam_optimizations"]),
        optimizer=str(config["optimizer"]),
        outlier_loss_factor=float(config["outlier_loss_factor"]),
        schedule_enabled=True,
        gain_delta_weight=float(config["gain_delta_weight"]),
        static_tune=False,
        presearch_relative_range=float(config["presearch_relative_range"]) if warm_start else 0.0,
        warm_start_schedule=warm_start,
        init_offset_radius=float(config["init_offset_radius"]),
        init_offset_angle=float(config["init_offset_angle"]),
        residual_model=residual_model,
        freeze_base_gains=base_from_static,
    )
    print_controller_gains("Parametrized base gains:", result["optimized_gains"])
    schedule = parametrization_to_cfg(result["schedule_params"])
    payload = {
        "variant": variant,
        "overrides": overrides,
        "started_from": source,
        "init_gains": [float(gain) for gain in base_gains],
        "gains": [float(gain) for gain in result["optimized_gains"]],
        "schedule": schedule,
        "used_residual_model": residual_model is not None,
        "final_loss": float(result["final_loss"]),
        "final_validation_loss": (
            None if result["final_validation_loss"] is None else float(result["final_validation_loss"])
        ),
        "checks": result["checks"],
        "warnings": warnings_for(result["checks"]),
    }
    save_yaml(result_path(experiment, iteration, variant), payload)

    # Deploy into the next iteration, with the parameters it was identified with.
    robot_config = load_yaml(next_paths.robot_config)
    robot_config["controller"]["gains"] = payload["gains"]
    robot_config["controller"]["gain_parametrization"] = {**schedule, "enabled": True}
    target = deploy_dir(experiment, iteration + 1, variant)
    target.mkdir(parents=True, exist_ok=True)
    save_yaml(target / "robot_config.yaml", robot_config)
    write_iteration_problem(experiment.config["problem"], robot_config, target / "problem.yaml")
    export_robot_config(
        target / "ROBOTCFG.CFG",
        physical_params=PhysicalParams(
            wheel_radius=robot_config["robot"]["wheel_radius"],
            base_diameter=robot_config["robot"]["base_diameter"],
            max_wheel_speed=robot_config["robot"]["max_wheel_speed"],
        ),
        controller_gains=payload["gains"],
        template_path=experiment.config.get("robotcfg_template"),
    )
    if stages._export_gain_mlp_if_configured(target / "problem.yaml", target / "GAINMLP.JSN") is None:
        raise RuntimeError(f"No gain network exported for variant {variant!r} into {target}.")
    print(f"Deployed variant {variant!r} for iteration {iteration + 1}: {target}")
    return payload


def finished(experiment: Experiment, variant: str, iterations: int) -> bool:
    """Every iteration tuned and every resulting controller benchmarked."""
    from wmr_simulator.active_learning.stages import iteration_status

    for iteration in range(1, iterations + 1):
        if not result_path(experiment, iteration, variant).is_file():
            return False
        if not iteration_status(experiment, iteration + 1)["benchmark"]:
            return False
    return True
