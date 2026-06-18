import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax_tqdm import scan_tqdm

from wmr_simulator.identification.pipeline import SystemIdentificationPipeline
from wmr_simulator.identification.losses import window_replay_mse
from wmr_simulator.planner import compute_reference_trajectory
from wmr_simulator.types import PhysicalParams, clip_physical_params, physical_params_mse


def load_trajectory_configs(config_path: str, section_name: str = "experiments", required: bool = True) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    entries = config.get(section_name, [])
    if required and not entries:
        raise ValueError(f"No '{section_name}' entries found in '{config_path}'")
    return entries


def build_reference_states(problem_cfg: dict, dt: float, trajectory_cfg: dict | None = None):
    if trajectory_cfg is None:
        planner_cfg = problem_cfg["planner"]
        start = problem_cfg["start"]
        goal = problem_cfg["goal"]
        waypoints = planner_cfg.get("waypoints", [])
        trajectory_time = float(planner_cfg["time"])
    else:
        start = trajectory_cfg["start"]
        goal = trajectory_cfg["goal"]
        waypoints = trajectory_cfg.get("waypoints", [])
        trajectory_time = float(trajectory_cfg["time"])

    num_steps = int(trajectory_time / dt)
    time_grid = np.linspace(0.0, num_steps * dt, num_steps + 1)
    reference_states, _ = compute_reference_trajectory(start, goal, waypoints, time_grid)
    return reference_states, np.asarray(start, dtype=float)


def apply_reference_config(pipeline, trajectory_cfg: dict | None):
    reference_states, start_pose = build_reference_states(pipeline.problem, pipeline.dt, trajectory_cfg)
    estimator_start = trajectory_cfg.get("estimator_start", start_pose) if trajectory_cfg is not None else start_pose

    pipeline.problem["start"] = list(np.asarray(start_pose, dtype=float))
    pipeline.estimator_cfg["start"] = list(np.asarray(estimator_start, dtype=float))
    pipeline.reference_states = jnp.asarray(
        pipeline._extend_reference_states(np.asarray(reference_states, dtype=float)),
        dtype=jnp.float32,
    )
    return pipeline


def run_window_replay_identification(
    problem_path: str,
    initial_params: PhysicalParams,
    num_steps: int,
    learning_rate: float,
    window_length: int,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
    )
    estimated_params, loss_history, motor_loss_history, parameter_mse_history = pipeline.optimize(
        init_params=initial_params,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )
    return {
        "pipeline": pipeline,
        "estimated_params": estimated_params,
        "loss_history": loss_history,
        "motor_loss_history": motor_loss_history,
        "parameter_mse_history": parameter_mse_history,
    }


def build_configured_pipeline(
    problem_path: str,
    initial_params: PhysicalParams,
    seed: int,
    trajectory_cfg: dict | None,
    window_length: int | None,
):
    pipeline = SystemIdentificationPipeline(
        problem_path=problem_path,
        initial_params=initial_params,
        seed=seed,
        window_length=window_length,
    )
    if trajectory_cfg is not None:
        apply_reference_config(pipeline, trajectory_cfg)
        pipeline.target_log = pipeline.run_closed_loop(
            initial_params,
            use_hidden_robot=True,
            controller_gains=pipeline.gains,
        )
    return pipeline


def run_multi_experiment_identification(
    problem_path: str,
    initial_params: PhysicalParams,
    experiments_path: str,
    num_steps: int,
    learning_rate: float,
    seed: int = 0,
    window_length: int | None = None,
    validation_section: str | None = None,
):
    experiment_configs = load_trajectory_configs(experiments_path, section_name="experiments", required=True)
    pipelines = [
        build_configured_pipeline(
            problem_path=problem_path,
            initial_params=initial_params,
            seed=seed + idx,
            trajectory_cfg=config,
            window_length=window_length,
        )
        for idx, config in enumerate(experiment_configs)
    ]

    validation_pipelines = []
    if validation_section is not None:
        validation_configs = load_trajectory_configs(experiments_path, section_name=validation_section, required=False)
        validation_pipelines = [
            build_configured_pipeline(
                problem_path=problem_path,
                initial_params=initial_params,
                seed=seed + len(pipelines) + idx,
                trajectory_cfg=config,
                window_length=window_length,
            )
            for idx, config in enumerate(validation_configs)
        ]

    optimizer = optax.adam(learning_rate)
    current_params = clip_physical_params(initial_params)
    opt_state = optimizer.init(current_params)
    has_validation = bool(validation_pipelines)

    if num_steps <= 0:
        return {
            "pipelines": pipelines,
            "validation_pipelines": validation_pipelines,
            "estimated_params": current_params,
            "loss_history": [],
            "parameter_mse_history": [],
            "validation_loss_history": [],
        }

    def training_loss(params):
        losses = [
            window_replay_mse(
                pipeline=pipeline,
                params=params,
                target_log=pipeline.target_log,
                est_params=initial_params,
                window_length=window_length,
            )
            for pipeline in pipelines
        ]
        return jnp.mean(jnp.asarray(losses))

    @scan_tqdm(num_steps, desc="Optimization")
    def train_step(carry, step):
        params, optimizer_state = carry
        loss_value, grads = jax.value_and_grad(training_loss)(params)
        updates, next_optimizer_state = optimizer.update(grads, optimizer_state, params)
        next_params = optax.apply_updates(params, updates)
        next_params = clip_physical_params(next_params)

        if has_validation:
            validation_losses = [
                window_replay_mse(
                    pipeline=pipeline,
                    params=next_params,
                    target_log=pipeline.target_log,
                    est_params=initial_params,
                    window_length=window_length,
                )
                for pipeline in validation_pipelines
            ]
            validation_loss = jnp.mean(jnp.asarray(validation_losses))
        else:
            validation_loss = jnp.asarray(jnp.nan, dtype=loss_value.dtype)

        parameter_mse = physical_params_mse(next_params, pipelines[0].hidden_params)
        return (next_params, next_optimizer_state), (loss_value, parameter_mse, validation_loss)

    (current_params, _), (loss_history, parameter_mse_history, validation_loss_history) = jax.lax.scan(
        train_step,
        (current_params, opt_state),
        jnp.arange(num_steps),
    )

    return {
        "pipelines": pipelines,
        "validation_pipelines": validation_pipelines,
        "estimated_params": current_params,
        "loss_history": np.asarray(loss_history, dtype=float).tolist(),
        "parameter_mse_history": np.asarray(parameter_mse_history, dtype=float).tolist(),
        "validation_loss_history": (
            np.asarray(validation_loss_history, dtype=float).tolist()
            if has_validation
            else []
        ),
    }


def run_validation_identification(**kwargs):
    return run_multi_experiment_identification(validation_section="validation", **kwargs)
