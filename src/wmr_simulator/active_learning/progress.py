"""Cross-iteration evaluation of a full active-learning pipeline run.

For every iteration, the benchmark runs of the controller it deployed
(``data/benchmark``, else ``data/benchmark_static``) are scored with the
gain-tuning loss terms in two ways:

- ``sim``: closed-loop simulation of the recording controller on the
  iteration's identified nominal model, along each run's reference;
- ``real``: the recorded run itself.

Each iteration's point is the mean over its runs; the individual runs are
returned as well. Every recorded run is also scored for yaw ringing
(``baseline_runs.run_yaw_ringing``), per shape. ``visualization.pipeline_progress``
renders the returned records.
"""

from pathlib import Path

import numpy as np

from wmr_simulator.active_learning.experiment import load_yaml
from wmr_simulator.controller import controller_gains_list

TERM_NAMES = (
    "position_tracking",
    "heading_tracking",
    "linear_velocity_tracking",
    "angular_velocity_tracking",
    "input",
    "input_delta",
    "omega_delta",
)


def _resolve_gain_tuning_config(experiment):
    """The gain-tuning config the tune-gains stage uses."""
    if experiment.config["use_standalone_gain_tuning_defaults"]:
        from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS

        return GAIN_TUNING_DEFAULTS
    return experiment.config["gain_tuning"]


def _reference_states_for_sim(run_reference):
    """Convert a recorded reference (body forward speed ``v_ff`` in column 3)
    into the simulator's global convention (``vx = v_ff cos theta``,
    ``vy = v_ff sin theta``)."""
    ref = np.asarray(run_reference, dtype=float)
    theta = ref[:, 2]
    v = ref[:, 3]
    omega = ref[:, 5]
    zeros = np.zeros_like(v)
    return np.column_stack(
        [ref[:, 0], ref[:, 1], theta, v * np.cos(theta), v * np.sin(theta), omega, zeros, zeros]
    )


def _sim_terms(pipeline, base_gains, schedule_params, reference_states, robot_keys, estimator_keys, weights):
    """The base sim loss terms (``TERM_NAMES``) for a reference already in the
    sim/global convention."""
    from wmr_simulator.gain_tuning.objectives import (
        closed_loop_objective_terms,
        scheduled_closed_loop_objective_terms,
    )

    ptw, htw, lvtw, avtw, iw, idw, odw = weights
    if schedule_params is None:
        terms = closed_loop_objective_terms(
            pipeline,
            base_gains,
            robot_keys,
            estimator_keys,
            position_tracking_weight=ptw,
            heading_tracking_weight=htw,
            linear_velocity_tracking_weight=lvtw,
            angular_velocity_tracking_weight=avtw,
            input_weight=iw,
            input_delta_weight=idw,
            omega_delta_weight=odw,
            reference_states=reference_states,
        )
    else:
        # Drop the trailing gain-schedule term; keep the base terms.
        terms = scheduled_closed_loop_objective_terms(
            pipeline,
            base_gains,
            schedule_params,
            robot_keys,
            estimator_keys,
            position_tracking_weight=ptw,
            heading_tracking_weight=htw,
            linear_velocity_tracking_weight=lvtw,
            angular_velocity_tracking_weight=avtw,
            input_weight=iw,
            input_delta_weight=idw,
            omega_delta_weight=odw,
            gain_delta_weight=0.0,
            reference_states=reference_states,
        )[: len(TERM_NAMES)]
    return np.asarray(terms, dtype=float)


def _sim_run_terms(pipeline, base_gains, schedule_params, run_reference, robot_keys, estimator_keys, weights):
    """Sim terms along a recorded run's reference (body ``v_ff`` -> global vx,vy)."""
    import jax.numpy as jnp

    reference_states = jnp.asarray(_reference_states_for_sim(run_reference), dtype=jnp.float32)
    return _sim_terms(pipeline, base_gains, schedule_params, reference_states, robot_keys, estimator_keys, weights)


def _real_run_terms(pipeline, log, weights):
    """Loss terms of the recorded run (as in ``_base_loss_terms``), aligned to
    the reference timestamps."""
    import jax.numpy as jnp

    from wmr_simulator.gain_tuning.objectives import pose_tracking_losses

    ptw, htw, lvtw, avtw, iw, idw, odw = weights
    ref_states = np.asarray(log.reference.states, dtype=float)
    ref_time = np.asarray(log.reference.time_s, dtype=float)
    if len(ref_states) < 2:
        return None

    target_poses = ref_states[1:, :3]
    reference_v = ref_states[1:, 3]  # body forward speed (v_ff)
    reference_omega = ref_states[1:, 5]
    target_time = ref_time[1:]

    pose_time = np.asarray(log.pose.time_s, dtype=float)
    pose_states = np.asarray(log.pose.states, dtype=float)  # smoothed, canonical
    pred_x = np.interp(target_time, pose_time, pose_states[:, 0])
    pred_y = np.interp(target_time, pose_time, pose_states[:, 1])
    pred_theta = np.interp(target_time, pose_time, np.unwrap(pose_states[:, 2]))
    predicted_poses = np.stack([pred_x, pred_y, pred_theta], axis=1)
    position_tracking, heading_tracking = (
        float(value)
        for value in pose_tracking_losses(jnp.asarray(predicted_poses), jnp.asarray(target_poses))
    )

    wheel_time = np.asarray(log.wheel.time_s, dtype=float)
    vel_omega = np.asarray(log.wheel.vel_omega, dtype=float)  # [v, omega]
    pred_v = np.interp(target_time, wheel_time, vel_omega[:, 0])
    pred_omega = np.interp(target_time, wheel_time, vel_omega[:, 1])
    velocity_scale = np.array([float(pipeline.v_max), float(pipeline.omega_max)])
    velocity_error = (
        np.stack([pred_v, pred_omega], axis=1) - np.stack([reference_v, reference_omega], axis=1)
    ) / velocity_scale
    linear_velocity = float(np.mean(velocity_error[:, 0] ** 2))
    angular_velocity = float(np.mean(velocity_error[:, 1] ** 2))

    duty = np.asarray(log.wheel.duty_cycle, dtype=float)
    window = (wheel_time >= target_time[0]) & (wheel_time <= target_time[-1])
    duty_window = duty[window] if int(window.sum()) >= 2 else duty
    input_loss = float(np.mean(np.sum(duty_window**2, axis=1)))
    duty_delta = np.diff(duty_window, axis=0)
    input_delta = float(np.mean(np.sum(duty_delta**2, axis=1)))

    # Yaw-rate change of the recorded run, normalized by omega_max.
    omega_series = vel_omega[window, 1] if int(window.sum()) >= 2 else vel_omega[:, 1]
    omega_delta = float(np.mean(np.diff(omega_series / float(pipeline.omega_max)) ** 2))

    return np.array(
        [
            ptw * position_tracking,
            htw * heading_tracking,
            lvtw * linear_velocity,
            avtw * angular_velocity,
            iw * input_loss,
            idw * input_delta,
            odw * omega_delta,
        ]
    )


def _benchmark_directory(paths) -> Path | None:
    """The deployed controller's benchmark runs of one iteration:
    ``data/benchmark``, else ``data/benchmark_static``, else None."""
    for name in ("benchmark", "benchmark_static"):
        directory = paths.data_dir / name
        if directory.is_dir():
            return directory
    return None


def _load_benchmark_logs(paths) -> list:
    """``(shape, run)`` for every benchmark recording of one iteration."""
    from wmr_simulator.active_learning.baseline_runs import load_run_logs

    directory = _benchmark_directory(paths)
    if directory is None:
        return []
    logs = load_run_logs(directory)
    if logs:
        return [(directory.name, run) for run in logs]
    # A benchmark set has one subdirectory per reference.
    return [
        (shape_dir.name, run)
        for shape_dir in sorted(path for path in directory.glob("*") if path.is_dir())
        for run in load_run_logs(shape_dir)
    ]


def _evaluate_iteration(experiment, index, weights, num_realizations, seed):
    """Benchmark sim/real terms of one iteration: the two run means, the
    per-run real terms behind the second of them, and each run's yaw ringing
    keyed by shape (None where the log has no score, see ``run_yaw_ringing``)."""
    import jax

    from wmr_simulator.active_learning.stages import _log_gain_parametrization
    from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline, resolve_gain_robot_params
    import jax.numpy as jnp

    from wmr_simulator.active_learning.baseline_runs import run_yaw_ringing

    paths = experiment.paths(index)
    logs = _load_benchmark_logs(paths)
    if not logs:
        return None
    problem_path = paths.problem_identified if paths.problem_identified.is_file() else paths.problem
    if not problem_path.is_file():
        return None

    robot_params = resolve_gain_robot_params(str(problem_path), None, None)
    pipeline = ControllerTuningPipeline(
        str(problem_path), robot_params=robot_params, seed=seed, residual_model=None
    )
    pipeline.extend_time_grids(max(len(run.log.reference.states) for _, run in logs))
    base_gains_list, schedule_params = _log_gain_parametrization(paths)
    base_gains = jnp.asarray(base_gains_list, dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    robot_key, estimator_key = jax.random.split(key)
    robot_keys = jax.random.split(robot_key, num_realizations)
    estimator_keys = jax.random.split(estimator_key, num_realizations)

    sim_terms = []
    real_terms = []
    ringing: dict[str, list] = {}
    for shape, run in logs:
        log = run.log
        real = _real_run_terms(pipeline, log, weights)
        if real is None:
            continue
        ringing.setdefault(shape, []).append(run_yaw_ringing(run))
        sim = _sim_run_terms(
            pipeline, base_gains, schedule_params, log.reference.states, robot_keys, estimator_keys, weights
        )
        real_terms.append(real)
        sim_terms.append(sim)

    if not sim_terms:
        return None
    return (
        np.mean(np.stack(sim_terms), axis=0),
        np.mean(np.stack(real_terms), axis=0),
        np.stack(real_terms),
        ringing,
    )


def _terms_to_dict(terms):
    record = {name: float(value) for name, value in zip(TERM_NAMES, terms)}
    record["total"] = float(np.sum(terms))
    return record


def _static_gains(paths):
    """The iteration's static controller gains (``robot_config_static_gains.yaml``,
    else ``problem.yaml``), or None."""
    if paths.robot_config_static.is_file():
        config = load_yaml(paths.robot_config_static)
    elif paths.problem.is_file():
        config = load_yaml(paths.problem)
    else:
        return None
    return np.asarray(controller_gains_list(config["controller"]["gains"]), dtype=float)


def evaluate_pipeline_progress(experiment):
    """Per-iteration gains and sim/real loss terms across the whole pipeline.

    Returns a list (sorted by iteration) of dicts with keys ``index``, ``gains``
    (the static controller gains), ``sim`` / ``real`` (loss-term dicts, None
    without benchmark runs), ``real_runs`` (one loss-term dict per run) and
    ``yaw_ringing_by_shape`` (shape -> one ringing value per run).
    """
    config = _resolve_gain_tuning_config(experiment)
    weights = (
        float(config["position_tracking_weight"]),
        float(config["heading_tracking_weight"]),
        float(config["linear_velocity_tracking_weight"]),
        float(config["angular_velocity_tracking_weight"]),
        float(config["input_weight"]),
        float(config["input_delta_weight"]),
        float(config.get("omega_delta_weight", 0.0)),
    )
    num_realizations = int(config["num_realizations"])
    seed = int(experiment.config["seed"])

    records = []
    for index in experiment.iteration_indices():
        paths = experiment.paths(index)
        gains = _static_gains(paths)

        evaluated = _evaluate_iteration(experiment, index, weights, num_realizations, seed)
        sim = real = None
        real_runs = []
        yaw_ringing_by_shape = {}
        if evaluated is not None:
            sim_terms, real_terms, per_run_real_terms, yaw_ringing_by_shape = evaluated
            sim = _terms_to_dict(sim_terms)
            real = _terms_to_dict(real_terms)
            real_runs = [_terms_to_dict(terms) for terms in per_run_real_terms]

        records.append(
            {
                "index": index,
                "gains": gains,
                "sim": sim,
                "real": real,
                "real_runs": real_runs,
                "yaw_ringing_by_shape": yaw_ringing_by_shape,
            }
        )
    return records
