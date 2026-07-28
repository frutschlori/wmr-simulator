"""Cross-iteration evaluation of a full active-learning pipeline run.

For every iteration this evaluates every recorded Gain-MLP circle benchmark in
``data/with gain MLP/circle`` (also accepting the older
``data/with_gain_MLP/circle`` spelling) against the reference each run
tracked. It computes the same four gain-tuning loss terms
(``tracking, velocity_tracking, input, input_delta``) in two ways:

- ``sim``  — closed-loop simulation of that iteration's *recording* controller
  (the base gains + error-MLP schedule stored in ``problem_identified.yaml``,
  which per project convention are exactly the gains used to record the runs)
  on the identified nominal model, along each run's reference;
- ``real`` — the actual recorded mocap/encoder trajectory of the run.

The two are directly comparable (same controller, same reference).  Each
iteration's plotted point is the mean over all of its available circle runs,
which shows both how the real tracking improves across iterations and how well
the identified sim predicts reality.

The heavy lifting (building the pipeline, the closed-loop rollout) lives here;
``visualization.pipeline_progress`` only renders the returned records.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np

TERM_NAMES = ("tracking", "velocity_tracking", "input", "input_delta", "omega_delta")


def _resolve_gain_tuning_config(experiment):
    """Same resolution the tune-gains stage uses, so the loss weights and
    realization count match how each iteration was actually tuned."""
    if experiment.config["use_standalone_gain_tuning_defaults"]:
        from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS

        return GAIN_TUNING_DEFAULTS
    return experiment.config["gain_tuning"]


def _reference_states_for_sim(run_reference):
    """Convert a recorded log reference (body forward speed ``v_ff`` in col 3,
    col 4 = 0) into the sim/global convention (``vx = v_ff cos theta``,
    ``vy = v_ff sin theta``) that ``_reference_targets`` and the controller
    feedforward expect."""
    ref = np.asarray(run_reference, dtype=float)
    theta = ref[:, 2]
    v = ref[:, 3]
    omega = ref[:, 5]
    zeros = np.zeros_like(v)
    return np.column_stack(
        [ref[:, 0], ref[:, 1], theta, v * np.cos(theta), v * np.sin(theta), omega, zeros, zeros]
    )


def _sim_terms(pipeline, base_gains, schedule_params, reference_states, robot_keys, estimator_keys, weights):
    """Five sim loss terms for a reference already in the sim/global convention."""
    from wmr_simulator.gain_tuning.objectives import (
        closed_loop_objective_terms,
        scheduled_closed_loop_objective_terms,
    )

    vtw, iw, idw, odw = weights
    if schedule_params is None:
        terms = closed_loop_objective_terms(
            pipeline,
            base_gains,
            robot_keys,
            estimator_keys,
            velocity_tracking_weight=vtw,
            input_weight=iw,
            input_delta_weight=idw,
            omega_delta_weight=odw,
            reference_states=reference_states,
        )
    else:
        # Drop the trailing gain-schedule term; keep the five base terms.
        terms = scheduled_closed_loop_objective_terms(
            pipeline,
            base_gains,
            schedule_params,
            robot_keys,
            estimator_keys,
            velocity_tracking_weight=vtw,
            input_weight=iw,
            input_delta_weight=idw,
            omega_delta_weight=odw,
            gain_delta_weight=0.0,
            reference_states=reference_states,
        )[:5]
    return np.asarray(terms, dtype=float)


def _sim_run_terms(pipeline, base_gains, schedule_params, run_reference, robot_keys, estimator_keys, weights):
    """Sim terms along a recorded run's reference (body ``v_ff`` -> global vx,vy)."""
    import jax.numpy as jnp

    reference_states = jnp.asarray(_reference_states_for_sim(run_reference), dtype=jnp.float32)
    return _sim_terms(pipeline, base_gains, schedule_params, reference_states, robot_keys, estimator_keys, weights)


def _real_run_terms(pipeline, log, weights):
    """Loss terms of the recorded run, time-aligned to the reference timestamps.

    Mirrors ``_base_loss_terms``: pose_mse tracking, [v, omega] velocity error
    normalized by (v_max, omega_max), duty-cycle input/input-delta energy, and
    the normalized yaw-rate-chatter penalty.
    """
    import jax.numpy as jnp

    vtw, iw, idw, odw = weights
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
    tracking = float(pipeline.pose_mse(jnp.asarray(predicted_poses), jnp.asarray(target_poses)))

    wheel_time = np.asarray(log.wheel.time_s, dtype=float)
    vel_omega = np.asarray(log.wheel.vel_omega, dtype=float)  # [v, omega]
    pred_v = np.interp(target_time, wheel_time, vel_omega[:, 0])
    pred_omega = np.interp(target_time, wheel_time, vel_omega[:, 1])
    velocity_scale = np.array([float(pipeline.v_max), float(pipeline.omega_max)])
    velocity_error = (
        np.stack([pred_v, pred_omega], axis=1) - np.stack([reference_v, reference_omega], axis=1)
    ) / velocity_scale
    velocity = float(np.mean(np.sum(velocity_error**2, axis=1)))

    duty = np.asarray(log.wheel.duty_cycle, dtype=float)
    window = (wheel_time >= target_time[0]) & (wheel_time <= target_time[-1])
    duty_window = duty[window] if int(window.sum()) >= 2 else duty
    input_loss = float(np.mean(np.sum(duty_window**2, axis=1)))
    duty_delta = np.diff(duty_window, axis=0)
    input_delta = float(np.mean(np.sum(duty_delta**2, axis=1)))

    # Yaw-rate chatter of the recorded run (same window, normalized by omega_max):
    # the real oscillation the sim omega-delta term is meant to reproduce.
    omega_series = vel_omega[window, 1] if int(window.sum()) >= 2 else vel_omega[:, 1]
    omega_delta = float(np.mean(np.diff(omega_series / float(pipeline.omega_max)) ** 2))

    return np.array(
        [tracking, vtw * velocity, iw * input_loss, idw * input_delta, odw * omega_delta]
    )


def _gain_mlp_circle_directory(paths) -> Path | None:
    """Return the direct Gain-MLP circle benchmark directory, if recorded."""
    for directory_name in ("with gain MLP", "with_gain_MLP"):
        circle_dir = paths.data_dir / directory_name / "circle"
        if circle_dir.is_dir():
            return circle_dir
    return None


def _load_gain_mlp_circle_logs(paths) -> list:
    """Decode and load all direct circle benchmark recordings for one iteration.

    Benchmark logs are intentionally nested below ``data/`` and normally kept
    as SD-card binaries.  Decode them in a temporary directory so progress
    plots remain read-only with respect to the experiment data.
    """
    from wmr_simulator.pololu.decode_binary import decode_file
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log

    circle_dir = _gain_mlp_circle_directory(paths)
    if circle_dir is None:
        return []

    logs = []
    with tempfile.TemporaryDirectory(prefix="pipeline_progress_circle_") as temp_name:
        temporary_dir = Path(temp_name)
        for log_path in sorted(path for path in circle_dir.glob("TR*") if path.is_file()):
            csv_path = log_path if log_path.suffix.lower() == ".csv" else temporary_dir / f"{log_path.name}.csv"
            if csv_path != log_path:
                with contextlib.redirect_stdout(io.StringIO()):
                    if not decode_file(str(log_path), str(csv_path)):
                        print(f"Progress evaluation skipped unreadable log: {log_path}")
                        continue
            try:
                logs.append(load_pololu_traj_control_log(csv_path, clip_after_first_trajectory=True))
            except ValueError as error:
                print(f"Progress evaluation skipped {log_path}: {error}")
    return logs


def _evaluate_iteration(experiment, index, weights, num_realizations, seed):
    """Return circle-benchmark sim/real terms averaged over one iteration."""
    import jax

    from wmr_simulator.active_learning.stages import _log_gain_parametrization
    from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline, resolve_gain_robot_params
    import jax.numpy as jnp

    paths = experiment.paths(index)
    logs = _load_gain_mlp_circle_logs(paths)
    if not logs:
        return None
    problem_path = paths.problem_identified if paths.problem_identified.is_file() else paths.problem
    if not problem_path.is_file():
        return None

    robot_params = resolve_gain_robot_params(str(problem_path), None, None)
    pipeline = ControllerTuningPipeline(
        str(problem_path), robot_params=robot_params, seed=seed, residual_model=None
    )
    base_gains_list, schedule_params = _log_gain_parametrization(paths)
    base_gains = jnp.asarray(base_gains_list, dtype=jnp.float32)

    key = jax.random.PRNGKey(seed)
    robot_key, estimator_key = jax.random.split(key)
    robot_keys = jax.random.split(robot_key, num_realizations)
    estimator_keys = jax.random.split(estimator_key, num_realizations)

    sim_terms = []
    real_terms = []
    for log in logs:
        real = _real_run_terms(pipeline, log, weights)
        if real is None:
            continue
        sim = _sim_run_terms(
            pipeline, base_gains, schedule_params, log.reference.states, robot_keys, estimator_keys, weights
        )
        real_terms.append(real)
        sim_terms.append(sim)

    if not sim_terms:
        return None
    return np.mean(np.stack(sim_terms), axis=0), np.mean(np.stack(real_terms), axis=0)


def _terms_to_dict(terms):
    record = {name: float(value) for name, value in zip(TERM_NAMES, terms)}
    record["total"] = float(np.sum(terms))
    return record


def evaluate_pipeline_progress(experiment):
    """Per-iteration gains and sim/real loss terms across the whole pipeline.

    Returns a list (sorted by iteration) of dicts with keys ``index``,
    ``gains`` (the base controller gains deployed to *record* the iteration,
    from its ``problem.yaml``; iteration 1 is the initial hand-set gains, later
    iterations are the previous iteration's tuned result) and ``sim`` / ``real``
    (loss-term dicts, None when the iteration has no Gain-MLP circle benchmark
    runs). The recording gains are used (rather than the iteration's own tuned
    ``results/gains.yaml``) so every iteration is represented and the gains line
    up with the controller the sim/real losses are evaluated under.
    """
    from wmr_simulator.active_learning.stages import _log_gain_parametrization

    config = _resolve_gain_tuning_config(experiment)
    weights = (
        float(config["velocity_tracking_weight"]),
        float(config["input_weight"]),
        float(config["input_delta_weight"]),
        float(config.get("omega_delta_weight", 0.0)),
    )
    num_realizations = int(config["num_realizations"])
    seed = int(experiment.config["seed"])

    records = []
    for index in experiment.iteration_indices():
        paths = experiment.paths(index)
        gains = None
        if paths.problem.is_file():
            base_gains, _ = _log_gain_parametrization(paths)
            gains = np.asarray(base_gains, dtype=float)

        evaluated = _evaluate_iteration(experiment, index, weights, num_realizations, seed)
        sim = real = None
        if evaluated is not None:
            sim_terms, real_terms = evaluated
            sim = _terms_to_dict(sim_terms)
            real = _terms_to_dict(real_terms)

        records.append({"index": index, "gains": gains, "sim": sim, "real": real})
    return records
