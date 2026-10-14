"""Post-hoc checks on a gain-tuning result, recorded as warnings next to the gains.

* **Stall.** The solve never left its start point and returned its input gains.
* **Box bound.** A gain ended on its search-box bound (``zero`` for kimotor or
  kdmotor at exactly 0).
* **Divergence.** Tuning rollouts whose position error leaves the divergence
  radius; for a parametrized run also scored on the bare base gains.
"""

from __future__ import annotations

import numpy as np

from wmr_simulator.controller import GAIN_NAMES
from wmr_simulator.gain_tuning.objectives import TRAINING_KEY_NAMESPACE, VALIDATION_KEY_NAMESPACE
from wmr_simulator.gain_tuning.optimizers import controller_gains_to_optimizer_values

# Optimizer-space distance (unit box) under which a gain counts as on its bound.
BOUND_TOLERANCE = 1e-3
# Optimizer-space movement under which a start counts as never having moved.
STALL_TOLERANCE = 1e-6
# Position error [m] above which a rollout diverged.
DIVERGENCE_RADIUS = 0.25
_NUM_LOG_GAINS = 4


def bound_hits(gains, k_min_stab: float, k_max_stab: float, k_max_rest: float) -> list[dict]:
    """Every gain within ``BOUND_TOLERANCE`` of its search-box bound."""
    gains = np.asarray(gains, dtype=float)
    values = np.asarray(controller_gains_to_optimizer_values(gains, k_min_stab, k_max_stab, k_max_rest))
    hits = []
    for index, (name, value, gain) in enumerate(zip(GAIN_NAMES, values, gains)):
        log_space = index < _NUM_LOG_GAINS
        if value <= BOUND_TOLERANCE:
            bound = "k_min_stab" if log_space else "zero"
        elif value >= 1.0 - BOUND_TOLERANCE:
            bound = "k_max_stab" if log_space else "k_max_rest"
        else:
            continue
        hits.append({"gain": name, "bound": bound, "value": float(gain)})
    return hits


def stalled(optimization: dict) -> bool:
    """Whether the returned start ended exactly where it began (also when the
    solve moved but never beat its start)."""
    best = int(optimization["best_start_index"])
    initial = np.asarray(optimization["initial_values_per_start"][best], dtype=float)
    final = np.asarray(optimization["final_values_per_start"][best], dtype=float)
    return bool(np.max(np.abs(final - initial)) <= STALL_TOLERANCE)


def start_and_returned_losses(optimization: dict) -> dict:
    """Training and validation loss of the returned start at its start point
    and at the returned point."""
    best = int(optimization["best_start_index"])
    training = np.asarray(optimization["loss_history_per_start"], dtype=float)[:, best]
    validation_history = optimization["validation_loss_history_per_start"]
    returned_validation = optimization["best_validation_loss"]
    start_training = float(training[0])
    returned_training = float(optimization["best_training_loss"])
    return {
        "start_training_loss": start_training,
        "returned_training_loss": returned_training,
        "training_loss_improvement": (
            float(1.0 - returned_training / start_training) if start_training > 0.0 else None
        ),
        "start_validation_loss": (
            None if validation_history is None else float(np.asarray(validation_history)[0, best])
        ),
        "returned_validation_loss": None if returned_validation is None else float(returned_validation),
    }


def gains_unchanged(returned_gains, init_gains, k_min_stab: float, k_max_stab: float, k_max_rest: float) -> bool:
    """Whether the returned gains equal the gains the run was handed."""
    returned = np.asarray(controller_gains_to_optimizer_values(returned_gains, k_min_stab, k_max_stab, k_max_rest))
    initial = np.asarray(controller_gains_to_optimizer_values(init_gains, k_min_stab, k_max_stab, k_max_rest))
    return bool(np.max(np.abs(returned - initial)) <= STALL_TOLERANCE)


def rollout_divergence(
    pipeline,
    robot_params,
    realizations,
    training_start_offsets,
    validation_start_offsets,
    controller_gains,
    schedule_params=None,
    radius: float = DIVERGENCE_RADIUS,
) -> dict:
    """Diverged rollouts of one controller over the training and validation
    sets: position error above ``radius`` in the second half of the rollout
    (the first half drives out the start offset), or not finite."""
    from wmr_simulator.visualization.gain_tuning import realization_keys_for_set, rollout_realizations

    errors = []
    for references, offsets, namespace in (
        (pipeline.training_reference_trajectories, training_start_offsets, TRAINING_KEY_NAMESPACE),
        (pipeline.validation_reference_trajectories, validation_start_offsets, VALIDATION_KEY_NAMESPACE),
    ):
        num_trajectories = int(references.shape[0])
        if num_trajectories == 0:
            continue
        robot_keys, estimator_keys = realization_keys_for_set(realizations, num_trajectories, namespace)
        poses = rollout_realizations(
            pipeline, robot_params, references, offsets, robot_keys, estimator_keys,
            controller_gains=controller_gains, schedule_params=schedule_params,
        )
        # Sample the wheel-clock poses on the geometry clock.
        stride = int(pipeline.inner_steps_per_geometry_step)
        reference = np.asarray(references, dtype=float)[:, 1:, :2]
        indices = np.arange(stride, reference.shape[1] * stride + 1, stride)[: reference.shape[1]]
        sampled = poses[:, :, indices, :2]
        errors.append(np.linalg.norm(sampled - reference[:, None, :, :], axis=-1).reshape(-1, len(indices)))
    error = np.concatenate(errors, axis=0)
    second_half = error[:, error.shape[1] // 2 :]
    finite = np.all(np.isfinite(error), axis=1)
    diverged = ~finite | (np.nan_to_num(second_half, nan=np.inf).max(axis=1) > radius)
    rmse = np.sqrt(np.mean(np.where(np.isfinite(error), error, np.nan) ** 2, axis=1))
    return {
        "rollouts": int(error.shape[0]),
        "diverged": int(diverged.sum()),
        "median_position_rmse": float(np.nanmedian(rmse)),
    }


def warnings_for(checks: dict) -> list[str]:
    """One human-readable line per failed check in a ``tuning checks`` block."""
    lines = []
    for run, run_checks in checks.items():
        if run_checks.get("stalled"):
            improvement = run_checks.get("training_loss_improvement")
            detail = "" if improvement is None else f", training loss improved {100 * improvement:.2f} %"
            lines.append(f"{run}: the returned start never left its start point (stalled solve{detail}).")
        if run_checks.get("unchanged_from_init"):
            lines.append(f"{run}: returned gains equal the input gains.")
        for hit in run_checks.get("bound_hits", []):
            lines.append(f"{run}: {hit['gain']} = {hit['value']:.6g} on the {hit['bound']} bound.")
        for label, divergence in run_checks.get("divergence", {}).items():
            if divergence["diverged"]:
                lines.append(
                    f"{run}: {divergence['diverged']}/{divergence['rollouts']} tuning-plant rollouts "
                    f"diverge under the {label} controller."
                )
    return lines
