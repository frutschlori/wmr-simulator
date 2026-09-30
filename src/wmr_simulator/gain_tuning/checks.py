"""Post-hoc checks on a gain-tuning result.

Each one is a known way the tuner fails silently, so each is a result to report
rather than an error: the stage records them next to the gains and carries on.

* **Stall.** BFGS sometimes never leaves its start point (the line search
  rejects every trial, or no iterate beats the start on the validation set),
  and the tuner then ships its input gains unchanged -- about one run in 18,
  measured on ``real02/iteration_05`` and ``test07``.
* **Box bound.** The sim has no interior optimum in some gains (``kpmotor``
  monotone to the lower bound, ``ky`` unobservable on slow designs), so a gain
  on the search box is a failure of the model or the design set, not a tuned
  value. The sqrt-space gains (``kimotor``, ``kdmotor``) may reach exactly 0 by
  design; that is reported as its own bound, ``zero``.
* **Divergence.** Rollouts in the tuning plant whose position error leaves the
  benchmark's divergence radius. For a parametrized run this is also scored on
  the *bare* base gains: they are not a controller anybody drives, but when
  they diverge the network is cancelling a runaway base gain (memory
  ``gain-mlp-undoes-kpmotor``) rather than scheduling a working one.
"""

from __future__ import annotations

import numpy as np

from wmr_simulator.controller import GAIN_NAMES
from wmr_simulator.gain_tuning.objectives import TRAINING_KEY_NAMESPACE, VALIDATION_KEY_NAMESPACE
from wmr_simulator.gain_tuning.optimizers import controller_gains_to_optimizer_values

# Optimizer-space distance (unit box) under which a gain counts as on its bound.
# In log space over [1e-3, 20] this is a factor of 1.01 from the bound.
BOUND_TOLERANCE = 1e-3
# Optimizer-space movement under which a start counts as never having moved.
STALL_TOLERANCE = 1e-6
# Same radius the benchmark and the deployment call a diverged run.
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
    """Whether the returned start ended exactly where it began.

    The returned point is the best-scoring iterate of its start, the start
    itself included, so this is also true when the solve moved but never beat
    its starting point on the selection score.
    """
    best = int(optimization["best_start_index"])
    initial = np.asarray(optimization["initial_values_per_start"][best], dtype=float)
    final = np.asarray(optimization["final_values_per_start"][best], dtype=float)
    return bool(np.max(np.abs(final - initial)) <= STALL_TOLERANCE)


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
    """Diverged rollouts of one controller over the whole tuning set.

    Every (trajectory, realization) pair of the training and validation sets,
    under exactly the keys and start offsets the objective scored them with. A
    rollout diverged when its position error exceeds ``radius`` anywhere in its
    second half (the first half still carries the start offset being driven
    out) or is not finite.
    """
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
        # Poses are on the wheel-loop clock, the reference on the geometry
        # clock; sample the poses where the objective does.
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
            lines.append(f"{run}: the returned start never left its start point (stalled solve).")
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
