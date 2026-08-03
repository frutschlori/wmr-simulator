"""Benchmark harness: sequential vs alternating gain/trajectory optimization.

One JSON record per run, identical schema for every run, so runs are comparable
by construction and a missing key is a bug rather than a variant. The record is
what the study is read off; nothing here prints a conclusion.

Three things it measures that the optimization loss cannot answer on its own:

* **The plateau test** -- sweep each gain across its whole allowed range with
  the others held, and record ``(max - min) / min`` of the tuning loss. A
  trajectory on which the loss barely moves over the entire range of ``ky``
  cannot identify ``ky``, whatever its own loss says. Sequential design leaves
  ~5% on ``ky``; widening that is the point of designing the trajectory against
  the gains the tuner is actually converging to.
* **Held-out evaluation** -- every rollout in the loop uses one frozen
  realization bundle (common random numbers), so the training loss is scored on
  the same noise draws and start poses the design was optimized for. The
  held-out bundle (``EVAL_SEED``) is a fresh draw, and the problem's planner
  reference is a trajectory nobody designed.
* **Both information criteria** -- A-optimality ``trace(FIM^-1)`` (what the
  objective minimizes) and D-optimality ``-logdet(FIM)``, both computed from the
  FIM *factor* (``fim.py``); the explicit FIM is never assembled, because
  squaring the condition number is what made the criteria return NaN in float32.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
import optax

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.objectives import (
    Realizations,
    closed_loop_objective_terms,
    make_realizations,
)
from wmr_simulator.gain_tuning.optimizers import (
    clip_optimizer_values,
    controller_gains_from_optimizer_values,
    controller_gains_to_optimizer_values,
)
from wmr_simulator.gain_tuning.pipeline import ControllerTuningPipeline, resolve_gain_robot_params
from wmr_simulator.joint_tuning.pipeline import (
    CONSTRAINT_COMPONENT_NAMES,
    GAIN_NAMES,
    MODE_ALTERNATING,
    JointTuningResult,
    run_joint_tuning,
)
from wmr_simulator.trajectory_optimization.constraints import (
    constraint_loss_components_from_reference_states,
)
from wmr_simulator.trajectory_optimization.fim import (
    fim_eigenvalues,
    logdet_criterion,
    marginal_information,
    trace_inverse_criterion,
)


# Distinct from any training seed on purpose: scoring the tuned gains on the
# bundle they were tuned against measures how well the optimizer fit its own
# common random numbers, not how well the gains work.
EVAL_SEED = 1234
# The held-out set is *fixed*, not a mirror of the run's own configuration.
# Drawing it at the run's ``num_realizations`` / ``num_trajectories`` would make
# the realization sweep score its conditions against a different held-out set at
# every point -- measured: R=2 and R=8 came out at 0.055 and 0.053 m against
# R=4's 0.064 m purely because their evaluation bundles held different start
# offsets. Every run is now scored on the same 8 realizations and the same 6
# random trajectories, so held-out numbers compare across the whole study.
EVAL_NUM_REALIZATIONS = 8
EVAL_NUM_TRAJECTORIES = 6
# Points per gain in the plateau sweep. Odd, so the sweep contains the range
# midpoint; 31 resolves a plateau's edges without costing more than a few
# seconds at R=4.
NUM_PLATEAU_POINTS = 31


# --------------------------------------------------------------------- helpers
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - metadata only
        return "unknown"


def _file_sha(path: str) -> str:
    try:
        with open(path, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()[:16]
    except OSError:  # pragma: no cover - metadata only
        return "unknown"


def _to_jsonable(value):
    """NumPy/JAX arrays and scalars to plain JSON, recursively."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _to_jsonable(np.asarray(value).tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the rank correlation."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Rank correlation, so two criteria on wildly different scales (a variance
    sum against a log-determinant) are compared on the only thing a design
    criterion has to get right: the ordering."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    finite = np.isfinite(first) & np.isfinite(second)
    if finite.sum() < 3:
        return float("nan")
    first_ranks = _ranks(first[finite])
    second_ranks = _ranks(second[finite])
    if first_ranks.std() == 0.0 or second_ranks.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


# ------------------------------------------------------------------- criteria
def fim_report(
    trajectory_pipeline,
    reference_states_batch: jnp.ndarray,
    gains: jnp.ndarray,
    realizations: Realizations,
) -> dict:
    """Both design criteria, the spectrum and the per-gain marginal information,
    per trajectory and for the pooled design.

    Pooling stacks the factors rather than summing FIMs: ``[J1; J2]^T [J1; J2] =
    J1^T J1 + J2^T J2`` is the FIM of running both trajectories, in factored
    form, so the pooled criteria are computed without ever assembling a FIM.
    """
    factors = [
        trajectory_pipeline.compute_fim_factor(
            reference_states=reference_states,
            gains=gains,
            realizations=realizations,
        )
        for reference_states in reference_states_batch
    ]
    pooled_factor = jnp.concatenate(factors, axis=0) / np.sqrt(len(factors))

    def one(factor) -> dict:
        eigenvalues = np.asarray(fim_eigenvalues(factor), dtype=float)
        return {
            "a_criterion": float(trace_inverse_criterion(factor)),
            "logdet_criterion": float(logdet_criterion(factor)),
            "eigenvalues": eigenvalues.tolist(),
            "condition_number": float(eigenvalues[0] / max(eigenvalues[-1], 1e-300)),
            "marginal_information": dict(
                zip(GAIN_NAMES, np.asarray(marginal_information(factor), dtype=float).tolist())
            ),
        }

    return {
        "pooled": one(pooled_factor),
        "per_trajectory": [one(factor) for factor in factors],
    }


# ---------------------------------------------------------------- gain losses
def make_gain_loss(
    gain_pipeline,
    reference_states_batch: jnp.ndarray,
    realizations: Realizations,
    weights: dict,
):
    """``gains -> mean tuning loss`` over a trajectory batch and a realization
    bundle -- exactly the quantity the gain block minimizes, so the plateau test
    and the held-out evaluation are the tuner's own yardstick and not a
    look-alike."""

    def terms_for_reference(gains, reference_states):
        return closed_loop_objective_terms(
            gain_pipeline,
            gains,
            realizations.robot_keys,
            realizations.estimator_keys,
            velocity_tracking_weight=weights["velocity_tracking_weight"],
            input_weight=weights["input_weight"],
            input_delta_weight=weights["input_delta_weight"],
            omega_delta_weight=weights["omega_delta_weight"],
            reference_states=reference_states,
            initial_pose_offsets=realizations.start_offsets,
        )

    @jax.jit
    def loss(gains):
        terms = jax.vmap(terms_for_reference, in_axes=(None, 0))(gains, reference_states_batch)
        return jnp.sum(jnp.mean(terms, axis=0))

    return loss


def held_out_trajectories(
    trajectory_pipeline,
    num_control_points: int,
    num_trajectories: int,
    seed: int = EVAL_SEED,
) -> jnp.ndarray:
    """Unoptimized trajectories from the same B-spline family, drawn off a seed
    no run trains on.

    This is the generalization test that matters: gains tuned on a design that
    was itself optimized against those gains could be fitting the design rather
    than the plant, and only a trajectory nobody designed can tell the
    difference.
    """
    decision_variables = jnp.stack(
        trajectory_pipeline.initial_decision_variable_candidates(
            num_control_points=num_control_points,
            num_trajectories=num_trajectories,
            seed=seed,
        ),
        axis=0,
    )
    return jax.vmap(
        lambda variables: trajectory_pipeline.reference_states_from_control_points(
            trajectory_pipeline.control_points_from_decision_variables(variables)
        )
    )(decision_variables)


def pose_rmse(
    gain_pipeline,
    gains: jnp.ndarray,
    reference_states: jnp.ndarray,
    realizations: Realizations,
) -> jnp.ndarray:
    """Closed-loop position RMSE against the reference, averaged over the
    realization bundle. Scored on the *true* poses, not the noisy estimates."""
    reference_positions = reference_states[1:, :2]
    indices = jnp.arange(
        gain_pipeline.inner_steps_per_geometry_step,
        reference_states.shape[0] * gain_pipeline.inner_steps_per_geometry_step,
        gain_pipeline.inner_steps_per_geometry_step,
        dtype=jnp.int32,
    )
    reference_start = gain_pipeline.initial_reference_pose(reference_states)

    def one(robot_key, estimator_key, offset):
        predicted_log = gain_pipeline.run_closed_loop(
            gain_pipeline.robot_params,
            controller_gains=gains,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
            initial_pose=reference_start + offset,
        )
        error = predicted_log.pose.true_states[indices][:, :2] - reference_positions
        return jnp.sqrt(jnp.mean(jnp.sum(error**2, axis=1)))

    return jnp.mean(
        jax.vmap(one)(realizations.robot_keys, realizations.estimator_keys, realizations.start_offsets)
    )


# ------------------------------------------------------------- plateau test
def _plateau_grid(gain_index: int, k_min_stab: float, k_max_stab: float, k_max_rest: float,
                  num_points: int) -> np.ndarray:
    """The range the tuner may search, sampled the way it searches it: log for
    the four gains that must stay strictly positive, sqrt for ``kimotor``, which
    is the one gain allowed to be exactly 0."""
    if gain_index == 4:
        return np.linspace(0.0, np.sqrt(k_max_rest), num_points) ** 2
    return np.exp(np.linspace(np.log(k_min_stab), np.log(k_max_stab), num_points))


def plateau_test(
    loss_fn,
    gains: jnp.ndarray,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
    num_points: int = NUM_PLATEAU_POINTS,
) -> dict:
    """Sweep each gain across its allowed range with the others held fixed.

    ``(max - min) / min`` of the tuning loss is how much of the loss the whole
    range of that gain is worth. Near 0 means the trajectory cannot identify the
    gain at all -- the tuner's answer for it is then noise, whatever its
    optimization loss did. Sequential design leaves ~5% on ``ky``.
    """
    gains = jnp.asarray(gains, dtype=jnp.float32)
    report = {}
    for gain_index, name in enumerate(GAIN_NAMES):
        grid = _plateau_grid(gain_index, k_min_stab, k_max_stab, k_max_rest, num_points)
        losses = np.asarray(
            [float(loss_fn(gains.at[gain_index].set(float(value)))) for value in grid], dtype=float
        )
        finite = np.isfinite(losses)
        if finite.sum() < 2 or losses[finite].min() <= 0.0:
            report[name] = {
                "relative_range": float("nan"),
                "grid": grid.tolist(),
                "losses": losses.tolist(),
                "argmin": float("nan"),
            }
            continue
        minimum = float(losses[finite].min())
        report[name] = {
            "relative_range": (float(losses[finite].max()) - minimum) / minimum,
            # The relative range is dominated by whatever the far end of the
            # range does -- a gain large enough to destabilize the loop makes it
            # huge without saying anything about identifiability near the
            # optimum. The plateau fraction is the complementary reading: the
            # share of the swept range on which the loss is within 5% of its
            # best, i.e. how wide the band of gains the trajectory cannot tell
            # apart is. Smaller is better identified.
            "plateau_fraction": float(np.mean(losses[finite] <= 1.05 * minimum)),
            "grid": grid.tolist(),
            "losses": losses.tolist(),
            "argmin": float(grid[finite][int(np.argmin(losses[finite]))]),
        }
    return report


# --------------------------------------------------------- per-trajectory tune
def tune_gains_per_trajectory(
    gain_pipeline,
    reference_states_batch: jnp.ndarray,
    realizations: Realizations,
    weights: dict,
    initial_gains: jnp.ndarray,
    num_steps: int,
    learning_rate: float,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
) -> jnp.ndarray:
    """Tune an independent gain vector on each trajectory alone.

    This is what makes the criterion comparison a *prediction* test: each
    trajectory's criterion value is a claim about how well gains tuned on that
    trajectory alone will do, and this produces the gains whose held-out
    performance settles it. Vmapped over trajectories -- they are independent
    problems, so one batched Adam does all of them.
    """
    initial_values = controller_gains_to_optimizer_values(
        initial_gains, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
    )
    optimizer = optax.adam(learning_rate)

    def loss(values, reference_states):
        gains = controller_gains_from_optimizer_values(
            values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
        )
        return jnp.sum(
            closed_loop_objective_terms(
                gain_pipeline,
                gains,
                realizations.robot_keys,
                realizations.estimator_keys,
                velocity_tracking_weight=weights["velocity_tracking_weight"],
                input_weight=weights["input_weight"],
                input_delta_weight=weights["input_delta_weight"],
                omega_delta_weight=weights["omega_delta_weight"],
                reference_states=reference_states,
                initial_pose_offsets=realizations.start_offsets,
            )
        )

    @jax.jit
    def run(reference_states):
        def step(carry, _):
            values, optimizer_state = carry
            gradients = jax.grad(loss)(values, reference_states)
            updates, optimizer_state = optimizer.update(gradients, optimizer_state, values)
            values = clip_optimizer_values(optax.apply_updates(values, updates))
            return (values, optimizer_state), None

        (values, _), _ = jax.lax.scan(
            step, (initial_values, optimizer.init(initial_values)), None, length=num_steps
        )
        return controller_gains_from_optimizer_values(
            values, k_min_stab=k_min_stab, k_max_stab=k_max_stab, k_max_rest=k_max_rest
        )

    return jax.vmap(run)(reference_states_batch)


def criterion_comparison(
    trajectory_pipeline,
    gain_pipeline,
    reference_states_batch: jnp.ndarray,
    gains: jnp.ndarray,
    training_realizations: Realizations,
    evaluation_realizations: Realizations,
    evaluation_reference_states: jnp.ndarray,
    weights: dict,
    k_min_stab: float,
    k_max_stab: float,
    k_max_rest: float,
    num_steps: int = 100,
    learning_rate: float = 1e-3,
) -> dict:
    """How the two criteria rank the trajectories, and how well each ranking
    predicts held-out gain-tuning performance.

    Both criteria are read off the same FIM factors, so the comparison is purely
    a comparison of criteria and not of two different rollouts.
    """
    report = fim_report(trajectory_pipeline, reference_states_batch, gains, training_realizations)
    a_criteria = np.asarray([entry["a_criterion"] for entry in report["per_trajectory"]])
    logdet_criteria = np.asarray([entry["logdet_criterion"] for entry in report["per_trajectory"]])

    tuned_gains = tune_gains_per_trajectory(
        gain_pipeline,
        reference_states_batch,
        training_realizations,
        weights,
        gains,
        num_steps=num_steps,
        learning_rate=learning_rate,
        k_min_stab=k_min_stab,
        k_max_stab=k_max_stab,
        k_max_rest=k_max_rest,
    )
    # Held out in both senses: a fresh realization bundle and trajectories no
    # optimizer in this run ever saw.
    held_out_loss = make_gain_loss(
        gain_pipeline, evaluation_reference_states, evaluation_realizations, weights
    )
    held_out_losses = np.asarray(
        [float(held_out_loss(one_tuned)) for one_tuned in tuned_gains], dtype=float
    )
    held_out_rmse = np.asarray(
        [
            float(
                np.mean(
                    [
                        float(pose_rmse(gain_pipeline, one_tuned, reference_states, evaluation_realizations))
                        for reference_states in evaluation_reference_states
                    ]
                )
            )
            for one_tuned in tuned_gains
        ],
        dtype=float,
    )
    return {
        "a_criterion": a_criteria.tolist(),
        "logdet_criterion": logdet_criteria.tolist(),
        "tuned_gains": np.asarray(tuned_gains, dtype=float).tolist(),
        "held_out_loss": held_out_losses.tolist(),
        "held_out_rmse": held_out_rmse.tolist(),
        "tune_steps": int(num_steps),
        # Both criteria are "smaller is better", as is the held-out loss, so a
        # criterion that predicts performance gives a *positive* rho.
        "spearman_a_vs_held_out_loss": spearman(a_criteria, held_out_losses),
        "spearman_logdet_vs_held_out_loss": spearman(logdet_criteria, held_out_losses),
        "spearman_a_vs_held_out_rmse": spearman(a_criteria, held_out_rmse),
        "spearman_logdet_vs_held_out_rmse": spearman(logdet_criteria, held_out_rmse),
        "spearman_a_vs_logdet": spearman(a_criteria, logdet_criteria),
    }


# ------------------------------------------------------------------ movement
def movement_report(trajectory_pipeline, result: JointTuningResult) -> dict:
    """How far the trajectories actually travelled, from the start and from the
    end of the warm start. The second is the one that answers "did the
    alternation do anything, or was the warm start the whole result"."""

    def control_points(decision_variables):
        return np.asarray(
            jax.vmap(trajectory_pipeline.control_points_from_decision_variables)(decision_variables),
            dtype=float,
        )

    def reference_path(decision_variables):
        return np.asarray(
            jax.vmap(trajectory_pipeline.reference_states_from_control_points)(
                jax.vmap(trajectory_pipeline.control_points_from_decision_variables)(decision_variables)
            )[..., :2],
            dtype=float,
        )

    initial = control_points(result.initial_decision_variables)
    warm_start = control_points(result.warm_start_decision_variables)
    final = control_points(result.state.decision_variables)
    initial_path = reference_path(result.initial_decision_variables)
    final_path = reference_path(result.state.decision_variables)

    def displacement(first, second):
        return np.linalg.norm(second - first, axis=-1)

    path_length = lambda path: float(
        np.mean(np.sum(np.linalg.norm(np.diff(path, axis=1), axis=-1), axis=1))
    )
    pairwise = [
        float(np.mean(np.linalg.norm(final_path[i] - final_path[j], axis=-1)))
        for i in range(len(final_path))
        for j in range(i + 1, len(final_path))
    ]
    return {
        "control_point_displacement_mean": float(np.mean(displacement(initial, final))),
        "control_point_displacement_max": float(np.max(displacement(initial, final))),
        "control_point_displacement_from_warm_start_mean": float(
            np.mean(displacement(warm_start, final))
        ),
        "control_point_displacement_from_warm_start_max": float(
            np.max(displacement(warm_start, final))
        ),
        "reference_path_displacement_mean": float(
            np.mean(np.linalg.norm(final_path - initial_path, axis=-1))
        ),
        "path_length_initial": path_length(initial_path),
        "path_length_final": path_length(final_path),
        "trajectory_pairwise_diversity": float(np.mean(pairwise)) if pairwise else float("nan"),
    }


def constraint_report(trajectory_pipeline, reference_states_batch, smooth_max_beta: float = 20.0) -> dict:
    """Per-component constraint loss and the fractional over-limit behind it.

    The components are ``weight * g**2`` at unit weights, so the fractional
    violation reads straight back off the square root -- the number that decides
    whether an optimized trajectory is feasible on the real robot or has quietly
    bought its information by exceeding the limits.
    """
    limits = trajectory_pipeline.motion_limits()
    weights = trajectory_pipeline.constraint_weights()
    components = [
        constraint_loss_components_from_reference_states(
            reference_states=reference_states,
            dt=trajectory_pipeline.problem.dt,
            limits=limits,
            weights=weights,
            smooth_max_beta=smooth_max_beta,
        )
        for reference_states in reference_states_batch
    ]
    per_component = {
        name: np.asarray([float(entry[name]) for entry in components], dtype=float)
        for name in CONSTRAINT_COMPONENT_NAMES
    }
    stacked = np.stack([per_component[name] for name in CONSTRAINT_COMPONENT_NAMES], axis=1)
    total = stacked.sum(axis=1)
    return {
        "component_loss_mean": {
            name: float(values.mean()) for name, values in per_component.items()
        },
        "component_share": {
            name: float(values.sum() / total.sum()) if total.sum() > 0 else 0.0
            for name, values in per_component.items()
        },
        "max_fractional_violation_per_trajectory": np.sqrt(
            np.maximum(stacked, 0.0).max(axis=1)
        ).tolist(),
        "max_fractional_violation": float(np.sqrt(np.maximum(stacked, 0.0).max())),
        "total_constraint_loss_mean": float(total.mean()),
    }


# ------------------------------------------------------------------- the run
def benchmark_run(
    problem_path: str = "problems/pololu_gains.yaml",
    *,
    mode: str = MODE_ALTERNATING,
    stage: str = "adhoc",
    out_dir: str = "results/joint_tuning_benchmark",
    plateau: bool = True,
    criteria: bool = True,
    criterion_tune_steps: int = 100,
    plot_dir: str | None = None,
    write: bool = True,
    **joint_tuning_kwargs,
) -> dict:
    """Run one configuration and return (and by default write) its record."""
    total_start = time.time()
    result = run_joint_tuning(problem_path, mode=mode, **joint_tuning_kwargs)
    evaluation_start = time.time()

    trajectory_pipeline = result.trajectory_pipeline
    gain_pipeline = result.gain_pipeline
    config = result.config
    weights = {
        name: float(joint_tuning_kwargs.get(name, GAIN_TUNING_DEFAULTS[name]))
        for name in (
            "velocity_tracking_weight",
            "input_weight",
            "input_delta_weight",
            "omega_delta_weight",
        )
    }
    reference_states_batch = result.reference_states
    training_realizations = result.realizations
    evaluation_realizations = make_realizations(
        jax.random.PRNGKey(EVAL_SEED),
        jax.random.PRNGKey(EVAL_SEED + 1),
        EVAL_NUM_REALIZATIONS,
        float(config["init_offset_radius"]),
        float(config["init_offset_angle"]),
    )

    stock_gains = jnp.asarray(trajectory_pipeline.controller_gains, dtype=jnp.float32)
    gain_vectors = {"stock": stock_gains, "tuned": result.gains}
    held_out_reference_batch = held_out_trajectories(
        trajectory_pipeline,
        num_control_points=int(config["num_control_points"]),
        num_trajectories=EVAL_NUM_TRAJECTORIES,
    )

    # Three families, because "held out" means two different things here.
    # ``designed``: the run's own trajectories under a fresh realization bundle
    # -- generalization over conditions. ``random``: unoptimized trajectories
    # from the same B-spline family drawn off EVAL_SEED -- generalization over
    # trajectories, and the headline number. ``planner``: the problem's fixed
    # waypoint reference, reported but *not* a discriminating test, because it
    # asks for |omega| up to 19 rad/s against an omega_max of 10 and is
    # therefore untrackable by any gains.
    evaluation_sets = {
        "designed": (reference_states_batch, evaluation_realizations),
        "random": (held_out_reference_batch, evaluation_realizations),
        "planner": (gain_pipeline.reference_states[None, ...], evaluation_realizations),
    }
    training_loss = make_gain_loss(
        gain_pipeline, reference_states_batch, training_realizations, weights
    )
    held_out_losses = {
        name: make_gain_loss(gain_pipeline, batch, bundle, weights)
        for name, (batch, bundle) in evaluation_sets.items()
    }

    quality = {
        name: {
            "gains": np.asarray(vector, dtype=float).tolist(),
            "training_loss": float(training_loss(vector)),
            **{
                f"held_out_loss_{set_name}": float(loss_fn(vector))
                for set_name, loss_fn in held_out_losses.items()
            },
            **{
                f"held_out_rmse_{set_name}": float(
                    np.mean(
                        [
                            float(pose_rmse(gain_pipeline, vector, reference_states, bundle))
                            for reference_states in batch
                        ]
                    )
                )
                for set_name, (batch, bundle) in evaluation_sets.items()
            },
        }
        for name, vector in gain_vectors.items()
    }
    held_out_designed_loss = held_out_losses["designed"]

    record = {
        "meta": {
            "stage": stage,
            "git_sha": _git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "jax_version": jax.__version__,
            "problem_yaml_sha": _file_sha(problem_path),
            "config": _to_jsonable(config),
            "objective_weights": weights,
            "eval_seed": EVAL_SEED,
        },
        "timing": _to_jsonable(
            {
                **result.timing,
                "eval_s": None,  # filled in below
                "total_s": None,
            }
        ),
        "history": _to_jsonable(
            {
                name: result.history[name]
                for name in (
                    "gain_loss_pre", "gain_loss_post", "trajectory_loss_pre",
                    "trajectory_loss_post", "fim_loss", "constraint_loss",
                    "max_fractional_violation", "gains", "constraint_components",
                )
            }
        ),
        "stability": _to_jsonable(
            {
                "uphill_fraction_gain": result.history["uphill_fraction_gain"],
                "uphill_fraction_trajectory": result.history["uphill_fraction_trajectory"],
                "uphill_fraction_joint": result.history["uphill_fraction_joint"],
                "converged_at_round": result.history["converged_at_round"],
                "gain_loss_slope_last_100": _slope(result.history["gain_loss_pre"], 100),
            }
        ),
        "movement": _to_jsonable(movement_report(trajectory_pipeline, result)),
        "constraints": _to_jsonable(
            constraint_report(trajectory_pipeline, reference_states_batch)
        ),
        "fim": _to_jsonable(
            fim_report(trajectory_pipeline, reference_states_batch, result.gains, training_realizations)
        ),
        "quality": _to_jsonable(quality),
        "result": _to_jsonable(
            {
                "final_gains": dict(zip(GAIN_NAMES, np.asarray(result.gains, dtype=float).tolist())),
                "stock_gains": dict(zip(GAIN_NAMES, np.asarray(stock_gains, dtype=float).tolist())),
                "start_offsets": result.start_offsets,
                "reference_states": reference_states_batch,
            }
        ),
    }

    if plateau:
        record["plateau"] = {
            "training": _to_jsonable(
                plateau_test(
                    training_loss,
                    result.gains,
                    config["k_min_stab"],
                    config["k_max_stab"],
                    config["k_max_rest"],
                )
            ),
            "held_out": _to_jsonable(
                plateau_test(
                    held_out_designed_loss,
                    result.gains,
                    config["k_min_stab"],
                    config["k_max_stab"],
                    config["k_max_rest"],
                )
            ),
        }
    if criteria:
        record["criteria"] = _to_jsonable(
            criterion_comparison(
                trajectory_pipeline,
                gain_pipeline,
                reference_states_batch,
                result.gains,
                training_realizations,
                evaluation_realizations,
                held_out_reference_batch,
                weights,
                config["k_min_stab"],
                config["k_max_stab"],
                config["k_max_rest"],
                num_steps=criterion_tune_steps,
            )
        )

    if plot_dir is not None:
        # Imported here so the harness stays usable headless: matplotlib is only
        # needed when figures are actually asked for.
        from wmr_simulator.visualization.joint_tuning import (
            plot_joint_tuning_evaluation,
            plot_joint_tuning_history,
        )

        os.makedirs(plot_dir, exist_ok=True)
        stem = f"{stage}_{config['mode']}_seed{config['seed']}"
        training_path, validation_path = plot_joint_tuning_evaluation(
            result,
            held_out_reference_batch,
            evaluation_realizations,
            out_prefix=stem,
            out_dir=plot_dir,
        )
        record["artifacts"] = {
            "history": plot_joint_tuning_history(
                result.history, out_path=os.path.join(plot_dir, f"{stem}_history.pdf")
            ),
            "training_trajectories": training_path,
            "validation_trajectories": validation_path,
        }

    record["timing"]["eval_s"] = time.time() - evaluation_start
    record["timing"]["total_s"] = time.time() - total_start
    if write:
        record["meta"]["path"] = write_record(record, out_dir)
    return record


def _slope(values, window: int) -> float:
    """Least-squares slope of the last ``window`` finite entries -- the direct
    test of the non-stationarity failure mode, where the gain block chases a
    target the trajectory block keeps moving."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)][-window:]
    if len(values) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def record_name(record: dict) -> str:
    config = record["meta"]["config"]
    payload = json.dumps(config, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{record['meta']['stage']}_{config['mode']}_{digest}_seed{config['seed']}.json"


def write_record(record: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, record_name(record))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=1)
    return path


def load_runs(out_dir: str) -> list[dict]:
    records = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(out_dir, name), "r", encoding="utf-8") as file:
            records.append(json.load(file))
    return records


def comparison_row(record: dict) -> dict:
    """One record flattened to the numbers a sweep is decided on.

    Deliberately small: a comparison that needs more than this is a comparison
    of something the sweep was not designed to answer.
    """
    config = record["meta"]["config"]
    plateau = record.get("plateau", {})
    criteria = record.get("criteria", {})
    quality = record["quality"]

    def plateau_value(family: str, name: str, field: str):
        return plateau.get(family, {}).get(name, {}).get(field, float("nan"))

    return {
        "mode": config["mode"],
        "seed": config["seed"],
        "R": config["num_realizations"],
        "T": config["num_trajectories"],
        "rounds": config["num_rounds"],
        "traj_lr": config["trajectory_learning_rate"],
        "gain_lr": config["gain_learning_rate"],
        "offsets": config["start_offset_mode"],
        # Records written before the criterion switch existed were all
        # A-optimality; naming that is more useful than dropping the column.
        "criterion": config.get("criterion", "a-optimality"),
        "wheel_lp_tau": config["wheel_lp_tau"],
        "s_per_round": record["timing"]["seconds_per_round"],
        "uphill_gain": record["stability"]["uphill_fraction_gain"],
        "uphill_traj": record["stability"]["uphill_fraction_trajectory"],
        "uphill_joint": record["stability"]["uphill_fraction_joint"],
        "gain_slope_100": record["stability"]["gain_loss_slope_last_100"],
        "a_criterion": record["fim"]["pooled"]["a_criterion"],
        "logdet_criterion": record["fim"]["pooled"]["logdet_criterion"],
        "fim_condition": record["fim"]["pooled"]["condition_number"],
        "max_violation": record["constraints"]["max_fractional_violation"],
        "cp_move_from_warm_start": record["movement"][
            "control_point_displacement_from_warm_start_mean"
        ],
        "plateau_ky": plateau_value("training", "ky", "relative_range"),
        "plateau_ky_width": plateau_value("training", "ky", "plateau_fraction"),
        "plateau_kx": plateau_value("training", "kx", "relative_range"),
        "plateau_kx_width": plateau_value("training", "kx", "plateau_fraction"),
        "train_loss": quality["tuned"]["training_loss"],
        "held_out_loss_random": quality["tuned"]["held_out_loss_random"],
        "held_out_rmse_random": quality["tuned"]["held_out_rmse_random"],
        "held_out_rmse_designed": quality["tuned"]["held_out_rmse_designed"],
        "stock_held_out_rmse_random": quality["stock"]["held_out_rmse_random"],
        "rho_a": criteria.get("spearman_a_vs_held_out_loss", float("nan")),
        "rho_logdet": criteria.get("spearman_logdet_vs_held_out_loss", float("nan")),
        "rho_a_vs_logdet": criteria.get("spearman_a_vs_logdet", float("nan")),
        "kx": record["result"]["final_gains"]["kx"],
        "ky": record["result"]["final_gains"]["ky"],
        "kth": record["result"]["final_gains"]["kth"],
        "kpmotor": record["result"]["final_gains"]["kpmotor"],
        "kimotor": record["result"]["final_gains"]["kimotor"],
    }


def comparison_table(records: list[dict], columns: tuple[str, ...] | None = None) -> str:
    rows = [comparison_row(record) for record in records]
    if not rows:
        return "(no records)"
    columns = tuple(rows[0]) if columns is None else columns
    widths = {
        name: max(len(name), *(len(_cell(row.get(name))) for row in rows)) for name in columns
    }
    header = "  ".join(name.rjust(widths[name]) for name in columns)
    lines = [header, "-" * len(header)]
    lines.extend(
        "  ".join(_cell(row.get(name)).rjust(widths[name]) for name in columns) for row in rows
    )
    return "\n".join(lines)


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        return f"{value:.4g}"
    return str(value)


def summarize(record: dict) -> str:
    """The handful of numbers a sweep row is read by."""
    config = record["meta"]["config"]
    plateau = record.get("plateau", {}).get("training", {})
    return (
        f"{config['mode']:<12} seed {config['seed']} R{config['num_realizations']} "
        f"T{config['num_trajectories']} lr {config['trajectory_learning_rate']:.0e}/"
        f"{config['gain_learning_rate']:.0e} lp {config['wheel_lp_tau']:.3g} | "
        f"{record['timing']['seconds_per_round']:.3f} s/round | "
        f"uphill g {record['stability']['uphill_fraction_gain']:.3f} "
        f"t {record['stability']['uphill_fraction_trajectory']:.3f} "
        f"j {record['stability']['uphill_fraction_joint']:.3f} | "
        f"plateau ky {plateau.get('ky', {}).get('relative_range', float('nan')):.4g}"
        f"/{plateau.get('ky', {}).get('plateau_fraction', float('nan')):.3f} | "
        f"held-out random {record['quality']['tuned']['held_out_loss_random']:.5f} "
        f"rmse {record['quality']['tuned']['held_out_rmse_random']:.5f} | "
        f"A {record['fim']['pooled']['a_criterion']:.4e} "
        f"logdet {record['fim']['pooled']['logdet_criterion']:.4f} "
        f"cond {record['fim']['pooled']['condition_number']:.3e}"
    )
