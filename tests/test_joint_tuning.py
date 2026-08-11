"""Joint (alternating) gain/trajectory tuning.

Everything here runs at 2 realizations, 4 control points, 2 trajectories and 2
rounds, which is the smallest configuration that still exercises both blocks.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS
from wmr_simulator.gain_tuning.objectives import closed_loop_objective_terms, make_realizations
from wmr_simulator.joint_tuning.pipeline import run_joint_tuning
from wmr_simulator.trajectory_optimization.start_offsets import (
    START_OFFSET_MODE_MASKS,
    inverse_squash_start_offsets,
    normalize_start_offset_mode,
    resolve_start_offsets,
    squash_start_offsets,
    start_offset_mask,
    static_start_offsets,
)
from wmr_simulator.trajectory_optimization.constraints import constraint_loss_from_reference_states
from wmr_simulator.trajectory_optimization.objectives import fim_loss, trajectory_objective
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

PROBLEM = "problems/pololu_gains.yaml"
NUM_REALIZATIONS = 2
NUM_TRAJECTORIES = 2
NUM_CONTROL_POINTS = 4
OFFSET_RADIUS = 0.2
OFFSET_ANGLE = 0.2


@pytest.fixture(scope="module")
def realizations():
    return make_realizations(
        jax.random.PRNGKey(0), jax.random.PRNGKey(1), NUM_REALIZATIONS, OFFSET_RADIUS, OFFSET_ANGLE
    )


@pytest.fixture(scope="module")
def trajectory_pipeline(realizations):
    return TrajectoryOptimizationPipeline(
        PROBLEM, objective_mode="gain-tuning", realizations=realizations
    )


@pytest.fixture(scope="module")
def joint_result():
    return run_joint_tuning(
        PROBLEM,
        num_rounds=2,
        warm_start_rounds=0,
        num_trajectories=NUM_TRAJECTORIES,
        num_control_points=NUM_CONTROL_POINTS,
        num_realizations=NUM_REALIZATIONS,
        verbose=False,
    )


def test_shared_realizations_identical(joint_result, realizations, trajectory_pipeline):
    """Both sides must see the same bundle -- this is what the joint loop buys,
    and two independent draws is the bug it replaces."""
    assert trajectory_pipeline.realizations is realizations
    for designed, used in zip(
        joint_result.trajectory_pipeline.realizations, joint_result.realizations
    ):
        assert jnp.array_equal(designed, used)
    assert joint_result.realizations.robot_keys.shape == (NUM_REALIZATIONS, 2)
    assert joint_result.realizations.start_offsets.shape == (NUM_REALIZATIONS, 3)


def test_joint_objective_deterministic(trajectory_pipeline, realizations, joint_result):
    """Common random numbers: both objectives must be bitwise reproducible, jit
    and eager. Verified on the trajectory side before the realizations were
    unified; it has to survive the unification."""
    control_points = joint_result.control_points[0]
    gains = joint_result.gains

    def trajectory_loss(control_points):
        return trajectory_pipeline.fim_loss_from_control_points(
            control_points, gains=gains, realizations=realizations
        )

    eager = [float(trajectory_loss(control_points)) for _ in range(2)]
    jitted = float(jax.jit(trajectory_loss)(control_points))
    assert eager[0] == eager[1]
    assert eager[0] == pytest.approx(jitted, rel=1e-5)

    gain_pipeline = joint_result.gain_pipeline
    reference_states = joint_result.reference_states[0]

    def gain_loss(gains):
        return jnp.sum(
            closed_loop_objective_terms(
                gain_pipeline,
                gains,
                realizations.robot_keys,
                realizations.estimator_keys,
                reference_states=reference_states,
                initial_pose_offsets=realizations.start_offsets,
            )
        )

    gain_eager = [float(gain_loss(gains)) for _ in range(2)]
    assert gain_eager[0] == gain_eager[1]
    assert gain_eager[0] == pytest.approx(float(jax.jit(gain_loss)(gains)), rel=1e-5)


def test_pipeline_objective_matches_its_terms(trajectory_pipeline, realizations, joint_result):
    """The joint loop writes the trajectory objective out inline so the FIM and
    constraint terms fall out as aux; pin that against the pipeline's own."""
    control_points = trajectory_pipeline.clamp_control_points(joint_result.control_points[0])
    reference_states = trajectory_pipeline.reference_states_from_control_points(control_points)
    fim_factor = trajectory_pipeline.compute_fim_factor(
        reference_states=reference_states, gains=joint_result.gains, realizations=realizations
    )
    constraint = constraint_loss_from_reference_states(
        reference_states=reference_states,
        dt=trajectory_pipeline.problem.dt,
        limits=trajectory_pipeline.motion_limits(),
        weights=trajectory_pipeline.constraint_weights(),
        smooth_max_beta=20.0,
    )
    inlined = float(jnp.log(fim_loss(fim_factor)) + constraint / 0.05)
    from_pipeline = float(
        trajectory_pipeline.fim_loss_from_control_points(
            control_points,
            gains=joint_result.gains,
            realizations=realizations,
            constraint_violation_tolerance=0.05,
        )
    )
    assert inlined == pytest.approx(from_pipeline, rel=1e-5)


@pytest.mark.parametrize("mode", sorted(START_OFFSET_MODE_MASKS))
def test_offset_mode_masks(mode):
    """Frozen coordinates get exactly zero gradient; free ones get a finite,
    non-zero one."""
    mask = start_offset_mask(mode)
    frozen = static_start_offsets(NUM_REALIZATIONS, OFFSET_RADIUS, OFFSET_ANGLE)
    free = jnp.asarray([[0.3, -0.4, 0.5], [-0.2, 0.1, -0.6]], dtype=jnp.float32)

    def scalar(free_offsets):
        offsets = resolve_start_offsets(
            free_offsets, frozen, mask, OFFSET_RADIUS, OFFSET_ANGLE
        )
        return jnp.sum(offsets**2 + offsets)

    gradient = np.asarray(jax.grad(scalar)(free), dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient[:, ~mask_array] == 0.0)
    if mask_array.any():
        assert np.all(np.abs(gradient[:, mask_array]) > 0.0)


def test_offset_squash_bounds():
    """The squash must stay inside the feasible set for any input, and be
    differentiable at u = 0 (the finding-#2 bug class: a new decision variable
    with a 0/0 in it)."""
    extreme = jnp.asarray(
        [[1e3, -1e3, 1e3], [-1e3, 1e3, -1e3], [0.0, 0.0, 0.0], [1e-8, 1e-8, 1e-8]],
        dtype=jnp.float32,
    )
    offsets = np.asarray(
        squash_start_offsets(extreme, OFFSET_RADIUS, OFFSET_ANGLE), dtype=float
    )
    assert np.all(np.linalg.norm(offsets[:, :2], axis=1) <= OFFSET_RADIUS + 1e-6)
    assert np.all(np.abs(offsets[:, 2]) <= OFFSET_ANGLE + 1e-6)
    assert np.all(offsets[2] == 0.0)

    def scalar(free_offsets):
        return jnp.sum(squash_start_offsets(free_offsets, OFFSET_RADIUS, OFFSET_ANGLE))

    gradient = np.asarray(jax.grad(scalar)(jnp.zeros((2, 3), dtype=jnp.float32)), dtype=float)
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient != 0.0)


def test_offset_squash_round_trip():
    """The optimizing modes start at the frozen offsets, which requires the
    inverse map to actually invert."""
    offsets = static_start_offsets(4, OFFSET_RADIUS, OFFSET_ANGLE)
    free = inverse_squash_start_offsets(offsets, OFFSET_RADIUS, OFFSET_ANGLE)
    recovered = np.asarray(squash_start_offsets(free, OFFSET_RADIUS, OFFSET_ANGLE), dtype=float)
    # static offsets sit exactly on the boundary, which the inverse pulls to 99%
    # of it; the direction must survive, the magnitude is allowed to shrink.
    assert np.allclose(recovered, np.asarray(offsets, dtype=float), atol=0.02)


def test_trajectory_adam_state_carries_across_rounds(joint_result):
    """The *trajectory* optimizer is built once and threaded through; a fresh
    state each round would re-pay Adam's bias-correction warmup every round. The
    gain block deliberately has no such state -- it re-solves from scratch each
    round (see joint_tuning.gain_solvers), so `gain_opt_state` is a placeholder.
    """
    state = joint_result.state
    trajectory_leaves = [
        np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state.trajectory_opt_state)
    ]
    assert any(np.any(leaf != 0.0) for leaf in trajectory_leaves)
    assert jax.tree_util.tree_leaves(state.gain_opt_state) == []

    history = joint_result.history
    # Not "the gains differ between rounds": a quasi-Newton block reaches its
    # conditional optimum inside round 0 and correctly returns the same point in
    # round 1 when the trajectories have barely moved. That assertion was a
    # measure of Adam being slow. What matters is that the block ran at all.
    assert not np.allclose(
        history["gains"][0],
        np.asarray(joint_result.trajectory_pipeline.controller_gains, dtype=float),
    )
    assert history["fim_loss"].shape == (2, NUM_TRAJECTORIES)


def test_objective_normalization(trajectory_pipeline, joint_result):
    """``trajectory_objective`` is exactly ``log(fim_loss) + constraint / g_tol``,
    and the tolerance is the only scale knob on the penalty."""
    control_points = trajectory_pipeline.clamp_control_points(joint_result.control_points[0])
    reference_states = trajectory_pipeline.reference_states_from_control_points(control_points)
    # A violating reference: 40% over the speed limit on every sample.
    limits = trajectory_pipeline.motion_limits()
    violating = reference_states.at[:, 3].set(1.4 * limits["v_max"]).at[:, 4].set(0.0)
    fim_factor = jnp.eye(3, dtype=jnp.float32) * 2.0
    weights = trajectory_pipeline.constraint_weights()
    constraint = float(
        constraint_loss_from_reference_states(
            reference_states=violating,
            dt=trajectory_pipeline.problem.dt,
            limits=limits,
            weights=weights,
            smooth_max_beta=20.0,
        )
    )
    assert constraint > 0.0
    for tolerance in (0.02, 0.05, 0.2):
        total = float(
            trajectory_objective(
                fim_factor=fim_factor,
                reference_states=violating,
                dt=trajectory_pipeline.problem.dt,
                limits=limits,
                weights=weights,
                constraint_violation_tolerance=tolerance,
            )
        )
        expected = float(jnp.log(fim_loss(fim_factor))) + constraint / tolerance
        assert total == pytest.approx(expected, rel=1e-5)


def test_offsets_are_per_trajectory(joint_result):
    """Every trajectory designs its own start bundle, exactly as the standalone
    batch designer does and as the gain tuner reads them back."""
    assert joint_result.start_offsets.shape == (NUM_TRAJECTORIES, NUM_REALIZATIONS, 3)
    assert joint_result.state.free_offsets.shape == (NUM_TRAJECTORIES, NUM_REALIZATIONS, 3)
    # The root bundle keeps its own single draw: it is the noise roots plus the
    # point the per-trajectory offsets started from, not the design's answer.
    assert joint_result.realizations.start_offsets.shape == (NUM_REALIZATIONS, 3)


def test_warm_start_resumes_a_designed_set(tmp_path, joint_result):
    """A warm start reads back a design's control points *and* the offsets it
    was scored under, so the loop restarts exactly where the export left off
    rather than at a fresh line batch. The counts of trajectories, realizations
    and control points come from the directory, not from the arguments."""
    from wmr_simulator.trajectory_optimization.pipeline import (
        load_reference_states_exports,
        reference_states_export_payload,
    )
    import pickle

    export_dir = tmp_path / "designed"
    export_dir.mkdir()
    reference_states = np.asarray(joint_result.reference_states, dtype=float)
    start_offsets = np.asarray(joint_result.start_offsets, dtype=float)
    control_points = np.asarray(joint_result.control_points, dtype=float)
    for index, states in enumerate(reference_states):
        with open(export_dir / f"reference_states_{index:02d}.pkl", "wb") as file:
            pickle.dump(
                reference_states_export_payload(
                    states,
                    float(joint_result.trajectory_pipeline.problem.dt),
                    start_offsets=start_offsets[index],
                    control_points=control_points[index],
                ),
                file,
            )

    loaded = load_reference_states_exports(str(export_dir))
    assert loaded.reference_states.shape == reference_states.shape
    np.testing.assert_allclose(loaded.start_offsets, start_offsets, rtol=1e-6)
    np.testing.assert_allclose(loaded.control_points, control_points, rtol=1e-6)

    warm = run_joint_tuning(
        PROBLEM,
        num_rounds=1,
        # Deliberately wrong, and deliberately ignored: the design sets all of
        # these.
        warm_start_rounds=5,
        num_trajectories=NUM_TRAJECTORIES + 3,
        num_realizations=NUM_REALIZATIONS + 2,
        num_control_points=NUM_CONTROL_POINTS + 2,
        warm_start_trajectories_dir=str(export_dir),
        verbose=False,
    )
    assert warm.config["num_trajectories"] == NUM_TRAJECTORIES
    assert warm.config["num_realizations"] == NUM_REALIZATIONS
    assert warm.config["num_control_points"] == NUM_CONTROL_POINTS
    assert warm.config["warm_start_rounds"] == 0
    # The gain block ran from round 0: nothing was spent re-warming.
    assert np.isfinite(warm.history["gain_loss_pre"][0])
    # The loop resumed on the exported curve itself, not a reconstruction of it:
    # one Adam round moves it a little, nothing more.
    np.testing.assert_allclose(
        np.asarray(warm.initial_decision_variables, dtype=float).reshape(control_points.shape),
        control_points,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(warm.reference_states[:, :, :2], dtype=float),
        reference_states[:, :, :2],
        atol=5e-2,
    )


@pytest.mark.parametrize(
    "payload_kwargs, message",
    [
        # Only a gain-tuning-mode design can warm-start this loop; without the
        # offsets it says nothing about the gains.
        ({}, "no start offsets"),
        # The warm start keeps optimizing the curve, so the sampled states alone
        # are not enough -- it needs the decision variables.
        ({"start_offsets": np.zeros((2, 3))}, "no control points"),
    ],
)
def test_warm_start_rejects_incomplete_exports(tmp_path, payload_kwargs, message):
    from wmr_simulator.trajectory_optimization.pipeline import reference_states_export_payload
    import pickle

    export_dir = tmp_path / "incomplete"
    export_dir.mkdir()
    with open(export_dir / "reference_states_00.pkl", "wb") as file:
        pickle.dump(
            reference_states_export_payload(np.zeros((10, 8)), 0.02, **payload_kwargs), file
        )
    with pytest.raises(ValueError, match=message):
        run_joint_tuning(
            PROBLEM,
            num_rounds=1,
            warm_start_trajectories_dir=str(export_dir),
            verbose=False,
        )


def test_pinned_kimotor_scale_keeps_a_zero_design_point_informative(tmp_path, realizations):
    """Active learning inherits the tuned gains, and the tuner reliably returns
    kimotor = 0. Tied to the design point, the scale has no nominal value to use
    there and falls back to the search range, which leaves kimotor contributing
    almost nothing to trace(FIM^-1) -- the share that buys curvature, so the
    designs come out dull. Pinning the scale is what fixes that; the gain itself
    is never touched."""
    import yaml

    from wmr_simulator.trajectory_optimization.pipeline import (
        KIMOTOR_FIM_SCALE_FALLBACK,
        KIMOTOR_INDEX,
    )

    with open(PROBLEM, "r", encoding="utf-8") as file:
        problem_cfg = yaml.safe_load(file)
    nominal = float(problem_cfg["controller"]["gains"][KIMOTOR_INDEX])
    problem_cfg["controller"]["gains"][KIMOTOR_INDEX] = 0.0
    zeroed_problem = tmp_path / "kimotor_zero.yaml"
    with zeroed_problem.open("w", encoding="utf-8") as file:
        yaml.safe_dump(problem_cfg, file, sort_keys=False)

    tied = TrajectoryOptimizationPipeline(
        str(zeroed_problem), objective_mode="gain-tuning", realizations=realizations
    )
    assert tied.kimotor_fim_scale == pytest.approx(KIMOTOR_FIM_SCALE_FALLBACK)

    pinned = TrajectoryOptimizationPipeline(
        str(zeroed_problem),
        objective_mode="gain-tuning",
        realizations=realizations,
        kimotor_fim_scale=nominal,
    )
    assert pinned.kimotor_fim_scale == pytest.approx(nominal)
    # The scale is a property of the criterion, not of the plant: both design
    # against the problem's own gains, kimotor = 0 included.
    np.testing.assert_allclose(
        np.asarray(pinned.controller_gains), np.asarray(tied.controller_gains)
    )
    assert float(pinned.controller_gains[KIMOTOR_INDEX]) == 0.0

    def kimotor_share(pipeline):
        factor = pipeline.compute_fim_factor(reference_states=pipeline.reference_states)
        variances = np.diag(
            np.linalg.inv(np.asarray(factor).T @ np.asarray(factor) + 1e-6 * np.eye(factor.shape[1]))
        )
        return variances[KIMOTOR_INDEX] / variances.sum()

    # Measured at the stock problem with kimotor zeroed: 0.16% tied against
    # 3.1% pinned, i.e. ~19x. Not the ~38% a plant actually running kimotor = 5
    # gives -- the column is the sensitivity at 0 either way -- but the design
    # can see the gain again.
    tied_share, pinned_share = kimotor_share(tied), kimotor_share(pinned)
    assert tied_share < 0.005
    assert pinned_share > 10.0 * tied_share

    with pytest.raises(ValueError, match="positive"):
        TrajectoryOptimizationPipeline(
            PROBLEM, objective_mode="gain-tuning", realizations=realizations, kimotor_fim_scale=0.0
        )


def test_kimotor_fim_column_survives_zero(trajectory_pipeline):
    """kimotor is the one gain allowed to be exactly 0, so its FIM scale is a
    constant -- the problem's nominal kimotor -- not its own value. Relative
    scaling blanks the column as the tuner drives it to 0 and leaves
    A-optimality dominated by an unfixable term.

    Pinning the constant at the nominal value (rather than anything larger) is
    what keeps a design that never moves the gains bit-identical to relative
    scaling, and keeps the kimotor share of trace(FIM^-1) -- which is what buys
    curvature in the design -- where it was."""
    from wmr_simulator.trajectory_optimization.fim import (
        marginal_information,
        trace_inverse_criterion,
    )
    from wmr_simulator.trajectory_optimization.pipeline import KIMOTOR_INDEX

    from wmr_simulator.trajectory_optimization.pipeline import KIMOTOR_FIM_SCALE_FALLBACK

    gains = jnp.asarray(trajectory_pipeline.controller_gains, dtype=jnp.float32)
    nominal_kimotor = float(gains[KIMOTOR_INDEX])
    # The rule is "the problem's nominal kimotor, or the search-range fallback
    # when the yaml starts it at 0" -- asserting the first half alone made this
    # fail spuriously the moment a problem shipped kimotor: 0.0.
    expected_scale = nominal_kimotor if nominal_kimotor > 0.0 else KIMOTOR_FIM_SCALE_FALLBACK
    assert trajectory_pipeline.kimotor_fim_scale == pytest.approx(expected_scale)
    if nominal_kimotor <= 0.0:
        pytest.skip(
            "problem yaml ships kimotor: 0.0, so there is no nominal design point at "
            "which relative and constant scaling coincide"
        )
    # At the nominal design point the whole scaling vector is what relative
    # scaling would have produced, so nothing that holds the gains fixed moves.
    np.testing.assert_allclose(
        np.asarray(trajectory_pipeline.fim_parameter_scaling(gains)),
        np.asarray(jnp.maximum(gains, 1e-3)),
        rtol=1e-6,
    )
    zeroed = gains.at[KIMOTOR_INDEX].set(0.0)
    assert float(trajectory_pipeline.fim_parameter_scaling(zeroed)[KIMOTOR_INDEX]) == pytest.approx(
        nominal_kimotor
    )
    # The other four keep relative scaling.
    np.testing.assert_allclose(
        np.asarray(trajectory_pipeline.fim_parameter_scaling(gains)[:KIMOTOR_INDEX]),
        np.asarray(gains[:KIMOTOR_INDEX]),
        rtol=1e-6,
    )

    criteria, marginals = [], []
    for kimotor in (5.0, 0.0):
        factor = trajectory_pipeline.compute_fim_factor(
            reference_states=trajectory_pipeline.reference_states,
            gains=gains.at[KIMOTOR_INDEX].set(kimotor),
        )
        criteria.append(float(trace_inverse_criterion(factor)))
        marginals.append(float(marginal_information(factor)[KIMOTOR_INDEX]))
    # Driving kimotor to 0 must not blow the criterion up (it went 2.1e-3 -> 4.2,
    # i.e. 2000x, under relative scaling) and must not kill its own column.
    assert criteria[1] < 10.0 * criteria[0]
    assert marginals[1] > 0.1 * marginals[0]


def test_start_offset_mode_rejects_unknown():
    assert normalize_start_offset_mode("Optimize_Heading") == "optimize-heading"
    with pytest.raises(ValueError):
        normalize_start_offset_mode("optimise-heading")


# --------------------------------------------------------------------------
# Start offsets as decision variables of the *standalone* trajectory optimizer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def offset_optimizing_pipeline():
    # Drawn at the pipeline's own offset scale (which is what it squashes
    # against), so the free variables can sit exactly on the drawn offsets --
    # a bundle from a *wider* disk would be pulled back to the boundary and the
    # round trip below would be testing the clip, not the parametrization.
    return TrajectoryOptimizationPipeline(
        PROBLEM,
        objective_mode="gain-tuning",
        realizations=make_realizations(
            jax.random.PRNGKey(0),
            jax.random.PRNGKey(1),
            NUM_REALIZATIONS,
            float(GAIN_TUNING_DEFAULTS["init_offset_radius"]),
            float(GAIN_TUNING_DEFAULTS["init_offset_angle"]),
        ),
        start_offset_mode="optimize",
    )


def test_decision_vector_carries_offsets_only_when_optimized(
    trajectory_pipeline, offset_optimizing_pipeline
):
    control_points = trajectory_pipeline.initial_control_points(NUM_CONTROL_POINTS)
    frozen = trajectory_pipeline.decision_variables_from_control_points(control_points)
    free = offset_optimizing_pipeline.decision_variables_from_control_points(control_points)
    assert trajectory_pipeline.num_offset_variables == 0
    assert frozen.shape == (2 * NUM_CONTROL_POINTS,)
    assert free.shape == (2 * NUM_CONTROL_POINTS + 3 * NUM_REALIZATIONS,)
    # The curve half round-trips, and the offsets start at the drawn bundle, so
    # an optimizing run begins under exactly the frozen run's conditions.
    np.testing.assert_allclose(
        np.asarray(offset_optimizing_pipeline.control_points_from_decision_variables(free)),
        np.asarray(trajectory_pipeline.control_points_from_decision_variables(frozen)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(offset_optimizing_pipeline.start_offsets_from_decision_variables(free)),
        np.asarray(offset_optimizing_pipeline.realizations.start_offsets),
        atol=1e-4,
    )


def test_offset_variables_move_the_objective(offset_optimizing_pipeline):
    """The offset tail has to carry gradient, else the mode is decoration."""
    control_points = offset_optimizing_pipeline.initial_control_points(NUM_CONTROL_POINTS)
    decision_variables = offset_optimizing_pipeline.decision_variables_from_control_points(
        control_points
    )
    grads = jax.grad(offset_optimizing_pipeline.loss_from_decision_variables)(decision_variables)
    offset_grads = np.asarray(grads[-3 * NUM_REALIZATIONS:])
    assert np.all(np.isfinite(offset_grads))
    assert np.max(np.abs(offset_grads)) > 0.0


def test_offsets_stay_inside_their_feasible_set(offset_optimizing_pipeline):
    control_points = offset_optimizing_pipeline.initial_control_points(NUM_CONTROL_POINTS)
    extreme = jnp.concatenate(
        [
            jnp.ravel(offset_optimizing_pipeline.clamp_control_points(control_points)),
            jnp.full((3 * NUM_REALIZATIONS,), 50.0),
        ]
    )
    offsets = np.asarray(offset_optimizing_pipeline.start_offsets_from_decision_variables(extreme))
    radius = np.hypot(offsets[:, 0], offsets[:, 1])
    assert np.all(radius <= offset_optimizing_pipeline.offset_radius + 1e-6)
    assert np.all(np.abs(offsets[:, 2]) <= offset_optimizing_pipeline.offset_angle + 1e-6)


def test_identification_mode_rejects_offset_optimization():
    with pytest.raises(ValueError, match="gain-tuning"):
        TrajectoryOptimizationPipeline(
            PROBLEM, objective_mode="identification", start_offset_mode="optimize"
        )


# ---------------------------------------------------------------------------
# Best-iterate selection and the stopping rule.
#
# All three exist because neither block's own loss is a progress measure for the
# pair: each is scored against the other block's current iterate, so consecutive
# values answer different questions. See the module docstring of
# joint_tuning/pipeline.py.
# ---------------------------------------------------------------------------


def test_best_gains_are_selected_not_the_last_iterate(joint_result):
    """``gains`` is the best iterate on the frozen scoring set; ``final_gains``
    is what the loop happened to end on. Keeping only the latter is how a run
    that peaked at round 35 shipped round-350 gains eight times worse."""
    history = joint_result.history
    scores = np.asarray(history["gain_validation_loss"], dtype=float)
    finite = scores[np.isfinite(scores)]
    assert finite.size > 0
    # The recorded best is genuinely the minimum of what was scored, and the
    # round it is attributed to is one that actually ran.
    assert history["best_gain_score"] == pytest.approx(float(np.min(finite)))
    assert 0 <= history["best_gain_round"] < len(scores)
    assert joint_result.gains.shape == joint_result.final_gains.shape == (5,)


def test_stopping_rule_is_two_sided():
    """The standalone rule is one-sided (``improvement <= tol``) and so reads a
    *rising* loss as convergence -- which is how this loop used to quit at round
    150 with a -0.137 'improvement' while its gains were still moving steadily.
    Only flatness may stop a run; a rise must not."""
    from wmr_simulator.joint_tuning.pipeline import has_stagnated

    tolerance = 5e-3
    # Flat: stagnated, from either side of the tolerance band.
    assert has_stagnated(1.0, 1.0, tolerance)
    assert has_stagnated(1.0, 1.0 - 0.4 * tolerance, tolerance)
    assert has_stagnated(1.0, 1.0 + 0.4 * tolerance, tolerance)
    # Improving fast: not stagnated (and the old rule agreed).
    assert not has_stagnated(1.0, 0.5, tolerance)
    # Worsening fast: not stagnated. The old one-sided rule called this
    # "converged" -- this assertion is the regression.
    assert not has_stagnated(1.0, 2.0, tolerance)
    assert not has_stagnated(0.012095, 0.013729, tolerance)  # the round-150 stop


# ---------------------------------------------------------------------------
# Validation set.
# ---------------------------------------------------------------------------

VALIDATION_DIR = "trajectory_exports/validation_trajectories"


def test_validation_loader_accepts_ragged_lengths(tmp_path):
    """A validation set is a short, diverse collection, and diversity includes
    curve length. The batched exports reader stacks into one array and so
    rejects that -- correctly for a batched design, wrongly for this -- which is
    why the validation reader is a separate one. Uses its own fixture: the
    curated directory's shapes are the user's to change."""
    import pickle
    from wmr_simulator.joint_tuning.validation import load_validation_trajectories
    from wmr_simulator.trajectory_optimization.pipeline import load_reference_states_exports

    for index, length in enumerate((101, 161)):
        with open(tmp_path / f"ragged_{index}.pkl", "wb") as file:
            pickle.dump(
                {
                    "reference_states": np.zeros((length, 8)),
                    "start_offsets": np.zeros((2, 3)),
                    "dt": 0.05,
                },
                file,
            )
    trajectories = load_validation_trajectories(str(tmp_path))
    assert {trajectory.reference_states.shape[0] for trajectory in trajectories} == {101, 161}
    with pytest.raises(ValueError, match="differing shapes"):
        load_reference_states_exports(str(tmp_path))


def test_validation_directory_loads():
    """The curated set itself, whatever shape it currently has."""
    from wmr_simulator.joint_tuning.validation import load_validation_trajectories

    trajectories = load_validation_trajectories(VALIDATION_DIR)
    assert len(trajectories) >= 2
    for trajectory in trajectories:
        assert trajectory.reference_states.ndim == 2
        assert trajectory.start_offsets.shape[1] == 3


def test_validation_loader_requires_offsets(tmp_path):
    """A validation score without fixed start poses is not comparable across
    rounds: driving the start offset out dominates the tracking term."""
    import pickle
    from wmr_simulator.joint_tuning.validation import load_validation_trajectories

    with open(tmp_path / "no_offsets.pkl", "wb") as file:
        pickle.dump({"reference_states": np.zeros((10, 8)), "dt": 0.05}, file)
    with pytest.raises(ValueError, match="no start offsets"):
        load_validation_trajectories(str(tmp_path))

    with pytest.raises(ValueError, match="No trajectory pickles"):
        load_validation_trajectories(str(tmp_path / "empty"))


def test_validation_set_selects_the_shipped_gains():
    """With a validation directory the recorded score is the held-out one, and
    the shipped gains are its argmin -- not the training loss's."""
    result = run_joint_tuning(
        PROBLEM,
        num_rounds=3,
        warm_start_rounds=0,
        num_trajectories=NUM_TRAJECTORIES,
        num_control_points=NUM_CONTROL_POINTS,
        num_realizations=NUM_REALIZATIONS,
        validation_trajectories_dir=VALIDATION_DIR,
        verbose=False,
    )
    assert result.config["num_validation_trajectories"] >= 2
    scores = np.asarray(result.history["gain_validation_loss"], dtype=float)
    finite = scores[np.isfinite(scores)]
    assert result.history["best_gain_score"] == pytest.approx(float(np.min(finite)))
    # The held-out score is its own number, not a copy of the training loss.
    training = np.asarray(result.history["gain_loss_pre"], dtype=float)
    assert not np.allclose(scores[np.isfinite(scores)], training[np.isfinite(training)])


def test_early_stopping_is_off_by_default_and_can_be_re_enabled():
    """The loop plateaus long before it is finished (a basin transition lands at
    ~180 trajectory steps), so a stagnation rule quits mid-run. Default off; the
    knob still works when asked for."""
    common = dict(
        num_rounds=4,
        warm_start_rounds=0,
        num_trajectories=NUM_TRAJECTORIES,
        num_control_points=NUM_CONTROL_POINTS,
        num_realizations=NUM_REALIZATIONS,
        verbose=False,
    )
    default = run_joint_tuning(PROBLEM, **common)
    assert default.config["convergence_rel_tol"] < 0.0
    assert default.history["converged_at_round"] is None
    assert len(default.history["gain_loss_pre"]) == 4

    # A tolerance of 1.0 with a window of 1 stops at the first check.
    stopped = run_joint_tuning(
        PROBLEM, convergence_rel_tol=1.0, convergence_window=1, **common
    )
    assert stopped.history["converged_at_round"] is not None
    assert stopped.history["converged_reason"] == "validation loss stagnated"
    assert len(stopped.history["gain_loss_pre"]) < 4


# ---------------------------------------------------------------------------
# Gain-block inner solvers.
# ---------------------------------------------------------------------------


def test_gain_solver_stays_in_the_box():
    """The solver clips inside its objective rather than projecting, so this is
    the assertion that the clip actually reaches the result."""
    result = run_joint_tuning(
        PROBLEM,
        num_rounds=2,
        warm_start_rounds=0,
        num_trajectories=NUM_TRAJECTORIES,
        num_control_points=NUM_CONTROL_POINTS,
        num_realizations=NUM_REALIZATIONS,
        gain_steps_per_round=20,
        verbose=False,
    )
    gains = np.asarray(result.gains, dtype=float)
    assert np.all(np.isfinite(gains))
    assert np.all(gains >= 0.0)
    assert np.all(gains[:4] <= float(GAIN_TUNING_DEFAULTS["k_max_stab"]))
    assert gains[4] <= float(GAIN_TUNING_DEFAULTS["k_max_rest"])


def test_quasi_newton_never_returns_an_uphill_iterate():
    """A quasi-Newton step on a near-singular Hessian can return garbage --
    kimotor's optimum is on the box boundary, where clipping makes the direction
    flat. The stepper must refuse such a step rather than poison the loop."""
    result = run_joint_tuning(
        PROBLEM,
        num_rounds=3,
        warm_start_rounds=0,
        num_trajectories=NUM_TRAJECTORIES,
        num_control_points=NUM_CONTROL_POINTS,
        num_realizations=NUM_REALIZATIONS,
        gain_steps_per_round=20,
        verbose=False,
    )
    pre = np.asarray(result.history["gain_loss_pre"], dtype=float)
    post = np.asarray(result.history["gain_loss_post"], dtype=float)
    finite = np.isfinite(pre) & np.isfinite(post)
    assert finite.any()
    # Within a round, with the trajectories held fixed, the block may not go up.
    assert np.all(post[finite] <= pre[finite] + 1e-9)


def test_too_small_an_inner_budget_is_refused():
    """A budget below the line-search floor leaves the gains exactly where they
    started. Shipping that footgun once already cost a full run that returned
    the stock gains unchanged, so it must raise, not silently no-op."""
    from wmr_simulator.joint_tuning.gain_solvers import (
        DEFAULT_GAIN_STEPS_PER_ROUND,
        resolve_steps_per_round,
    )

    with pytest.raises(ValueError, match="too small"):
        resolve_steps_per_round(5)
    assert resolve_steps_per_round(None) == DEFAULT_GAIN_STEPS_PER_ROUND == 40
    assert resolve_steps_per_round(40) == 40


def test_warm_start_rounds_cannot_swallow_the_whole_budget():
    """warm_start_rounds >= num_rounds means the gain block never runs at all,
    which looks exactly like a tuning failure. Refuse it."""
    with pytest.raises(ValueError, match="never run"):
        run_joint_tuning(
            PROBLEM, num_rounds=2, warm_start_rounds=5,
            num_trajectories=NUM_TRAJECTORIES, num_control_points=NUM_CONTROL_POINTS,
            num_realizations=NUM_REALIZATIONS, verbose=False,
        )
