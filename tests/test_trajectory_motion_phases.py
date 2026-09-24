"""Motion-phased trajectories: one curve, consecutive phases with their own envelope.

The identification trajectory is a slow phase the identification fits followed
by a fast phase driven for the residual model. What has to hold: the reference
rests at every phase boundary and stays C^2, each phase is judged against its
own limits and floors, the identification FIM only reads the identified phases,
and the identify stage fits exactly those seconds of a log.
"""

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.active_learning.experiment import IterationPaths
from wmr_simulator.active_learning.stages import _identified_duration
from wmr_simulator.pololu.log_loader import clip_log_to_duration
from wmr_simulator.trajectory_optimization.constraints import min_speed_loss
from wmr_simulator.trajectory_optimization.parametrization import time_scaling_derivatives
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline
from wmr_simulator.types import PoseLog, ReferenceLog, SimulationLog, WheelLog

PROBLEM = "problems/pololu_gains.yaml"  # sim_time 4.0
PHASES = [
    {"duration": 1.5, "identify": True, "motion_limits": {"v_max": 1.0, "alpha_max": 8.0}, "min_speed": 0.4},
    {"duration": 2.5, "identify": False, "motion_limits": {"v_max": 2.5}, "min_speed": 1.3},
]


@pytest.fixture(scope="module")
def phased_pipeline():
    return TrajectoryOptimizationPipeline(problem_path=PROBLEM, objective_mode="identification", motion_phases=PHASES)


def test_phased_s_curve_rests_at_the_boundary_and_is_c2():
    total_time = 7.0
    t = jnp.linspace(0.0, total_time, 7001)
    s, s_dot, s_ddot = (np.asarray(value, dtype=float) for value in time_scaling_derivatives(t, total_time, "s_curve", (3.0 / 7.0,)))
    dt = total_time / 7000

    assert s[0] == 0.0 and s[-1] == pytest.approx(1.0)
    assert s[3000] == pytest.approx(3.0 / 7.0, abs=1e-6)
    assert s_dot[3000] == pytest.approx(0.0, abs=1e-6)
    assert s_ddot[3000] == pytest.approx(0.0, abs=1e-4)
    # The analytic derivatives are the derivatives of s, across the boundary too.
    np.testing.assert_allclose(np.gradient(s, dt)[5:-5], s_dot[5:-5], atol=2e-3)
    np.testing.assert_allclose(np.gradient(s_dot, dt)[5:-5], s_ddot[5:-5], atol=2e-3)


def test_phases_need_the_s_curve():
    with pytest.raises(ValueError, match="s-curve"):
        time_scaling_derivatives(jnp.linspace(0.0, 1.0, 5), 1.0, "linear", (0.5,))


def test_each_phase_is_held_to_its_own_limits(phased_pipeline):
    limits = phased_pipeline.motion_limits()
    time_grid = phased_pipeline.problem.sim_time_grid()
    slow = time_grid < 1.5

    assert np.all(np.asarray(limits["v_max"])[slow] == pytest.approx(1.0))
    assert np.all(np.asarray(limits["v_max"])[~slow] == pytest.approx(2.5))
    # Keys a phase leaves out fall back to the problem yaml's robot block.
    assert np.all(np.asarray(limits["alpha_max"])[~slow] == pytest.approx(float(phased_pipeline.problem.robot_cfg["alpha_max"])))
    np.testing.assert_allclose(np.asarray(limits["v_min"]), [0.4, 1.3])
    np.testing.assert_array_equal(np.asarray(limits["phase_masks"]).sum(axis=0), np.ones(len(time_grid)))


def test_a_floor_is_judged_on_its_own_phase_mean():
    """A fast second half cannot pay for a slow first half's floor."""
    speeds = jnp.concatenate([jnp.full(10, 0.2), jnp.full(10, 2.0)])
    masks = jnp.stack([jnp.arange(20) < 10, jnp.arange(20) >= 10]).astype(jnp.float32)

    phased = float(min_speed_loss(speeds, jnp.asarray([1.0, 1.0]), phase_masks=masks))
    whole = float(min_speed_loss(speeds, 1.0))

    assert whole == pytest.approx(0.0, abs=1e-3)  # overall mean 1.1 clears the floor
    assert phased > 0.5  # the slow phase's mean 0.2 does not


def test_the_identification_fim_reads_only_identified_phases(phased_pipeline):
    factor = np.asarray(phased_pipeline.compute_fim_factor(window_length=50))
    pose_time = np.asarray(phased_pipeline.closed_loop_log.pose.time_s)[1:]
    row_time = np.repeat(pose_time, 3)

    assert np.any(np.abs(factor[row_time < 1.5]) > 0.0)
    assert np.all(factor[row_time >= 1.5] == 0.0)


def test_phase_durations_must_fill_the_problem_clock():
    with pytest.raises(ValueError, match="sim_time"):
        TrajectoryOptimizationPipeline(
            problem_path=PROBLEM, objective_mode="identification",
            motion_phases=[{"duration": 1.0}, {"duration": 1.0}],
        )


def test_identified_phases_must_lead():
    """The identify stage cuts each log at the end of the identified phases."""
    with pytest.raises(ValueError, match="leading"):
        TrajectoryOptimizationPipeline(
            problem_path=PROBLEM, objective_mode="identification",
            motion_phases=[{"duration": 1.5, "identify": False}, {"duration": 2.5, "identify": True}],
        )


def test_the_pickle_records_the_identified_duration(phased_pipeline, tmp_path):
    paths = IterationPaths(tmp_path / "iteration_01")
    paths.identification_trajectory_dir.mkdir(parents=True)
    assert _identified_duration(paths) is None  # nothing designed: fit the whole log

    phased_pipeline.save_reference_states_pickle(
        out_dir=str(paths.identification_trajectory_dir), filename_prefix="identification_trajectory"
    )

    assert _identified_duration(paths) == pytest.approx(1.5)


def test_clip_log_to_duration_cuts_every_stream_on_its_own_clock():
    def stream(dt, end=4.0):
        return jnp.arange(0.0, end + 1e-9, dt)

    wheel_time, pose_time, reference_time = stream(0.01), stream(0.02), stream(0.05)
    log = SimulationLog(
        reference=ReferenceLog(time_s=reference_time, states=jnp.zeros((len(reference_time), 8))),
        wheel=WheelLog(
            time_s=wheel_time,
            speeds=jnp.zeros((len(wheel_time), 2)),
            vel_omega=jnp.zeros((len(wheel_time), 2)),
            duty_cycle=jnp.zeros((len(wheel_time), 2)),
        ),
        pose=PoseLog(
            time_s=pose_time,
            states=jnp.zeros((len(pose_time), 3)),
            true_states=jnp.zeros((len(pose_time), 3)),
            command_time_s=reference_time,
            wheel_cmd=jnp.zeros((len(reference_time), 2)),
            twists=jnp.zeros((len(pose_time), 3)),
        ),
    )

    clipped = clip_log_to_duration(log, 1.5)

    for time_s, values in (
        (clipped.wheel.time_s, clipped.wheel.duty_cycle),
        (clipped.pose.time_s, clipped.pose.twists),
        (clipped.reference.time_s, clipped.reference.states),
    ):
        assert float(time_s[-1]) <= 1.5 + 1e-6
        assert float(time_s[-1]) > 1.5 - 0.05
        assert values.shape[0] == time_s.shape[0]
    assert clipped.pose.clean_time_s is None and clipped.pose.gains is None
