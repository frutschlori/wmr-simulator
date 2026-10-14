"""Motion-phased trajectories: one curve, consecutive phases with their own limits."""


import jax.numpy as jnp
import numpy as np
import pytest

from wmr_simulator.active_learning.experiment import IterationPaths
from wmr_simulator.active_learning.stages import _identified_duration
from wmr_simulator.trajectory_optimization.parametrization import time_scaling_derivatives
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline

PROBLEM = "problems/pololu_gains.yaml"  # sim_time 5.0
PHASES = [
    {"duration": 1.5, "identify": True, "motion_limits": {"v_max": 1.0, "alpha_max": 8.0}, "min_speed": 0.4},
    {"duration": 3.5, "identify": False, "motion_limits": {"v_max": 2.5}, "min_speed": 1.3},
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


def test_the_identification_fim_reads_only_identified_phases(phased_pipeline):
    factor = np.asarray(phased_pipeline.compute_fim_factor(window_length=50))
    pose_time = np.asarray(phased_pipeline.closed_loop_log.pose.time_s)[1:]
    row_time = np.repeat(pose_time, 3)

    assert np.any(np.abs(factor[row_time < 1.5]) > 0.0)
    assert np.all(factor[row_time >= 1.5] == 0.0)


def test_the_pickle_records_the_identified_duration(phased_pipeline, tmp_path):
    paths = IterationPaths(tmp_path / "iteration_01")
    paths.identification_trajectory_dir.mkdir(parents=True)
    assert _identified_duration(paths) is None  # nothing designed: fit the whole log

    phased_pipeline.save_reference_states_pickle(
        out_dir=str(paths.identification_trajectory_dir), filename_prefix="identification_trajectory"
    )

    assert _identified_duration(paths) == pytest.approx(1.5)
