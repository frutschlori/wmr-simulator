from wmr_simulator.gain_tuning.pipeline import run_gain_tuning_experiment
from wmr_simulator.identification.pipeline import run_single_experiment_identification
from wmr_simulator.trajectory_optimization.pipeline import TrajectoryOptimizationPipeline
from wmr_simulator.types import PhysicalParams


def optimize_informative_trajectory(
    problem_path: str,
    time_scaling: str | None = None,
    bezier_order: int = 7,
    num_steps: int = 0,
    learning_rate: float = 1e-2,
    window_length: int | None = None,
    save_trace: bool = False,
    trace_stride: int = 50,
):
    pipeline = TrajectoryOptimizationPipeline(
        problem_path,
        time_scaling=time_scaling,
    )
    initial_control_points = pipeline.initial_bezier_control_points(bezier_order)
    pipeline.set_bezier_control_points(initial_control_points)
    if num_steps:
        pipeline.optimize_bezier_trajectory(
            order=bezier_order,
            num_steps=num_steps,
            learning_rate=learning_rate,
            window_length=window_length,
            save_trace=save_trace,
            trace_stride=trace_stride,
        )
    return pipeline


def run_si_then_gain_tuning(
    problem_path: str,
    initial_params: PhysicalParams,
    steps_si: int,
    learning_rate_si: float,
    steps_gain_tuning: int,
    learning_rate_gain_tuning: float,
    num_realizations: int,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
    window_length: int | None = None,
):
    identification_result = run_single_experiment_identification(
        problem_path=problem_path,
        initial_params=initial_params,
        num_steps=steps_si,
        learning_rate=learning_rate_si,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
    )
    gain_tuning_result = run_gain_tuning_experiment(
        problem_path=problem_path,
        robot_params=identification_result["estimated_params"],
        num_steps=steps_gain_tuning,
        learning_rate=learning_rate_gain_tuning,
        num_realizations=num_realizations,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
    )
    return {
        "identification": identification_result,
        "gain_tuning": gain_tuning_result,
    }


def run_active_learning_iteration(
    problem_path: str,
    initial_params: PhysicalParams,
    steps_si: int,
    learning_rate_si: float,
    steps_gain_tuning: int,
    learning_rate_gain_tuning: float,
    num_realizations: int,
    seed: int = 0,
    reference_trajectories_dir: str | None = None,
    window_length: int | None = None,
):
    return run_si_then_gain_tuning(
        problem_path=problem_path,
        initial_params=initial_params,
        steps_si=steps_si,
        learning_rate_si=learning_rate_si,
        steps_gain_tuning=steps_gain_tuning,
        learning_rate_gain_tuning=learning_rate_gain_tuning,
        num_realizations=num_realizations,
        seed=seed,
        reference_trajectories_dir=reference_trajectories_dir,
        window_length=window_length,
    )


def run_sequential_pipeline(*args, **kwargs):
    return run_active_learning_iteration(*args, **kwargs)
