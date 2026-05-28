__all__ = [
    "BezierTrajectoryGenerator",
    "PlannerTrajectoryGenerator",
    "ProblemDefinition",
    "Trajectory",
    "TrajectoryGenerator",
    "TrajectoryOptimizationPipeline",
    "build_trajectory_generator",
]


def __getattr__(name):
    if name == "BezierTrajectoryGenerator":
        from wmr_simulator.trajectory_optimization.bezier import BezierTrajectoryGenerator

        return BezierTrajectoryGenerator
    if name in {"PlannerTrajectoryGenerator", "Trajectory", "TrajectoryGenerator", "build_trajectory_generator"}:
        from wmr_simulator.trajectory_optimization import parametrization

        return getattr(parametrization, name)
    if name in {"ProblemDefinition", "TrajectoryOptimizationPipeline"}:
        from wmr_simulator.trajectory_optimization import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
