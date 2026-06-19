__all__ = [
    "BezierCurve",
    "ProblemDefinition",
    "TrajectoryOptimizationPipeline",
    "compute_bezier_reference",
]


def __getattr__(name):
    if name in {"BezierCurve", "compute_bezier_reference"}:
        from wmr_simulator.trajectory_optimization import bezier

        return getattr(bezier, name)
    if name in {"ProblemDefinition", "TrajectoryOptimizationPipeline"}:
        from wmr_simulator.trajectory_optimization import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
