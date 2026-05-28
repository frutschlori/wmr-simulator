__all__ = [
    "optimize_informative_trajectory",
    "run_active_learning_iteration",
    "run_sequential_pipeline",
    "run_si_then_gain_tuning",
]


def __getattr__(name):
    if name in __all__:
        from wmr_simulator.active_learning import sequential_pipeline

        return getattr(sequential_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
