__all__ = ["SystemIdentificationPipeline", "run_multi_log_identification"]


def __getattr__(name):
    if name in __all__:
        from wmr_simulator.identification import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
