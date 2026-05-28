__all__ = [
    "SystemIdentificationPipeline",
    "run_multi_experiment_identification",
    "run_single_experiment_identification",
    "run_validation_identification",
    "run_window_replay_identification",
]


def __getattr__(name):
    if name == "SystemIdentificationPipeline":
        from wmr_simulator.identification.pipeline import SystemIdentificationPipeline

        return SystemIdentificationPipeline
    if name == "run_single_experiment_identification":
        from wmr_simulator.identification.pipeline import run_single_experiment_identification

        return run_single_experiment_identification
    if name in {
        "run_multi_experiment_identification",
        "run_validation_identification",
        "run_window_replay_identification",
    }:
        from wmr_simulator.identification import variants

        return getattr(variants, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
