__all__ = ["ControllerTuningPipeline", "resolve_gain_robot_params", "run_gain_tuning_experiment"]


def __getattr__(name):
    if name in {"ControllerTuningPipeline", "resolve_gain_robot_params", "run_gain_tuning_experiment"}:
        from wmr_simulator.gain_tuning.pipeline import (
            ControllerTuningPipeline,
            resolve_gain_robot_params,
            run_gain_tuning_experiment,
        )

        return {
            "ControllerTuningPipeline": ControllerTuningPipeline,
            "resolve_gain_robot_params": resolve_gain_robot_params,
            "run_gain_tuning_experiment": run_gain_tuning_experiment,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
