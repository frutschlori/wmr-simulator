__all__ = [
    "Experiment",
    "stages",
]


def __getattr__(name):
    if name == "Experiment":
        from wmr_simulator.active_learning.experiment import Experiment

        return Experiment
    if name == "stages":
        # importlib avoids recursing into this __getattr__ (the attribute name
        # equals the submodule name).
        import importlib

        return importlib.import_module("wmr_simulator.active_learning.stages")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
