__all__ = [
    "Controller",
    "DiffDrive",
    "DiffDriveEstimator",
    "DiffDriveState",
    "EstimatorState",
    "PhysicalParams",
    "SimulationLog",
    "SimulationPipeline",
]


def __getattr__(name):
    if name == "Controller":
        from wmr_simulator.controller import Controller

        return Controller
    if name in {"DiffDriveEstimator", "EstimatorState"}:
        from wmr_simulator.estimator import DiffDriveEstimator, EstimatorState

        return {"DiffDriveEstimator": DiffDriveEstimator, "EstimatorState": EstimatorState}[name]
    if name in {"DiffDrive", "DiffDriveState"}:
        from wmr_simulator.robot import DiffDrive, DiffDriveState

        return {"DiffDrive": DiffDrive, "DiffDriveState": DiffDriveState}[name]
    if name == "SimulationPipeline":
        from wmr_simulator.simulation import SimulationPipeline

        return SimulationPipeline
    if name in {"PhysicalParams", "SimulationLog"}:
        from wmr_simulator.types import PhysicalParams, SimulationLog

        return {"PhysicalParams": PhysicalParams, "SimulationLog": SimulationLog}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
