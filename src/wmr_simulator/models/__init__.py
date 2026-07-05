from wmr_simulator.models.io import load_residual_model, save_residual_model
from wmr_simulator.models.residual import (
    RESIDUAL_FEATURE_NAMES,
    ResidualDynamicsModel,
    apply_residual_model,
    init_residual_model,
    residual_features,
    residual_corrected_twist,
)

__all__ = [
    "DiffDrive",
    "DiffDriveState",
    "RESIDUAL_FEATURE_NAMES",
    "ResidualDynamicsModel",
    "apply_residual_model",
    "init_residual_model",
    "load_residual_model",
    "residual_corrected_twist",
    "residual_features",
    "save_residual_model",
]


def __getattr__(name):
    # Lazy: models.robot imports models.residual, so importing it eagerly here
    # would make `import wmr_simulator.models.residual` resolve robot first.
    if name in {"DiffDrive", "DiffDriveState"}:
        from wmr_simulator.models.robot import DiffDrive, DiffDriveState

        return {"DiffDrive": DiffDrive, "DiffDriveState": DiffDriveState}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
