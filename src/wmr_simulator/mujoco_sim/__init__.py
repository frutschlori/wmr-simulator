"""MuJoCo stand-in for the real robot in the active-learning loop.

The plant's true parameters are hidden from the pipeline: only this package
reads ``models/pololu_hidden.yaml`` and ``models/pololu_calibrated.xml``.
"""

from wmr_simulator.mujoco_sim.binlog import BinaryLogWriter, LogTag
from wmr_simulator.mujoco_sim.deploy import DeploymentResult, run_deployment
from wmr_simulator.mujoco_sim.firmware import Firmware, FirmwareClock, FirmwareConfig
from wmr_simulator.mujoco_sim.truth import PlantTruth, measure_plant_truth
from wmr_simulator.mujoco_sim.plant import (
    HiddenPlantConfig,
    MujocoPlant,
    build_plant_xml,
    load_hidden_plant_config,
)

__all__ = [
    "BinaryLogWriter",
    "DeploymentResult",
    "Firmware",
    "FirmwareClock",
    "FirmwareConfig",
    "HiddenPlantConfig",
    "LogTag",
    "MujocoPlant",
    "PlantTruth",
    "build_plant_xml",
    "load_hidden_plant_config",
    "measure_plant_truth",
    "run_deployment",
]
