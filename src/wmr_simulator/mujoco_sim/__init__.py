"""MuJoCo stand-in for the real robot in the active-learning loop.

The plant's true parameters are hidden from the pipeline: only this package
reads ``models/pololu_hidden.yaml`` and ``models/pololu_calibrated.xml``.
"""

from wmr_simulator.mujoco_sim.binlog import BinaryLogWriter, LogTag
from wmr_simulator.mujoco_sim.deploy import DeploymentResult, run_deployment
from wmr_simulator.mujoco_sim.firmware import Firmware, FirmwareClock, FirmwareConfig
from wmr_simulator.mujoco_sim.plant import (
    HiddenPlantConfig,
    MujocoPlant,
    build_plant_xml,
    load_hidden_plant_config,
)
from wmr_simulator.mujoco_sim.render import (
    DeploymentVideoRecorder,
    FrameObserver,
    VariantRun,
    View,
    pane_rects,
    record_pose_track,
    render_deployment,
    render_deployment_comparison,
    split_screen_views,
)
from wmr_simulator.mujoco_sim.truth import PlantTruth, measure_plant_truth

__all__ = [
    "BinaryLogWriter",
    "DeploymentResult",
    "DeploymentVideoRecorder",
    "Firmware",
    "FirmwareClock",
    "FirmwareConfig",
    "FrameObserver",
    "HiddenPlantConfig",
    "LogTag",
    "MujocoPlant",
    "PlantTruth",
    "VariantRun",
    "View",
    "build_plant_xml",
    "load_hidden_plant_config",
    "measure_plant_truth",
    "pane_rects",
    "record_pose_track",
    "render_deployment",
    "render_deployment_comparison",
    "run_deployment",
    "split_screen_views",
]
