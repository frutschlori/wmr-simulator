"""Joint (alternating) optimization of controller gains and tuning trajectories.

Its own package because the existing dependency direction is
``trajectory_optimization -> gain_tuning.objectives``, and this loop needs
``gain_tuning.pipeline`` as well; it cannot live inside either side.
"""

from wmr_simulator.joint_tuning.pipeline import (
    JointRoundSnapshot,
    JointState,
    JointTuningResult,
    run_joint_tuning,
)
from wmr_simulator.trajectory_optimization.start_offsets import (
    START_OFFSET_MODES,
    normalize_start_offset_mode,
)

__all__ = [
    "JointRoundSnapshot",
    "JointState",
    "JointTuningResult",
    "START_OFFSET_MODES",
    "normalize_start_offset_mode",
    "run_joint_tuning",
]
