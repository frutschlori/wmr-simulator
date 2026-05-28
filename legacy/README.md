# Legacy Reference Code

This directory contains archived scripts and old backends kept for reference while
the maintained package lives under `src/wmr_simulator/`.

The files here are not part of the active package API. Some imports may require
manual adjustment because the active code has been reorganized around the JAX
implementation, `wmr_simulator.simulation`, `wmr_simulator.identification`,
`wmr_simulator.gain_tuning`, `wmr_simulator.trajectory_optimization`, and
`wmr_simulator.visualization`.

Notable contents:

- `monoliths/system_identification.py`: restored from git history before the
  minimal wrapper refactor.
- `numpy_backend/`: original NumPy robot, estimator, controller, and
  visualization modules.
- `monoliths/`: old analysis, system-identification, gain-tuning, simultaneous
  optimization, and trajectory-optimization scripts.
- `scripts/`: old executable simulator scripts.
