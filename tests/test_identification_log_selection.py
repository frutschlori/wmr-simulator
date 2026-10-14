import numpy as np

from wmr_simulator.active_learning.experiment import IterationPaths
from wmr_simulator.active_learning.stages import _resolve_log_paths
from wmr_simulator.identification.outliers import robust_parameter_outliers
from wmr_simulator.pololu.log_loader import POLOLU_TRAJ_CONTROL_COLUMNS

# Nominal identified parameters (r, L, u_max, tau) with realistic
# log-to-log scatter; index 3 is the broken run that must be caught.
_GOOD_PARAMS = np.array(
    [
        [0.0171, 0.0912, 238.0, 0.170],
        [0.0173, 0.0908, 240.0, 0.163],
        [0.0170, 0.0915, 236.0, 0.175],
        [0.0172, 0.0910, 239.0, 0.168],
        [0.0169, 0.0913, 237.0, 0.172],
    ]
)


def _write_log_csv(path, rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(POLOLU_TRAJ_CONTROL_COLUMNS)
    body = "\n".join(",".join("0" for _ in POLOLU_TRAJ_CONTROL_COLUMNS) for _ in range(rows))
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")


def test_outliers_flags_the_disagreeing_log():
    samples = _GOOD_PARAMS.copy()
    samples[3, 2] = 217.0  # max_wheel_speed way off

    report = robust_parameter_outliers(samples, z_threshold=3.5)

    assert report.evaluated
    assert list(np.flatnonzero(report.is_outlier)) == [3]
    # The flag comes from max_wheel_speed, not from a parameter that agrees.
    assert int(np.argmax(report.z_scores[3])) == 2


def test_outliers_keeps_a_consistent_batch():
    report = robust_parameter_outliers(_GOOD_PARAMS, z_threshold=3.5)

    assert report.evaluated
    assert not report.is_outlier.any()


def test_identification_logs_skip_baseline_subdirectories(tmp_path):
    paths = IterationPaths(tmp_path / "iteration_01")
    paths.data_dir.mkdir(parents=True)
    _write_log_csv(paths.data_dir / "TR00.csv")
    _write_log_csv(paths.data_dir / "TR01.csv")
    _write_log_csv(paths.data_dir / "baselines" / "TR02.csv")

    resolved = _resolve_log_paths(paths, log=None)

    assert [path.name for path in resolved] == ["TR00.csv", "TR01.csv"]
