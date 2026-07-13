"""Overlay baseline runs (auto tuned vs previous manual tune) per trajectory.

Takes a baselines directory laid out as
"Pololu Data/Experiments/exp02/baselines/{tuned,untuned}/<shape>/<speed>/decoded/TRxx.csv"
and writes one comparison PDF per shape/speed combo present in both groups: all
runs of a group as thin same-colored lines, so the run-to-run repeatability is
visible. Mocap velocities come from the Savitzky-Golay smoothing in
pololu.measurement_smoothing (applied inside the log loader).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
from wmr_simulator.visualization.pololu import plot_run_comparison

GROUPS = (("tuned", "auto tuned"), ("untuned", "previous manual tune"))


def _load_runs(decoded_dir: Path, max_runs: int) -> list:
    paths = sorted(decoded_dir.glob("*.csv"))
    if max_runs:
        paths = paths[:max_runs]
    return [load_pololu_traj_control_log(path, clip_after_first_trajectory=True) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baselines_dir",
        type=Path,
        nargs="?",
        default=Path("Pololu Data/Experiments/exp02/baselines"),
        help="Directory containing the tuned/ and untuned/ run trees",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Runs per group and combo (0 = all)")
    parser.add_argument("--out-dir", type=str, default="visualize")
    args = parser.parse_args()

    tuned_root = args.baselines_dir / GROUPS[0][0]
    untuned_root = args.baselines_dir / GROUPS[1][0]
    combos = sorted(path.parent.relative_to(tuned_root) for path in tuned_root.rglob("decoded"))
    if not combos:
        raise SystemExit(f"No decoded/ directories found under {tuned_root}")

    for combo in combos:
        untuned_decoded = untuned_root / combo / "decoded"
        if not untuned_decoded.is_dir():
            print(f"Skipping {combo}: no untuned counterpart")
            continue
        log_groups = [
            _load_runs(tuned_root / combo / "decoded", args.max_runs),
            _load_runs(untuned_decoded, args.max_runs),
        ]
        plot_run_comparison(
            log_groups,
            [label for _, label in GROUPS],
            out_prefix=f"baseline_comparison_{'-'.join(combo.parts)}",
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
