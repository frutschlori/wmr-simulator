"""Mocap rigid-body calibration (pololu.mocap_calibration), once per robot.

    # 1. the calibration run, to drive once with no mocap_* keys in ROBOTCFG.CFG
    python scripts/mocap_calibration.py trajectory --out-dir trajectory_exports/mocap_calibration
    # 2. estimate from its logs (SD-card binaries are decoded next to themselves)
    python scripts/mocap_calibration.py estimate "Pololu Data/mocap_calibration/robot2"/TR0? \\
        --wheel-radius 0.0161 --out "Pololu Data/mocap_calibration/robot2/mocap_calibration.yaml"

Point experiment.yaml's mocap_calibration at the yaml, and every exported
ROBOTCFG.CFG carries the correction. Estimating again on logs recorded with the
correction in place returns the residual (--previous composes it onto the old one).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.pololu.decode_binary import decode_file
from wmr_simulator.pololu.mocap_calibration import (
    MocapCalibration,
    estimate_mocap_calibration,
    export_calibration_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    trajectory = commands.add_parser("trajectory", help="Write the calibration run (pickle + JSN).")
    trajectory.add_argument("--out-dir", default="trajectory_exports/mocap_calibration")
    estimate = commands.add_parser("estimate", help="Estimate the calibration from binary logs.")
    estimate.add_argument("logs", nargs="+", help="Logs of the calibration run (SD-card binaries or decoded csvs).")
    estimate.add_argument("--wheel-radius", type=float, required=True,
                          help="Encoder speed scale only; the heading offset does not depend on it.")
    estimate.add_argument("--previous", default=None,
                          help="Calibration the logs were recorded with; the result is composed onto it.")
    estimate.add_argument("--out", required=True, help="Calibration yaml to write.")
    args = parser.parse_args()

    if args.command == "trajectory":
        for path in export_calibration_reference(args.out_dir):
            print(f"Wrote {path}")
        return
    logs = []
    for log in map(Path, args.logs):
        if log.suffix.lower() != ".csv":
            decoded = log.with_name(f"{log.name}.csv")
            if not decoded.exists() and not decode_file(str(log), str(decoded)):
                raise SystemExit(f"Could not decode {log}")
            log = decoded
        logs.append(log)
    calibration, diagnostics = estimate_mocap_calibration(logs, args.wheel_radius)
    if args.previous:
        print(f"Residual on top of {args.previous}: {calibration.to_mapping()}")
        calibration = MocapCalibration.load(args.previous).compose(calibration)
    print(f"Calibration: {calibration.to_mapping()}")
    print(f"Diagnostics: {diagnostics}")
    path = calibration.save(args.out, diagnostics=diagnostics, logs=[str(log) for log in logs])
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
