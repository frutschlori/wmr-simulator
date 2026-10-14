"""Append a wait and a bridge path back to the start to a Pololu reference JSN,
so a run can be repeated without moving the robot.

    python scripts/pololu_append_bridge_reference.py --input trajectory_exports/benchmark_set/free_form_1.JSN
"""

from wmr_simulator.pololu.bridge_exporter import main

if __name__ == "__main__":
    raise SystemExit(main())
