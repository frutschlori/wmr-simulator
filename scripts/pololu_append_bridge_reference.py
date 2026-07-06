"""Thin CLI wrapper: append a wait + bridge-back path to a Pololu reference JSN.

All logic lives in wmr_simulator.pololu.bridge_exporter.
"""

from wmr_simulator.pololu.bridge_exporter import main

if __name__ == "__main__":
    raise SystemExit(main())
