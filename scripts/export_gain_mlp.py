"""Export a trained gain schedule to GAINMLP.JSN, optionally with a golden test
file for the firmware crate (``--golden``); see ``pololu.gain_mlp_exporter``."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.pololu.gain_mlp_exporter import main

if __name__ == "__main__":
    raise SystemExit(main())
