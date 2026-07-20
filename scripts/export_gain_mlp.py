import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from wmr_simulator.pololu.gain_mlp_exporter import main

if __name__ == "__main__":
    raise SystemExit(main())
