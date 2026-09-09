---
name: verify
description: Verify wmr-simulator changes end-to-end — canonical test command and short-budget smoke invocations per pipeline (identification, trajectory optimization, gain tuning, residual training, baseline references), plus how to inspect the output plots.
---

# Verifying changes in wmr-simulator

Run the layers in order; stop escalating once the layers relevant to the change pass.
Only smoke-test the pipelines the change touches.

## 1. Fast checks (always)

```bash
python -m compileall -q src scripts tests   # syntax after refactors/renames
uv run pytest -q                            # full suite, ~13 s, CPU
```

## 2. Pipeline smoke runs (short budgets, explicit timeout)

Full-budget runs take minutes — never run them as verification.

```bash
# System identification (~30 s at 10 steps)
timeout 300 uv run python scripts/run_identification_pololu.py --steps 10

# Trajectory optimization (no artifacts saved)
timeout 300 uv run python scripts/run_trajectory_optimization.py \
    --num-trajectories 2 --opt-steps 20 --no-save-trajectory

# Gain tuning (minimal budget)
timeout 300 uv run python scripts/run_gain_tuning.py \
    --num-lhs-points 0 --num-adam-optimizations 1 --steps 3 --num-realizations 2

# Residual model training (tiny epoch counts)
timeout 300 uv run python scripts/train_residual_model.py \
    --epochs 20 --multistep-epochs 2 --out /tmp/residual_smoke.pkl

# Closed-loop simulation
timeout 300 uv run python scripts/run_simulation.py

# Baseline reference generation
timeout 300 uv run python scripts/generate_baseline_reference.py
```

For ad-hoc heredoc snippets prefix `JAX_PLATFORMS=cpu` (the scripts set it themselves).

## 3. Inspect output plots

Scripts write summary PDFs to `visualize/`. To actually look at one:

```bash
pdftoppm -png -r 100 visualize/<name>.pdf /tmp/Codex-plot
```

then Read the generated PNG. Crop with `-x/-y/-W/-H` if a detail matters.

## Cleanup

Smoke runs overwrite tracked PDFs in `visualize/` and may drop files in
`trajectory_exports/`. Restore/remove test artifacts before finishing:

```bash
git status --short
git checkout -- visualize/   # restore overwritten plots
```
