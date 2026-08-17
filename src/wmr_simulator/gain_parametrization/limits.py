"""Stability floor on the gains a parametrization hands the controller.

Every parametrization here is multiplicative, so a factor of 0 sets the gain it
scales to exactly 0 -- which is what the raw ``clip(1 + mlp(z), 0, bound)``
bound allows, and what the tuned error-MLP actually does (measured: ``kth`` and
``kpmotor`` hit 0 on ~4% of the feature space). That undoes the one invariant
the base-gain search space guarantees: the first four gains are searched in log
space over ``[k_min_stab, k_max_stab]`` precisely because they must stay
strictly positive for stability, and they can never reach 0 there.

So the *applied* gains get the same floor the base gains do. ``kimotor`` is
excluded: it is the one gain the tuner is allowed to set to exactly 0 (integral
action off), which is why it is searched in sqrt space over ``[0, k_max_rest]``.

The floor never binds at the identity parametrization -- a factor of 1 leaves a
base gain that is already >= ``k_min_stab`` -- so a zero trainable vector still
reproduces the static controller exactly.

Firmware note: ``firmware/libs/gain_mlp`` clamps the *factor* to ``[0, bound]``
and knows nothing about this floor, so it does not currently mirror it. The
equivalent factor-space statement is a per-gain lower bound
``k_min_stab / base_gains[i]`` -- the base gains are constant for a deployment,
so it is a constant vector that could be baked into ``GAINMLP.JSN``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from wmr_simulator.gain_tuning.defaults import GAIN_TUNING_DEFAULTS

# Gains that must stay strictly positive, i.e. [kx, ky, kth, kpmotor]; kimotor
# (index 4) is the one gain allowed to be exactly 0.
NUM_STABLE_GAINS = 4
# Matches the k_min_stab of the gain search (gain_tuning.defaults), the lower
# end of the log-space range those four gains live in.
GAIN_STABILITY_FLOOR = float(GAIN_TUNING_DEFAULTS["k_min_stab"])


def gain_floor(num_gains: int) -> jax.Array:
    """Per-gain lower bound, ``(num_gains,)``: the floor on the stable gains, 0 elsewhere."""
    return jnp.where(jnp.arange(num_gains) < NUM_STABLE_GAINS, GAIN_STABILITY_FLOOR, 0.0)


def clip_to_stability_floor(gains: jax.Array) -> jax.Array:
    """Floor a full parametrized gain vector (last axis is the gain vector)."""
    return jnp.maximum(gains, gain_floor(gains.shape[-1]))


def floor_at_indices(indices: jax.Array) -> jax.Array:
    """Floor for a gain subset addressed by ``indices`` (e.g. ``scheduled_indices``)."""
    return jnp.where(indices < NUM_STABLE_GAINS, GAIN_STABILITY_FLOOR, 0.0)
