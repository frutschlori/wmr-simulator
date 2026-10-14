"""Controller-gain parametrization: base gains + parameters + rollout context
-> controller gains (implemented by ``error_mlp``)."""

from __future__ import annotations

import jax

from wmr_simulator.gain_parametrization import error_mlp
from wmr_simulator.gain_parametrization.error_mlp import (
    ErrorMlpParams,
    apply,
    flat_params,
    num_params,
    to_cfg,
    with_flat_params,
    zero_params,
)
from wmr_simulator.gain_parametrization.error_mlp import (
    on_reference_gains_over_refs as outer_gains_over_refs,
)


def params_from_cfg(cfg: dict | None, feature_scale) -> ErrorMlpParams:
    """Build parametrization params from a controller config block.

    ``feature_scale`` is ``[v_max, omega_max]``; the MLP derives its feature
    normalization from it.
    """
    kind = str(({} if cfg is None else cfg).get("kind", error_mlp.KIND))
    if kind != error_mlp.KIND:
        raise ValueError(f"Unsupported gain parametrization kind {kind!r}; expected {error_mlp.KIND!r}.")
    return error_mlp.from_cfg(cfg, feature_scale)


def gains_over_samples(base_gains, params, ref_states, pose_states, twist_states):
    """Applied controller gains ``(N, num_gains)`` for stacked ``ref_states``
    ``(N, 8)``, ``pose_states`` ``(N, 3)`` and ``twist_states`` ``(N, 2)``."""
    import jax.numpy as jnp

    base = jnp.asarray(base_gains)
    return jax.vmap(lambda ref, pose, twist: apply(base, params, ref, pose, twist))(
        jnp.asarray(ref_states), jnp.asarray(pose_states), jnp.asarray(twist_states)
    )
