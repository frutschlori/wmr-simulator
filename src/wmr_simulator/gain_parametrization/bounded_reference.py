"""Bounded multiplicative gain scheduling for the outer geometric gains."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

KIND = "bounded_reference"

# Registry mapping YAML feature names to the column they occupy in ``W``/``z``.
FEATURE_NAMES = ("v_d", "abs_omega_d")
_V_D_EPS = 1e-12


class BoundedReferenceParams(NamedTuple):
    """Trainable and fixed parameters for the bounded reference scheduler."""

    W: jax.Array                    # (num_scheduled, num_features)
    rho: jax.Array                  # (num_scheduled,)
    feature_scale: jax.Array        # (num_features,)
    scheduled_indices: jax.Array    # (num_scheduled,) int indices into the gain vector


def features(ref_state: jax.Array, feature_scale: jax.Array) -> jax.Array:
    """Normalized feature vector ``z`` for one reference state.

    ``ref_state`` layout: [x, y, theta, vx, vy, omega, ax, ay].
    """
    vx_d = ref_state[3]
    vy_d = ref_state[4]
    omega_d = ref_state[5]
    v_d = jnp.sqrt(vx_d**2 + vy_d**2 + _V_D_EPS)
    raw = jnp.stack([v_d, jnp.abs(omega_d)])
    return raw / feature_scale


def factors(params: BoundedReferenceParams, ref_state: jax.Array) -> jax.Array:
    """Bounded multiplicative factors, shape ``(num_scheduled,)``."""
    z = features(ref_state, params.feature_scale)
    lin = params.W @ z
    return 1.0 + params.rho * jnp.clip(lin, -1.0, 1.0)


def apply(base_gains: jax.Array, params: BoundedReferenceParams, ref_state: jax.Array) -> jax.Array:
    """Full scheduled gain vector for one reference state."""
    scheduled_factors = factors(params, ref_state)
    return base_gains.at[params.scheduled_indices].multiply(scheduled_factors, unique_indices=True)


def scheduled_outer_gains_over_refs(
    base_gains: jax.Array,
    params: BoundedReferenceParams,
    reference_states: jax.Array,
) -> jax.Array:
    """Scheduled gains over a reference trajectory for diagnostics/penalties."""
    refs = reference_states[:-1]
    base_outer = base_gains[params.scheduled_indices]
    scheduled_factors = jax.vmap(lambda ref: factors(params, ref))(refs)
    return base_outer[None, :] * scheduled_factors


def num_params(params: BoundedReferenceParams) -> int:
    """Number of trainable scheduler parameters."""
    return int(params.W.size)


def with_flat_params(theta: jax.Array, template: BoundedReferenceParams) -> BoundedReferenceParams:
    """Rebuild params with ``W`` filled from a flat vector."""
    num_scheduled, num_features = template.W.shape
    W = jnp.asarray(theta, dtype=jnp.float32).reshape(num_scheduled, num_features)
    return template._replace(W=W)


def zero_params(template: BoundedReferenceParams) -> BoundedReferenceParams:
    """Identity scheduler with the same fixed config as ``template``."""
    return template._replace(W=jnp.zeros_like(template.W))


def from_cfg(cfg: dict | None, feature_scale) -> BoundedReferenceParams:
    """Parse a YAML block into bounded-reference scheduler params."""
    cfg = {} if cfg is None else cfg
    num_features = len(FEATURE_NAMES)

    feature_names = tuple(cfg.get("feature_names", FEATURE_NAMES))
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            f"gain parametrization feature_names must be {list(FEATURE_NAMES)} "
            f"for {KIND}, got {list(feature_names)}."
        )

    scheduled_indices = tuple(int(i) for i in cfg.get("scheduled_indices", (0, 1, 2)))
    num_scheduled = len(scheduled_indices)

    W = jnp.asarray(cfg.get("W", jnp.zeros((num_scheduled, num_features))), dtype=jnp.float32)
    rho = jnp.asarray(cfg.get("rho", [0.5] * num_scheduled), dtype=jnp.float32)
    if rho.ndim == 0:
        rho = jnp.broadcast_to(rho, (num_scheduled,))
    feature_scale = jnp.asarray(cfg.get("feature_scale", feature_scale), dtype=jnp.float32)

    if W.shape != (num_scheduled, num_features):
        raise ValueError(f"gain parametrization W must have shape {(num_scheduled, num_features)}, got {W.shape}.")
    if rho.shape != (num_scheduled,):
        raise ValueError(f"gain parametrization rho must have shape {(num_scheduled,)} or scalar, got {rho.shape}.")
    if feature_scale.shape != (num_features,):
        raise ValueError(
            f"gain parametrization feature_scale must have shape {(num_features,)}, got {feature_scale.shape}."
        )

    return BoundedReferenceParams(
        W=W,
        rho=rho,
        feature_scale=feature_scale,
        scheduled_indices=jnp.asarray(scheduled_indices, dtype=jnp.int32),
    )


def to_cfg(params: BoundedReferenceParams) -> dict:
    """Serialize params to a YAML-compatible dict; :func:`from_cfg` roundtrips it."""
    import numpy as np

    return {
        "kind": KIND,
        "scheduled_indices": [int(index) for index in np.asarray(params.scheduled_indices)],
        "rho": [float(value) for value in np.asarray(params.rho).ravel()],
        "W": np.asarray(params.W, dtype=float).tolist(),
    }
