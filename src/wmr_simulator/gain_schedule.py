"""Bounded multiplicative gain scheduling for the outer geometric tracking gains.

The outer pose-control gains (kx, ky, kth) are represented as a base gain times a
bounded, reference-dependent factor:

    z        = raw_feature / feature_scale             # normalized features, ~[0, 1]
    factor_i = 1.0 + rho_i * clip(W_i . z, -1.0, 1.0)  # bounded multiplicative factor
    gain_i   = base_i * factor_i                       # for i in scheduled_indices

``base_i`` are the ordinary static outer gains (entries of the full 6-gain vector).
``W`` is the only schedule-specific parameter; with ``W = 0`` the factors are exactly
``1.0`` for every reference state, so the controller reduces to the static baseline
with gains ``base``. Base gains and ``W`` are optimized jointly (single stage).

There is deliberately no additive bias inside the clip: a constant offset there would
just re-scale the base gain (i.e. duplicate it), so the constant level is owned by
``base`` and ``W`` owns the pure scheduling.

Features (v1, smooth reference-dependent only, no tracking error):
    z = [v_d / v_max, |omega_d| / omega_max]
where ``v_d = sqrt(vx_d**2 + vy_d**2)`` and ``omega_d`` is ``ref_state[5]``. The
feature scales are auto-derived from the robot velocity limits.

The final policy uses only multiply/add/clip, so it ports directly to RP2040 Rust.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Registry mapping YAML feature names to the column they occupy in ``W``/``z``.
FEATURE_NAMES = ("v_d", "abs_omega_d")
_V_D_EPS = 1e-12


class GainScheduleParams(NamedTuple):
    """Gain-schedule parameters.

    ``W`` is the trainable schedule parameter. ``rho``, ``feature_scale`` and
    ``scheduled_indices`` are fixed configuration.
    """

    W: jax.Array                    # (num_scheduled, num_features)
    rho: jax.Array                  # (num_scheduled,)
    feature_scale: jax.Array        # (num_features,)
    scheduled_indices: jax.Array    # (num_scheduled,) int indices into the 6-gain vector


def gain_schedule_features(ref_state: jax.Array, feature_scale: jax.Array) -> jax.Array:
    """Return the normalized feature vector ``z`` for a single reference state.

    ``ref_state`` layout: [x, y, theta, vx, vy, omega, ax, ay].
    """
    vx_d = ref_state[3]
    vy_d = ref_state[4]
    omega_d = ref_state[5]
    v_d = jnp.sqrt(vx_d**2 + vy_d**2 + _V_D_EPS)
    raw = jnp.stack([v_d, jnp.abs(omega_d)])
    return raw / feature_scale


def schedule_factors(params: GainScheduleParams, ref_state: jax.Array) -> jax.Array:
    """Bounded multiplicative factors, shape ``(num_scheduled,)``.

    Equals ``1.0`` for every input when ``W == 0``.
    """
    z = gain_schedule_features(ref_state, params.feature_scale)
    lin = params.W @ z
    return 1.0 + params.rho * jnp.clip(lin, -1.0, 1.0)


def apply_gain_schedule(
    base_gains: jax.Array,
    params: GainScheduleParams,
    ref_state: jax.Array,
) -> jax.Array:
    """Full scheduled 6-gain vector for a single reference state.

    Scheduled indices are multiplied by their factors; all other gains
    (the inner motor gains) are passed through unchanged.
    """
    factors = schedule_factors(params, ref_state)
    # scheduled_indices are distinct, so unique_indices=True is valid and enables
    # scatter_mul gradients (JAX rejects them otherwise).
    return base_gains.at[params.scheduled_indices].multiply(factors, unique_indices=True)


def scheduled_outer_gains_over_refs(
    base_gains: jax.Array,
    params: GainScheduleParams,
    reference_states: jax.Array,
) -> jax.Array:
    """Scheduled outer gains over a reference trajectory.

    Returns shape ``(T, num_scheduled)`` where ``T = reference_states.shape[0] - 1``,
    matching the geometry steps scanned in ``run_closed_loop``. Pure function of the
    reference trajectory (independent of the rollout), used for the smoothness penalty
    and for diagnostics.
    """
    refs = reference_states[:-1]
    base_outer = base_gains[params.scheduled_indices]
    factors = jax.vmap(lambda ref: schedule_factors(params, ref))(refs)
    return base_outer[None, :] * factors


def schedule_num_params(params: GainScheduleParams) -> int:
    """Number of trainable schedule parameters (the size of ``W``)."""
    return int(params.W.size)


def with_flat_W(theta_w: jax.Array, template: GainScheduleParams) -> GainScheduleParams:
    """Rebuild ``GainScheduleParams`` with ``W`` filled from a flat vector."""
    num_scheduled, num_features = template.W.shape
    W = jnp.asarray(theta_w, dtype=jnp.float32).reshape(num_scheduled, num_features)
    return template._replace(W=W)


def gain_schedule_params_from_cfg(cfg: dict | None, feature_scale) -> GainScheduleParams:
    """Parse a ``controller.gain_schedule`` YAML block into ``GainScheduleParams``.

    Tolerant of a missing/empty block (returns an identity template with ``W = 0``),
    so callers always get a valid template. ``feature_scale`` is provided by the
    caller (auto-derived from the robot velocity limits) but may be overridden by a
    ``feature_scale`` entry in ``cfg``.
    """
    cfg = {} if cfg is None else cfg
    num_features = len(FEATURE_NAMES)

    feature_names = tuple(cfg.get("feature_names", FEATURE_NAMES))
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            f"gain_schedule.feature_names must be {list(FEATURE_NAMES)} for v1, got {list(feature_names)}."
        )

    scheduled_indices = tuple(int(i) for i in cfg.get("scheduled_indices", (0, 1, 2)))
    num_scheduled = len(scheduled_indices)

    W = jnp.asarray(cfg.get("W", jnp.zeros((num_scheduled, num_features))), dtype=jnp.float32)
    rho = jnp.asarray(cfg.get("rho", [0.5] * num_scheduled), dtype=jnp.float32)
    if rho.ndim == 0:
        rho = jnp.broadcast_to(rho, (num_scheduled,))
    feature_scale = jnp.asarray(cfg.get("feature_scale", feature_scale), dtype=jnp.float32)

    if W.shape != (num_scheduled, num_features):
        raise ValueError(f"gain_schedule.W must have shape {(num_scheduled, num_features)}, got {W.shape}.")
    if rho.shape != (num_scheduled,):
        raise ValueError(f"gain_schedule.rho must have shape {(num_scheduled,)} or scalar, got {rho.shape}.")
    if feature_scale.shape != (num_features,):
        raise ValueError(
            f"gain_schedule.feature_scale must have shape {(num_features,)}, got {feature_scale.shape}."
        )

    return GainScheduleParams(
        W=W,
        rho=rho,
        feature_scale=feature_scale,
        scheduled_indices=jnp.asarray(scheduled_indices, dtype=jnp.int32),
    )
