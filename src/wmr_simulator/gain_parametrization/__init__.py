"""Controller-gain parametrizations.

This package owns the selectable mapping

    base gains + parametrization parameters + rollout context -> controller gains

for closed-loop simulation and gain tuning. Implementations: the reference-only
bounded multiplicative scheduler (``bounded_reference``) and a tracking-error
MLP over all gains (``error_mlp``); new parametrizations should add a module
here and a case in :func:`params_from_cfg` plus the dispatchers below.
"""

from __future__ import annotations

import jax

from wmr_simulator.gain_parametrization import bounded_reference, error_mlp
from wmr_simulator.gain_parametrization.bounded_reference import (
    BoundedReferenceParams,
    features as bounded_reference_features,
    factors as bounded_reference_factors,
)
from wmr_simulator.gain_parametrization.error_mlp import ErrorMlpParams

DEFAULT_KIND = bounded_reference.KIND
SUPPORTED_KINDS = (bounded_reference.KIND, error_mlp.KIND)


def parametrization_kind(cfg: dict | None) -> str:
    """Return the selected gain-parametrization kind from a config block."""
    cfg = {} if cfg is None else cfg
    return str(cfg.get("kind", cfg.get("type", DEFAULT_KIND)))


def params_from_cfg(cfg: dict | None, feature_scale):
    """Build parametrization params from a controller config block.

    ``feature_scale`` is ``[v_max, omega_max]``; each parametrization derives
    its own feature normalization from it.
    """
    kind = parametrization_kind(cfg)
    if kind == bounded_reference.KIND:
        return bounded_reference.from_cfg(cfg, feature_scale)
    if kind == error_mlp.KIND:
        return error_mlp.from_cfg(cfg, feature_scale)
    raise ValueError(f"Unsupported gain parametrization kind {kind!r}; expected one of {SUPPORTED_KINDS}.")


def apply(base_gains: jax.Array, params, ref_state: jax.Array, pose_est=None, twist_est=None) -> jax.Array:
    """Return the controller gains for one geometry step.

    ``pose_est``/``twist_est`` are the estimated pose and body twist ``[v, omega]``
    fed to the controller; reference-only parametrizations ignore them.
    """
    if isinstance(params, BoundedReferenceParams):
        return bounded_reference.apply(base_gains, params, ref_state)
    if isinstance(params, ErrorMlpParams):
        return error_mlp.apply(base_gains, params, ref_state, pose_est, twist_est)
    raise ValueError(f"Unsupported gain parametrization params: {type(params).__name__}.")


def gains_over_samples(base_gains, params, ref_states, pose_states, twist_states):
    """Applied controller gains for a batch of geometry-step samples.

    Vectorizes :func:`apply` over stacked ``ref_states`` ``(N, 8)``, ``pose_states``
    ``(N, 3)`` and ``twist_states`` ``(N, 2)`` and returns ``(N, num_gains)``.
    Reference-only parametrizations ignore the pose/twist inputs. This is the
    offline counterpart to the gains a closed-loop rollout records: replaying it
    on a recorded log's reference/pose/twist recovers the gains the on-robot
    controller applied without logging them on the firmware.
    """
    import jax.numpy as jnp

    base = jnp.asarray(base_gains)
    return jax.vmap(lambda ref, pose, twist: apply(base, params, ref, pose, twist))(
        jnp.asarray(ref_states), jnp.asarray(pose_states), jnp.asarray(twist_states)
    )


def outer_gains_over_refs(base_gains: jax.Array, params, reference_states: jax.Array) -> jax.Array:
    """Parametrized gains over a reference trajectory for diagnostics/penalties.

    State-dependent parametrizations are evaluated at the on-track condition
    (zero tracking error along the reference).
    """
    if isinstance(params, BoundedReferenceParams):
        return bounded_reference.scheduled_outer_gains_over_refs(base_gains, params, reference_states)
    if isinstance(params, ErrorMlpParams):
        return error_mlp.on_reference_gains_over_refs(base_gains, params, reference_states)
    raise ValueError(f"Unsupported gain parametrization params: {type(params).__name__}.")


def num_params(params) -> int:
    """Number of trainable parameters for this gain parametrization."""
    if isinstance(params, BoundedReferenceParams):
        return bounded_reference.num_params(params)
    if isinstance(params, ErrorMlpParams):
        return error_mlp.num_params(params)
    raise ValueError(f"Unsupported gain parametrization params: {type(params).__name__}.")


def with_flat_params(theta: jax.Array, template):
    """Rebuild params from a flat trainable vector and a template.

    A zero vector must reproduce the identity parametrization for every kind
    (the LHS presearch relies on it).
    """
    if isinstance(template, BoundedReferenceParams):
        return bounded_reference.with_flat_params(theta, template)
    if isinstance(template, ErrorMlpParams):
        return error_mlp.with_flat_params(theta, template)
    raise ValueError(f"Unsupported gain parametrization template: {type(template).__name__}.")


def zero_params(template):
    """Identity parametrization with the same fixed config as ``template``."""
    if isinstance(template, BoundedReferenceParams):
        return bounded_reference.zero_params(template)
    if isinstance(template, ErrorMlpParams):
        return error_mlp.zero_params(template)
    raise ValueError(f"Unsupported gain parametrization template: {type(template).__name__}.")


def flat_params(params) -> jax.Array:
    """Flat trainable vector reproducing ``params`` via :func:`with_flat_params`.

    Used to warm-start tuning from an already-trained parametrization (e.g. the
    previous active-learning iteration) rather than the identity mapping.
    """
    if isinstance(params, BoundedReferenceParams):
        return bounded_reference.flat_params(params)
    if isinstance(params, ErrorMlpParams):
        return error_mlp.flat_params(params)
    raise ValueError(f"Unsupported gain parametrization params: {type(params).__name__}.")


def to_cfg(params) -> dict:
    """Serialize params to a YAML-compatible dict (roundtrips via :func:`params_from_cfg`)."""
    if isinstance(params, BoundedReferenceParams):
        return bounded_reference.to_cfg(params)
    if isinstance(params, ErrorMlpParams):
        return error_mlp.to_cfg(params)
    raise ValueError(f"Unsupported gain parametrization params: {type(params).__name__}.")


# Backward-compatible names for the existing bounded scheduler.
GainScheduleParams = BoundedReferenceParams
gain_schedule_features = bounded_reference_features
schedule_factors = bounded_reference_factors
apply_gain_schedule = apply
scheduled_outer_gains_over_refs = outer_gains_over_refs
schedule_num_params = num_params
with_flat_W = with_flat_params
gain_schedule_params_from_cfg = params_from_cfg
