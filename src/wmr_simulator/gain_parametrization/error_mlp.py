"""MLP gain parametrization on tracking errors and reference speeds.

A small tanh MLP maps normalized body-frame tracking errors plus the reference
speed features to one bounded multiplicative factor per *scheduled* controller
gain (``scheduled_indices``, default all five; gains not listed pass through
untouched, e.g. ``[0, 1, 2]`` keeps the motor PI gains static):

    gains[i] = base_gains[i] * clip(1 + mlp(z)[i], 0, bound),  bound >= 1

The trainable flat vector is a *delta* on a frozen random init whose final
layer is zero, so a zero delta is exactly the identity parametrization (static
controller) while the hidden layers still produce nonzero activations -- the
LHS presearch over base gains starts from the static controller and Adam sees
useful gradients from step one (an all-zero MLP would be a dead saddle).

Regularization: every weight matrix is spectrally normalized at forward time
to ``spectral_norm_cap`` (Miyato et al. 2018), which caps the Lipschitz
constant of the whole network (tanh is 1-Lipschitz) in normalized feature
space. This bounds how fast the factors can change under noisy or unseen
inputs by construction, independent of the tuning objective. 0 disables it.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

KIND = "error_mlp"

# Body-frame tracking errors (as used by the pose controller) plus the
# reference speed features of the bounded_reference scheduler.
FEATURE_NAMES = ("x_e", "y_e", "theta_e", "v_e", "omega_e", "v_d", "abs_omega_d")
NUM_FEATURES = len(FEATURE_NAMES)
NUM_GAINS = 5

_V_D_EPS = 1e-12
# Default scales for the pose-error features; velocity features are scaled by
# the robot limits (v_max, omega_max) like in the bounded scheduler.
_POS_ERROR_SCALE = 0.5   # m
_ANGLE_ERROR_SCALE = 1.0  # rad


class ErrorMlpParams(NamedTuple):
    """Trainable and fixed parameters for the error-MLP parametrization."""

    layers: tuple                 # interleaved (W0, b0, ..., Wn, bn) current raw weights
    init_layers: tuple            # frozen random init with zero final layer (delta = 0 -> identity)
    bound: jax.Array              # scalar upper factor bound (clipped to >= 1 when applied)
    init_bound: jax.Array         # frozen init bound; learnable delta is added on top
    feature_scale: jax.Array      # (NUM_FEATURES,)
    scheduled_indices: jax.Array  # (num_scheduled,) int indices into the gain vector
    learn_bound: bool             # whether the bound is part of the flat trainable vector
    seed: int                     # PRNG seed of the frozen init (kept for serialization)
    spectral_norm_cap: float      # per-matrix spectral norm cap at forward time; 0 disables


def _layer_shapes(hidden_sizes: tuple, num_outputs: int) -> list:
    sizes = (NUM_FEATURES, *hidden_sizes, num_outputs)
    shapes = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        shapes.append((fan_out, fan_in))
        shapes.append((fan_out,))
    return shapes


def _init_layers(hidden_sizes: tuple, seed: int, num_outputs: int) -> tuple:
    key = jax.random.PRNGKey(seed)
    shapes = _layer_shapes(hidden_sizes, num_outputs)
    num_layers = len(shapes) // 2
    layers = []
    for index in range(num_layers):
        weight_shape = shapes[2 * index]
        key, weight_key = jax.random.split(key)
        if index == num_layers - 1:
            weight = jnp.zeros(weight_shape, dtype=jnp.float32)
        else:
            weight = jax.random.normal(weight_key, weight_shape, dtype=jnp.float32) / math.sqrt(weight_shape[1])
        layers.append(weight)
        layers.append(jnp.zeros(shapes[2 * index + 1], dtype=jnp.float32))
    return tuple(layers)


def _spectral_norm(weight: jax.Array, num_iters: int = 30) -> jax.Array:
    """Largest singular value via power iteration (deterministic start).

    The singular vectors are treated as constants (stop-gradient), so the
    gradient of the returned sigma is the standard rank-one u v^T estimate.
    """
    eps = 1e-12
    v = jnp.full((weight.shape[1],), 1.0 / math.sqrt(weight.shape[1]), dtype=weight.dtype)
    for _ in range(num_iters):
        u = weight @ v
        u = u / (jnp.linalg.norm(u) + eps)
        v = weight.T @ u
        v = v / (jnp.linalg.norm(v) + eps)
    u = weight @ v
    u = u / (jnp.linalg.norm(u) + eps)
    u = jax.lax.stop_gradient(u)
    v = jax.lax.stop_gradient(v)
    return u @ weight @ v


def effective_layers(params: "ErrorMlpParams") -> tuple:
    """Weights as used by the forward pass: spectrally normalized to the cap.

    Each weight matrix is rescaled by ``cap / max(sigma, cap)`` -- untouched
    while its spectral norm is below the cap (in particular the zero output
    layer at the identity), hard-limited above it. Biases are unconstrained.
    """
    cap = float(params.spectral_norm_cap)
    if cap <= 0.0:
        return params.layers
    layers = []
    for index in range(0, len(params.layers), 2):
        weight = params.layers[index]
        sigma = _spectral_norm(weight)
        layers.append(weight * (cap / jnp.maximum(sigma, cap)))
        layers.append(params.layers[index + 1])
    return tuple(layers)


def _forward(layers: tuple, z: jax.Array) -> jax.Array:
    h = z
    for index in range(0, len(layers) - 2, 2):
        h = jnp.tanh(layers[index] @ h + layers[index + 1])
    return layers[-2] @ h + layers[-1]


def hidden_sizes(params: ErrorMlpParams) -> tuple:
    return tuple(int(weight.shape[0]) for weight in params.layers[0:-2:2])


def features(
    ref_state: jax.Array,
    pose_est: jax.Array,
    twist_est: jax.Array,
    feature_scale: jax.Array,
) -> jax.Array:
    """Normalized feature vector ``z`` for one geometry step.

    ``ref_state`` layout: [x, y, theta, vx, vy, omega, ax, ay];
    ``pose_est`` is the estimated pose fed to the controller;
    ``twist_est`` is the estimated body twist ``[v, omega]`` from the encoders.
    """
    theta = pose_est[2]
    dx = ref_state[0] - pose_est[0]
    dy = ref_state[1] - pose_est[1]
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_e = dx * cos_t + dy * sin_t
    y_e = -dx * sin_t + dy * cos_t
    theta_e = (ref_state[2] - theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    v_d = jnp.sqrt(ref_state[3] ** 2 + ref_state[4] ** 2 + _V_D_EPS)
    omega_d = ref_state[5]
    v_e = v_d - twist_est[0]
    omega_e = omega_d - twist_est[1]
    raw = jnp.stack([x_e, y_e, theta_e, v_e, omega_e, v_d, jnp.abs(omega_d)])
    return raw / feature_scale


def factors(
    params: ErrorMlpParams,
    ref_state: jax.Array,
    pose_est: jax.Array,
    twist_est: jax.Array,
) -> jax.Array:
    """Bounded multiplicative factors, shape ``(num_scheduled,)``."""
    z = features(ref_state, pose_est, twist_est, params.feature_scale)
    raw = _forward(effective_layers(params), z)
    bound = jnp.clip(params.bound, min=1.0)
    return jnp.clip(1.0 + raw, min=0.0, max=bound)


def apply(
    base_gains: jax.Array,
    params: ErrorMlpParams,
    ref_state: jax.Array,
    pose_est: jax.Array,
    twist_est: jax.Array,
) -> jax.Array:
    """Full parametrized gain vector for one geometry step.

    Only the ``scheduled_indices`` entries are scaled; the rest pass through.
    """
    scheduled_factors = factors(params, ref_state, pose_est, twist_est)
    return base_gains.at[params.scheduled_indices].multiply(scheduled_factors, unique_indices=True)


def on_reference_gains_over_refs(
    base_gains: jax.Array,
    params: ErrorMlpParams,
    reference_states: jax.Array,
) -> jax.Array:
    """Gains along a reference at zero tracking error, for diagnostics/penalties.

    The rollout gains depend on the realized errors; this evaluates the
    schedule at the on-track condition (pose on the reference, twist matching
    it), so only the reference speed features vary. Used by the gain-rate
    smoothness penalty, which therefore shapes the on-track schedule only.
    """

    def on_track_gains(ref_state):
        pose_est = ref_state[:3]
        v_d = jnp.sqrt(ref_state[3] ** 2 + ref_state[4] ** 2 + _V_D_EPS)
        twist_est = jnp.stack([v_d, ref_state[5]])
        return apply(base_gains, params, ref_state, pose_est, twist_est)

    return jax.vmap(on_track_gains)(reference_states[:-1])


def _num_weight_params(params: ErrorMlpParams) -> int:
    return int(sum(layer.size for layer in params.init_layers))


def num_params(params: ErrorMlpParams) -> int:
    """Number of trainable parameters (weight deltas + optional bound delta)."""
    return _num_weight_params(params) + (1 if params.learn_bound else 0)


def with_flat_params(theta: jax.Array, template: ErrorMlpParams) -> ErrorMlpParams:
    """Rebuild params from a flat delta vector on the frozen init."""
    theta = jnp.asarray(theta, dtype=jnp.float32)
    layers = []
    offset = 0
    for init_layer in template.init_layers:
        size = init_layer.size
        delta = theta[offset : offset + size].reshape(init_layer.shape)
        layers.append(init_layer + delta)
        offset += size
    bound = template.init_bound + theta[offset] if template.learn_bound else template.init_bound
    return template._replace(layers=tuple(layers), bound=bound)


def zero_params(template: ErrorMlpParams) -> ErrorMlpParams:
    """Identity parametrization with the same fixed config as ``template``."""
    return template._replace(layers=template.init_layers, bound=template.init_bound)


def flat_params(params: ErrorMlpParams) -> jax.Array:
    """Flat delta vector reproducing ``params`` via :func:`with_flat_params`."""
    deltas = [
        (layer - init_layer).reshape(-1)
        for layer, init_layer in zip(params.layers, params.init_layers)
    ]
    if params.learn_bound:
        deltas.append(jnp.reshape(params.bound - params.init_bound, (1,)))
    return jnp.concatenate(deltas)


def from_cfg(cfg: dict | None, feature_scale) -> ErrorMlpParams:
    """Parse a YAML block into error-MLP parametrization params."""
    cfg = {} if cfg is None else cfg

    hidden = tuple(int(size) for size in cfg.get("hidden_sizes", (16,)))
    if not hidden or any(size <= 0 for size in hidden):
        raise ValueError(f"gain parametrization hidden_sizes must be positive, got {list(hidden)}.")
    seed = int(cfg.get("seed", 0))
    learn_bound = bool(cfg.get("learn_bound", False))
    bound = jnp.asarray(float(cfg.get("bound", 2.0)), dtype=jnp.float32)
    spectral_norm_cap = float(cfg.get("spectral_norm_cap", 1.0))
    scheduled_indices = tuple(int(index) for index in cfg.get("scheduled_indices", range(NUM_GAINS)))
    if (
        not scheduled_indices
        or len(set(scheduled_indices)) != len(scheduled_indices)
        or any(index < 0 or index >= NUM_GAINS for index in scheduled_indices)
    ):
        raise ValueError(
            f"gain parametrization scheduled_indices must be unique indices in [0, {NUM_GAINS - 1}], "
            f"got {list(scheduled_indices)}."
        )

    v_max, omega_max = (float(value) for value in feature_scale)
    default_scale = [
        _POS_ERROR_SCALE,
        _POS_ERROR_SCALE,
        _ANGLE_ERROR_SCALE,
        v_max,
        omega_max,
        v_max,
        omega_max,
    ]
    feature_scale = jnp.asarray(cfg.get("feature_scale", default_scale), dtype=jnp.float32)
    if feature_scale.shape != (NUM_FEATURES,):
        raise ValueError(
            f"gain parametrization feature_scale must have shape {(NUM_FEATURES,)}, got {feature_scale.shape}."
        )

    init_layers = _init_layers(hidden, seed, num_outputs=len(scheduled_indices))
    template = ErrorMlpParams(
        layers=init_layers,
        init_layers=init_layers,
        bound=bound,
        init_bound=bound,
        feature_scale=feature_scale,
        scheduled_indices=jnp.asarray(scheduled_indices, dtype=jnp.int32),
        learn_bound=learn_bound,
        seed=seed,
        spectral_norm_cap=spectral_norm_cap,
    )
    theta = jnp.asarray(cfg.get("theta", jnp.zeros(_num_weight_params(template))), dtype=jnp.float32)
    if theta.shape != (_num_weight_params(template),):
        raise ValueError(
            f"gain parametrization theta must have shape {(_num_weight_params(template),)}, got {theta.shape}."
        )
    layers = with_flat_params(
        jnp.concatenate([theta, jnp.zeros(1, dtype=jnp.float32)]) if learn_bound else theta,
        template,
    ).layers
    return template._replace(layers=layers)


def to_cfg(params: ErrorMlpParams) -> dict:
    """Serialize params to a YAML-compatible dict; :func:`from_cfg` roundtrips it."""
    import numpy as np

    theta = np.asarray(
        jnp.concatenate(
            [
                (layer - init_layer).reshape(-1)
                for layer, init_layer in zip(params.layers, params.init_layers)
            ]
        ),
        dtype=float,
    )
    return {
        "kind": KIND,
        "hidden_sizes": [int(size) for size in hidden_sizes(params)],
        "scheduled_indices": [int(index) for index in np.asarray(params.scheduled_indices)],
        "seed": int(params.seed),
        "bound": float(params.bound),
        "learn_bound": bool(params.learn_bound),
        "spectral_norm_cap": float(params.spectral_norm_cap),
        "feature_scale": [float(value) for value in np.asarray(params.feature_scale)],
        "theta": theta.tolist(),
    }
