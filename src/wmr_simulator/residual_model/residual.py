"""Learned residual dynamics on the body-frame twist.

The nominal model (first-order motor lag + ideal kinematics) predicts a body
twist ``(v_nom, omega_nom)``. The residual corrects it additively:

    [delta_v, delta_omega] = residual(v_nom, omega_nom, v_cmd, omega_cmd)
    v = v_nom + delta_v,   omega = omega_nom + delta_omega,   v_y = 0

where ``(v_cmd, omega_cmd)`` is the twist of the commanded wheel speeds
``max_wheel_speed * duty``. The residual is a mixture of ``K`` small MLP experts
with spectrally normalized weights. A Gaussian gate over k-means centers in the
normalized descriptor space routes between them, and a null expert drives the
residual to zero more than ``ood_sigma`` bandwidths away from every center. The
last layer is zero-initialized, so an untrained ensemble is the nominal model.

This module also builds the training data from Pololu logs and trains the
ensemble (one-step, or closed-loop via ``residual_model.closed_loop``).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml


# Descriptor of the residual and its gate: nominal twist and commanded twist.
RESIDUAL_FEATURE_NAMES = ("v_nom", "omega_nom", "v_cmd", "omega_cmd")
RESIDUAL_INPUT_DIM = len(RESIDUAL_FEATURE_NAMES)
RESIDUAL_OUTPUT_DIM = 2  # [delta_v, delta_omega]

TARGET_LABELS = (r"$\Delta v$ [m/s]", r"$\Delta \omega$ [rad/s]")

# Defaults for the ensemble geometry / regularization.
DEFAULT_NUM_EXPERTS = 4
DEFAULT_HIDDEN_SIZES = (16, 16)
DEFAULT_SPECTRAL_NORM_CAP = 1.0
# Gaussian gate bandwidth = (root-mean intra-cluster distance) * this factor.
DEFAULT_GATE_BANDWIDTH_SCALE = 1.5
# Distance [bandwidths] beyond which the null expert wins (residual -> 0).
DEFAULT_OOD_SIGMA = 3.0


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class ResidualEnsemble(eqx.Module):
    """Mixture of spectrally-normalized expert MLPs with an OOD gate.

    ``layers`` holds one stacked array per (weight, bias) with a leading expert
    axis of length K. ``centers``/``scales`` are the fixed k-means gate in
    normalized descriptor space; the normalization stats are part of the model.
    """

    layers: tuple                 # stacked expert layers: (W0[K,h,in], b0[K,h], ..., Wn[K,out,h], bn[K,out])
    centers: jax.Array            # (K, input_dim) gate centers in normalized space
    scales: jax.Array             # (K,) Gaussian gate bandwidths (normalized space)
    ood_sigma: jax.Array          # scalar: null-expert distance in bandwidths
    input_mean: jax.Array
    input_std: jax.Array
    target_mean: jax.Array        # kept at zero (residual: zero output == nominal)
    target_std: jax.Array
    spectral_norm_cap: float = eqx.field(static=True)


def _layer_shapes(input_dim: int, hidden_sizes: tuple, output_dim: int) -> list:
    sizes = (input_dim, *hidden_sizes, output_dim)
    shapes = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        shapes.append((fan_out, fan_in))
        shapes.append((fan_out,))
    return shapes


def _init_expert_layers(
    key: jax.Array, num_experts: int, input_dim: int, hidden_sizes: tuple, output_dim: int
) -> tuple:
    """Stacked (leading K axis) weights/biases; last layer zero (zero residual)."""
    shapes = _layer_shapes(input_dim, hidden_sizes, output_dim)
    num_layers = len(shapes) // 2
    layers = []
    for index in range(num_layers):
        weight_shape = (num_experts, *shapes[2 * index])
        bias_shape = (num_experts, *shapes[2 * index + 1])
        if index == num_layers - 1:
            weight = jnp.zeros(weight_shape, dtype=jnp.float32)
        else:
            key, weight_key = jax.random.split(key)
            fan_in = shapes[2 * index][1]
            weight = jax.random.normal(weight_key, weight_shape, dtype=jnp.float32) / math.sqrt(fan_in)
        layers.append(weight)
        layers.append(jnp.zeros(bias_shape, dtype=jnp.float32))
    return tuple(layers)


def init_residual_model(
    key: jax.Array,
    input_dim: int = RESIDUAL_INPUT_DIM,
    output_dim: int = RESIDUAL_OUTPUT_DIM,
    num_experts: int = DEFAULT_NUM_EXPERTS,
    hidden_sizes: tuple = DEFAULT_HIDDEN_SIZES,
    spectral_norm_cap: float = DEFAULT_SPECTRAL_NORM_CAP,
    ood_sigma: float = DEFAULT_OOD_SIGMA,
) -> ResidualEnsemble:
    """Fresh ensemble with identity normalization and a placeholder gate
    (filled from k-means by training)."""
    hidden_sizes = tuple(int(h) for h in hidden_sizes)
    return ResidualEnsemble(
        layers=_init_expert_layers(key, num_experts, input_dim, hidden_sizes, output_dim),
        centers=jnp.zeros((num_experts, input_dim), dtype=jnp.float32),
        scales=jnp.ones((num_experts,), dtype=jnp.float32),
        ood_sigma=jnp.asarray(ood_sigma, dtype=jnp.float32),
        input_mean=jnp.zeros(input_dim, dtype=jnp.float32),
        input_std=jnp.ones(input_dim, dtype=jnp.float32),
        target_mean=jnp.zeros(output_dim, dtype=jnp.float32),
        target_std=jnp.ones(output_dim, dtype=jnp.float32),
        spectral_norm_cap=float(spectral_norm_cap),
    )


def residual_features(nominal_v, nominal_omega, commanded_v, commanded_omega) -> jax.Array:
    """Assemble the descriptor [v_nom, omega_nom, v_cmd, omega_cmd]."""
    return jnp.stack(
        [
            jnp.asarray(nominal_v, dtype=jnp.float32),
            jnp.asarray(nominal_omega, dtype=jnp.float32),
            jnp.asarray(commanded_v, dtype=jnp.float32),
            jnp.asarray(commanded_omega, dtype=jnp.float32),
        ]
    )


def _spectral_norm(weight: jax.Array, num_iters: int = 15) -> jax.Array:
    """Largest singular value via power iteration (singular vectors stop-gradient)."""
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


def _normalized_weight(weight: jax.Array, cap: float) -> jax.Array:
    """Weight rescaled by cap / max(sigma, cap): untouched below the cap,
    hard-limited above it (cap <= 0 disables)."""
    if cap <= 0.0:
        return weight
    sigma = _spectral_norm(weight)
    return weight * (cap / jnp.maximum(sigma, cap))


def _expert_forward(layers: tuple, z: jax.Array, cap: float) -> jax.Array:
    """One expert's normalized-residual output (tanh hidden, linear out)."""
    h = z
    for index in range(0, len(layers) - 2, 2):
        weight = _normalized_weight(layers[index], cap)
        h = jnp.tanh(weight @ h + layers[index + 1])
    weight = _normalized_weight(layers[-2], cap)
    return weight @ h + layers[-1]


def _gate_weights(model: ResidualEnsemble, z: jax.Array) -> jax.Array:
    """Softmax responsibilities of the K experts, ``r_k = exp(-1/2 ||z - c_k||^2 / s_k^2)``,
    against a null expert with ``r_0 = exp(-1/2 ood_sigma^2)`` (the remainder)."""
    distance_sq = jnp.sum((z - model.centers) ** 2, axis=1)  # (K,)
    responsibilities = jnp.exp(-0.5 * distance_sq / (model.scales**2 + 1e-12))
    null = jnp.exp(-0.5 * model.ood_sigma**2)
    return responsibilities / (null + jnp.sum(responsibilities) + 1e-12)


def _shaped_delta(model: ResidualEnsemble, z: jax.Array) -> jax.Array:
    """Physical residual delta from a single normalized descriptor ``z``."""
    cap = model.spectral_norm_cap
    outputs = jax.vmap(lambda layers: _expert_forward(layers, z, cap))(model.layers)  # (K, out)
    weights = _gate_weights(model, z)  # (K,)
    blended = weights @ outputs  # (out,) normalized residual
    return blended * model.target_std + model.target_mean


def apply_residual_model(model: ResidualEnsemble, features: jax.Array) -> jax.Array:
    """Physical residual twist [delta_v, delta_omega] for one or a batch of features."""
    features = jnp.asarray(features, dtype=jnp.float32)

    def single(feature_vector):
        z = (feature_vector - model.input_mean) / model.input_std
        return _shaped_delta(model, z)

    if features.ndim == 1:
        return single(features)
    return jax.vmap(single)(features)


def gate_weights(model: ResidualEnsemble, features: jax.Array) -> jax.Array:
    """Per-expert gate weights (and, as the remainder to 1, the null weight) for
    one or a batch of *physical* feature vectors -- for diagnostics/plots."""
    features = jnp.asarray(features, dtype=jnp.float32)

    def single(feature_vector):
        z = (feature_vector - model.input_mean) / model.input_std
        return _gate_weights(model, z)

    if features.ndim == 1:
        return single(features)
    return jax.vmap(single)(features)


def residual_corrected_twist(
    model: ResidualEnsemble,
    features: jax.Array,
    nominal_v: jax.Array,
    nominal_omega: jax.Array,
):
    """Corrected body twist (v_x, v_y = 0, omega) from the nominal twist and features."""
    delta = apply_residual_model(model, features)
    v_x = nominal_v + delta[..., 0]
    v_y = jnp.zeros_like(v_x)
    omega = nominal_omega + delta[..., 1]
    return v_x, v_y, omega


# ---------------------------------------------------------------------------
# Dataset construction from Pololu logs
# ---------------------------------------------------------------------------


def build_residual_dataset(
    log,
    params,
    *,
    min_dt: float = 1e-6,
    max_dt_factor: float = 5.0,
    resample_uniform: bool = True,
) -> dict:
    """Supervised one-step samples from one experiment log.

    Measured twist: the log's smoothed body twists (finite differences of the
    poses when it has none). Nominal twist: the encoder wheel speeds through
    the motor lag (as in ``DiffDrive.step``) and the ideal kinematics.
    Commanded twist: the kinematics of ``max_wheel_speed * duty``. The target
    is ``measured - nominal``.

    ``resample_uniform`` interpolates poses (and twists) onto a uniform
    median-dt grid; ``max_dt_factor`` drops mocap intervals longer than that
    multiple of the median. Returns float32 ``features`` (N, 4), ``targets``
    (N, 2), ``time_s``, ``dt``, and the twists and poses for diagnostics.
    """
    pose_time_s = np.asarray(log.pose.time_s, dtype=float)
    pose_states = np.asarray(log.pose.states, dtype=float)
    pose_twists = getattr(log.pose, "twists", None)
    if pose_twists is not None:
        pose_twists = np.asarray(pose_twists, dtype=float)
    wheel_time_s = np.asarray(log.wheel.time_s, dtype=float)
    wheel_speeds = np.asarray(log.wheel.speeds, dtype=float)
    duty_cycle = np.asarray(log.wheel.duty_cycle, dtype=float)

    if resample_uniform:
        pose_time_s, pose_states, pose_twists = _resample_poses_uniform(
            pose_time_s, pose_states, pose_twists
        )

    dt = np.diff(pose_time_s)
    if pose_twists is not None:
        measured_twist = pose_twists[:-1]
    else:
        theta_unwrapped = np.unwrap(pose_states[:, 2])
        safe_dt = np.maximum(dt, 1e-9)
        dx = np.diff(pose_states[:, 0])
        dy = np.diff(pose_states[:, 1])
        theta = pose_states[:-1, 2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        v_x_meas = (dx * cos_t + dy * sin_t) / safe_dt
        v_y_meas = (-dx * sin_t + dy * cos_t) / safe_dt
        omega_meas = np.diff(theta_unwrapped) / safe_dt
        measured_twist = np.column_stack([v_x_meas, v_y_meas, omega_meas])

    interval_time = pose_time_s[:-1]
    u_r_enc = np.interp(interval_time, wheel_time_s, wheel_speeds[:, 0])
    u_l_enc = np.interp(interval_time, wheel_time_s, wheel_speeds[:, 1])
    r = float(params.wheel_radius)
    effective_wheelbase = float(params.base_diameter)

    # Motor lag as in DiffDrive.step:
    # lag[t] = alpha * u_enc[t-1] + (1 - alpha) * max_speed * duty[t].
    duty = _zero_order_hold(interval_time, wheel_time_s, duty_cycle)
    max_speed = float(params.max_wheel_speed)
    tau = float(params.time_constant)
    alpha = np.exp(-dt / tau) if tau >= 1e-3 else np.zeros_like(dt)
    prev_u_r = np.concatenate([u_r_enc[:1], u_r_enc[:-1]])
    prev_u_l = np.concatenate([u_l_enc[:1], u_l_enc[:-1]])
    lag_r = alpha * prev_u_r + (1.0 - alpha) * max_speed * duty[:, 0]
    lag_l = alpha * prev_u_l + (1.0 - alpha) * max_speed * duty[:, 1]

    # Nominal twist from the lag wheel through the ideal kinematics.
    v_nom = 0.5 * r * (lag_r + lag_l)
    omega_nom = r * (lag_r - lag_l) / effective_wheelbase
    nominal_twist = np.column_stack([v_nom, np.zeros_like(v_nom), omega_nom])

    # Commanded twist of the target wheel speeds.
    target_r = max_speed * duty[:, 0]
    target_l = max_speed * duty[:, 1]
    v_cmd = 0.5 * r * (target_r + target_l)
    omega_cmd = r * (target_r - target_l) / effective_wheelbase

    features = np.column_stack([v_nom, omega_nom, v_cmd, omega_cmd])
    targets = np.column_stack(
        [measured_twist[:, 0] - v_nom, measured_twist[:, 2] - omega_nom]
    )
    assert features.shape[1] == RESIDUAL_INPUT_DIM
    assert targets.shape[1] == RESIDUAL_OUTPUT_DIM

    median_dt = float(np.median(dt[dt > min_dt])) if np.any(dt > min_dt) else 0.0
    mask = (dt > min_dt) & (dt <= max_dt_factor * max(median_dt, min_dt))

    return {
        "time_s": interval_time[mask].astype(np.float32),
        "dt": dt[mask].astype(np.float32),
        "features": features[mask].astype(np.float32),
        "targets": targets[mask].astype(np.float32),
        "measured_twist": measured_twist[mask].astype(np.float32),
        "nominal_twist": nominal_twist[mask].astype(np.float32),
        "initial_pose": pose_states[0].astype(np.float32),
        "pose_states": pose_states.astype(np.float32),
        "pose_time_s": pose_time_s.astype(np.float32),
        "mask": mask,
    }


def stack_datasets(datasets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([dataset["features"] for dataset in datasets], axis=0)
    targets = np.concatenate([dataset["targets"] for dataset in datasets], axis=0)
    return features, targets


def split_datasets_by_file(
    datasets: list[dict],
    validation_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Train/validation indices into ``datasets``, split by whole file (at
    least one file is held out)."""
    if len(datasets) < 2:
        raise ValueError("Need at least 2 logs to hold one out for validation.")
    indices = np.arange(len(datasets))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    num_validation = int(np.floor(len(indices) * validation_split))
    num_validation = min(max(num_validation, 1), len(indices) - 1)
    return list(indices[num_validation:]), list(indices[:num_validation])


def normalization_stats(values: np.ndarray, min_std: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), min_std)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Gate (k-means over the normalized descriptor)
# ---------------------------------------------------------------------------


def _kmeans(points: np.ndarray, num_clusters: int, seed: int, iters: int = 50) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding (numpy, deterministic)."""
    rng = np.random.default_rng(seed)
    num_points = len(points)
    num_clusters = min(num_clusters, num_points)
    # k-means++ initialization.
    centers = [points[rng.integers(num_points)]]
    for _ in range(1, num_clusters):
        distance_sq = np.min(
            [np.sum((points - center) ** 2, axis=1) for center in centers], axis=0
        )
        total = distance_sq.sum()
        probabilities = distance_sq / total if total > 0 else np.full(num_points, 1.0 / num_points)
        centers.append(points[rng.choice(num_points, p=probabilities)])
    centers = np.stack(centers)
    for _ in range(iters):
        assignments = np.argmin(
            np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
        )
        new_centers = np.stack(
            [
                points[assignments == k].mean(axis=0) if np.any(assignments == k) else centers[k]
                for k in range(num_clusters)
            ]
        )
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers.astype(np.float32)


def _fit_gate(
    normalized_inputs: np.ndarray,
    num_experts: int,
    bandwidth_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Centers (K, D) and Gaussian bandwidths (K,) in normalized descriptor space.

    A bandwidth is the root-mean distance of a cluster's points to its center
    times ``bandwidth_scale``; empty or singleton clusters get the median.
    """
    centers = _kmeans(normalized_inputs, num_experts, seed)
    assignments = np.argmin(
        np.sum((normalized_inputs[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1
    )
    scales = np.empty(len(centers), dtype=np.float32)
    for k in range(len(centers)):
        member_distance_sq = np.sum((normalized_inputs[assignments == k] - centers[k]) ** 2, axis=1)
        scales[k] = np.sqrt(member_distance_sq.mean()) if member_distance_sq.size > 1 else np.nan
    fallback = np.nanmedian(scales) if np.isfinite(scales).any() else 1.0
    scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, fallback)
    return centers, (scales * bandwidth_scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Training (single step)
# ---------------------------------------------------------------------------


def train_residual_ensemble(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int = 0,
    num_experts: int = DEFAULT_NUM_EXPERTS,
    hidden_sizes: tuple = DEFAULT_HIDDEN_SIZES,
    spectral_norm_cap: float = DEFAULT_SPECTRAL_NORM_CAP,
    gate_bandwidth_scale: float = DEFAULT_GATE_BANDWIDTH_SCALE,
    ood_sigma: float = DEFAULT_OOD_SIGMA,
    epochs: int = 400,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    output_reg_weight: float = 1e-2,
):
    """Train the expert ensemble on one-step twist residuals with Adam.

    Inputs and targets are normalized (targets by std only). The gate is fixed
    from k-means before training; only the expert weights are optimized. The
    loss is the prediction MSE plus ``output_reg_weight`` times the mean
    squared normalized output. Returns (model, history) with per-epoch
    normalized MSE.
    """
    import optax

    input_mean, input_std = normalization_stats(train_features)
    target_mean = np.zeros(train_targets.shape[1], dtype=np.float32)
    _, target_std = normalization_stats(train_targets)

    x_train = ((train_features - input_mean) / input_std).astype(np.float32)
    y_train = (train_targets / target_std).astype(np.float32)
    centers, scales = _fit_gate(x_train, num_experts, gate_bandwidth_scale, seed)
    num_experts = len(centers)

    model = ResidualEnsemble(
        layers=_init_expert_layers(
            jax.random.PRNGKey(seed),
            num_experts,
            train_features.shape[1],
            tuple(hidden_sizes),
            train_targets.shape[1],
        ),
        centers=jnp.asarray(centers),
        scales=jnp.asarray(scales),
        ood_sigma=jnp.asarray(ood_sigma, dtype=jnp.float32),
        input_mean=jnp.asarray(input_mean),
        input_std=jnp.asarray(input_std),
        target_mean=jnp.asarray(target_mean),
        target_std=jnp.asarray(target_std),
        spectral_norm_cap=float(spectral_norm_cap),
    )

    x_train_j = jnp.asarray(x_train)
    y_train_j = jnp.asarray(y_train)
    has_validation = len(validation_features) > 0
    if has_validation:
        x_val = jnp.asarray(((validation_features - input_mean) / input_std).astype(np.float32))
        y_val = jnp.asarray((validation_targets / target_std).astype(np.float32))

    # Only the expert weights (model.layers) are trained; the k-means gate and
    # the normalization stats are frozen.
    trainable_spec = jax.tree_util.tree_map(lambda _: False, model)
    trainable_spec = eqx.tree_at(
        lambda m: m.layers,
        trainable_spec,
        replace=jax.tree_util.tree_map(lambda _: True, model.layers),
    )
    trainable, frozen = eqx.partition(model, trainable_spec)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(trainable)

    def normalized_prediction(net, z):
        # Blended normalized residual.
        return _shaped_delta(net, z) / net.target_std

    def prediction_mse(net, x, y):
        predictions = jax.vmap(lambda z: normalized_prediction(net, z))(x)
        return jnp.mean((predictions - y) ** 2)

    def objective(trainable, frozen, x, y):
        net = eqx.combine(trainable, frozen)
        predictions = jax.vmap(lambda z: normalized_prediction(net, z))(x)
        mse = jnp.mean((predictions - y) ** 2)
        output_norm = jnp.mean(jnp.sum(predictions**2, axis=1))
        return mse + output_reg_weight * output_norm

    @eqx.filter_jit
    def train_step(trainable, frozen, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(objective)(trainable, frozen, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
        return trainable, opt_state, loss

    eval_mse = eqx.filter_jit(prediction_mse)

    rng = np.random.default_rng(seed)
    num_samples = len(x_train_j)
    batch_size = min(batch_size, num_samples)
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(epochs):
        permutation = rng.permutation(num_samples)
        epoch_losses = []
        for start in range(0, num_samples - batch_size + 1, batch_size):
            batch = permutation[start : start + batch_size]
            trainable, opt_state, loss = train_step(
                trainable, frozen, opt_state, x_train_j[batch], y_train_j[batch]
            )
            epoch_losses.append(float(loss))
        history["train_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
        if has_validation:
            history["validation_loss"].append(
                float(eval_mse(eqx.combine(trainable, frozen), x_val, y_val))
            )
        else:
            history["validation_loss"].append(float("nan"))
        if epoch % max(epochs // 20, 1) == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch + 1:4d}/{epochs}: train {history['train_loss'][-1]:.6f}"
                f"  validation {history['validation_loss'][-1]:.6f}"
            )

    model = eqx.combine(trainable, frozen)
    return model, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_predictions(model: ResidualEnsemble, features: np.ndarray, targets: np.ndarray) -> dict:
    """Per-channel RMSE of the model against the zero-residual baseline."""
    predictions = np.asarray(apply_residual_model(model, jnp.asarray(features)))
    return {
        "predictions": predictions,
        "rmse": np.sqrt(np.mean((predictions - targets) ** 2, axis=0)),
        "baseline_rmse": np.sqrt(np.mean(targets**2, axis=0)),
    }


def residual_diagnostics(
    model: ResidualEnsemble,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
) -> dict:
    """Diagnostics of a trained checkpoint.

    * ``target_rms``: RMS of what the nominal model misses, [dv, domega].
    * ``residual_rmse`` / ``explained_fraction``: how much of it the ensemble
      fits (1 - rmse^2 / target_rms^2).
    * ``input_std``: the per-feature normalization.
    * ``gate_weight_train``: total expert weight (1 - null weight) on the
      training features.
    * ``max_abs_prediction``: the largest correction on the training data.
    """
    def split(features, targets):
        if len(features) == 0:
            return None
        evaluation = evaluate_predictions(model, features, targets)
        explained = 1.0 - (evaluation["rmse"] / np.maximum(evaluation["baseline_rmse"], 1e-12)) ** 2
        return {
            "samples": int(len(features)),
            "target_rms": [round(float(value), 5) for value in evaluation["baseline_rmse"]],
            "residual_rmse": [round(float(value), 5) for value in evaluation["rmse"]],
            "explained_fraction": [round(float(value), 4) for value in explained],
        }

    weights = np.asarray(gate_weights(model, jnp.asarray(train_features))).sum(axis=1)
    predictions = np.asarray(apply_residual_model(model, jnp.asarray(train_features)))
    return {
        "train": split(train_features, train_targets),
        "validation": split(validation_features, validation_targets),
        "input_std": {
            name: round(float(value), 5)
            for name, value in zip(RESIDUAL_FEATURE_NAMES, np.asarray(model.input_std))
        },
        "gate_weight_train": {
            f"p{q}": round(float(np.percentile(weights, q)), 4) for q in (1, 50, 99)
        },
        "max_abs_prediction": [round(float(value), 5) for value in np.abs(predictions).max(axis=0)],
    }


def simulate_closed_loop_on_log_reference(
    model: ResidualEnsemble | None,
    problem: str,
    log_path: str,
    *,
    seed: int = 0,
    clip_after_first_trajectory: bool = True,
):
    """Closed-loop simulation (nominal plant + optional residual) of the
    reference recorded in a Pololu log; returns ``(sim_log, measured_log)``.

    The plant is ``problem``'s robot block (the residual's nominal model); the
    controller is the one that recorded the log (``controller_problem_for_log``).
    """
    import yaml

    from wmr_simulator.gain_parametrization import params_from_cfg
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.robot import DiffDrive
    from wmr_simulator.simulation import SimulationPipeline

    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=clip_after_first_trajectory,
    )
    problem_path = controller_problem_for_log(log_path, problem)
    config = yaml.safe_load(open(problem_path, "r", encoding="utf-8"))
    plant_robot_cfg = yaml.safe_load(open(problem, "r", encoding="utf-8"))["robot"]
    controller_cfg = config["controller"]
    parametrization_cfg = controller_cfg.get("gain_parametrization")
    schedule_params = (
        params_from_cfg(parametrization_cfg, [config["robot"]["v_max"], config["robot"]["omega_max"]])
        if parametrization_cfg and parametrization_cfg.get("enabled", False)
        else None
    )

    reference_time = np.asarray(log.reference.time_s, dtype=float)
    pipeline = SimulationPipeline(
        problem_path=problem_path,
        seed=seed,
        residual_model=model,
        sim_time=float(reference_time[-1] - reference_time[0]),
    )
    # The plant is the residual's nominal model, not the controller's belief.
    pipeline.robot = DiffDrive(robot_cfg=plant_robot_cfg, dt=pipeline.wheel_dt)
    reference_states = jnp.asarray(
        pipeline._fit_reference_states(np.asarray(log.reference.states, dtype=float)),
        dtype=jnp.float32,
    )
    # Start at the measured pose at the time of the first reference sample.
    t0 = float(reference_time[0])
    pose_time = np.asarray(log.pose.time_s, dtype=float)
    pose = np.asarray(log.pose.states, dtype=float)
    initial_pose = [np.interp(t0, pose_time, pose[:, 0]), np.interp(t0, pose_time, pose[:, 1]),
                    np.interp(t0, pose_time, np.unwrap(pose[:, 2]))]
    sim_log = pipeline.run_closed_loop(
        pipeline.hidden_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
        schedule_params=schedule_params,
        reference_states=reference_states,
        initial_pose=jnp.asarray(initial_pose, dtype=jnp.float32),
    )
    return shift_simulation_log_time(sim_log, t0), log


def shift_simulation_log_time(sim_log, offset: float):
    """The same simulated log with every time axis moved by ``offset`` seconds."""
    return sim_log._replace(
        reference=sim_log.reference._replace(time_s=sim_log.reference.time_s + offset),
        wheel=sim_log.wheel._replace(time_s=sim_log.wheel.time_s + offset),
        pose=sim_log.pose._replace(
            time_s=sim_log.pose.time_s + offset, command_time_s=sim_log.pose.command_time_s + offset
        ),
    )


def robot_params_from_problem(problem_path: str):
    import yaml

    from wmr_simulator.types import PhysicalParams

    with open(problem_path, "r", encoding="utf-8") as file:
        robot_cfg = yaml.safe_load(file)["robot"]
    return PhysicalParams(
        wheel_radius=jnp.asarray(robot_cfg["wheel_radius"], dtype=jnp.float32),
        base_diameter=jnp.asarray(robot_cfg["base_diameter"], dtype=jnp.float32),
        max_wheel_speed=jnp.asarray(robot_cfg["max_wheel_speed"], dtype=jnp.float32),
        time_constant=jnp.asarray(robot_cfg["time_constant"], dtype=jnp.float32),
    )


def controller_problem_for_log(log_path: str, default_problem_path: str) -> str:
    """The problem config of the controller that recorded a log: the
    ``problem.yaml`` of its iteration folder, else ``default_problem_path``."""
    from pathlib import Path

    iteration_problem = Path(log_path).parent.parent / "problem.yaml"
    return str(iteration_problem) if iteration_problem.is_file() else default_problem_path


DATA_DIR_NAME = "data"


def gather_log_paths(log_dirs: list[str], recursive: bool = True) -> list:
    """Loadable Pololu logs under the given directories, sorted and deduplicated.

    ``recursive`` descends into an experiment tree but stops at each ``data/``
    directory (its subfolders hold benchmark runs).
    """
    from pathlib import Path

    from wmr_simulator.pololu.log_loader import _looks_like_pololu_log, list_pololu_log_paths

    paths = []
    for log_dir in log_dirs:
        directory = Path(log_dir)
        if not directory.is_dir():
            raise ValueError(f"Log path must be a directory: {directory}")
        if recursive:
            paths.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and (path.parent.name == DATA_DIR_NAME or path.parent == directory)
                and _looks_like_pololu_log(path)
            )
        else:
            paths.extend(list_pololu_log_paths(directory))
    unique = sorted(set(paths))
    if not unique:
        raise ValueError(f"No loadable Pololu logs found under: {log_dirs}")
    return unique


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_from_logs(
    problem: str,
    log_dirs: list[str],
    out: str,
    *,
    log_paths: list | None = None,
    num_experts: int = DEFAULT_NUM_EXPERTS,
    hidden_sizes: tuple = DEFAULT_HIDDEN_SIZES,
    spectral_norm_cap: float = DEFAULT_SPECTRAL_NORM_CAP,
    gate_bandwidth_scale: float = DEFAULT_GATE_BANDWIDTH_SCALE,
    ood_sigma: float = DEFAULT_OOD_SIGMA,
    epochs: int = 400,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    output_reg_weight: float = 1e-2,
    validation_split: float = 0.2,
    seed: int = 0,
    clip_after_first_trajectory: bool = True,
    resample_uniform: bool = True,
    recursive: bool = True,
    out_dir: str = "visualize",
    training: str = "one_step",
    closed_loop_window_s: float = 1.0,
    closed_loop_stride_s: float = 1.0,
    closed_loop_steps: int = 400,
    closed_loop_learning_rate: float = 1e-3,
) -> ResidualEnsemble:
    """Build datasets from the logs, train, and save the checkpoint and plots.

    ``training`` is ``"one_step"`` (``epochs`` of Adam on the one-step targets)
    or ``"closed_loop"`` (gate and normalization from the one-step data, then
    closed-loop windows, ``residual_model.closed_loop``). Every log's targets
    are relative to ``problem``'s nominal model, which the checkpoint records.
    """
    from datetime import datetime
    from pathlib import Path

    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.residual_model.io import save_residual_model
    from wmr_simulator.visualization.residual import (
        plot_gate_map,
        plot_predictions,
        plot_training_history,
    )

    # An explicit list wins over scanning log_dirs.
    log_paths = gather_log_paths(log_dirs, recursive=recursive) if log_paths is None else list(log_paths)
    params = robot_params_from_problem(problem)
    print(f"Residual targets against the nominal model of {problem}")
    if training not in ("one_step", "closed_loop"):
        raise ValueError(f"training must be 'one_step' or 'closed_loop', got {training!r}")
    datasets = []
    logs = []
    for path in log_paths:
        log = load_pololu_traj_control_log(path, clip_after_first_trajectory=clip_after_first_trajectory)
        dataset = build_residual_dataset(log, params, resample_uniform=resample_uniform)
        datasets.append(dataset)
        logs.append(log)
        print(f"{path}: {len(dataset['features'])} samples")

    train_indices, validation_indices = split_datasets_by_file(datasets, validation_split, seed)
    train_features, train_targets = stack_datasets([datasets[i] for i in train_indices])
    validation_features, validation_targets = stack_datasets([datasets[i] for i in validation_indices])
    train_segments = [len(datasets[i]["features"]) for i in train_indices]
    validation_segments = [len(datasets[i]["features"]) for i in validation_indices]
    train_files = [str(log_paths[i]) for i in train_indices]
    validation_files = [str(log_paths[i]) for i in validation_indices]
    print(f"Split by file: {len(train_indices)} train, {len(validation_indices)} validation")
    print(f"  validation files: {[Path(f).name for f in validation_files]}")

    print(f"Training samples: {len(train_features)}, validation samples: {len(validation_features)}")

    model, history = train_residual_ensemble(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        seed=seed,
        num_experts=num_experts,
        hidden_sizes=hidden_sizes,
        spectral_norm_cap=spectral_norm_cap,
        gate_bandwidth_scale=gate_bandwidth_scale,
        ood_sigma=ood_sigma,
        epochs=epochs if training == "one_step" else 0,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_reg_weight=output_reg_weight,
    )
    closed_loop_report = None
    if training == "closed_loop":
        model, closed_loop_report = _train_closed_loop_from_logs(
            model, problem, logs, log_paths, train_indices, validation_indices,
            window_s=closed_loop_window_s, stride_s=closed_loop_stride_s,
            steps=closed_loop_steps, learning_rate=closed_loop_learning_rate,
        )

    config = {
        "input_dim": int(train_features.shape[1]),
        "output_dim": int(train_targets.shape[1]),
        "num_experts": int(model.centers.shape[0]),
        "hidden_sizes": list(hidden_sizes),
        "spectral_norm_cap": float(spectral_norm_cap),
    }
    metadata = {
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "problem": problem,
        "log_dirs": list(log_dirs),
        "train_files": train_files,
        "validation_files": validation_files,
        "resample_uniform": resample_uniform,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "output_reg_weight": output_reg_weight,
        "gate_bandwidth_scale": gate_bandwidth_scale,
        "ood_sigma": ood_sigma,
        "seed": seed,
        "training": training,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_validation_loss": history["validation_loss"][-1] if history["validation_loss"] else None,
        "closed_loop": closed_loop_report,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_residual_model(out, model, config, params, metadata)
    print(f"Saved residual model to {out}")

    diagnostics = residual_diagnostics(
        model, train_features, train_targets, validation_features, validation_targets
    )
    diagnostics["num_logs"] = len(log_paths)
    if closed_loop_report is not None:
        diagnostics["closed_loop"] = closed_loop_report
    diagnostics_path = Path(out).with_name("residual_diagnostics.yaml")
    with diagnostics_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(diagnostics, file, sort_keys=False)
    print(
        f"Residual diagnostics ({diagnostics_path}): target RMS [dv, domega] "
        f"{diagnostics['train']['target_rms']}, explained {diagnostics['train']['explained_fraction']}, "
        f"gate weight p1/p50 {diagnostics['gate_weight_train']['p1']:.3f}/"
        f"{diagnostics['gate_weight_train']['p50']:.3f}"
    )

    if history["train_loss"]:
        plot_training_history(history, out_dir=out_dir)
    plot_gate_map(model, train_features, out_dir=out_dir)
    for split_name, features, targets, segments in (
        ("train", train_features, train_targets, train_segments),
        ("validation", validation_features, validation_targets, validation_segments),
    ):
        if len(features) == 0:
            continue
        evaluation = evaluate_predictions(model, features, targets)
        print(f"{split_name} RMSE  [dv, domega]: {evaluation['rmse']}")
        print(f"{split_name} baseline (zero residual): {evaluation['baseline_rmse']}")
        plot_predictions(
            evaluation["predictions"], targets, split_name, out_dir=out_dir, segment_lengths=segments
        )

    # Closed-loop sanity check along the first training log's reference.
    from wmr_simulator.visualization.residual import plot_closed_loop_rollout

    first_train_log = train_files[0]
    sim_log, measured_log = simulate_closed_loop_on_log_reference(
        model, problem, first_train_log, seed=seed, clip_after_first_trajectory=clip_after_first_trajectory
    )
    # The same loop without the residual, for comparison.
    nominal_sim_log, _ = simulate_closed_loop_on_log_reference(
        None, problem, first_train_log, seed=seed, clip_after_first_trajectory=clip_after_first_trajectory
    )
    plot_closed_loop_rollout(
        sim_log,
        out_prefix=f"residual_closed_loop_{Path(first_train_log).stem}",
        nominal_sim_log=nominal_sim_log,
        measured_log=measured_log,
        out_dir=out_dir,
    )
    print(f"Plots saved to {out_dir}/")
    return model


def _train_closed_loop_from_logs(
    model, problem, logs, log_paths, train_indices, validation_indices, *, window_s, stride_s, steps, learning_rate
):
    """Closed-loop training on the training logs' windows; reports the window
    loss of the nominal plant and of the trained residual on both splits."""
    from wmr_simulator.residual_model import closed_loop

    plant_robot = yaml.safe_load(open(problem, "r", encoding="utf-8"))["robot"]
    pipeline = closed_loop.window_pipeline(problem, plant_robot, window_s)

    def windows_of(indices):
        return closed_loop.build_windows(
            [logs[i] for i in indices],
            [controller_problem_for_log(str(log_paths[i]), problem) for i in indices],
            window_s=window_s, stride_s=stride_s,
            geometry_dt=pipeline.geometry_dt, wheel_dt=pipeline.wheel_dt,
        )

    train_windows = windows_of(train_indices)
    if train_windows is None:
        raise ValueError(f"No training log is longer than one closed-loop window ({window_s} s).")
    validation_windows = windows_of(validation_indices)
    model = closed_loop.zero_residual(model)
    train_loss = closed_loop.window_loss_fn(pipeline, train_windows)
    print(f"Closed-loop residual training: {len(train_windows)} windows of {window_s} s "
          f"from {len(train_indices)} log(s), {steps} Adam steps at {learning_rate}")
    model, history = closed_loop.train_closed_loop(model, train_loss, steps=steps, learning_rate=learning_rate)
    report = {
        "window_s": float(window_s),
        "train_windows": len(train_windows),
        "train_loss_nominal": history[0] if history else None,
        "train_loss": float(jax.jit(train_loss)(model)[0]),
    }
    if validation_windows is not None:
        validation_loss = jax.jit(closed_loop.window_loss_fn(pipeline, validation_windows))
        report["validation_windows"] = len(validation_windows)
        report["validation_loss_nominal"] = float(validation_loss(closed_loop.zero_residual(model))[0])
        report["validation_loss"] = float(validation_loss(model)[0])
    print(f"Closed-loop window loss (nominal -> residual): train {report['train_loss_nominal']:.4f} -> "
          f"{report['train_loss']:.4f}" + (
              f", validation {report['validation_loss_nominal']:.4f} -> {report['validation_loss']:.4f}"
              if "validation_loss" in report else ""))
    return model, report


def _resample_poses_uniform(
    pose_time_s: np.ndarray,
    pose_states: np.ndarray,
    pose_twists: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Linearly interpolate poses (and twists) onto a uniform median-dt grid.

    Yaw is unwrapped before interpolation and re-wrapped after. Keeps the
    original start/end times so downstream stream alignment is unaffected.
    """
    if len(pose_time_s) < 3:
        return pose_time_s, pose_states, pose_twists
    dt = np.diff(pose_time_s)
    median_dt = float(np.median(dt[dt > 0.0])) if np.any(dt > 0.0) else 0.0
    if median_dt <= 0.0:
        return pose_time_s, pose_states, pose_twists
    num_intervals = int(np.round((pose_time_s[-1] - pose_time_s[0]) / median_dt))
    if num_intervals < 2:
        return pose_time_s, pose_states, pose_twists
    uniform_time = pose_time_s[0] + median_dt * np.arange(num_intervals + 1)
    yaw_unwrapped = np.unwrap(pose_states[:, 2])
    resampled = np.column_stack(
        [
            np.interp(uniform_time, pose_time_s, pose_states[:, 0]),
            np.interp(uniform_time, pose_time_s, pose_states[:, 1]),
            _wrap_to_pi(np.interp(uniform_time, pose_time_s, yaw_unwrapped)),
        ]
    )
    if pose_twists is not None:
        pose_twists = np.column_stack(
            [np.interp(uniform_time, pose_time_s, pose_twists[:, i]) for i in range(3)]
        )
    return uniform_time, resampled, pose_twists


def _zero_order_hold(query_time: np.ndarray, time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Latest logged value at or before each query time (first value before the log starts)."""
    indices = np.clip(np.searchsorted(time, query_time, side="right") - 1, 0, len(time) - 1)
    return values[indices]


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
