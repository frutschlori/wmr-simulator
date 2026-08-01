"""Learned state-action residual dynamics on the body-frame twist.

Overview
--------
The nominal differential-drive model (first-order motor lag -> traction limit ->
ideal kinematics) predicts a body twist ``(v_nom, omega_nom)`` with zero lateral
velocity. A small learned residual corrects the leftover gap between that
nominal twist and the measured (mocap) twist during *nominal* (non-drifting)
operation:

    delta = [delta_v, delta_omega] = residual(v_nom, omega_nom, v_cmd, omega_cmd)

    v     = v_nom     + delta_v
    v_y   = 0                                # no side-slip in nominal operation
    omega = omega_nom + delta_omega

so the corrected twist is integrated with the ideal planar kinematics. This is
the additive residual of Sym2Real (arXiv:2509.15412), reduced to the two
degrees of freedom a two-wheel differential drive actually has, conditioned on
a (state, action) pair as in that paper.

Why the descriptor is (state, action) and not the state alone
-------------------------------------------------------------
The *state* half is the nominal twist ``[v_nom, omega_nom]``: a bijective linear
re-parametrization of the (ground-contact) wheel speeds, since
``u_r = (2v + L omega) / 2r`` and ``u_l = (2v - L omega) / 2r``. Feeding the
wheel speeds *as well* would add literally no information -- it is the same two
numbers in a different basis -- and measurably does not help.

The *action* half is the commanded twist ``[v_cmd, omega_cmd]``, i.e. the
kinematics of the target wheel speeds ``max_wheel_speed * duty`` the motor lag is
heading toward. This is the information the nominal model has and the state
alone does not. It matters because the leftover physics splits in two:
quasi-static effects (Coulomb/viscous friction, wheelbase error) are functions
of the operating point and were already representable, whereas *transient*
effects -- a mis-identified motor time constant, transport delay, backlash on
torque reversal -- appear in proportion to (command - state), which a memoryless
function of the state alone cannot express at any capacity. Note the motor lag
gives ``u^+ - u = (1 - alpha) (u_target - u)``, so the commanded twist is the
nominal acceleration up to a constant; empirically it roughly doubles the share
of the yaw residual the model explains (held-out RMSE 0.265 -> 0.240 rad/s
against a 0.287 zero-residual baseline), while the forward channel is already
saturated by the state alone.

Both halves are available identically in training (encoder speeds through the
lag; logged duty) and in simulation, where there is no encoder -- ``robot.step``
builds them from its own lag state and its duty argument. The action is a
function of the actuation, never of the measured target, so there is no leakage.

Expert ensemble + out-of-distribution gate
-------------------------------------------
The residual is a mixture of ``K`` small expert MLPs, each specialized to a
region of the operating envelope. K-means over the normalized descriptor
(the full (state, action) vector) fixes one center/bandwidth per expert.
At query time a softmax over Gaussian responsibilities routes to the nearest
expert(s); a *null* "zero expert" with a fixed responsibility wins whenever the
query is farther than ``ood_sigma`` bandwidths from every center, so the residual
smoothly decays to zero in operating regimes the training data never covered
(no spiky extrapolation off the manifold). The whole gate is differentiable, so
gradients still flow controller gains -> duty -> nominal twist -> residual ->
rollout loss during gain tuning.

Smoothness
----------
Every expert weight matrix is spectrally normalized to ``spectral_norm_cap`` at
forward time (Miyato et al. 2018, power iteration -- the same regularizer as the
error-MLP gain parametrization). tanh activations are 1-Lipschitz, so the cap
bounds each expert's Lipschitz constant in normalized feature space: the
residual cannot change arbitrarily fast with the twist, which keeps the
residual -> twist -> feature closed loop from oscillating. A small L2 penalty on
the residual output shrinks the model toward the nominal dynamics wherever the
data does not clearly demand a correction. The last layer is zero-initialized,
so an untrained ensemble is exactly the nominal model.

Training is one-step only (predict this step's twist residual from this step's
nominal twist): the target is measured-minus-nominal, and since the input is the
nominal twist (a function of the actuation, not of the measured target) there is
no target leakage. Everything is pure JAX so a saved model is a single
self-contained pytree (see residual_model.io).

This module also hosts the dataset construction from Pololu logs, the training
loop, and evaluation helpers; plotting lives in
wmr_simulator.visualization.residual and the CLI wrappers in scripts/ are thin.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from wmr_simulator.residual_model.burnout import rate_limited_series

# (state, action) descriptor: the nominal predicted body twist and the commanded
# twist the motor lag is heading toward (see the module docstring). Doubles as
# the gate's regime descriptor (the k-means / softmax space).
RESIDUAL_FEATURE_NAMES = ("v_nom", "omega_nom", "v_cmd", "omega_cmd")
RESIDUAL_INPUT_DIM = len(RESIDUAL_FEATURE_NAMES)
RESIDUAL_OUTPUT_DIM = 2  # [delta_v, delta_omega]

TARGET_LABELS = (r"$\Delta v$ [m/s]", r"$\Delta \omega$ [rad/s]")

# Defaults for the ensemble geometry / regularization.
DEFAULT_NUM_EXPERTS = 4
DEFAULT_HIDDEN_SIZES = (16, 16)
DEFAULT_SPECTRAL_NORM_CAP = 1.0
# Gaussian gate bandwidth = (root-mean intra-cluster distance) * this factor;
# > 1 overlaps neighbouring experts so the hand-off between regimes is smooth.
DEFAULT_GATE_BANDWIDTH_SCALE = 1.5
# The null "zero expert" wins (residual -> 0) once the nearest cluster is more
# than this many bandwidths away in normalized descriptor space.
DEFAULT_OOD_SIGMA = 3.0


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class ResidualEnsemble(eqx.Module):
    """Mixture of spectrally-normalized expert MLPs with an OOD gate.

    ``layers`` holds one stacked array per (weight, bias) with a leading expert
    axis of length K, so the whole ensemble evaluates with a single ``vmap``.
    ``centers``/``scales`` are the fixed k-means gate (normalized descriptor
    space); ``ood_sigma`` sets the null-expert distance. Normalization stats are
    carried inside the module so a checkpoint is self-contained.
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
    """Fresh ensemble with identity normalization and an unset (zero) gate.

    The gate centers/scales are placeholders here; ``train_residual_ensemble``
    fills them from k-means. ``io.load_residual_model`` uses this to rebuild the
    checkpoint skeleton before swapping in the saved leaves, so the shapes must
    match the trained model (fixed by ``input_dim``/``output_dim``/
    ``num_experts``/``hidden_sizes``).
    """
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
    """Assemble the descriptor [v_nom, omega_nom, v_cmd, omega_cmd].

    See RESIDUAL_FEATURE_NAMES: the nominal twist is the state, the commanded
    twist (kinematics of ``max_wheel_speed * duty``) is the action.
    """
    return jnp.stack(
        [
            jnp.asarray(nominal_v, dtype=jnp.float32),
            jnp.asarray(nominal_omega, dtype=jnp.float32),
            jnp.asarray(commanded_v, dtype=jnp.float32),
            jnp.asarray(commanded_omega, dtype=jnp.float32),
        ]
    )


def _spectral_norm(weight: jax.Array, num_iters: int = 15) -> jax.Array:
    """Largest singular value via power iteration (deterministic start).

    The singular vectors are treated as constants (stop-gradient), so the
    gradient of the returned sigma is the standard rank-one u v^T estimate.
    Matches gain_parametrization.error_mlp._spectral_norm.
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
    """Softmax responsibilities over the K experts (rows sum to <= 1).

    Each expert gets ``r_k = exp(-1/2 * ||z - c_k||^2 / s_k^2)``; a null expert
    with fixed ``r_0 = exp(-1/2 * ood_sigma^2)`` competes in the softmax so the
    experts' weights vanish (residual -> 0) when the query is far from every
    center. Returns the K expert weights (the null weight is the remainder).
    """
    distance_sq = jnp.sum((z - model.centers) ** 2, axis=1)  # (K,)
    responsibilities = jnp.exp(-0.5 * distance_sq / (model.scales**2 + 1e-12))
    null = jnp.exp(-0.5 * model.ood_sigma**2)
    return responsibilities / (null + jnp.sum(responsibilities) + 1e-12)


def _shaped_delta(model: ResidualEnsemble, z: jax.Array) -> jax.Array:
    """Physical residual delta from a single normalized descriptor ``z``.

    Blends the experts by their gate weights (OOD -> zero) and denormalizes.
    ``target_mean`` is zero, so a zero blend is exactly zero physical residual.
    """
    cap = model.spectral_norm_cap
    outputs = jax.vmap(lambda layers: _expert_forward(layers, z, cap))(model.layers)  # (K, out)
    weights = _gate_weights(model, z)  # (K,)
    blended = weights @ outputs  # (out,) normalized residual
    return blended * model.target_std + model.target_mean


def apply_residual_model(model: ResidualEnsemble, features: jax.Array) -> jax.Array:
    """Physical residual twist [delta_v, delta_omega] for one or a batch of features.

    Accepts a single feature vector or a batch (leading axis). The gate zeroes
    the residual out of distribution, so no input clamping is needed.
    """
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
    """Corrected body twist (v_x, v_y, omega) from the nominal twist and features.

    ``v_y`` is always zero (no side-slip in nominal operation); returned for a
    uniform (v_x, v_y, omega) interface.
    """
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

    Measured twist: the log's filter-derived body twists (``log.pose.twists``,
    see pololu.measurement_smoothing) at the interval starts, consistent with
    Euler integration; logs without twists fall back to finite-differencing the
    poses over each mocap interval and rotating the world displacement into the
    body frame at the interval start.

    Nominal twist: encoder wheel speeds interpolated to the interval starts,
    passed through the first-order motor lag (matching DiffDrive.step -- the
    simulator has no encoder), the identified traction limit
    (``a_slip_max``, residual_model.burnout) and the ideal differential-drive
    kinematics.

    Commanded twist: the same kinematics applied to the *target* wheel speeds
    ``max_wheel_speed * duty`` (the action half of the descriptor). The residual
    input is the concatenation of the two and the target is
    ``measured - nominal``; because both inputs are functions of the actuation
    (not of the measured target) there is no target leakage and every valid
    interval is a sample (no off-by-one drop).

    ``resample_uniform`` interpolates poses (and twists) onto a uniform
    median-dt grid; ``max_dt_factor`` drops mocap intervals longer than that
    multiple of the median (stream gaps make the twist meaningless). Returns
    float32 ``features`` (N, 4), ``targets`` (N, 2), ``time_s``, ``dt``, and the
    measured/nominal twists plus pose arrays for diagnostics.
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

    # Zero-order-hold duty at the interval starts, then the first-order motor
    # lag: lag[t] = alpha * u_enc[t-1] + (1 - alpha) * max_speed * duty[t],
    # matching DiffDrive.step. Both the nominal twist and (if ever needed) the
    # wheel feature are built from this lag, never the raw encoder speed.
    duty = _zero_order_hold(interval_time, wheel_time_s, duty_cycle)
    max_speed = float(params.max_wheel_speed)
    tau = float(params.time_constant)
    alpha = np.exp(-dt / tau) if tau >= 1e-3 else np.zeros_like(dt)
    prev_u_r = np.concatenate([u_r_enc[:1], u_r_enc[:-1]])
    prev_u_l = np.concatenate([u_l_enc[:1], u_l_enc[:-1]])
    lag_r = alpha * prev_u_r + (1.0 - alpha) * max_speed * duty[:, 0]
    lag_l = alpha * prev_u_l + (1.0 - alpha) * max_speed * duty[:, 1]

    # Nominal twist from the lag wheel through the traction limit and ideal
    # kinematics (so the residual does not have to relearn the burnout).
    max_rate = float(params.a_slip_max) / r if float(params.a_slip_max) > 0.0 else np.inf
    u_r_nom = rate_limited_series(lag_r, dt, max_rate)
    u_l_nom = rate_limited_series(lag_l, dt, max_rate)
    v_nom = 0.5 * r * (u_r_nom + u_l_nom)
    omega_nom = r * (u_r_nom - u_l_nom) / effective_wheelbase
    nominal_twist = np.column_stack([v_nom, np.zeros_like(v_nom), omega_nom])

    # Action half of the descriptor: the twist of the *target* wheel speeds the
    # lag is heading toward (matching DiffDrive.step, which builds it from its
    # duty argument). Not traction-limited -- it is a command, not a state.
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
    """Train/validation indices into ``datasets``, split by whole file.

    Always whole files, never a within-file split: the residual has to learn a
    complete trajectory, so holding out the tail of every log would train it on
    truncated runs. With too few files for ``validation_split`` to reach one
    file, one file is held out anyway and the rest train.
    """
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

    A bandwidth is the root-mean distance of a cluster's own points to its
    center, scaled by ``bandwidth_scale`` for overlap; empty/singleton clusters
    fall back to the median bandwidth so the gate stays well-defined.
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
    """Train the expert ensemble on single-step twist residuals with Adam.

    Inputs/targets are normalized (targets std-only: mean fixed at zero, so a
    zero output is exactly the nominal model). The gate is fixed from k-means
    over the normalized inputs before training; only the expert weights are
    optimized (jointly, through the fixed gate, so each expert specializes to
    where its gate weight is high). The loss is the blended prediction MSE plus
    ``output_reg_weight`` times the mean squared normalized residual output
    (shrinks the correction toward zero where the data does not demand it).

    Returns (model, history) with per-epoch normalized MSE (train = full
    objective incl. the output penalty, validation = prediction MSE only).
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
        # Blended normalized residual (target_mean is zero, so this is the
        # physical delta scaled by 1/target_std).
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


def integrate_twist_trajectory(
    initial_pose: np.ndarray,
    twists: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Euler-integrate a body twist series [v_x, v_y, omega] into poses (numpy).

    Returns len(twists) + 1 poses starting at ``initial_pose``.
    """
    poses = np.empty((len(twists) + 1, 3), dtype=float)
    poses[0] = np.asarray(initial_pose, dtype=float)
    for index, (twist, step_dt) in enumerate(zip(twists, dt)):
        x, y, theta = poses[index]
        v_x, v_y, omega = twist
        poses[index + 1] = [
            x + (v_x * np.cos(theta) - v_y * np.sin(theta)) * step_dt,
            y + (v_x * np.sin(theta) + v_y * np.cos(theta)) * step_dt,
            _wrap_to_pi(theta + omega * step_dt),
        ]
    return poses


def evaluate_predictions(model: ResidualEnsemble, features: np.ndarray, targets: np.ndarray) -> dict:
    """Per-channel RMSE of the model against the zero-residual baseline."""
    predictions = np.asarray(apply_residual_model(model, jnp.asarray(features)))
    return {
        "predictions": predictions,
        "rmse": np.sqrt(np.mean((predictions - targets) ** 2, axis=0)),
        "baseline_rmse": np.sqrt(np.mean(targets**2, axis=0)),
    }


def pose_rmse(poses: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((poses[:, :2] - reference[:, :2]) ** 2, axis=1))))


def simulate_closed_loop_on_log_reference(
    model: ResidualEnsemble | None,
    problem: str,
    log_path: str,
    *,
    seed: int = 0,
    clip_after_first_trajectory: bool = True,
):
    """Closed-loop simulation (hidden robot + optional residual) tracking the
    reference trajectory recorded in a Pololu log.

    Returns ``(sim_log, measured_log)``: the simulated SimulationLog and the
    decoded Pololu log it was driven from, so callers can compare the simulated
    robot against the real one that recorded the same reference.

    Runs the log's *own* recorded controller: the iteration's ``problem.yaml``
    supplies the tuned base gains, the error-MLP ``theta`` that was exported to
    ``GAINMLP.JSN`` and the identified robot/estimator parameters. Using the
    global ``problem`` config instead compares against a controller that never
    recorded the log (the stock kx is 4.5 where the tuned one is ~0.003), which
    makes the comparison meaningless -- so ``problem`` is only the fallback for
    logs outside an iteration folder.
    """
    import yaml

    from wmr_simulator.gain_parametrization import params_from_cfg
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.simulation import SimulationPipeline

    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=clip_after_first_trajectory,
    )
    problem_path = problem_path_for_log(log_path, problem)
    config = yaml.safe_load(open(problem_path, "r", encoding="utf-8"))
    controller_cfg = config["controller"]
    parametrization_cfg = controller_cfg.get(
        "gain_parametrization", controller_cfg.get("gain_schedule")
    )
    schedule_params = (
        params_from_cfg(parametrization_cfg, [config["robot"]["v_max"], config["robot"]["omega_max"]])
        if parametrization_cfg and parametrization_cfg.get("enabled", False)
        else None
    )

    pipeline = SimulationPipeline(problem_path=problem_path, seed=seed, residual_model=model)
    reference_states = jnp.asarray(
        pipeline._fit_reference_states(np.asarray(log.reference.states, dtype=float)),
        dtype=jnp.float32,
    )
    # Start where the real robot started, not where the reference starts: the
    # robot is placed by hand and the initial offset it has to drive out is part
    # of the recorded tracking error. Simulating from the reference start hands
    # the sim a head start the real run never had.
    sim_log = pipeline.run_closed_loop(
        pipeline.hidden_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
        schedule_params=schedule_params,
        reference_states=reference_states,
        initial_pose=jnp.asarray(log.pose.states[0], dtype=jnp.float32),
    )
    return sim_log, log


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
        a_slip_max=jnp.asarray(robot_cfg.get("a_slip_max", 0.0), dtype=jnp.float32),
    )


def problem_path_for_log(log_path: str, default_problem_path: str) -> str:
    """Problem config the robot was actually running when this log was recorded.

    Active-learning logs live in ``<experiment>/<iteration_XX>/data/TRxx.csv``
    and each iteration folder carries the ``problem.yaml`` that produced its
    ``ROBOTCFG.CFG`` / ``GAINMLP.JSN`` export -- the identified robot parameters
    and tuned controller gains that were running for those runs. They differ per
    iteration (the motor time constant alone moves by ~50% between the stock
    config and the identified ones), so a single global config misstates the
    nominal model for every log but its own. Falls back to
    ``default_problem_path`` for logs outside such a folder.
    """
    from pathlib import Path

    iteration_problem = Path(log_path).parent.parent / "problem.yaml"
    return str(iteration_problem) if iteration_problem.is_file() else default_problem_path


def robot_params_for_log(log_path: str, default_problem_path: str):
    """Physical params the robot was configured with when this log was recorded."""
    return robot_params_from_problem(problem_path_for_log(log_path, default_problem_path))


DATA_DIR_NAME = "data"


def gather_log_paths(log_dirs: list[str], recursive: bool = True) -> list:
    """Loadable Pololu logs under the given directories.

    ``recursive`` descends into an experiment tree (logs live in
    ``iteration_XX/data/``); otherwise only direct children are scanned. Files
    are deduplicated and returned in sorted order.

    Recursion deliberately stops *at* each ``data/`` directory: only logs
    sitting directly inside one are collected. Subfolders of ``data/`` hold
    special runs -- baseline comparisons, no-gain-MLP ablations, single-shape
    sweeps (``circle/``, ``lemniscate/``, ``*_no_mlp/``) -- which must not be
    baked into the residual training set. Logs lying directly in a scanned
    directory are kept too, so pointing at a flat folder still works.
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
# High-level train / evaluate entry points (called by the thin CLI wrappers)
# ---------------------------------------------------------------------------


def train_from_logs(
    problem: str,
    log_dirs: list[str],
    out: str,
    *,
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
) -> ResidualEnsemble:
    """Build datasets from all logs under ``log_dirs``, train, save checkpoint + plots.

    ``log_dirs`` may span several active-learning iterations: one-step training is
    gain-independent (the descriptor is built from the logged duty, so whatever
    controller produced it is irrelevant), which is what makes pooling iterations
    with different tuned gains sound. Per-log robot params keep the nominal model
    correct across the pool (see ``robot_params_for_log``).
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

    log_paths = gather_log_paths(log_dirs, recursive=recursive)
    datasets = []
    for path in log_paths:
        # Per-log params: the nominal model each log is a residual *of* is the one
        # its own iteration was running (see problem_path_for_log).
        params = robot_params_for_log(str(path), problem)
        log = load_pololu_traj_control_log(path, clip_after_first_trajectory=clip_after_first_trajectory)
        dataset = build_residual_dataset(log, params, resample_uniform=resample_uniform)
        datasets.append(dataset)
        print(f"{path}: {len(dataset['features'])} samples  (params from {problem_path_for_log(str(path), problem)})")

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
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_reg_weight=output_reg_weight,
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
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_residual_model(out, model, config, metadata)
    print(f"Saved residual model to {out}")

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
    # Same loop with the residual switched off: the control that says whether
    # the residual moved the simulated robot toward the real one or past it.
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


def evaluate_on_log(
    model_path: str,
    log_path: str,
    *,
    problem: str = "problems/pololu_gains.yaml",
    clip_after_first_trajectory: bool = True,
    resample_uniform: bool = True,
    out_dir: str = "visualize",
    out_prefix: str | None = None,
) -> dict:
    """Compare nominal vs residual-augmented open-loop rollout against mocap."""
    from pathlib import Path

    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.residual_model.io import load_residual_model
    from wmr_simulator.visualization.residual import plot_rollout_comparison

    model, checkpoint = load_residual_model(model_path)
    print(f"Loaded residual model {model_path}")
    print(f"  config: {checkpoint['config']}")
    if checkpoint.get("metadata"):
        validation_files = checkpoint["metadata"].get("validation_files", [])
        print(f"  trained validation files: {[Path(f).name for f in validation_files]}")

    params = robot_params_for_log(log_path, problem)
    log = load_pololu_traj_control_log(log_path, clip_after_first_trajectory=clip_after_first_trajectory)
    dataset = build_residual_dataset(log, params, resample_uniform=resample_uniform)

    predicted_residual = np.asarray(apply_residual_model(model, jnp.asarray(dataset["features"])))
    corrected_twist = dataset["nominal_twist"].copy()
    corrected_twist[:, 0] = dataset["nominal_twist"][:, 0] + predicted_residual[:, 0]
    corrected_twist[:, 2] = dataset["nominal_twist"][:, 2] + predicted_residual[:, 1]

    initial_pose = dataset["initial_pose"]
    dt = dataset["dt"]
    nominal_poses = integrate_twist_trajectory(initial_pose, dataset["nominal_twist"], dt)
    corrected_poses = integrate_twist_trajectory(initial_pose, corrected_twist, dt)
    measured_poses = integrate_twist_trajectory(initial_pose, dataset["measured_twist"], dt)

    time_s = dataset["time_s"]
    print(f"Open-loop position RMSE vs mocap-twist trajectory over {time_s[-1] - time_s[0]:.1f} s:")
    print(f"  nominal model:      {pose_rmse(nominal_poses, measured_poses):.4f} m")
    print(f"  residual-augmented: {pose_rmse(corrected_poses, measured_poses):.4f} m")
    evaluation = evaluate_predictions(model, dataset["features"], dataset["targets"])
    print(f"Residual RMSE  [dv, domega]: {evaluation['rmse']}")
    print(f"Zero-residual baseline:      {evaluation['baseline_rmse']}")

    prefix = out_prefix if out_prefix is not None else f"residual_eval_{Path(log_path).stem}"
    plot_rollout_comparison(
        dataset=dataset,
        predicted_residual=predicted_residual,
        corrected_twist=corrected_twist,
        nominal_poses=nominal_poses,
        corrected_poses=corrected_poses,
        out_prefix=prefix,
        out_dir=out_dir,
    )
    print(f"Plots saved to {out_dir}/{prefix}_*.pdf")

    return {
        "dataset": dataset,
        "predicted_residual": predicted_residual,
        "corrected_twist": corrected_twist,
        "nominal_poses": nominal_poses,
        "corrected_poses": corrected_poses,
        "measured_poses": measured_poses,
        "rmse": evaluation["rmse"],
        "baseline_rmse": evaluation["baseline_rmse"],
    }


# ---------------------------------------------------------------------------
# CLI (scripts/train_residual_model.py and scripts/eval_residual_model.py are
# thin wrappers around these)
# ---------------------------------------------------------------------------


def train_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Train the residual dynamics model from Pololu logs.")
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    # One or more directories; loadable logs are gathered recursively (default:
    # exp04 + exp05 across all iterations).
    parser.add_argument(
        "--log-dir",
        type=str,
        nargs="+",
        default=["Pololu Data/archive/Experiments/2026_07_27 Full Pipeline (with Residual and Gain MLP)/exp01"],
    )
    parser.add_argument("--out", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    # Hidden layer widths, e.g. --hidden-sizes 32 32 for two layers of 32.
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES))
    # Per-matrix spectral-norm cap on the experts (Lipschitz bound); 0 disables.
    parser.add_argument("--spectral-norm-cap", type=float, default=DEFAULT_SPECTRAL_NORM_CAP)
    # Gaussian gate bandwidth = intra-cluster RMS distance * this (>1 overlaps).
    parser.add_argument("--gate-bandwidth-scale", type=float, default=DEFAULT_GATE_BANDWIDTH_SCALE)
    # Null "zero expert" distance (bandwidths): beyond it the residual -> 0.
    parser.add_argument("--ood-sigma", type=float, default=DEFAULT_OOD_SIGMA)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    # L2 penalty on the (normalized) residual output; shrinks toward nominal.
    parser.add_argument("--output-reg-weight", type=float, default=1e-2)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    parser.add_argument("--resample-uniform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, default="visualize")
    args = parser.parse_args(argv)

    train_from_logs(
        problem=args.problem,
        log_dirs=args.log_dir,
        out=args.out,
        num_experts=args.num_experts,
        hidden_sizes=tuple(args.hidden_sizes),
        spectral_norm_cap=args.spectral_norm_cap,
        gate_bandwidth_scale=args.gate_bandwidth_scale,
        ood_sigma=args.ood_sigma,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_reg_weight=args.output_reg_weight,
        validation_split=args.validation_split,
        seed=args.seed,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
        resample_uniform=args.resample_uniform,
        recursive=args.recursive,
        out_dir=args.out_dir,
    )


def evaluate_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a residual dynamics model on a Pololu log.")
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--model", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--log", type=str, default="Pololu Data/Experiments/exp05/iteration_01/data/TR00.csv")
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    parser.add_argument("--resample-uniform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, default="visualize")
    parser.add_argument("--output", type=str, default=None, help="Output filename prefix")
    args = parser.parse_args(argv)

    evaluate_on_log(
        model_path=args.model,
        log_path=args.log,
        problem=args.problem,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
        resample_uniform=args.resample_uniform,
        out_dir=args.out_dir,
        out_prefix=args.output,
    )


main = train_main


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


if __name__ == "__main__":
    train_main()
