"""Learned state-action residual dynamics on the body-frame twist.

Model
-----
The nominal differential-drive model (wheel dynamics + ideal kinematics)
predicts a body twist (v_x, omega) with zero lateral velocity. A small MLP
learns the *residual* twist between that prediction and the twist measured by
mocap on the real robot, conditioned on the current **state and action**:

    state  = current body twist [vx_body, vy_body, omega]
             (so the model knows whether the robot is already slipping)
    action = current actuation  [wheel_speed_r, wheel_speed_l, duty_r, duty_l]

    [delta_vx_body, delta_vy_body, delta_omega] = MLP(state, action)

    vx_body_next = vx_nominal + delta_vx_body
    vy_body_next =              delta_vy_body
    omega_next   = omega_nominal + delta_omega

The corrected twist is integrated with the full planar kinematics (including
the lateral term), which lets the model capture chassis side-slip during
high-speed turns and braking that the ideal kinematics cannot represent.

Target convention (documented choice)
-------------------------------------
The residual is defined on the **current-step twist**: a training sample at
mocap interval t is

    input_t  = [measured body twist of interval t-1, actuation at interval t]
    target_t = measured_body_twist_t - nominal_body_twist_t

i.e. one-step prediction: given the motion the robot *enters* the step with and
the current action, predict this step's twist residual. This mirrors the
simulator, where DiffDrive.step conditions on the previous step's (corrected)
twist stored in the state. Using interval t's own measured twist as input would
leak the target. Note the state twist in simulation is the model's corrected
twist at the control dt, while training uses the mocap-differenced twist at the
mocap dt - an approximation that holds because the targets are velocities, not
increments, and therefore sample-rate independent.

Everything used during simulation/gain tuning is pure JAX, so gradients flow
controller gains -> duty cycles -> nominal dynamics + residual -> rollout loss;
the residual parameters are constants of the rollout closure and stay fixed.
Input/target normalization statistics live inside the module as arrays, so a
saved model is a single self-contained pytree (see residual_model.io for
checkpoints).

This module also hosts the dataset construction from Pololu logs, the training
loop, and evaluation helpers; plotting lives in
wmr_simulator.visualization.residual and the CLI wrappers in scripts/ are thin.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

RESIDUAL_FEATURE_NAMES = (
    "vx_body",
    "vy_body",
    "omega",
    "wheel_speed_r",
    "wheel_speed_l",
    "duty_r",
    "duty_l",
)

RESIDUAL_INPUT_DIM = len(RESIDUAL_FEATURE_NAMES)
RESIDUAL_OUTPUT_DIM = 3  # [delta_vx_body, delta_vy_body, delta_omega]

TARGET_LABELS = (r"$\Delta v_x$ [m/s]", r"$\Delta v_y$ [m/s]", r"$\Delta \omega$ [rad/s]")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class ResidualDynamicsModel(eqx.Module):
    mlp: eqx.nn.MLP
    input_mean: jax.Array
    input_std: jax.Array
    target_mean: jax.Array
    target_std: jax.Array


def init_residual_model(
    key: jax.Array,
    input_dim: int = RESIDUAL_INPUT_DIM,
    hidden_width: int = 200,
    hidden_depth: int = 4,
    output_dim: int = RESIDUAL_OUTPUT_DIM,
) -> ResidualDynamicsModel:
    """Fresh model with identity normalization (mean 0, std 1)."""
    mlp = eqx.nn.MLP(
        in_size=input_dim,
        out_size=output_dim,
        width_size=hidden_width,
        depth=hidden_depth,
        activation=jax.nn.tanh,
        key=key,
    )
    return ResidualDynamicsModel(
        mlp=mlp,
        input_mean=jnp.zeros(input_dim, dtype=jnp.float32),
        input_std=jnp.ones(input_dim, dtype=jnp.float32),
        target_mean=jnp.zeros(output_dim, dtype=jnp.float32),
        target_std=jnp.ones(output_dim, dtype=jnp.float32),
    )


def residual_features(
    body_twist: jax.Array,
    wheel_speeds: jax.Array,
    duty_cycle: jax.Array,
) -> jax.Array:
    """Assemble the state-action input in the order of RESIDUAL_FEATURE_NAMES.

    ``body_twist`` is the *current* twist [vx_body, vy_body, omega]; wheel
    speeds and duty cycles are [right, left], matching WheelLog.speeds.
    """
    return jnp.concatenate(
        [
            jnp.asarray(body_twist, dtype=jnp.float32).reshape(-1),
            jnp.asarray(wheel_speeds, dtype=jnp.float32).reshape(-1),
            jnp.asarray(duty_cycle, dtype=jnp.float32).reshape(-1),
        ]
    )


def apply_residual_model(model: ResidualDynamicsModel, features: jax.Array) -> jax.Array:
    """Residual twist [delta_vx_body, delta_vy_body, delta_omega].

    Accepts a single feature vector or a batch (leading axis).
    """
    features = jnp.asarray(features, dtype=jnp.float32)

    def single(feature_vector):
        normalized = (feature_vector - model.input_mean) / model.input_std
        return model.mlp(normalized) * model.target_std + model.target_mean

    if features.ndim == 1:
        return single(features)
    return jax.vmap(single)(features)


def residual_corrected_twist(
    model: ResidualDynamicsModel,
    features: jax.Array,
    nominal_v: jax.Array,
    nominal_omega: jax.Array,
):
    """Corrected body twist (v_x, v_y, omega) from the nominal twist and features."""
    delta = apply_residual_model(model, features)
    v_x = nominal_v + delta[..., 0]
    v_y = delta[..., 1]
    omega = nominal_omega + delta[..., 2]
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
    """Supervised state-action samples from one experiment log.

    Measured twist: the log's spline-derived body twists (``log.pose.twists``,
    see pololu.pose_smoothing) taken at the interval starts, consistent with
    Euler integration. Logs without twists (older pickles, simulated logs)
    fall back to finite-differencing the poses over each mocap interval and
    rotating the world-frame displacement into the body frame at the interval
    start. Nominal twist: encoder wheel speeds interpolated to the interval
    starts, passed through the ideal differential-drive kinematics.
    Sample t uses interval t-1's measured twist as the state input (see module
    docstring), so the first interval is consumed as state only.

    ``resample_uniform`` interpolates the poses (and twists) onto a uniform
    median-dt grid. For the finite-difference fallback this protects against
    logging-timestamp jitter (dividing a real ~10 ms displacement by a
    jittered 3-5 ms dt fabricates 2-3 m/s velocity spikes); with spline twists
    it merely regularizes the sample spacing.

    ``max_dt_factor`` drops mocap intervals longer than that multiple of the
    median interval (stream gaps make the finite-difference twist meaningless);
    a sample also needs a valid *previous* interval for its state input.
    Returns float32 ``features`` (N, 7), ``targets`` (N, 3), ``time_s``, ``dt``,
    and the measured/nominal twists plus pose arrays for diagnostics.
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
        # Spline-derived instantaneous twist at the interval start, consistent
        # with the Euler-integration convention used for the nominal twist.
        measured_twist = pose_twists[:-1]
    else:
        # Fallback: measured body twist from mocap finite differences.
        theta_unwrapped = np.unwrap(pose_states[:, 2])
        safe_dt = np.maximum(dt, 1e-9)
        dx = np.diff(pose_states[:, 0])
        dy = np.diff(pose_states[:, 1])
        theta = pose_states[:-1, 2]  # interval start, consistent with Euler integration
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        v_x_meas = (dx * cos_t + dy * sin_t) / safe_dt
        v_y_meas = (-dx * sin_t + dy * cos_t) / safe_dt
        omega_meas = np.diff(theta_unwrapped) / safe_dt
        measured_twist = np.column_stack([v_x_meas, v_y_meas, omega_meas])

    # Nominal twist from encoder speeds through the ideal kinematics.
    interval_time = pose_time_s[:-1]
    u_r_enc = np.interp(interval_time, wheel_time_s, wheel_speeds[:, 0])
    u_l_enc = np.interp(interval_time, wheel_time_s, wheel_speeds[:, 1])
    r = float(params.wheel_radius)
    effective_wheelbase = float(params.base_diameter)
    v_nom = 0.5 * r * (u_r_enc + u_l_enc)
    omega_nom = r * (u_r_enc - u_l_enc) / effective_wheelbase
    nominal_twist = np.column_stack([v_nom, np.zeros_like(v_nom), omega_nom])

    # Action context, zero-order hold at the interval starts.
    duty = _zero_order_hold(interval_time, wheel_time_s, duty_cycle)

    # State (previous interval's measured twist) + action at interval t.
    features = np.column_stack(
        [measured_twist[:-1], u_r_enc[1:], u_l_enc[1:], duty[1:, 0], duty[1:, 1]]
    )
    targets = (measured_twist - nominal_twist)[1:]
    assert features.shape[1] == len(RESIDUAL_FEATURE_NAMES)

    median_dt = float(np.median(dt[dt > min_dt])) if np.any(dt > min_dt) else 0.0
    interval_ok = (dt > min_dt) & (dt <= max_dt_factor * max(median_dt, min_dt))
    mask = interval_ok[1:] & interval_ok[:-1]  # sample t needs interval t and t-1

    return {
        "time_s": interval_time[1:][mask].astype(np.float32),
        "dt": dt[1:][mask].astype(np.float32),
        "features": features[mask].astype(np.float32),
        "targets": targets[mask].astype(np.float32),
        "measured_twist": measured_twist[1:][mask].astype(np.float32),
        "nominal_twist": nominal_twist[1:][mask].astype(np.float32),
        "initial_pose": pose_states[0].astype(np.float32),
        "pose_states": pose_states.astype(np.float32),
        "pose_time_s": pose_time_s.astype(np.float32),
        "mask": mask,
        # Contiguous (unmasked) per-interval arrays for multi-step rollout
        # training; interval_ok flags which intervals a window may cover.
        "seq": {
            "dt": dt.astype(np.float32),
            "measured_twist": measured_twist.astype(np.float32),
            "nominal_twist": nominal_twist.astype(np.float32),
            "actions": np.column_stack([u_r_enc, u_l_enc, duty]).astype(np.float32),
            "poses": pose_states.astype(np.float32),
            "interval_ok": interval_ok,
        },
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

    With a single file (or a split too small to reserve one), the caller should
    fall back to a within-file tail split (see split_dataset_tail).
    """
    indices = np.arange(len(datasets))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    num_validation = int(np.floor(len(indices) * validation_split))
    if len(indices) - num_validation == 0:
        raise ValueError("validation_split leaves no training files.")
    return list(indices[num_validation:]), list(indices[:num_validation])


def split_dataset_tail(dataset: dict, validation_split: float) -> tuple[dict, dict]:
    """Within-file split: the last ``validation_split`` fraction (contiguous in
    time) becomes validation, avoiding leakage from shuffled neighbors."""
    num_samples = len(dataset["features"])
    num_validation = int(np.floor(num_samples * validation_split))
    split_index = num_samples - num_validation
    keys = ("time_s", "dt", "features", "targets", "measured_twist", "nominal_twist")
    train = {key: dataset[key][:split_index] for key in keys}
    validation = {key: dataset[key][split_index:] for key in keys}
    return train, validation


def normalization_stats(values: np.ndarray, min_std: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), min_std)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_residual_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int = 0,
    hidden_width: int = 200,
    hidden_depth: int = 4,
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    output_reg_weight: float = 0.0,
):
    """Train the residual MLP with Adam on normalized inputs/targets.

    ``output_reg_weight`` adds a penalty on the squared norm of the model's
    *physical* residual output (per channel scaled by the target std so the
    three channels are commensurate) to the next-step prediction MSE. This
    shrinks the model toward the nominal dynamics wherever the data does not
    clearly demand a correction, which keeps closed-loop rollouts (gain tuning)
    from being destabilized by large extrapolated residuals. 0 disables.

    Returns (model, history) where the model already carries the normalization
    stats and ``history`` has per-epoch losses in normalized space: train is
    the full training objective (prediction MSE + output penalty), validation
    is the prediction MSE only.
    """
    import optax

    input_mean, input_std = normalization_stats(train_features)
    target_mean, target_std = normalization_stats(train_targets)
    # Physical output scaled by std: (z * std + mean) / std = z + mean / std.
    target_mean_over_std = jnp.asarray(target_mean / target_std, dtype=jnp.float32)

    x_train = jnp.asarray((train_features - input_mean) / input_std, dtype=jnp.float32)
    y_train = jnp.asarray((train_targets - target_mean) / target_std, dtype=jnp.float32)
    has_validation = len(validation_features) > 0
    if has_validation:
        x_val = jnp.asarray((validation_features - input_mean) / input_std, dtype=jnp.float32)
        y_val = jnp.asarray((validation_targets - target_mean) / target_std, dtype=jnp.float32)

    model = init_residual_model(
        jax.random.PRNGKey(seed),
        input_dim=train_features.shape[1],
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        output_dim=train_targets.shape[1],
    )
    mlp = model.mlp
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(mlp, eqx.is_array))

    def prediction_loss(mlp, x, y):
        predictions = jax.vmap(mlp)(x)
        return jnp.mean((predictions - y) ** 2)

    def batch_loss(mlp, x, y):
        predictions = jax.vmap(mlp)(x)
        mse = jnp.mean((predictions - y) ** 2)
        # Squared norm of the physical residual output (std-scaled per channel).
        output_norm = jnp.mean(jnp.sum((predictions + target_mean_over_std) ** 2, axis=1))
        return mse + output_reg_weight * output_norm

    @eqx.filter_jit
    def train_step(mlp, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(mlp, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        mlp = eqx.apply_updates(mlp, updates)
        return mlp, opt_state, loss

    eval_loss = eqx.filter_jit(prediction_loss)

    rng = np.random.default_rng(seed)
    num_samples = len(x_train)
    batch_size = min(batch_size, num_samples)
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(epochs):
        permutation = rng.permutation(num_samples)
        epoch_losses = []
        for start in range(0, num_samples - batch_size + 1, batch_size):
            batch = permutation[start : start + batch_size]
            mlp, opt_state, loss = train_step(mlp, opt_state, x_train[batch], y_train[batch])
            epoch_losses.append(float(loss))
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["validation_loss"].append(float(eval_loss(mlp, x_val, y_val)) if has_validation else float("nan"))
        if epoch % max(epochs // 20, 1) == 0 or epoch == epochs - 1:
            print(
                f"epoch {epoch + 1:4d}/{epochs}: train {history['train_loss'][-1]:.6f}"
                f"  validation {history['validation_loss'][-1]:.6f}"
            )

    model = ResidualDynamicsModel(
        mlp=mlp,
        input_mean=jnp.asarray(input_mean, dtype=jnp.float32),
        input_std=jnp.asarray(input_std, dtype=jnp.float32),
        target_mean=jnp.asarray(target_mean, dtype=jnp.float32),
        target_std=jnp.asarray(target_std, dtype=jnp.float32),
    )
    return model, history


def build_residual_sequences(dataset: dict, window_length: int, stride: int) -> dict:
    """Contiguous rollout windows from one log's dataset for multi-step training.

    A window starting at interval s needs intervals s-1 (initial state twist)
    through s+K-1 all valid. Returns stacked float32 arrays:
    ``init_twist`` (W, 3), ``nominal_twist`` (W, K, 3), ``actions`` (W, K, 4),
    ``poses`` (W, K+1, 3) and ``dt`` (W, K); poses[i, 0] is the window's start
    pose on the (uniform) mocap grid.
    """
    seq = dataset["seq"]
    interval_ok = np.asarray(seq["interval_ok"], dtype=bool)
    num_intervals = len(seq["dt"])
    starts = [
        s
        for s in range(1, num_intervals - window_length + 1, max(stride, 1))
        if interval_ok[s - 1 : s + window_length].all()
    ]
    if not starts:
        empty = lambda *shape: np.empty(shape, dtype=np.float32)  # noqa: E731
        return {
            "init_twist": empty(0, 3),
            "nominal_twist": empty(0, window_length, 3),
            "actions": empty(0, window_length, 4),
            "poses": empty(0, window_length + 1, 3),
            "dt": empty(0, window_length),
        }
    return {
        "init_twist": np.stack([seq["measured_twist"][s - 1] for s in starts]),
        "nominal_twist": np.stack([seq["nominal_twist"][s : s + window_length] for s in starts]),
        "actions": np.stack([seq["actions"][s : s + window_length] for s in starts]),
        "poses": np.stack([seq["poses"][s : s + window_length + 1] for s in starts]),
        "dt": np.stack([seq["dt"][s : s + window_length] for s in starts]),
    }


def stack_sequences(sequence_sets: list[dict]) -> dict:
    keys = ("init_twist", "nominal_twist", "actions", "poses", "dt")
    return {key: np.concatenate([s[key] for s in sequence_sets], axis=0) for key in keys}


def train_residual_model_multistep(
    train_sequences: dict,
    validation_sequences: dict,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    initial_mlp=None,
    seed: int = 0,
    hidden_width: int = 200,
    hidden_depth: int = 4,
    epochs: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    heading_weight: float = 0.1,
    output_reg_weight: float = 1e-3,
):
    """Fine-tune the residual MLP on multi-step open-loop rollouts.

    Each window is rolled out with the model feeding back its *own* corrected
    twist as the next state input (exactly the regime it faces in closed-loop
    simulation), integrating the corrected twists into poses. The loss is the
    mean squared position error against the (smooth) mocap poses plus a
    heading term ``2 - 2 cos(dtheta)`` weighted by ``heading_weight`` and a
    small std-scaled output-magnitude penalty. Because the supervision lives
    at the pose level, the twist-differentiation noise that dominates the
    one-step targets is integrated away, and the compounding-error/covariate
    shift of self-fed rollouts is trained on instead of ignored.

    ``normalization`` is (input_mean, input_std, target_mean, target_std) from
    the one-step dataset; ``initial_mlp`` warm-starts from a one-step
    pretrained MLP. History losses are the full rollout objective (train) and
    the same objective without the output penalty (validation).
    """
    import optax

    input_mean, input_std, target_mean, target_std = (
        jnp.asarray(v, dtype=jnp.float32) for v in normalization
    )

    if initial_mlp is None:
        initial_mlp = init_residual_model(
            jax.random.PRNGKey(seed),
            input_dim=int(input_mean.shape[0]),
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
            output_dim=int(target_mean.shape[0]),
        ).mlp
    mlp = initial_mlp
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(mlp, eqx.is_array))

    def to_jnp(sequences):
        return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in sequences.items()}

    train_data = to_jnp(train_sequences)
    has_validation = len(validation_sequences["dt"]) > 0
    if has_validation:
        validation_data = to_jnp(validation_sequences)

    def rollout_terms(net, init_twist, nominal, actions, poses, dt):
        def step(carry, inputs):
            twist, pose = carry
            nominal_t, action_t, pose_ref, step_dt = inputs
            features = jnp.concatenate([twist, action_t])
            normalized = (features - input_mean) / input_std
            delta = net(normalized) * target_std + target_mean
            new_twist = jnp.array(
                [nominal_t[0] + delta[0], delta[1], nominal_t[2] + delta[2]]
            )
            x, y, theta = pose
            cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
            new_pose = jnp.array(
                [
                    x + (new_twist[0] * cos_t - new_twist[1] * sin_t) * step_dt,
                    y + (new_twist[0] * sin_t + new_twist[1] * cos_t) * step_dt,
                    theta + new_twist[2] * step_dt,  # unwrapped; compared via cos below
                ]
            )
            position_error = jnp.sum((new_pose[:2] - pose_ref[:2]) ** 2)
            heading_error = 2.0 - 2.0 * jnp.cos(new_pose[2] - pose_ref[2])
            output_norm = jnp.sum((delta / target_std) ** 2)
            return (new_twist, new_pose), (position_error, heading_error, output_norm)

        _, (position_error, heading_error, output_norm) = jax.lax.scan(
            step, (init_twist, poses[0]), (nominal, actions, poses[1:], dt)
        )
        return jnp.mean(position_error), jnp.mean(heading_error), jnp.mean(output_norm)

    def batch_loss(net, data, with_reg: bool):
        position_error, heading_error, output_norm = jax.vmap(
            lambda i, n, a, p, d: rollout_terms(net, i, n, a, p, d)
        )(data["init_twist"], data["nominal_twist"], data["actions"], data["poses"], data["dt"])
        loss = jnp.mean(position_error) + heading_weight * jnp.mean(heading_error)
        if with_reg:
            loss = loss + output_reg_weight * jnp.mean(output_norm)
        return loss

    @eqx.filter_jit
    def train_step(net, opt_state, data):
        loss, grads = eqx.filter_value_and_grad(lambda m, d: batch_loss(m, d, True))(net, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss

    eval_loss = eqx.filter_jit(lambda net, data: batch_loss(net, data, False))

    rng = np.random.default_rng(seed)
    num_windows = len(train_sequences["dt"])
    batch_size = min(batch_size, num_windows)
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(epochs):
        permutation = rng.permutation(num_windows)
        for start in range(0, num_windows, batch_size):
            batch = permutation[start : start + batch_size]
            batch_data = {key: value[batch] for key, value in train_data.items()}
            mlp, opt_state, _ = train_step(mlp, opt_state, batch_data)
        history["train_loss"].append(float(eval_loss(mlp, train_data)))
        history["validation_loss"].append(
            float(eval_loss(mlp, validation_data)) if has_validation else float("nan")
        )
        if epoch % max(epochs // 20, 1) == 0 or epoch == epochs - 1:
            print(
                f"rollout epoch {epoch + 1:4d}/{epochs}: train {history['train_loss'][-1]:.6f}"
                f"  validation {history['validation_loss'][-1]:.6f}"
            )

    model = ResidualDynamicsModel(
        mlp=mlp,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    )
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

    Returns len(twists) + 1 poses starting at ``initial_pose``. Used by the
    residual diagnostics to compare open-loop nominal vs residual-augmented
    rollouts against the mocap trajectory.
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


def evaluate_predictions(model: ResidualDynamicsModel, features: np.ndarray, targets: np.ndarray) -> dict:
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
    model: ResidualDynamicsModel | None,
    problem: str,
    log_path: str,
    *,
    seed: int = 0,
    clip_after_first_trajectory: bool = True,
    mocap_filter_window_s: float = 0.0,
):
    """Closed-loop simulation (hidden robot + optional residual) tracking the
    reference trajectory recorded in a Pololu log.

    Returns the SimulationLog; the reference is the log's desired-pose stream
    fitted to the problem's simulation grid.
    """
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.simulation import SimulationPipeline

    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=clip_after_first_trajectory,
        mocap_filter_window_s=mocap_filter_window_s,
    )
    pipeline = SimulationPipeline(problem_path=problem, seed=seed, residual_model=model)
    reference_states = jnp.asarray(
        pipeline._fit_reference_states(np.asarray(log.reference.states, dtype=float)),
        dtype=jnp.float32,
    )
    return pipeline.run_closed_loop(
        pipeline.hidden_params,
        use_hidden_robot=True,
        controller_gains=pipeline.gains,
        reference_states=reference_states,
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


# ---------------------------------------------------------------------------
# High-level train / evaluate entry points (called by the thin CLI wrappers)
# ---------------------------------------------------------------------------


def train_from_logs(
    problem: str,
    log_dir: str,
    out: str,
    *,
    epochs: int = 300,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_width: int = 200,
    hidden_depth: int = 4,
    validation_split: float = 0.25,
    seed: int = 0,
    output_reg_weight: float = 1.0,
    multistep_window: int = 20,
    multistep_epochs: int = 2000,
    multistep_stride: int = 5,
    multistep_batch_size: int = 64,
    multistep_learning_rate: float = 1e-4,
    multistep_heading_weight: float = 0.1,
    multistep_output_reg_weight: float = 1e-3,
    clip_after_first_trajectory: bool = True,
    mocap_filter_window_s: float = 0.0,
    mocap_delay_s: float = 0.0,
    spline_order: int = 3,
    spline_noise_std_xy: float = 1e-3,
    spline_noise_std_yaw: float = 5e-3,
    spline_smoothing_factor: float = 1.0,
    resample_uniform: bool = True,
    out_dir: str = "visualize",
) -> ResidualDynamicsModel:
    """Build datasets from all logs in ``log_dir``, train, save checkpoint + plots.

    Besides the prediction diagnostics, runs a closed-loop simulation with the
    trained residual enabled along the reference of the first training log and
    saves the tracking plot (visualization.residual.plot_closed_loop_rollout).
    """
    from datetime import datetime
    from pathlib import Path

    from wmr_simulator.residual_model.io import save_residual_model
    from wmr_simulator.pololu.log_loader import list_pololu_log_paths, load_pololu_traj_control_log
    from wmr_simulator.visualization.residual import plot_predictions, plot_training_history

    params = robot_params_from_problem(problem)
    log_paths = list_pololu_log_paths(log_dir)
    datasets = []
    for path in log_paths:
        log = load_pololu_traj_control_log(
            path,
            clip_after_first_trajectory=clip_after_first_trajectory,
            mocap_filter_window_s=mocap_filter_window_s,
            mocap_delay_s=mocap_delay_s,
            spline_order=spline_order,
            spline_noise_std_xy=spline_noise_std_xy,
            spline_noise_std_yaw=spline_noise_std_yaw,
            spline_smoothing_factor=spline_smoothing_factor,
        )
        dataset = build_residual_dataset(log, params, resample_uniform=resample_uniform)
        datasets.append(dataset)
        print(f"{path}: {len(dataset['features'])} samples")

    num_validation_files = int(np.floor(len(datasets) * validation_split))
    file_split = num_validation_files >= 1 and len(datasets) - num_validation_files >= 1
    if file_split:
        train_indices, validation_indices = split_datasets_by_file(datasets, validation_split, seed)
        train_features, train_targets = stack_datasets([datasets[i] for i in train_indices])
        validation_features, validation_targets = stack_datasets([datasets[i] for i in validation_indices])
        train_files = [str(log_paths[i]) for i in train_indices]
        validation_files = [str(log_paths[i]) for i in validation_indices]
        print(f"Split by file: {len(train_indices)} train, {len(validation_indices)} validation")
        print(f"  validation files: {[Path(f).name for f in validation_files]}")
    else:
        splits = [split_dataset_tail(dataset, validation_split) for dataset in datasets]
        train_features, train_targets = stack_datasets([train for train, _ in splits])
        validation_features, validation_targets = stack_datasets([val for _, val in splits])
        train_files = validation_files = [str(path) for path in log_paths]
        print(f"Too few files for a file split; using per-file tail split ({validation_split:.0%})")

    print(f"Training samples: {len(train_features)}, validation samples: {len(validation_features)}")

    model, history = train_residual_model(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        seed=seed,
        hidden_width=hidden_width,
        hidden_depth=hidden_depth,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_reg_weight=output_reg_weight,
    )

    # Multi-step rollout fine-tuning: train on self-fed rollouts against the
    # (smooth) mocap poses, the same regime the model faces in closed loop.
    multistep_history = None
    if multistep_window > 0:
        if file_split:
            train_windows = stack_sequences(
                [build_residual_sequences(datasets[i], multistep_window, multistep_stride) for i in train_indices]
            )
            validation_windows = stack_sequences(
                [build_residual_sequences(datasets[i], multistep_window, multistep_stride) for i in validation_indices]
            )
        else:
            train_parts, validation_parts = [], []
            for dataset in datasets:
                windows = build_residual_sequences(dataset, multistep_window, multistep_stride)
                count = len(windows["dt"])
                cut = count - int(np.floor(count * validation_split))
                train_parts.append({key: value[:cut] for key, value in windows.items()})
                validation_parts.append({key: value[cut:] for key, value in windows.items()})
            train_windows = stack_sequences(train_parts)
            validation_windows = stack_sequences(validation_parts)
        print(
            f"Rollout fine-tuning: {len(train_windows['dt'])} train / {len(validation_windows['dt'])} validation "
            f"windows of {multistep_window} steps (stride {multistep_stride})"
        )
        normalization = (
            np.asarray(model.input_mean),
            np.asarray(model.input_std),
            np.asarray(model.target_mean),
            np.asarray(model.target_std),
        )
        model, multistep_history = train_residual_model_multistep(
            train_windows,
            validation_windows,
            normalization,
            initial_mlp=model.mlp,
            seed=seed,
            epochs=multistep_epochs,
            batch_size=multistep_batch_size,
            learning_rate=multistep_learning_rate,
            heading_weight=multistep_heading_weight,
            output_reg_weight=multistep_output_reg_weight,
        )

    config = {
        "input_dim": int(train_features.shape[1]),
        "hidden_width": hidden_width,
        "hidden_depth": hidden_depth,
        "output_dim": int(train_targets.shape[1]),
    }
    metadata = {
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "problem": problem,
        "log_dir": log_dir,
        "train_files": train_files,
        "validation_files": validation_files,
        "mocap_filter_window_s": mocap_filter_window_s,
        "mocap_delay_s": mocap_delay_s,
        "spline_order": spline_order,
        "spline_noise_std_xy": spline_noise_std_xy,
        "spline_noise_std_yaw": spline_noise_std_yaw,
        "spline_smoothing_factor": spline_smoothing_factor,
        "resample_uniform": resample_uniform,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "output_reg_weight": output_reg_weight,
        "multistep_window": multistep_window,
        "multistep_epochs": multistep_epochs,
        "multistep_stride": multistep_stride,
        "multistep_batch_size": multistep_batch_size,
        "multistep_learning_rate": multistep_learning_rate,
        "multistep_heading_weight": multistep_heading_weight,
        "multistep_output_reg_weight": multistep_output_reg_weight,
        "seed": seed,
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_rollout_train_loss": multistep_history["train_loss"][-1] if multistep_history else None,
        "final_rollout_validation_loss": multistep_history["validation_loss"][-1] if multistep_history else None,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_residual_model(out, model, config, metadata)
    print(f"Saved residual model to {out}")

    plot_training_history(history, out_dir=out_dir)
    if multistep_history is not None:
        plot_training_history(
            multistep_history,
            out_dir=out_dir,
            out_name="residual_rollout_loss.pdf",
            ylabel="rollout pose loss",
        )
    for split_name, features, targets in (
        ("train", train_features, train_targets),
        ("validation", validation_features, validation_targets),
    ):
        if len(features) == 0:
            continue
        evaluation = evaluate_predictions(model, features, targets)
        print(f"{split_name} RMSE  [dvx, dvy, domega]: {evaluation['rmse']}")
        print(f"{split_name} baseline (zero residual): {evaluation['baseline_rmse']}")
        plot_predictions(evaluation["predictions"], targets, split_name, out_dir=out_dir)

    # Closed-loop sanity check: simulate with the residual enabled along the
    # reference of the first training log (catches models that fit the data but
    # destabilize the rollout).
    from wmr_simulator.visualization.residual import plot_closed_loop_rollout

    first_train_log = train_files[0]
    sim_log = simulate_closed_loop_on_log_reference(
        model,
        problem,
        first_train_log,
        seed=seed,
        clip_after_first_trajectory=clip_after_first_trajectory,
    )
    plot_closed_loop_rollout(
        sim_log,
        out_prefix=f"residual_closed_loop_{Path(first_train_log).stem}",
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
    mocap_filter_window_s: float = 0.0,
    spline_order: int = 3,
    spline_noise_std_xy: float = 1e-3,
    spline_noise_std_yaw: float = 5e-3,
    spline_smoothing_factor: float = 1.0,
    resample_uniform: bool = True,
    out_dir: str = "visualize",
    out_prefix: str | None = None,
) -> dict:
    """Compare nominal vs residual-augmented open-loop rollout against mocap.

    Integrates the nominal twist, the residual-corrected twist, and the
    measured twist over the mocap grid from the initial mocap pose, saves the
    comparison plots, and returns the arrays/metrics.
    """
    from pathlib import Path

    from wmr_simulator.residual_model.io import load_residual_model
    from wmr_simulator.pololu.log_loader import load_pololu_traj_control_log
    from wmr_simulator.visualization.residual import plot_rollout_comparison

    model, checkpoint = load_residual_model(model_path)
    print(f"Loaded residual model {model_path}")
    print(f"  config: {checkpoint['config']}")
    if checkpoint.get("metadata"):
        validation_files = checkpoint["metadata"].get("validation_files", [])
        print(f"  trained validation files: {[Path(f).name for f in validation_files]}")

    params = robot_params_from_problem(problem)
    log = load_pololu_traj_control_log(
        log_path,
        clip_after_first_trajectory=clip_after_first_trajectory,
        mocap_filter_window_s=mocap_filter_window_s,
        spline_order=spline_order,
        spline_noise_std_xy=spline_noise_std_xy,
        spline_noise_std_yaw=spline_noise_std_yaw,
        spline_smoothing_factor=spline_smoothing_factor,
    )
    dataset = build_residual_dataset(log, params, resample_uniform=resample_uniform)

    predicted_residual = np.asarray(apply_residual_model(model, jnp.asarray(dataset["features"])))
    corrected_twist = dataset["nominal_twist"] + predicted_residual

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
    print(f"Residual RMSE  [dvx, dvy, domega]: {evaluation['rmse']}")
    print(f"Zero-residual baseline:            {evaluation['baseline_rmse']}")

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
    parser.add_argument("--log-dir", type=str, default="Pololu Data/Experiments/2026_07_06/04_New_Mocap_timestamps/decoded/")
    parser.add_argument("--out", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--validation-split", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    # Weight of the squared-norm penalty on the (std-scaled) physical residual
    # output, added to the prediction MSE; shrinks the model toward the nominal
    # dynamics to keep closed-loop rollouts stable. 0 disables. Empirically on
    # the 2026_07_01 logs, 1e-2 and 1e-1 still destabilize the closed-loop
    # rollout (residual feeds back on its own twist state); 1.0 is stable.
    parser.add_argument("--output-reg-weight", type=float, default=0.1)
    # Multi-step rollout fine-tuning (after one-step pretraining): windows of K
    # uniform mocap intervals are rolled out with the model feeding back its own
    # corrected twist, supervised by pose error against the smooth mocap poses.
    # This trains the self-fed regime the model faces in closed loop. 0 disables.
    parser.add_argument("--multistep-window", type=int, default=50)
    parser.add_argument("--multistep-epochs", type=int, default=10000)
    parser.add_argument("--multistep-stride", type=int, default=20)
    parser.add_argument("--multistep-batch-size", type=int, default=12000)
    parser.add_argument("--multistep-learning-rate", type=float, default=1e-5)
    parser.add_argument("--multistep-heading-weight", type=float, default=0.2)
    parser.add_argument("--multistep-output-reg-weight", type=float, default=1e-3)
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    # Resample filtered poses onto a uniform median-dt grid before differencing;
    # protects the twist targets against logging-timestamp jitter (see
    # build_residual_dataset). --no-resample-uniform differences at raw event times.
    parser.add_argument("--resample-uniform", action=argparse.BooleanOptionalAction, default=True)
    # Zero-phase moving-average window (seconds) on mocap positions before the
    # spline fit; 0 disables. The spline smoothing below supersedes it.
    parser.add_argument("--mocap-filter-window", type=float, default=0.0)
    # Mocap transport latency (seconds); timestamps shifted back before
    # differencing so twist targets align with the actions (identification/mocap_delay.py).
    parser.add_argument("--mocap-delay", type=float, default=0.0)
    # Smoothing-spline parameters for the mocap poses/twists (see
    # pololu.pose_smoothing.fit_pose_splines and the log loader).
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--spline-noise-std-xy", type=float, default=2e-3)
    parser.add_argument("--spline-noise-std-yaw", type=float, default=5e-3)
    parser.add_argument("--spline-smoothing-factor", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="visualize")
    args = parser.parse_args(argv)

    train_from_logs(
        problem=args.problem,
        log_dir=args.log_dir,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        validation_split=args.validation_split,
        seed=args.seed,
        output_reg_weight=args.output_reg_weight,
        multistep_window=args.multistep_window,
        multistep_epochs=args.multistep_epochs,
        multistep_stride=args.multistep_stride,
        multistep_batch_size=args.multistep_batch_size,
        multistep_learning_rate=args.multistep_learning_rate,
        multistep_heading_weight=args.multistep_heading_weight,
        multistep_output_reg_weight=args.multistep_output_reg_weight,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
        mocap_filter_window_s=args.mocap_filter_window,
        mocap_delay_s=args.mocap_delay,
        spline_order=args.spline_order,
        spline_noise_std_xy=args.spline_noise_std_xy,
        spline_noise_std_yaw=args.spline_noise_std_yaw,
        spline_smoothing_factor=args.spline_smoothing_factor,
        resample_uniform=args.resample_uniform,
        out_dir=args.out_dir,
    )


def evaluate_main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a residual dynamics model on a Pololu log.")
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml")
    parser.add_argument("--model", type=str, default="models/residual_pololu.pkl")
    parser.add_argument("--log", type=str, default="Pololu Data/Experiments/2026_07_06/04_New_Mocap_timestamps/TR06")
    parser.add_argument("--clip-after-first-trajectory", action="store_true", default=True)
    # Resample filtered poses onto a uniform median-dt grid before differencing;
    # protects the twist targets against logging-timestamp jitter (see
    # build_residual_dataset). --no-resample-uniform differences at raw event times.
    parser.add_argument("--resample-uniform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mocap-filter-window", type=float, default=0.0)
    # Smoothing-spline parameters for the mocap poses/twists (see
    # pololu.pose_smoothing.fit_pose_splines and the log loader).
    parser.add_argument("--spline-order", type=int, default=3)
    parser.add_argument("--spline-noise-std-xy", type=float, default=1e-3)
    parser.add_argument("--spline-noise-std-yaw", type=float, default=1e-3)
    parser.add_argument("--spline-smoothing-factor", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="visualize")
    parser.add_argument("--output", type=str, default=None, help="Output filename prefix")
    args = parser.parse_args(argv)

    evaluate_on_log(
        model_path=args.model,
        log_path=args.log,
        problem=args.problem,
        clip_after_first_trajectory=args.clip_after_first_trajectory,
        mocap_filter_window_s=args.mocap_filter_window,
        spline_order=args.spline_order,
        spline_noise_std_xy=args.spline_noise_std_xy,
        spline_noise_std_yaw=args.spline_noise_std_yaw,
        spline_smoothing_factor=args.spline_smoothing_factor,
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

    Yaw is unwrapped before interpolation (interpolating across the +-pi wrap
    would cut through zero) and re-wrapped after. Keeps the original start/end
    times so downstream stream alignment (encoder/duty interpolation by time)
    is unaffected.
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
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    train_main()
