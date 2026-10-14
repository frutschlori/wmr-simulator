import jax
import jax.numpy as jnp


DEFAULT_CONSTRAINT_WEIGHTS = {
    "v": 1.0,
    "a": 1.0,
    "lateral": 1.0,
    "omega": 1.0,
    "alpha": 1.0,
    # Lower bound on the trajectory's *mean* speed; see min_speed_loss.
    "v_min": 1.0,
    # Lower bound on its mean lateral acceleration; see
    # min_lateral_acceleration_loss.
    "a_lat_min": 1.0,
}


def smooth_max(x: jnp.ndarray, beta: float = 20.0) -> jnp.ndarray:
    return jax.nn.logsumexp(beta * x) / beta


def smooth_positive_max(x: jnp.ndarray, beta: float = 20.0) -> jnp.ndarray:
    zero = jnp.zeros((1,), dtype=x.dtype)
    return smooth_max(jnp.concatenate([zero, x], axis=0), beta=beta)


def smooth_norm(x: jnp.ndarray, axis: int = -1, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)


def motion_limits_from_robot_config(robot_cfg: dict) -> dict[str, jnp.ndarray]:
    return {
        "v_max": jnp.asarray(robot_cfg["v_max"], dtype=jnp.float32),
        "a_max": jnp.asarray(robot_cfg["a_max"], dtype=jnp.float32),
        "a_lat_max": jnp.asarray(robot_cfg.get("a_max_lateral", robot_cfg["a_max"]), dtype=jnp.float32),
        "omega_max": jnp.asarray(robot_cfg["omega_max"], dtype=jnp.float32),
        "alpha_max": jnp.asarray(robot_cfg["alpha_max"], dtype=jnp.float32),
        # Lower bounds (0 = off); the pipeline sets them per trajectory.
        "v_min": jnp.asarray(robot_cfg.get("v_min", 0.0), dtype=jnp.float32),
        "a_lat_min": jnp.asarray(robot_cfg.get("a_lat_min", 0.0), dtype=jnp.float32),
    }


def min_speeds_for_batch(num_trajectories: int, min_speed: float, fraction: float = 0.5):
    """Per-trajectory ``v_min`` for a batch design: ``fraction`` of the batch
    gets a floor, spread linearly over ``[0.5 * min_speed, min_speed]``, the
    rest none (0)."""
    import numpy as np

    speeds = np.zeros(int(num_trajectories), dtype=float)
    if float(min_speed) <= 0.0 or float(fraction) <= 0.0 or num_trajectories <= 0:
        return speeds
    count = int(np.clip(round(float(fraction) * num_trajectories), 1, num_trajectories))
    if count == 1:
        speeds[0] = float(min_speed)
        return speeds
    speeds[:count] = float(min_speed) * np.linspace(0.5, 1.0, count)
    return speeds


def motion_floors_for_batch(
    num_trajectories: int,
    min_speed: float,
    min_speed_fraction: float = 0.5,
    min_lateral_acceleration: float = 0.0,
    min_lateral_acceleration_fraction: float = 0.25,
):
    """Per-trajectory ``[v_min, a_lat_min]`` for a batch design, shape ``(T, 2)``.

    Each column is spread like :func:`min_speeds_for_batch`. The turning floors
    end where the speed floors end, so the hardest turn goes to the fastest
    trajectory.
    """
    import numpy as np

    speeds = min_speeds_for_batch(num_trajectories, min_speed, min_speed_fraction)
    lateral_head = min_speeds_for_batch(num_trajectories, min_lateral_acceleration, min_lateral_acceleration_fraction)
    num_lateral = int(np.count_nonzero(lateral_head))
    num_speed = int(np.count_nonzero(speeds))
    end = max(num_speed, num_lateral)
    lateral = np.zeros(int(num_trajectories), dtype=float)
    lateral[end - num_lateral : end] = lateral_head[:num_lateral]
    return np.stack([speeds, lateral], axis=1)


def constraint_weights(scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
    weights = dict(DEFAULT_CONSTRAINT_WEIGHTS)
    if component_weights is not None:
        weights.update(component_weights)
    return {
        name: jnp.asarray(scale * float(weights[name]), dtype=jnp.float32)
        for name in DEFAULT_CONSTRAINT_WEIGHTS
    }


def finite_difference(values: jnp.ndarray, dt: float) -> jnp.ndarray:
    dt = jnp.asarray(dt, dtype=values.dtype)
    if values.shape[0] < 2:
        return jnp.zeros_like(values)

    first = (values[1:2] - values[0:1]) / dt
    last = (values[-1:] - values[-2:-1]) / dt
    if values.shape[0] == 2:
        return jnp.concatenate([first, last], axis=0)

    middle = (values[2:] - values[:-2]) / (2.0 * dt)
    return jnp.concatenate([first, middle, last], axis=0)


def constraint_loss(
    v: jnp.ndarray,
    a: jnp.ndarray,
    omega: jnp.ndarray,
    alpha_ang: jnp.ndarray,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    components = constraint_loss_components(
        v=v,
        a=a,
        omega=omega,
        alpha_ang=alpha_ang,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
    return sum(components.values())


def _mean_floor_loss(values: jnp.ndarray, floor, phase_masks=None, smooth_max_beta: float = 20.0) -> jnp.ndarray:
    """Squared fractional shortfall of the mean of ``values`` below ``floor``.

    ``phase_masks`` (P, N) splits the samples into motion phases, and ``floor``
    is then one value per phase, each judged on its own phase's mean; without
    masks it is one floor on the whole trajectory. A floor ``<= 0`` is off and
    contributes exactly 0.
    """
    floors = jnp.atleast_1d(jnp.asarray(floor, dtype=values.dtype))
    if phase_masks is None:
        means = jnp.mean(values)[None]
    else:
        masks = jnp.asarray(phase_masks, dtype=values.dtype)
        means = (masks @ values) / jnp.maximum(jnp.sum(masks, axis=1), 1.0)
    shortfall = 1.0 - means / jnp.maximum(floors, 1e-6)
    losses = jax.vmap(lambda value: smooth_positive_max(value[None], beta=smooth_max_beta))(shortfall) ** 2
    return jnp.sum(jnp.where(floors > 0.0, losses, jnp.zeros_like(losses)))


def min_speed_loss(v_norm: jnp.ndarray, v_min, smooth_max_beta: float = 20.0, phase_masks=None) -> jnp.ndarray:
    """Squared fractional shortfall of the mean speed below ``v_min`` (the
    reference rests at both ends, so a per-sample bound would not work).

    ``v_min <= 0`` disables it. With ``phase_masks`` each motion phase has its
    own ``v_min`` on its own mean.
    """
    return _mean_floor_loss(v_norm, v_min, phase_masks, smooth_max_beta)


def min_lateral_acceleration_loss(
    a_lat: jnp.ndarray, a_lat_min, smooth_max_beta: float = 20.0, phase_masks=None
) -> jnp.ndarray:
    """Squared fractional shortfall of the mean |v * omega| below ``a_lat_min``.

    ``a_lat_min <= 0`` disables it. With ``phase_masks`` each motion phase has
    its own floor on its own mean.
    """
    return _mean_floor_loss(a_lat, a_lat_min, phase_masks, smooth_max_beta)


def constraint_loss_components(
    v: jnp.ndarray,
    a: jnp.ndarray,
    omega: jnp.ndarray,
    alpha_ang: jnp.ndarray,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> dict[str, jnp.ndarray]:
    # Upper limits broadcast per sample; floors are per phase (``phase_masks``).
    v_norm = smooth_norm(v, axis=-1)
    a_norm = smooth_norm(a, axis=-1)
    a_lat = jnp.abs(v_norm * omega)
    phase_masks = limits.get("phase_masks")

    g_v_samples = v_norm / limits["v_max"] - 1.0
    g_a_samples = a_norm / limits["a_max"] - 1.0
    g_lat_samples = a_lat / limits["a_lat_max"] - 1.0
    g_w_samples = jnp.abs(omega) / limits["omega_max"] - 1.0
    g_alpha_samples = jnp.abs(alpha_ang) / limits["alpha_max"] - 1.0

    v_loss = smooth_positive_max(g_v_samples, beta=smooth_max_beta) ** 2
    a_loss = smooth_positive_max(g_a_samples, beta=smooth_max_beta) ** 2
    lateral_loss = smooth_positive_max(g_lat_samples, beta=smooth_max_beta) ** 2
    omega_loss = smooth_positive_max(g_w_samples, beta=smooth_max_beta) ** 2
    alpha_loss = smooth_positive_max(g_alpha_samples, beta=smooth_max_beta) ** 2

    return {
        "v": weights["v"] * v_loss,
        "a": weights["a"] * a_loss,
        "lateral": weights["lateral"] * lateral_loss,
        "omega": weights["omega"] * omega_loss,
        "alpha": weights["alpha"] * alpha_loss,
        "v_min": weights["v_min"]
        * min_speed_loss(v_norm, limits.get("v_min", 0.0), smooth_max_beta=smooth_max_beta, phase_masks=phase_masks),
        "a_lat_min": weights["a_lat_min"]
        * min_lateral_acceleration_loss(
            a_lat, limits.get("a_lat_min", 0.0), smooth_max_beta=smooth_max_beta, phase_masks=phase_masks
        ),
    }


def constraint_loss_from_reference_states(
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    v = reference_states[:, 3:5]
    omega = reference_states[:, 5]
    a = reference_states[:, 6:8]
    alpha_ang = finite_difference(omega, dt)
    return constraint_loss(
        v=v,
        a=a,
        omega=omega,
        alpha_ang=alpha_ang,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )


# Where the motion limits come from -----------------------------------------

GRAVITY = 9.81
