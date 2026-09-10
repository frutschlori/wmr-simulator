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
        # Lower bound, 0 = off. Not a property of the robot -- it is a property
        # of the *experiment being designed*, so the pipeline substitutes a
        # per-trajectory value; the robot config only supplies a fallback.
        "v_min": jnp.asarray(robot_cfg.get("v_min", 0.0), dtype=jnp.float32),
    }


def min_speeds_for_batch(num_trajectories: int, min_speed: float, fraction: float = 0.5):
    """Per-trajectory ``v_min`` for a batch design: some fast, some left free.

    A design set exists to be informative *and* to cover the regimes the
    controller is judged in, and those pull in opposite directions -- an
    unconstrained FIM buys slow tight wiggles, and a set that is uniformly fast
    stops covering the slow regime at all. So ``fraction`` of the batch carries
    a minimum and the rest carries none, and the constrained ones are spread
    linearly over ``[0.5 * min_speed, min_speed]`` so the fast group is itself a
    range rather than one operating point.

    ``min_speed <= 0`` or ``fraction <= 0`` returns all zeros, which the
    constraint reads as "off".
    """
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


def min_speed_loss(v_norm: jnp.ndarray, v_min, smooth_max_beta: float = 20.0) -> jnp.ndarray:
    """Squared fractional shortfall of the *mean* speed below ``v_min``.

    Every other limit here is an upper bound read off the worst sample, but a
    minimum speed cannot be: under the s-curve time scaling the reference is at
    rest at both ends by construction, so a worst-sample rule would be violated
    on every feasible design and would only fight the time scaling. The mean is
    the statistic that expresses "this trajectory is driven fast" without
    saying anything about how it starts and stops.

    ``sim_time`` is fixed and not a decision variable, so mean speed is
    arc length / sim_time: the only way the optimizer can satisfy this is to
    lay out a *longer* path, which is exactly the intent -- long sustained
    sweeps instead of the tight slow wiggles an unconstrained FIM prefers.
    It is a quadratic penalty, not a barrier, so an infeasible ``v_min`` (one
    that would need more arc than the environment box holds) trades off against
    the FIM rather than breaking the run.

    ``v_min <= 0`` disables it and returns exactly 0, so a design that does not
    ask for a minimum speed is bit-identical to one from before this existed.
    """
    v_min = jnp.asarray(v_min, dtype=v_norm.dtype)
    shortfall = 1.0 - jnp.mean(v_norm) / jnp.maximum(v_min, 1e-6)
    loss = smooth_positive_max(jnp.reshape(shortfall, (1,)), beta=smooth_max_beta) ** 2
    return jnp.where(v_min > 0.0, loss, jnp.zeros_like(loss))


def constraint_loss_components(
    v: jnp.ndarray,
    a: jnp.ndarray,
    omega: jnp.ndarray,
    alpha_ang: jnp.ndarray,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> dict[str, jnp.ndarray]:
    v_norm = smooth_norm(v, axis=-1)
    a_norm = smooth_norm(a, axis=-1)
    a_lat = jnp.abs(v_norm * omega)

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
        * min_speed_loss(v_norm, limits.get("v_min", 0.0), smooth_max_beta=smooth_max_beta),
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


def constraint_loss_components_from_reference_states(
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> dict[str, jnp.ndarray]:
    v = reference_states[:, 3:5]
    omega = reference_states[:, 5]
    a = reference_states[:, 6:8]
    alpha_ang = finite_difference(omega, dt)
    return constraint_loss_components(
        v=v,
        a=a,
        omega=omega,
        alpha_ang=alpha_ang,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
