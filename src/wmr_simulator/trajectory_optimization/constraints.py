import numpy as np
import jax
import jax.numpy as jnp


DEFAULT_CONSTRAINT_WEIGHTS = {
    "v": 1.0,
    "a": 1.0,
    "lateral": 1.0,
    "omega": 1.0,
    "alpha": 1.0,
}


def initial_bezier_control_points(problem, order: int) -> jnp.ndarray:
    if order < 1:
        raise ValueError("Bezier order must be at least 1.")
    num_control_points = order + 1
    line_samples = np.linspace(0.0, 1.0, num_control_points)[:, None]
    start = problem.start[:2][None, :]
    goal = problem.goal[:2][None, :]
    control_points = start + line_samples * (goal - start)
    return jnp.asarray(control_points, dtype=jnp.float32)


def clamp_control_points(problem, control_points: jnp.ndarray) -> jnp.ndarray:
    env_min = jnp.asarray(problem.environment_min, dtype=jnp.float32)
    env_max = jnp.asarray(problem.environment_max, dtype=jnp.float32)
    clamped = jnp.clip(control_points, env_min, env_max)
    clamped = clamped.at[0].set(jnp.asarray(problem.start[:2], dtype=jnp.float32))
    return clamped.at[-1].set(jnp.asarray(problem.goal[:2], dtype=jnp.float32))


def control_points_from_decision_variables(problem, decision_variables: jnp.ndarray) -> jnp.ndarray:
    start_point = jnp.asarray(problem.start[:2], dtype=jnp.float32)[None, :]
    control_points = jnp.concatenate([start_point, decision_variables], axis=0)
    return clamp_control_points(problem, control_points)


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
    }


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
