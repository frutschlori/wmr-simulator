import jax.numpy as jnp


def sigma(t: jnp.ndarray, total_time: jnp.ndarray) -> jnp.ndarray:
    """5th-order S-curve on [0, T] -> [0, 1]."""
    tau = t / total_time
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def sigma_dot(t: jnp.ndarray, total_time: jnp.ndarray) -> jnp.ndarray:
    tau = t / total_time
    return (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / total_time


def sigma_ddot(t: jnp.ndarray, total_time: jnp.ndarray) -> jnp.ndarray:
    tau = t / total_time
    return (60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3) / (total_time**2)


def normalize_time_scaling(time_scaling: str | None) -> str:
    if time_scaling is None:
        return "s_curve"
    normalized = time_scaling.lower().replace("-", "_")
    if normalized in {"s_curve", "scurve"}:
        return "s_curve"
    if normalized == "linear":
        return "linear"
    raise ValueError(f"Unsupported time scaling '{time_scaling}'. Expected 's-curve' or 'linear'.")


def time_scaling_derivatives(
    t: jnp.ndarray,
    total_time: jnp.ndarray,
    time_scaling: str | None = "s_curve",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    selected_time_scaling = normalize_time_scaling(time_scaling)
    if selected_time_scaling == "linear":
        return t / total_time, jnp.ones_like(t) / total_time, jnp.zeros_like(t)
    return sigma(t, total_time), sigma_dot(t, total_time), sigma_ddot(t, total_time)
