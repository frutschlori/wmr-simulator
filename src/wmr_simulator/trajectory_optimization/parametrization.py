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


def _reference_from_derivatives(
    position: jnp.ndarray,
    dpos_ds: jnp.ndarray,
    d2pos_ds2: jnp.ndarray,
    s_dot: jnp.ndarray,
    s_ddot: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble the [T, 8] reference-state matrix from curve derivatives.
    Shared by every curve parametrization (bezier, quintic_spline, ...) --
    keep this bit-identical across curve kinds."""
    velocity = dpos_ds * s_dot[:, None]
    acceleration = dpos_ds * s_ddot[:, None] + d2pos_ds2 * (s_dot[:, None] ** 2)
    theta = jnp.arctan2(dpos_ds[:, 1], dpos_ds[:, 0])

    tangent_norm_sq = jnp.sum(dpos_ds**2, axis=1)
    dtheta_ds = (
        dpos_ds[:, 0] * d2pos_ds2[:, 1] - dpos_ds[:, 1] * d2pos_ds2[:, 0]
    ) / (tangent_norm_sq + 1e-8)
    omega = dtheta_ds * s_dot

    return jnp.column_stack(
        [
            position[:, 0],
            position[:, 1],
            theta,
            velocity[:, 0],
            velocity[:, 1],
            omega,
            acceleration[:, 0],
            acceleration[:, 1],
        ]
    )
