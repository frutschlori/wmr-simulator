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


def normalize_phase_breaks(phase_breaks) -> tuple[float, ...]:
    """Interior phase boundaries as strictly increasing fractions of the duration."""
    breaks = tuple(float(value) for value in (phase_breaks or ()))
    if any(not 0.0 < value < 1.0 for value in breaks) or any(b <= a for a, b in zip(breaks, breaks[1:])):
        raise ValueError(f"Phase breaks must be strictly increasing fractions in (0, 1), got {breaks}.")
    return breaks


def phase_indices(fraction, phase_breaks) -> jnp.ndarray:
    """Phase each time fraction in [0, 1] falls in; a boundary belongs to the later phase."""
    breaks = jnp.asarray(normalize_phase_breaks(phase_breaks), dtype=jnp.float32)
    return jnp.searchsorted(breaks, jnp.asarray(fraction, dtype=jnp.float32), side="right")


def time_scaling_derivatives(
    t: jnp.ndarray,
    total_time: jnp.ndarray,
    time_scaling: str | None = "s_curve",
    phase_breaks=(),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Path parameter s(t) in [0, 1] and its first two time derivatives.

    ``phase_breaks`` (fractions of the duration) split the s-curve into
    consecutive phases, each its own quintic s-curve over the same share of the
    path parameter as of the time. The reference comes to rest at every
    boundary with zero acceleration, so the trajectory stays C^2 in time and
    each phase can be held to its own motion envelope (a slow phase followed by
    a fast one, say) without one leaking into the other.
    """
    selected_time_scaling = normalize_time_scaling(time_scaling)
    breaks = normalize_phase_breaks(phase_breaks)
    if selected_time_scaling == "linear":
        if breaks:
            raise ValueError("Motion phases need the s-curve time scaling, which rests between phases.")
        return t / total_time, jnp.ones_like(t) / total_time, jnp.zeros_like(t)
    if not breaks:
        return sigma(t, total_time), sigma_dot(t, total_time), sigma_ddot(t, total_time)
    bounds = jnp.asarray((0.0, *breaks, 1.0), dtype=jnp.float32)
    fraction = t / total_time
    index = jnp.clip(phase_indices(fraction, breaks), 0, len(breaks))
    start = bounds[index]
    width = bounds[index + 1] - start
    local = jnp.clip((fraction - start) / width, 0.0, 1.0)
    s = start + width * (10.0 * local**3 - 15.0 * local**4 + 6.0 * local**5)
    s_dot = (30.0 * local**2 - 60.0 * local**3 + 30.0 * local**4) / total_time
    s_ddot = (60.0 * local - 180.0 * local**2 + 120.0 * local**3) / (total_time**2 * width)
    return s, s_dot, s_ddot
