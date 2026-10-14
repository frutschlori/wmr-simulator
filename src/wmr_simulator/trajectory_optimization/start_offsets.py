"""Start-pose offsets of the tuning rollouts.

Offsets are sampled at random (:func:`sample_initial_pose_offsets`), spread
deterministically (:func:`static_start_offsets`), left out (``none``), or made
decision variables of the trajectory design (:func:`squash_start_offsets` and
the ``optimize*`` modes), selected by a boolean mask over ``[dx, dy, dtheta]``.
The gain tuner reads the designed offsets from the trajectory pickles, or
draws its own with the sampler here.
"""

import jax
import jax.numpy as jnp


START_OFFSET_MODE_RANDOM = "random"
START_OFFSET_MODE_STATIC = "static"
START_OFFSET_MODE_NONE = "none"
START_OFFSET_MODE_OPTIMIZE = "optimize"
START_OFFSET_MODE_OPTIMIZE_HEADING = "optimize-heading"
START_OFFSET_MODE_OPTIMIZE_DISPLACEMENT = "optimize-displacement"

# Which of [dx, dy, dtheta] are decision variables; the rest keep their frozen value.
START_OFFSET_MODE_MASKS = {
    START_OFFSET_MODE_RANDOM: (False, False, False),
    START_OFFSET_MODE_STATIC: (False, False, False),
    START_OFFSET_MODE_NONE: (False, False, False),
    START_OFFSET_MODE_OPTIMIZE: (True, True, True),
    START_OFFSET_MODE_OPTIMIZE_HEADING: (False, False, True),
    START_OFFSET_MODE_OPTIMIZE_DISPLACEMENT: (True, True, False),
}
START_OFFSET_MODES = set(START_OFFSET_MODE_MASKS)


def normalize_start_offset_mode(start_offset_mode: str) -> str:
    start_offset_mode = start_offset_mode.strip().lower().replace("_", "-")
    if start_offset_mode not in START_OFFSET_MODES:
        raise ValueError(
            f"Unsupported start-offset mode '{start_offset_mode}'. "
            f"Expected one of {sorted(START_OFFSET_MODES)}."
        )
    return start_offset_mode


def start_offset_mask(start_offset_mode: str) -> jnp.ndarray:
    return jnp.asarray(START_OFFSET_MODE_MASKS[normalize_start_offset_mode(start_offset_mode)])


def static_start_offsets(num_realizations: int, offset_radius: float, offset_angle: float) -> jnp.ndarray:
    """Deterministic spread: bearings evenly spaced on the circle of
    ``offset_radius``, headings alternating +/-``offset_angle``."""
    bearings = jnp.linspace(0.0, 2.0 * jnp.pi, num_realizations, endpoint=False)
    signs = jnp.where(jnp.arange(num_realizations) % 2 == 0, 1.0, -1.0)
    return jnp.stack(
        [offset_radius * jnp.cos(bearings), offset_radius * jnp.sin(bearings), offset_angle * signs],
        axis=1,
    ).astype(jnp.float32)


def squash_start_offsets(free_offsets: jnp.ndarray, offset_radius: float, offset_angle: float) -> jnp.ndarray:
    """Map unconstrained ``u in R^{R x 3}`` smoothly into the feasible offset
    set (a squash rather than a clip, so the gradient stays smooth at the
    boundary; ``u = 0`` is zero offset)."""
    displacement = free_offsets[:, :2]
    scale = offset_radius / jnp.sqrt(1.0 + jnp.sum(displacement**2, axis=1, keepdims=True))
    heading = offset_angle * jnp.tanh(free_offsets[:, 2])
    return jnp.concatenate([displacement * scale, heading[:, None]], axis=1)


def inverse_squash_start_offsets(
    offsets: jnp.ndarray, offset_radius: float, offset_angle: float
) -> jnp.ndarray:
    """The inverse of :func:`squash_start_offsets` (inputs pulled just inside
    the boundary, which maps to infinity)."""
    radius_fraction = jnp.clip(
        jnp.sqrt(jnp.sum(offsets[:, :2] ** 2, axis=1, keepdims=True)) / offset_radius, 0.0, 0.99
    )
    displacement = offsets[:, :2] / (offset_radius * jnp.sqrt(1.0 - radius_fraction**2))
    heading = jnp.arctanh(jnp.clip(offsets[:, 2] / offset_angle, -0.99, 0.99))
    return jnp.concatenate([displacement, heading[:, None]], axis=1)


def resolve_start_offsets(
    free_offsets: jnp.ndarray,
    frozen_offsets: jnp.ndarray,
    mask: jnp.ndarray,
    offset_radius: float,
    offset_angle: float,
) -> jnp.ndarray:
    return jnp.where(
        mask, squash_start_offsets(free_offsets, offset_radius, offset_angle), frozen_offsets
    )


def sample_initial_pose_offsets(
    key: jax.Array,
    num_realizations: int,
    offset_radius: float,
    offset_angle: float,
) -> jax.Array:
    """Draw one start-pose offset ``[dx, dy, dtheta]`` per noise realization:
    positions uniform over the disk of ``offset_radius`` (by area), headings
    uniform over +/-``offset_angle``."""
    if offset_radius <= 0.0 and offset_angle <= 0.0:
        return jnp.zeros((num_realizations, 3), dtype=jnp.float32)
    radius_key, bearing_key, heading_key = jax.random.split(key, 3)
    radius = offset_radius * jnp.sqrt(jax.random.uniform(radius_key, (num_realizations,)))
    bearing = jax.random.uniform(bearing_key, (num_realizations,), minval=-jnp.pi, maxval=jnp.pi)
    heading = jax.random.uniform(heading_key, (num_realizations,), minval=-offset_angle, maxval=offset_angle)
    return jnp.stack(
        [radius * jnp.cos(bearing), radius * jnp.sin(bearing), heading], axis=1
    ).astype(jnp.float32)


def sample_initial_pose_offset_batch(
    key: jax.Array,
    num_trajectories: int,
    num_realizations: int,
    offset_radius: float,
    offset_angle: float,
) -> jax.Array:
    """Draw an independent ``(R, 3)`` offset bundle per trajectory, shape ``(T, R, 3)``."""
    trajectory_keys = jax.random.split(key, num_trajectories)
    return jax.vmap(
        lambda trajectory_key: sample_initial_pose_offsets(
            trajectory_key, num_realizations, offset_radius, offset_angle
        )
    )(trajectory_keys)
