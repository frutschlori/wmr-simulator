"""Start-pose offsets as (optionally) decision variables of the joint loop.

The rollout start offsets are what make the tracking gains observable at all
(``gain_tuning.objectives.sample_initial_pose_offsets``). The joint loop can
leave them as a frozen draw, fix them to a deterministic spread, or hand them
to the trajectory block as decision variables -- one enum, one code path, a
3-element boolean mask over ``[dx, dy, dtheta]``.

The *trajectory* block owns them, never the gain block: minimizing the gain
loss over the offsets would drive them to zero and destroy exactly the
excitation they exist to provide, while minimizing the FIM loss maximizes
information, which is the only self-consistent choice.
"""

import jax.numpy as jnp


START_OFFSET_MODE_RANDOM = "random"
START_OFFSET_MODE_STATIC = "static"
START_OFFSET_MODE_OPTIMIZE = "optimize"
START_OFFSET_MODE_OPTIMIZE_HEADING = "optimize-heading"
START_OFFSET_MODE_OPTIMIZE_DISPLACEMENT = "optimize-displacement"

# Which of [dx, dy, dtheta] are decision variables; the rest keep their frozen
# value, and the ``jnp.where`` below zeroes their gradient the same way
# ``bspline.clamp_control_points`` zeroes a pinned control point's.
START_OFFSET_MODE_MASKS = {
    START_OFFSET_MODE_RANDOM: (False, False, False),
    START_OFFSET_MODE_STATIC: (False, False, False),
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
    ``offset_radius``, headings alternating +/-``offset_angle``.

    The point of this mode is to remove the draw from the comparison entirely --
    every seed sees the same conditions, so a difference between runs is the
    optimizer's, not the sampler's.
    """
    bearings = jnp.linspace(0.0, 2.0 * jnp.pi, num_realizations, endpoint=False)
    signs = jnp.where(jnp.arange(num_realizations) % 2 == 0, 1.0, -1.0)
    return jnp.stack(
        [offset_radius * jnp.cos(bearings), offset_radius * jnp.sin(bearings), offset_angle * signs],
        axis=1,
    ).astype(jnp.float32)


def squash_start_offsets(free_offsets: jnp.ndarray, offset_radius: float, offset_angle: float) -> jnp.ndarray:
    """Map unconstrained ``u in R^{R x 3}`` smoothly into the feasible offset set.

    Squash, not clip: FIM-optimized offsets *will* be driven to the boundary of
    the disk, and a clip there would reproduce the constraint-wall pathology
    that makes the trajectory objective choppy (a zero-then-jump derivative Adam
    bounces off). ``u = 0`` gives exactly zero offset with a finite gradient, so
    starting from "no offset" is not a singular point either.
    """
    displacement = free_offsets[:, :2]
    scale = offset_radius / jnp.sqrt(1.0 + jnp.sum(displacement**2, axis=1, keepdims=True))
    heading = offset_angle * jnp.tanh(free_offsets[:, 2])
    return jnp.concatenate([displacement * scale, heading[:, None]], axis=1)


def inverse_squash_start_offsets(
    offsets: jnp.ndarray, offset_radius: float, offset_angle: float
) -> jnp.ndarray:
    """The inverse of :func:`squash_start_offsets`, for initializing the free
    variables *at* a given offset set (so an optimizing mode starts from the
    same conditions the frozen modes use, rather than from zero offset).

    Inputs are pulled just inside the boundary first: the map sends the boundary
    to infinity.
    """
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
