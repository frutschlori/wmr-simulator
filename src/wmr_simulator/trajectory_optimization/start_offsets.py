"""Where a rollout starts: every way a start-pose offset comes into being.

Sampled at random (:func:`sample_initial_pose_offsets`), spread
deterministically (:func:`static_start_offsets`), or made a decision variable
of the design (:func:`squash_start_offsets` and the ``optimize*`` modes) --
one enum, one code path, a 3-element boolean mask over ``[dx, dy, dtheta]``.

This is a trajectory-design concern, which is why it lives here: the offsets
are part of the *experiment* being designed, alongside the curve, and both
designers optimize them against the same FIM -- the standalone run
(``pipeline.py``, offsets appended to the control points' decision vector) and
the alternating ``joint_tuning`` loop. The gain tuner is downstream of that: it
either reads the designed offsets off the trajectory pickles or, with no design
to read, calls the sampler here for its own draw.

The *trajectory* side owns them, never the gain side: minimizing the gain loss
over the offsets would drive them to zero and destroy exactly the excitation
they exist to provide, while minimizing the FIM loss maximizes information,
which is the only self-consistent choice.
"""

import jax
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


def sample_initial_pose_offsets(
    key: jax.Array,
    num_realizations: int,
    offset_radius: float,
    offset_angle: float,
) -> jax.Array:
    """Draw one start-pose offset ``[dx, dy, dtheta]`` per noise realization.

    Starting every rollout exactly on the reference leaves only the error the
    plant fails to track (~1 cm), which is why the tuning loss is nearly flat in
    kx and ky: those gains act on tracking error, and there is almost none to
    act on. Placing the robot off the reference start injects the transient that
    makes them observable -- and matches deployment, where the robot is placed
    by hand (31-100 mm and up to 9.5 deg across the exp04/exp05 logs).

    Positions are uniform over the disk of ``offset_radius`` (the sqrt keeps
    them uniform by area rather than clustered at the center); headings are
    uniform over +/-``offset_angle``. Both 0 returns zeros, i.e. the reference
    start, exactly as before.
    """
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
    """Draw independent frozen start-pose bundles for a trajectory batch.

    The result has shape ``(T, R, 3)``.  Each trajectory receives its own
    ``R`` realization starts; every bundle is nevertheless fixed for the
    entire optimization, preserving common random numbers within that
    trajectory's objective.
    """
    trajectory_keys = jax.random.split(key, num_trajectories)
    return jax.vmap(
        lambda trajectory_key: sample_initial_pose_offsets(
            trajectory_key, num_realizations, offset_radius, offset_angle
        )
    )(trajectory_keys)
