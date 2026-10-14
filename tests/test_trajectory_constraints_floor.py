"""Lower bounds on a design's mean motion: minimum speed and minimum lateral acceleration."""

import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.constraints import (
    min_lateral_acceleration_loss,
)


def test_the_turning_floor_prices_the_mean_shortfall_and_nothing_above_it():
    """A straight fast sweep (no lateral acceleration) is fully short of the
    floor; a sustained turn at the floor's mean is not penalized."""
    straight = min_lateral_acceleration_loss(jnp.zeros(50), 2.0)
    turning = min_lateral_acceleration_loss(jnp.full(50, 3.0), 2.0)
    assert float(straight) > 0.9
    assert float(turning) < 1e-3
