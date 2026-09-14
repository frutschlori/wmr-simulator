"""Lower bounds on a design's mean motion: minimum speed and minimum lateral acceleration."""

import jax.numpy as jnp
import numpy as np

from wmr_simulator.trajectory_optimization.constraints import (
    min_lateral_acceleration_loss,
    motion_floors_for_batch,
)


def test_a_disabled_turning_floor_costs_exactly_nothing():
    a_lat = jnp.asarray([0.0, 0.1, 0.2], dtype=jnp.float32)
    assert float(min_lateral_acceleration_loss(a_lat, 0.0)) == 0.0


def test_the_turning_floor_prices_the_mean_shortfall_and_nothing_above_it():
    """A straight fast sweep (no lateral acceleration) is fully short of the
    floor; a sustained turn at the floor's mean is not penalized."""
    straight = min_lateral_acceleration_loss(jnp.zeros(50), 2.0)
    turning = min_lateral_acceleration_loss(jnp.full(50, 3.0), 2.0)
    assert float(straight) > 0.9
    assert float(turning) < 1e-3


def test_turning_floors_go_to_the_fastest_trajectories():
    """The turning floors end where the speed floors end, so every hard-turning
    design is also a fast one and the hardest turn is the fastest trajectory."""
    floors = motion_floors_for_batch(8, 1.1, 0.5, 2.0, 0.25)
    assert floors.shape == (8, 2)
    assert np.count_nonzero(floors[:, 0]) == 4
    assert np.count_nonzero(floors[:, 1]) == 2
    assert np.all(floors[floors[:, 1] > 0.0, 0] > 0.0)
    np.testing.assert_allclose(floors[2:4, 1], [1.0, 2.0])
    assert np.argmax(floors[:, 1]) == np.argmax(floors[:, 0])
