import numpy as np
import jax

from wmr_simulator.gain_tuning.objectives import split_realization_keys_by_trajectory


def test_trajectory_keys_are_distinct_and_reproducible():
    realization_keys = jax.random.split(jax.random.PRNGKey(7), 4)

    keys = split_realization_keys_by_trajectory(realization_keys, 6)
    repeated = split_realization_keys_by_trajectory(realization_keys, 6)

    assert keys.shape == (6, 4, 2)
    np.testing.assert_array_equal(keys, repeated)
    assert len({tuple(key) for key in np.asarray(keys).reshape(-1, 2)}) == 24
