from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from wmr_simulator.identification.optimizers import optimize_physical_params_adam
from wmr_simulator.types import PhysicalParams, physical_params_to_array

INIT = PhysicalParams(
    wheel_radius=jnp.asarray(0.020),
    base_diameter=jnp.asarray(0.100),
    max_wheel_speed=jnp.asarray(200.0),
    time_constant=jnp.asarray(0.200),
    a_slip_max=jnp.asarray(5.0),
)
TARGET = PhysicalParams(
    wheel_radius=jnp.asarray(0.017),
    base_diameter=jnp.asarray(0.090),
    max_wheel_speed=jnp.asarray(240.0),
    time_constant=jnp.asarray(0.160),
    a_slip_max=jnp.asarray(2.0),
)


class _StubPipeline:
    """Replay pipeline stand-in whose losses are a quadratic bowl in the five
    physical parameters, so the optimizer's parameter handling can be tested
    without a log or a rollout."""

    use_inverse_variance_pose_loss = False
    window_length = None
    replay_segment_plan = None

    def __init__(self):
        self.initial_params = INIT
        self.hidden_params = TARGET
        self.target_log = SimpleNamespace(pose=SimpleNamespace(states=jnp.zeros((3, 3))))

    def _relative_error(self, params):
        return physical_params_to_array(params) / physical_params_to_array(TARGET) - 1.0

    def replay_rollout(self, params, target_log=None, window_length=None, replay_segment_plan=None):
        error = self._relative_error(params)
        # Two predicted poses against an all-zero target: every parameter error
        # reaches the pose loss, a_slip_max included.
        states = jnp.stack([error[:3], jnp.array([error[3], error[4], 0.0])])
        return SimpleNamespace(pose=SimpleNamespace(states=states))

    def motor_wheel_speed_mse(self, params, target_log, window_length=None):
        return jnp.sum(self._relative_error(params) ** 2)


def _optimize(identify_a_slip_max):
    params, *_ = optimize_physical_params_adam(
        pipelines=[_StubPipeline()],
        init_params=INIT,
        num_steps=200,
        learning_rate=0.05,
        identify_a_slip_max=identify_a_slip_max,
    )
    return np.asarray(physical_params_to_array(params), dtype=float)


def test_a_slip_max_is_identified_by_default():
    values = _optimize(identify_a_slip_max=True)

    assert values[4] < 4.0  # moved off the 5.0 init toward the 2.0 target


def test_a_slip_max_is_held_when_identification_is_disabled():
    values = _optimize(identify_a_slip_max=False)

    assert values[4] == float(INIT.a_slip_max)
    # The other four are unaffected by the frozen dimension.
    init = np.asarray(physical_params_to_array(INIT), dtype=float)
    target = np.asarray(physical_params_to_array(TARGET), dtype=float)
    assert np.all(np.abs(values[:4] - target[:4]) < np.abs(init[:4] - target[:4]))
