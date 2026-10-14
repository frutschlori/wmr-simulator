"""The hidden MuJoCo plant: XML patching, sensor units and the pose interface."""

from __future__ import annotations

import math

import pytest

from wmr_simulator.mujoco_sim import (
    MujocoPlant,
    load_hidden_plant_config,
)

TWO_PI = 2.0 * math.pi


@pytest.fixture(scope="module")
def config():
    return load_hidden_plant_config()


@pytest.fixture(scope="module")
def plant(config):
    return MujocoPlant(config, seed=0)


def test_hidden_config_values_reach_the_compiled_model(config, plant):
    """The yaml is authoritative: every value it carries is patched into the XML."""
    import mujoco

    model = plant.model
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
    assert model.actuator_gainprm[left, 0] == pytest.approx(config.motor.kv)
    assert model.actuator_ctrlrange[left, 1] == pytest.approx(config.motor.ctrl_limit)

    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
    # `armature` is EXTRA inertia, so what has to match the spec is the TOTAL
    # about the hinge: the patched armature plus the wheel geom's own share.
    from wmr_simulator.mujoco_sim.plant import _wheel_spin_inertia

    total = model.dof_armature[model.jnt_dofadr[joint]] + _wheel_spin_inertia(config.geometry)
    assert total == pytest.approx(config.motor.armature)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    assert model.body_pos[body, 1] == pytest.approx(config.geometry.half_track)

    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    assert model.geom_size[geom, 0] == pytest.approx(config.geometry.wheel_radius)
    assert model.geom_friction[geom, 0] == pytest.approx(config.friction.wheel)

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert model.geom_friction[floor, 0] == pytest.approx(config.friction.floor)
    # Caster and skid friction live on explicit pairs (looked up by name).
    for name, expected in (("floor_caster", config.friction.caster), ("floor_nose", config.friction.nose)):
        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        assert model.pair_friction[pair, 0] == pytest.approx(expected)


def test_encoder_counts_invert_to_the_true_wheel_speed(plant):
    """The firmware's count -> rad/s conversion, run against the plant's counts."""
    cpr = plant.config.encoder.counts_per_revolution
    dt = 0.01
    plant.reset()
    plant.set_duty(0.4, 0.2)
    plant.step(plant.steps_for(1.0))  # past the motor transient

    before = plant.encoder_counts()
    truth = plant.wheel_speeds()
    plant.step(plant.steps_for(dt))
    after = plant.encoder_counts()

    firmware = [TWO_PI * (after[i] - before[i]) / (cpr * dt) for i in range(2)]
    # One count is 2*pi/183 = 0.034 rad, i.e. 3.4 rad/s at 100 Hz - and the count
    # is rounded at BOTH ends of the window, so the bound is two counts, not one.
    assert firmware == pytest.approx(truth, abs=7.0)
    assert firmware[0] > firmware[1] > 0.0


def test_duty_is_clamped_like_the_h_bridge(plant):
    plant.set_duty(2.5, -3.0)
    assert plant.data.ctrl[0] == pytest.approx(plant.config.motor.omega_max)
    assert plant.data.ctrl[1] == pytest.approx(-plant.config.motor.omega_max)


def test_unknown_config_keys_are_rejected():
    """A typo in the hidden yaml must not silently leave a parameter at its XML value."""
    import copy

    import yaml

    from wmr_simulator.mujoco_sim.plant import DEFAULT_HIDDEN_CONFIG_PATH, HiddenPlantConfig

    with open(DEFAULT_HIDDEN_CONFIG_PATH, "r", encoding="utf-8") as file:
        values = yaml.safe_load(file)

    top_level = copy.deepcopy(values)
    top_level["wheel_radius"] = 0.016  # right value, wrong nesting
    with pytest.raises(ValueError, match="wheel_radius"):
        HiddenPlantConfig.from_mapping(top_level)

    nested = copy.deepcopy(values)
    nested["motor"]["omega_maxx"] = 240.0
    with pytest.raises(ValueError, match="omega_maxx"):
        HiddenPlantConfig.from_mapping(nested)

    incomplete = copy.deepcopy(values)
    del incomplete["friction"]["caster"]
    with pytest.raises(ValueError, match="caster"):
        HiddenPlantConfig.from_mapping(incomplete)
