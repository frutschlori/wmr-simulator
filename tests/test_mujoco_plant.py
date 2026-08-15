"""The hidden MuJoCo plant: XML patching, sensor units and the pose interface."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wmr_simulator.mujoco_sim import (
    MujocoPlant,
    build_plant_xml,
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
    assert model.dof_armature[model.jnt_dofadr[joint]] == pytest.approx(config.motor.armature)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel")
    assert model.body_pos[body, 1] == pytest.approx(config.geometry.half_track)

    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_geom")
    assert model.geom_size[geom, 0] == pytest.approx(config.geometry.wheel_radius)
    assert model.geom_friction[geom, 0] == pytest.approx(config.friction.wheel)

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert model.geom_friction[floor, 0] == pytest.approx(config.friction.floor)
    # The caster's low friction only survives through the explicit pair, since
    # MuJoCo otherwise takes the elementwise maximum with the floor's.
    assert model.pair_friction[0, 0] == pytest.approx(config.friction.caster)


def test_patching_keeps_the_non_sliding_friction_components(config):
    """Torsional and rolling friction are XML-only and must not be overwritten."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(build_plant_xml(config))
    floor = root.find(".//geom[@name='floor']").get("friction").split()
    assert [float(value) for value in floor] == [config.friction.floor, 0.005, 0.0001]
    pair = [float(value) for value in root.find(".//pair").get("friction").split()]
    assert pair[:2] == [config.friction.caster] * 2
    assert pair[2:] == [0.0001, 0.0001, 0.0001]


def test_reset_places_the_robot_at_rest(plant):
    plant.reset(0.35, -0.2, 0.9)
    pose = plant.pose()
    assert pose == pytest.approx([0.35, -0.2, 0.9], abs=1e-9)
    assert plant.encoder_counts() == (0, 0)
    assert plant.wheel_speeds() == pytest.approx([0.0, 0.0])
    assert plant.time == 0.0


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
    # One count is 2*pi/183 = 0.034 rad, i.e. 3.4 rad/s of quantization at 100 Hz.
    assert firmware == pytest.approx(truth, abs=4.0)
    assert firmware[0] > firmware[1] > 0.0


def test_imu_is_reported_in_hardware_units(plant):
    plant.reset()
    accel, gyro = plant.read_imu()
    # Accelerometer in g, gravity included: at rest it reads +1 g up.
    assert accel[2] == pytest.approx(1.0 + plant.config.imu.accel_bias_g[2], abs=0.05)
    assert abs(gyro[2]) < 5.0
    lsb = plant.config.imu.accel_lsb_g
    assert np.allclose(np.rint(accel / lsb), accel / lsb)

    plant.reset()
    plant.set_duty(-0.25, 0.25)
    plant.step(plant.steps_for(1.0))
    _, gyro = plant.read_imu()
    # Gyro in deg/s, z aligned with the body's yaw axis.
    assert gyro[2] == pytest.approx(math.degrees(plant.twist()[1]), rel=0.05, abs=5.0)


def test_mocap_noise_is_the_configured_size(plant):
    plant.reset(1.0, 0.5, 0.3)
    truth = plant.marker_pose()
    samples = np.array([plant.read_mocap() for _ in range(400)])
    assert samples[:, :3].std(axis=0) == pytest.approx(plant.config.mocap.position_noise_std, rel=0.2)
    assert samples[:, 3:].std(axis=0) == pytest.approx(plant.config.mocap.angle_noise_std, rel=0.2)
    assert samples.mean(axis=0) == pytest.approx(truth, abs=5e-4)


def test_duty_is_clamped_like_the_h_bridge(plant):
    plant.set_duty(2.5, -3.0)
    assert plant.data.ctrl[0] == pytest.approx(plant.config.motor.omega_max)
    assert plant.data.ctrl[1] == pytest.approx(-plant.config.motor.omega_max)


def test_free_wheel_speed_matches_the_hidden_motor_spec(config):
    """omega_max and the time constant are what the yaml says, off the ground."""
    import mujoco

    motor = config.motor
    free = MujocoPlant(config, seed=0)
    # Off the ground and out of gravity, the wheel is a first-order system with
    # exactly the spec's gain and time constant - no contact, no load torque.
    free.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    free.model.opt.gravity[:] = 0.0
    free.reset()
    free.set_duty(1.0, 1.0)
    free.step(free.steps_for(motor.time_constant))
    rising = free.wheel_speeds()[0]
    free.step(free.steps_for(6.0 * motor.time_constant))
    assert free.wheel_speeds()[0] == pytest.approx(motor.omega_max, rel=0.02)
    assert rising == pytest.approx((1.0 - math.exp(-1.0)) * motor.omega_max, rel=0.05)


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


def test_measured_plant_truth_still_matches_the_benchmark_reference(config):
    """The identification benchmark is scored against these; a plant edit invalidates it.

    Bands are wide enough for solver noise and narrow enough that any real
    change to pololu_calibrated.xml or pololu_hidden.yaml trips them.
    """
    from wmr_simulator.mujoco_sim.truth import measure_plant_truth

    truth = measure_plant_truth(MujocoPlant(config, seed=0))
    # Emergent, not declared: contact penetration, not the 16 mm geom size.
    assert truth.wheel_radius == pytest.approx(0.01597, abs=2e-4)
    # Scrub: the geometric wheelbase is 84.2 mm and both regimes sit just under it.
    assert truth.wheel_base_arc == pytest.approx(0.0833, abs=1e-3)
    assert truth.wheel_base_spin == pytest.approx(0.0836, abs=2e-3)
    # Loaded, so a little under the hidden yaml's 230 rad/s free speed.
    assert truth.max_wheel_speed == pytest.approx(229.3, abs=1.0)
    # The logged signal is behind the firmware's 3 Hz low-pass, which is what
    # identification is fitted to - score it against this one, not the other.
    assert truth.time_constant_mechanical == pytest.approx(0.110, abs=0.02)
    assert truth.time_constant_logged == pytest.approx(0.180, abs=0.02)
    assert truth.time_constant_logged > truth.time_constant_mechanical
