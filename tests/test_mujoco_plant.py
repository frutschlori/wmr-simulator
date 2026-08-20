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
    # The caster's and the nose skid's frictions only survive through the
    # explicit pairs, since MuJoCo otherwise takes the elementwise maximum with
    # the floor's. Look them up by name: MuJoCo does not keep pairs in
    # declaration order.
    for name, expected in (("floor_caster", config.friction.caster), ("floor_nose", config.friction.nose)):
        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name)
        assert model.pair_friction[pair, 0] == pytest.approx(expected)


def test_patching_keeps_the_non_sliding_friction_components(config):
    """Torsional and rolling friction are XML-only and must not be overwritten."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(build_plant_xml(config))
    floor = root.find(".//geom[@name='floor']").get("friction").split()
    assert [float(value) for value in floor] == [config.friction.floor, 0.005, 0.0001]
    pair = [float(value) for value in root.find(".//pair[@name='floor_caster']").get("friction").split()]
    assert pair[:2] == [config.friction.caster] * 2
    # Torsional then the two rolling terms. The rolling terms must stay small:
    # they are what let the ball caster roll instead of skid.
    assert pair[2:] == [0.001, 0.0001, 0.0001]


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
    # One count is 2*pi/183 = 0.034 rad, i.e. 3.4 rad/s at 100 Hz - and the count
    # is rounded at BOTH ends of the window, so the bound is two counts, not one.
    assert firmware == pytest.approx(truth, abs=7.0)
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


def test_mass_and_com_match_the_robot_that_was_weighed(config):
    """The mass split in the XML is a decomposition of one measurement, not four
    independent guesses, so it has to add back up.

    Measured on the robot 2026-08-19: 171 g with batteries, CoM 2-3 mm behind
    the drive axle. The height is NOT measured - it is inferred from the
    batteries sitting in the underside tub - so it is pinned loosely here to
    catch a geom move, not asserted as truth.
    """
    import mujoco

    # A fresh, un-driven copy: `subtree_com` is a world vector, so subtracting
    # the body origin only gives the body-frame CoM at the identity orientation.
    model = mujoco.MjModel.from_xml_string(build_plant_xml(config))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    assert model.body_subtreemass[chassis] == pytest.approx(0.171, abs=1e-4)
    com = data.subtree_com[chassis] - data.xpos[chassis]
    assert com[0] == pytest.approx(-0.0025, abs=2e-4)
    assert com[1] == pytest.approx(0.0, abs=1e-9)
    # Body frame sits at the wheel axle, so the floor is one wheel radius down.
    assert com[2] + 0.016 == pytest.approx(0.015, abs=3e-4)


def test_the_robot_rests_on_its_caster_very_slightly_nose_up(config):
    """The resting attitude is contact geometry, not a setting, and it is the
    check that the drawing was read correctly.

    Observed on the robot: it rests on the caster ball, very nearly level, nose
    a shade up, and the nose has ~5 mm of travel before it reaches the floor.
    Every 0.1 mm of caster drop is 0.14 deg of pitch on the 40.4 mm lever, so
    this is a tight assertion on the geometry.
    """
    import mujoco

    plant = MujocoPlant(config, seed=0)
    plant.reset()
    w, x, y, z = plant.data.qpos[3:7]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
    assert -1.2 < pitch < -0.2, "should rest nose-up by well under a degree"

    caster = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "caster_geom")
    nose = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, "nose_skid")
    ball_bottom = plant.data.geom_xpos[caster][2] - plant.model.geom_size[caster, 0]
    assert abs(ball_bottom) < 5e-4, "the caster carries the rear, so it is on the floor"
    nose_clearance = plant.data.geom_xpos[nose][2] - plant.model.geom_size[nose, 0]
    assert nose_clearance == pytest.approx(0.0054, abs=6e-4)

    # And the caster is loaded, which is the whole reason the CoM is behind the
    # axle: static share is the CoM offset over the caster's setback.
    weight = plant.model.body_subtreemass[
        mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    ] * 9.81
    on_caster = 0.0
    for index in range(plant.data.ncon):
        contact = plant.data.contact[index]
        if caster in (contact.geom1, contact.geom2):
            force = np.zeros(6)
            mujoco.mj_contactForce(plant.model, plant.data, index, force)
            on_caster += abs(force[0])
    assert on_caster / weight == pytest.approx(0.0025 / 0.04037, rel=0.2)


def test_measured_plant_truth_still_matches_the_benchmark_reference(config):
    """The identification benchmark is scored against these; a plant edit invalidates it.

    Bands are wide enough for solver noise and narrow enough that any real
    change to pololu_calibrated.xml or pololu_hidden.yaml trips them.
    """
    from wmr_simulator.mujoco_sim.truth import measure_plant_truth

    truth = measure_plant_truth(MujocoPlant(config, seed=0))
    # Re-pinned 2026-08-20 for the rebuilt model: drawing geometry, the measured
    # 171 g, material-value frictions and a near-rigid floor.
    #
    # The rolling radius is now essentially the geometric one, and that is the
    # honest answer rather than a lost feature: a 171 g robot does not sink into
    # rigid PVC, and the contacts were stiffened until the resting penetration
    # was 0.002 mm. Scrub still makes the wheelbase emergent, and that is where
    # the identification benchmark's difficulty now lives.
    assert truth.wheel_radius == pytest.approx(0.016, abs=5e-5)
    # Scrub against a geometric 84.2 mm, measured in two regimes. They are noisy
    # at the 0.5 mm level because both are open-loop and the plant is free to
    # drift in yaw, so these are loose pins on an emergent quantity, not tight
    # assertions - the point is to catch a plant edit, not to certify a value.
    assert truth.wheel_base_arc == pytest.approx(0.0842, abs=2e-3)
    assert truth.wheel_base_spin == pytest.approx(0.0842, abs=3e-3)
    # Barely loaded: the rolling ball costs almost no drag, where the old rigid
    # skid dragged the steady speed well below the yaml's 230 rad/s.
    assert truth.max_wheel_speed == pytest.approx(229.9, abs=1.5)
    # BOTH of these are regression pins ONLY - do not read them as the plant's
    # motor lag and do not tune anything against them. They measure a free-run
    # step from rest, which is dominated by vehicle inertia reflected through the
    # wheel and is very slip-sensitive; the PT1 constant the identification
    # pipeline fits is a lumped stand-in for motor dynamics plus loading over
    # real driving. Measured like-for-like - the pipeline run on four closed-loop
    # plant logs - the plant identifies at 0.19 s, inside the 0.12-0.25 s band
    # real robot logs give. These numbers say nothing about that.
    assert truth.time_constant_mechanical == pytest.approx(0.08, abs=0.05)
    assert truth.time_constant_logged == pytest.approx(0.26, abs=0.04)
    assert truth.time_constant_logged > truth.time_constant_mechanical
