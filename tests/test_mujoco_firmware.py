"""The firmware port: config, clock, inner loop, odometry, EKF and the outer law."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from wmr_simulator.mujoco_sim.firmware import (
    DUTY_LIMIT,
    INTEGRAL_LIMIT,
    INNER_PHASE_S,
    Ekf,
    Firmware,
    FirmwareClock,
    FirmwareConfig,
    InnerLoop,
    TrajectoryFollower,
    wrap_angle,
)

STOCK_GAINS = [4.5, 6.0, 12.0, 2.5, 5.0]
TWO_PI = 2.0 * math.pi


def stock_values(gains=None):
    from wmr_simulator.pololu.robot_config import robot_config_values

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0)
    return robot_config_values(physical_params=physical, controller_gains=gains or STOCK_GAINS)


@pytest.fixture(scope="module")
def config():
    return FirmwareConfig.from_mapping(stock_values())


# ------------------------------------------------------------------- config


def test_config_reads_a_robotcfg_file(tmp_path, config):
    from wmr_simulator.pololu.robot_config import export_robot_config

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0)
    path = export_robot_config(tmp_path / "ROBOTCFG.CFG", physical_params=physical, controller_gains=STOCK_GAINS)
    loaded = FirmwareConfig.from_file(path)
    # The CFG is written rounded to 9 decimals, and the rounded values are the
    # only ones the robot ever sees - so this path is the faithful one.
    for field in FirmwareConfig.__dataclass_fields__:
        assert getattr(loaded, field) == pytest.approx(getattr(config, field), abs=5e-10)


def test_config_reads_a_simulator_robot_config_yaml(tmp_path, config):
    """The inner gains are converted to duty/(rad/s), like the exporter does."""
    import yaml

    path = tmp_path / "robot_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {"wheel_radius": 0.016, "base_diameter": 0.0825, "max_wheel_speed": 223.0},
                "controller": {"gains": STOCK_GAINS},
            }
        ),
        encoding="utf-8",
    )
    loaded = FirmwareConfig.from_file(path)
    assert loaded == config
    assert loaded.kp_inner == pytest.approx(2.5 / 223.0)
    assert loaded.ki_inner == pytest.approx(5.0 / 223.0)


def write_scheduled_robot_config(directory):
    import yaml

    path = directory / "robot_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "robot": {"wheel_radius": 0.016, "base_diameter": 0.0825, "max_wheel_speed": 223.0},
                "controller": {"gains": STOCK_GAINS, "gain_parametrization": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    return path


def write_gain_mlp(directory, seed=1, scheduled=(0, 1, 2, 3, 4), theta_scale=0.5):
    """A GAINMLP.JSN with non-identity factors, exported the way the robot gets one."""
    import numpy as np

    from wmr_simulator.gain_parametrization import error_mlp
    from wmr_simulator.pololu.gain_mlp_exporter import export_gain_mlp

    params = error_mlp.from_cfg(
        {"hidden_sizes": [8], "seed": seed, "bound": 3.0, "scheduled_indices": list(scheduled)},
        feature_scale=[2.5, 12.0],
    )
    rng = np.random.default_rng(seed)
    theta = theta_scale * rng.normal(size=error_mlp.num_params(params))
    return export_gain_mlp(directory / "GAINMLP.JSN", error_mlp.with_flat_params(theta, params))


def test_a_scheduled_gain_config_without_its_network_is_refused(tmp_path):
    """On the robot that combination silently runs static gains; reproducing
    that silently here would hide a real difference."""
    with pytest.raises(ValueError, match="GAINMLP.JSN"):
        FirmwareConfig.from_file(write_scheduled_robot_config(tmp_path))


def test_a_gain_mlp_next_to_the_config_is_picked_up(tmp_path):
    """``sdlog.rs`` loads it off the card; here it is read from the same dir."""
    write_gain_mlp(tmp_path)
    scheduled = FirmwareConfig.from_file(write_scheduled_robot_config(tmp_path))
    assert scheduled.gain_mlp is not None
    assert scheduled.gain_mlp["kind"] == "error_mlp"

    from wmr_simulator.pololu.robot_config import export_robot_config

    physical = SimpleNamespace(wheel_radius=0.016, base_diameter=0.0825, max_wheel_speed=223.0)
    cfg = export_robot_config(
        tmp_path / "ROBOTCFG.CFG", physical_params=physical, controller_gains=STOCK_GAINS
    )
    assert FirmwareConfig.from_file(cfg).gain_mlp is not None
    # Without one next to it, the same config is static.
    (tmp_path / "GAINMLP.JSN").unlink()
    assert FirmwareConfig.from_file(cfg).gain_mlp is None


# -------------------------------------------------------------------- clock


def test_clock_rates_match_the_firmware_tasks():
    clock = FirmwareClock(timestep=0.002, outer_period_s=0.05, boot_time_ms=15_000)
    ticks = list(clock.ticks(5.0))
    assert len(ticks) == 2500
    # 499, not 500: the half-period shift pushes the last inner tick past 5 s.
    assert sum(tick.inner for tick in ticks) == 499
    assert sum(tick.mocap for tick in ticks) == 500
    assert sum(tick.imu for tick in ticks) == 500
    assert sum(tick.outer for tick in ticks) == 100

    # Every task is an embassy Ticker: the first fire is one full period in,
    # and the inner loop is deliberately shifted half a period so it never
    # lands on the same instant as the outer one.
    # The 5 ms phase does not land on the 2 ms physics grid, so the tick snaps
    # to the nearest step.
    assert next(tick.time for tick in ticks if tick.inner) == pytest.approx(0.010 + INNER_PHASE_S, abs=0.001)
    assert next(tick.time for tick in ticks if tick.outer) == pytest.approx(0.050)
    assert not any(tick.inner and tick.outer for tick in ticks)
    assert ticks[0].t_ms == 15_002
    assert ticks[-1].t_ms == 20_000


# --------------------------------------------------------------- inner loop


def test_inner_loop_recovers_the_wheel_speed_from_counts(config):
    """counts -> rad/s at the config's CPR, then the 3 Hz low-pass."""
    inner = InnerLoop(config)
    inner.reset((0, 0))
    speed = 60.0  # rad/s on both wheels
    counts_per_tick = speed * inner.dt * config.encoder_cpr / TWO_PI
    for tick in range(200):  # 2 s, plenty for a 3 Hz low-pass
        inner.update((round(counts_per_tick * (tick + 1)), round(counts_per_tick * (tick + 1))))
    assert inner.omega_lp == pytest.approx([speed, speed], rel=0.02)


def test_inner_loop_low_pass_matches_the_firmware_alpha(config):
    inner = InnerLoop(config)
    assert inner.alpha == pytest.approx(0.1586, abs=1e-4)


def test_inner_loop_pi_saturates_and_clamps(config):
    inner = InnerLoop(config)
    inner.reset((0, 0))
    inner.set_command(200.0, -200.0)
    for _ in range(400):
        duty = inner.update((0, 0))  # wheels blocked: the error never goes away
    assert duty == pytest.approx([DUTY_LIMIT, -DUTY_LIMIT])
    assert np.abs(inner.integral).max() == pytest.approx(INTEGRAL_LIMIT)


def test_inner_loop_feedforward_divides_by_the_configs_wheel_max():
    """With the feedback zeroed the duty is pure feedforward, at `wheel_max` gain."""
    open_loop = FirmwareConfig.from_mapping({**stock_values(), "kp_inner": 0.0, "ki_inner": 0.0, "kd_inner": 0.0})
    inner = InnerLoop(open_loop)
    inner.reset((0, 0))
    speed = 100.0
    inner.set_command(speed, speed)
    duty = inner.update((0, 0))
    assert duty == pytest.approx([speed / open_loop.wheel_max] * 2)
    # The exporter writes the identified motor gain into wheel_max, so on an
    # exported config the feedforward is exact rather than ~12% weak.
    assert duty[0] == pytest.approx(speed / 223.0)


def test_a_hand_written_wheel_max_still_wins_over_the_identified_gain():
    """The firmware believes the card, so a stale CFG makes the feedforward weak."""
    stale = FirmwareConfig.from_mapping({**stock_values(), "wheel_max": 250.0, "kp_inner": 0.0, "ki_inner": 0.0})
    inner = InnerLoop(stale)
    inner.reset((0, 0))
    inner.set_command(100.0, 100.0)
    duty = inner.update((0, 0))
    assert duty[0] == pytest.approx(100.0 / 250.0)
    assert duty[0] < 100.0 / 223.0


def test_inner_loop_publishes_the_twist_from_the_raw_speeds(config):
    """`odometry.rs`: the EKF predicts on the unfiltered wheel speeds.

    Building it from `omega_lp` instead puts the 53 ms encoder lag into the
    pose the outer tracking law closes around, and disagrees with the JAX
    `estimator.py`. On the first tick out of reset the low-pass has only
    reached `alpha` of the step, so the two differ by ~6x here.
    """
    inner = InnerLoop(config)
    inner.reset((0, 0))
    counts = np.array([-12, -17])
    inner.update(counts)
    omega_raw = 2.0 * math.pi * counts / (config.encoder_cpr * inner.dt)
    left, right = omega_raw
    assert inner.twist[0] == pytest.approx(config.wheel_radius * (right + left) / 2.0)
    assert inner.twist[1] == pytest.approx(config.wheel_radius * (right - left) / config.wheel_base)

    lp_left, lp_right = inner.omega_lp
    assert inner.twist[0] != pytest.approx(config.wheel_radius * (lp_right + lp_left) / 2.0)


def test_the_twist_and_the_logged_speeds_converge_in_steady_state(config):
    """The raw/filtered split is a lag, not an offset: it vanishes at constant speed."""
    inner = InnerLoop(config)
    inner.reset((0, 0))
    step = np.array([-12, -17])
    counts = np.zeros(2, dtype=np.int64)
    for _ in range(400):
        counts = counts + step
        inner.update(counts)
    left, right = inner.omega_lp
    assert inner.twist[0] == pytest.approx(config.wheel_radius * (right + left) / 2.0, rel=1e-6)


def test_inner_loop_applies_the_motor_direction(config):
    flipped = FirmwareConfig.from_mapping({**stock_values(), "motor_direction_left": -1.0})
    inner = InnerLoop(flipped)
    inner.reset((0, 0))
    inner.set_command(50.0, 50.0)
    duty = inner.update((0, 0))
    assert duty[0] < 0.0 < duty[1]


# --------------------------------------------------------------------- EKF


def test_ekf_predicts_along_the_heading_and_wraps():
    ekf = Ekf((0.0, 0.0, math.pi - 0.05))
    ekf.predict(1.0, 1.0, 0.1)
    assert ekf.state[2] == pytest.approx(wrap_angle(math.pi + 0.05))
    assert ekf.state[2] < 0.0  # wrapped, not 3.19


def test_ekf_update_pulls_toward_the_measurement():
    ekf = Ekf((0.0, 0.0, 0.0))
    for _ in range(20):
        ekf.predict(0.0, 0.0, 0.05)
        ekf.update((1.0, -0.5, 0.3))
    assert ekf.state == pytest.approx([1.0, -0.5, 0.3], abs=1e-3)


def test_ekf_trusts_the_mocap_more_than_the_odometry():
    """R is two orders below Q in ekf.rs, so one update dominates one prediction."""
    ekf = Ekf((0.0, 0.0, 0.0))
    ekf.predict(1.0, 0.0, 0.05)
    predicted = ekf.state[0]
    ekf.update((0.0, 0.0, 0.0))
    assert abs(ekf.state[0]) < 0.2 * predicted


# ------------------------------------------------------------- outer loop


def trivial_trajectory(dt=0.05, num=6):
    states = np.column_stack([np.arange(num) * 0.1, np.zeros(num), np.zeros(num)])
    actions = np.column_stack([np.full(num - 1, 2.0), np.zeros(num - 1)])
    return states, actions


def test_setpoint_lookup_uses_the_next_state_and_the_current_action(config):
    states, actions = trivial_trajectory()
    follower = TrajectoryFollower(config, states, actions)
    assert follower.duration == pytest.approx(states.shape[0] * config.traj_following_dt_s)
    assert follower.setpoint(0.0).x_des == pytest.approx(states[1, 0])
    assert follower.setpoint(0.12).x_des == pytest.approx(states[3, 0])  # floor(0.12/0.05) = 2
    # Past the end the index clamps instead of raising.
    assert follower.setpoint(99.0).x_des == pytest.approx(states[-1, 0])


def test_kanayama_law_matches_the_firmware_expression(config):
    states, actions = trivial_trajectory()
    follower = TrajectoryFollower(config, states, actions)
    setpoint = follower.setpoint(0.0)
    pose = (0.05, -0.02, 0.1)
    outputs = follower.control(pose, setpoint)

    dx, dy = setpoint.x_des - pose[0], setpoint.y_des - pose[1]
    x_err = math.cos(pose[2]) * dx + math.sin(pose[2]) * dy
    y_err = -math.sin(pose[2]) * dx + math.cos(pose[2]) * dy
    yaw_err = wrap_angle(setpoint.yaw_des - pose[2])
    v = setpoint.v_ff * math.cos(yaw_err) + config.kx_traj * x_err
    w = setpoint.w_ff + setpoint.v_ff * (config.ky_traj * y_err + config.ktheta_traj * math.sin(yaw_err))
    assert (outputs.x_err, outputs.y_err, outputs.yaw_err) == pytest.approx((x_err, y_err, yaw_err))
    assert outputs.omega_right == pytest.approx((2 * v + config.wheel_base * w) / (2 * config.wheel_radius))
    assert outputs.omega_left == pytest.approx((2 * v - config.wheel_base * w) / (2 * config.wheel_radius))


def test_the_wheel_command_is_not_clipped(config):
    """Saturation happens at the duty clamp only - the firmware clips nothing here."""
    states, actions = trivial_trajectory()
    follower = TrajectoryFollower(config, states, actions)
    outputs = follower.control((0.0, -5.0, 0.0), follower.setpoint(0.0))
    # Past the config's own wheel_max, which the firmware carries on its
    # DiffdriveCascade and then never applies on this path.
    assert abs(outputs.omega_right) > config.wheel_max


# ----------------------------------------------------------------- gain MLP


def test_the_outer_loop_scales_its_gains_by_the_networks_factors(tmp_path):
    """``trajectory_control.rs`` overwrites controller.k* in place every tick,
    so the Kanayama law must run *on* the scheduled gains, not be corrected
    afterwards."""
    import json

    from wmr_simulator.pololu.gain_mlp_exporter import reference_forward

    network = write_gain_mlp(tmp_path)
    config = FirmwareConfig.from_mapping(stock_values(), gain_mlp=json.loads(network.read_text()))
    states, actions = trivial_trajectory()
    firmware = Firmware(config, states, actions, (0.05, -0.02, 0.1))

    outputs = firmware.outer_tick(0.0)
    setpoint = outputs.setpoint
    factors = reference_forward(
        config.gain_mlp,
        ref=[setpoint.x_des, setpoint.y_des, setpoint.yaw_des, setpoint.v_ff, setpoint.w_ff],
        pose=list(firmware.ekf.state),
        twist=[0.0, 0.0],
    )
    assert np.any(np.abs(factors - 1.0) > 1e-3)  # a network that does nothing proves nothing

    expected = TrajectoryFollower(config, states, actions).control(
        firmware.ekf.state, setpoint, config.base_gains[:3] * factors[:3]
    )
    assert outputs.omega_left == pytest.approx(expected.omega_left)
    assert outputs.omega_right == pytest.approx(expected.omega_right)
    # ... and it is not what the static gains would have produced.
    static = TrajectoryFollower(config, states, actions).control(firmware.ekf.state, setpoint)
    assert abs(outputs.omega_right - static.omega_right) > 1e-6


def test_the_inner_motor_gains_follow_the_published_factors(tmp_path):
    """``gain_mlp_store``: the outer loop publishes factors[3:5], the inner loop
    applies them at 100 Hz until the next outer tick replaces them."""
    import json

    network = write_gain_mlp(tmp_path)
    config = FirmwareConfig.from_mapping(stock_values(), gain_mlp=json.loads(network.read_text()))
    states, actions = trivial_trajectory()
    firmware = Firmware(config, states, actions, (0.05, -0.02, 0.1))

    # 1.0 until the first outer tick - reset_inner_gain_scales.
    assert firmware.inner.gain_scales == (1.0, 1.0)
    firmware.outer_tick(0.0)
    assert firmware.inner.gain_scales == pytest.approx(tuple(firmware.gain_factors[3:5]))
    assert firmware.inner.gain_scales != (1.0, 1.0)

    # The duty really is computed with the scaled kp: same tick, scale off vs on.
    scaled = InnerLoop(config)
    scaled.reset((0, 0))
    scaled.set_command(40.0, 40.0)
    scaled.set_gain_scales(*firmware.inner.gain_scales)
    static = InnerLoop(config)
    static.reset((0, 0))
    static.set_command(40.0, 40.0)
    assert not np.allclose(scaled.update((0, 0)), static.update((0, 0)))


def test_a_static_card_leaves_every_gain_untouched(config):
    """No GAINMLP.JSN means factors of exactly 1.0, bit-for-bit the old path."""
    states, actions = trivial_trajectory()
    firmware = Firmware(config, states, actions, (0.05, -0.02, 0.1))
    outputs = firmware.outer_tick(0.0)

    assert config.gain_mlp is None
    assert firmware.gain_factors == pytest.approx(np.ones(5))
    assert firmware.inner.gain_scales == (1.0, 1.0)
    expected = TrajectoryFollower(config, states, actions).control(firmware.ekf.state, outputs.setpoint)
    assert outputs.omega_right == pytest.approx(expected.omega_right)


# ------------------------------------------------------- closed loop on the plant


def test_closed_loop_tracks_a_slow_baseline_on_the_plant():
    """The whole stack against the hidden plant, on a trajectory it can execute."""
    from wmr_simulator.mujoco_sim import MujocoPlant
    from wmr_simulator.pololu.reference_importer import load_pololu_reference

    # The tuned static gains, not the stock ones: stock overdrives the yaw rate
    # past what this plant can hold (see finding B in the deployment plan).
    config = FirmwareConfig.from_mapping(stock_values([2.0, 6.7, 7.4, 3.2, 0.0]))
    reference = load_pololu_reference("trajectory_exports/baselines/circle_slow.JSN")
    plant = MujocoPlant(seed=0)
    plant.reset(*reference.states[0])
    mocap = plant.read_mocap()
    firmware = Firmware(config, reference.states, reference.actions, (mocap[0], mocap[1], mocap[5]))
    firmware.reset_encoders(plant.encoder_counts())

    errors = []
    for tick in firmware.clock(plant.timestep).ticks(firmware.duration):
        if tick.mocap:
            pose = plant.read_mocap()
            firmware.receive_mocap((pose[0], pose[1], pose[5]))
        if tick.inner:
            plant.set_duty(*firmware.inner_tick(plant.encoder_counts()))
        if tick.outer:
            outputs = firmware.outer_tick(tick.time)
            true_pose = plant.pose()
            errors.append(math.hypot(outputs.setpoint.x_des - true_pose[0], outputs.setpoint.y_des - true_pose[1]))
        plant.step()

    errors = np.array(errors)
    # Real logs of this robot sit at 0.05-0.09 m tracking RMSE.
    assert math.sqrt((errors**2).mean()) < 0.05
    # It went round the circle rather than sitting still.
    assert np.linalg.norm(plant.pose()[:2] - reference.states[-1, :2]) < 0.15
