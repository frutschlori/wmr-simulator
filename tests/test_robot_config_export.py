import pytest

from wmr_simulator.pololu.robot_config import (
    DEFAULT_ROBOT_CONFIG,
    export_robot_config,
    format_robot_config,
    load_robot_config_file,
    robot_config_values,
)


class FakeParams:
    wheel_radius = 0.017
    base_diameter = 0.09
    max_wheel_speed = 250.0


def test_format_parse_roundtrip(tmp_path):
    output = tmp_path / "ROBOTCFG.CFG"
    output.write_text(format_robot_config(DEFAULT_ROBOT_CONFIG), encoding="utf-8")
    assert load_robot_config_file(output) == pytest.approx(DEFAULT_ROBOT_CONFIG)


def test_robot_id_is_fixed_int_directly_above_joystick_control():
    text = format_robot_config({**DEFAULT_ROBOT_CONFIG, "robot_id": 12.5})
    lines = text.splitlines()

    robot_id_line = lines.index("robot_id=10")
    joystick_line = lines.index("joystick_control_dt_ms=20.0")
    assert robot_id_line + 1 == joystick_line
    assert "robot_id=10.0" not in lines


def test_gain_conversion_divides_inner_gains_by_motor_gain():
    gains = [9.5, 7.5, 6.0, 7.15, 11.9]
    values = robot_config_values(physical_params=FakeParams(), controller_gains=gains)
    assert values["wheel_radius"] == pytest.approx(0.017)
    assert values["wheel_base"] == pytest.approx(0.09)
    assert values["kx_traj"] == pytest.approx(9.5)
    assert values["ky_traj"] == pytest.approx(7.5)
    assert values["ktheta_traj"] == pytest.approx(6.0)
    assert values["kp_inner"] == pytest.approx(7.15 / 250.0)
    assert values["ki_inner"] == pytest.approx(11.9 / 250.0)
    assert values["kd_inner"] == pytest.approx(0.0)
    # Untouched firmware-only keys come from the template.
    assert values["gear_ratio"] == pytest.approx(DEFAULT_ROBOT_CONFIG["gear_ratio"])


def test_wheel_max_is_the_identified_motor_gain():
    # The firmware's feedforward is omega_cmd / wheel_max, so the exported
    # wheel_max has to be the same max_wheel_speed the inner gains were
    # divided by -- a template value left in place makes the feedforward weak
    # by exactly the identification's shortfall.
    class SlowParams(FakeParams):
        max_wheel_speed = 229.83

    values = robot_config_values(physical_params=SlowParams(), controller_gains=[1, 2, 3, 4, 5])
    assert values["wheel_max"] == pytest.approx(229.83)
    assert values["wheel_max"] != pytest.approx(DEFAULT_ROBOT_CONFIG["wheel_max"])
    assert values["kp_inner"] == pytest.approx(4.0 / 229.83)


def test_gains_require_physical_params():
    with pytest.raises(ValueError, match="physical_params"):
        robot_config_values(controller_gains=[1, 2, 3, 4, 5])


def test_export_with_template_and_overrides(tmp_path):
    template = tmp_path / "template.CFG"
    template.write_text(format_robot_config({**DEFAULT_ROBOT_CONFIG, "wheel_max": 300.0}), encoding="utf-8")
    output = export_robot_config(
        tmp_path / "out" / "ROBOTCFG.CFG",
        physical_params=FakeParams(),
        controller_gains=[1, 2, 3, 4, 5],
        template_path=template,
        overrides={"max_speed": 2.0},
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    values = load_robot_config_file(output)
    assert lines[3:5] == ["robot_id=10", "joystick_control_dt_ms=20.0"]
    assert values["robot_id"] == pytest.approx(10.0)
    # wheel_max is derived from physical_params, so it wins over the template.
    assert values["wheel_max"] == pytest.approx(250.0)
    assert values["gear_ratio"] == pytest.approx(DEFAULT_ROBOT_CONFIG["gear_ratio"])
    assert values["max_speed"] == pytest.approx(2.0)
    assert values["kp_inner"] == pytest.approx(4.0 / 250.0)
