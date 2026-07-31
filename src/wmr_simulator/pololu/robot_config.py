"""Read and write Pololu firmware robot configuration files (ROBOTCFG.CFG).

The firmware consumes a flat ``key=value`` text file (see e.g.
"Pololu Data/Experiments/2026_06_22/Logs/gain tuning/tuned_3/ROBOTCFG.CFG").
This module renders that format from simulator quantities:

- ``wheel_radius`` / ``wheel_base`` come from PhysicalParams (base_diameter is
  the effective wheelbase).
- ``kx_traj`` / ``ky_traj`` / ``ktheta_traj`` are the outer tracking gains,
  identical in both conventions.
- ``kp_inner`` / ``ki_inner`` are the inner wheel-speed gains. The simulator
  keeps them in wheel-speed units (dimensionless feedback on rad/s errors)
  while the firmware expects duty / (rad/s), so the simulator gains are divided
  by the motor gain (max_wheel_speed) on export. ``kd_inner`` is a firmware-only
  slot the simulator no longer models; it is forced to 0 on export.

Keys that have no simulator counterpart (joystick timing, motor directions,
encoder constants, ...) are taken from a template: either an existing CFG file
or the built-in defaults below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

HEADER_COMMENT = "# Robot configuration (key=value)\n# Polulu Configuration File\n"
ROBOT_ID = 10

# Key groups in file order; groups are separated by blank lines like the
# firmware examples.
KEY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("robot_id", "joystick_control_dt_ms", "traj_following_dt_s"),
    ("wheel_radius", "wheel_base"),
    ("motor_direction_left", "motor_direction_right"),
    ("motor_max_duty_left", "motor_max_duty_right"),
    ("k_clip",),
    ("kp_inner", "ki_inner", "kd_inner"),
    ("kx_traj", "ky_traj", "ktheta_traj"),
    ("gear_ratio", "encoder_cpr", "max_speed", "max_omega", "wheel_max"),
)

DEFAULT_ROBOT_CONFIG: dict[str, float] = {
    "robot_id": ROBOT_ID,
    "joystick_control_dt_ms": 20.0,
    "traj_following_dt_s": 0.05,
    "wheel_radius": 0.01637,
    "wheel_base": 0.08525,
    "motor_direction_left": 1.0,
    "motor_direction_right": 1.0,
    "motor_max_duty_left": 0.8,
    "motor_max_duty_right": 0.8,
    "k_clip": 1.0,
    "kp_inner": 0.0288,
    "ki_inner": 0.048,
    "kd_inner": 0.0,
    "kx_traj": 9.68,
    "ky_traj": 7.85,
    "ktheta_traj": 6.12,
    "gear_ratio": 15.25,
    "encoder_cpr": -183.0,
    "max_speed": 4.0,
    "max_omega": 1.0,
    "wheel_max": 250.0,
}


def load_robot_config_file(path: str | Path) -> dict[str, float]:
    """Parse a ``key=value`` firmware config file into a dict (comments skipped)."""
    values: dict[str, float] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed robot config line in {path}: {raw_line!r}")
        key, _, value = line.partition("=")
        values[key.strip()] = float(value.strip())
    return values


def format_robot_config(values: Mapping[str, float]) -> str:
    """Render config values in the firmware file format (header, grouped keys)."""
    render_values = dict(values)
    render_values["robot_id"] = ROBOT_ID

    known_keys = {key for group in KEY_GROUPS for key in group}
    missing = known_keys - set(render_values)
    if missing:
        raise ValueError(f"Robot config is missing keys: {sorted(missing)}")

    blocks = [HEADER_COMMENT]
    for group in KEY_GROUPS:
        blocks.append(
            "\n".join(f"{key}={_format_robot_config_value(key, render_values[key])}" for key in group) + "\n"
        )
    extra_keys = [key for key in render_values if key not in known_keys]
    if extra_keys:
        blocks.append(
            "\n".join(f"{key}={_format_robot_config_value(key, render_values[key])}" for key in extra_keys) + "\n"
        )
    return "\n".join(blocks)


def robot_config_values(
    physical_params=None,
    controller_gains=None,
    template: Mapping[str, float] | None = None,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Firmware config dict from simulator quantities.

    ``physical_params`` is a PhysicalParams (or anything with wheel_radius,
    base_diameter and max_wheel_speed attributes); ``controller_gains`` is the
    full simulator gain vector (``controller.GAIN_NAMES``), of which only the
    firmware-known prefix [kx, ky, kth, kpmotor, kimotor] is exported — the
    firmware implements the Kanayama law only, so the dynamic-feedback gains
    have no slot. Inner motor gains are converted to firmware duty/(rad/s) units
    by dividing by max_wheel_speed, so exporting gains requires physical_params
    too. The firmware ``kd_inner`` slot has no simulator counterpart and is
    forced to 0.
    """
    values = dict(DEFAULT_ROBOT_CONFIG if template is None else template)
    if physical_params is not None:
        values["wheel_radius"] = float(physical_params.wheel_radius)
        values["wheel_base"] = float(physical_params.base_diameter)
    if controller_gains is not None:
        if physical_params is None:
            raise ValueError("Exporting controller gains requires physical_params for the motor-gain conversion.")
        from wmr_simulator.controller import GAIN_NAMES, NUM_GAINS

        gains = [float(gain) for gain in controller_gains]
        if len(gains) != NUM_GAINS:
            raise ValueError(f"Expected {NUM_GAINS} controller gains {GAIN_NAMES}, got {len(gains)}.")
        motor_gain = float(physical_params.max_wheel_speed)
        if motor_gain <= 0.0:
            raise ValueError("max_wheel_speed must be positive for the inner-gain conversion.")
        values["kx_traj"], values["ky_traj"], values["ktheta_traj"] = gains[0:3]
        values["kp_inner"] = gains[3] / motor_gain
        values["ki_inner"] = gains[4] / motor_gain
        values["kd_inner"] = 0.0
    if overrides:
        values.update({key: float(value) for key, value in overrides.items()})
    values["robot_id"] = ROBOT_ID
    return values


def export_robot_config(
    output_path: str | Path,
    physical_params=None,
    controller_gains=None,
    template_path: str | Path | None = None,
    overrides: Mapping[str, float] | None = None,
) -> Path:
    """Write a ROBOTCFG.CFG for the firmware and return its path."""
    template = load_robot_config_file(template_path) if template_path is not None else None
    values = robot_config_values(
        physical_params=physical_params,
        controller_gains=controller_gains,
        template=template,
        overrides=overrides,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_robot_config(values), encoding="utf-8")
    return output_path


def _format_robot_config_value(key: str, value: float) -> str:
    if key == "robot_id":
        return str(ROBOT_ID)
    return _format_value(value)


def _format_value(value: float) -> str:
    # repr of a rounded float matches the firmware files ("20.0", "0.01637", ...)
    # while keeping enough precision for converted gains.
    return repr(round(float(value), 9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Pololu firmware ROBOTCFG.CFG.")
    parser.add_argument("output", type=str, help="Output path for the CFG file.")
    parser.add_argument("--template", type=str, default=None, help="Existing CFG file to use as template.")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem yaml; robot wheel_radius/base_diameter, max_wheel_speed and controller gains are exported.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override an individual config value (repeatable).")
    args = parser.parse_args(argv)

    physical_params = None
    controller_gains = None
    if args.problem is not None:
        import yaml

        with open(args.problem, "r", encoding="utf-8") as file:
            problem_cfg = yaml.safe_load(file)
        from types import SimpleNamespace

        robot_cfg = problem_cfg["robot"]
        physical_params = SimpleNamespace(
            wheel_radius=robot_cfg["wheel_radius"],
            base_diameter=robot_cfg["base_diameter"],
            max_wheel_speed=robot_cfg["max_wheel_speed"],
        )
        from wmr_simulator.controller import gains_from_cfg

        controller_gains = gains_from_cfg(problem_cfg["controller"])

    overrides = {}
    for item in args.overrides:
        key, _, value = item.partition("=")
        if not key or not value:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = float(value)

    output_path = export_robot_config(
        args.output,
        physical_params=physical_params,
        controller_gains=controller_gains,
        template_path=args.template,
        overrides=overrides,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
