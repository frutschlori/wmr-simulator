"""Export the error-MLP gain parametrization for the Pololu firmware.

The firmware crate ``firmware/libs/gain_mlp`` (pololu-rs) evaluates the MLP
every outer control tick from the same inputs the simulator uses: the current
setpoint, the EKF pose and the encoder body twist. This module renders the
trained :mod:`wmr_simulator.gain_parametrization.error_mlp` parameters into the
JSON file that crate parses (``GAINMLP.JSN`` on the SD card, next to
``TRJ0001.JSN`` and ``ROBOTCFG.CFG``):

    {
      "kind": "error_mlp",
      "sizes": [7, 16, 3],            # features, hidden..., num scheduled gains
      "scheduled_indices": [0, 1, 2], # gain-vector slots the factors apply to
      "bound": 5.0,
      "feature_scale": [... 7 floats ...],
      "weights": [... flat: per layer W row-major, then b ...]
    }

The spectral normalization of :func:`error_mlp.effective_layers` is baked into
the exported weights, so the firmware forward pass is a plain tanh MLP (no
power iteration on the MCU). Factors are multiplicative, so they apply
unchanged to the firmware gains even though the inner motor gains use
different units (duty/(rad/s) instead of the simulator's wheel-speed units).

A golden file with input/output pairs computed by JAX can be exported next to
the network; the firmware crate's host test (``cargo test``) checks its
implementation against it.

Firmware capacity limits are mirrored from ``firmware/libs/gain_mlp/src/lib.rs``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmr_simulator.controller import FIRMWARE_GAIN_SLICE
from wmr_simulator.controller import NUM_GAINS as SIMULATOR_NUM_GAINS
from wmr_simulator.gain_parametrization import error_mlp
from wmr_simulator.gain_parametrization.error_mlp import (
    NUM_FEATURES,
    NUM_GAINS,
    ErrorMlpParams,
)

# Mirrors firmware/libs/gain_mlp/src/lib.rs (MAX_HIDDEN_LAYERS, MAX_WIDTH, MAX_WEIGHTS).
FIRMWARE_MAX_HIDDEN_LAYERS = 2
FIRMWARE_MAX_WIDTH = 32
FIRMWARE_MAX_WEIGHTS = 1600


def gain_mlp_payload(params: ErrorMlpParams) -> dict:
    """JSON-ready dict for the firmware, spectral normalization baked in."""
    layers = [np.asarray(layer, dtype=np.float32) for layer in error_mlp.effective_layers(params)]
    weights = [w for layer in layers for w in layer.reshape(-1).tolist()]
    hidden = [int(w.shape[0]) for w in layers[0:-2:2]]
    sizes = [NUM_FEATURES, *hidden, int(layers[-2].shape[0])]
    scheduled = [int(index) for index in np.asarray(params.scheduled_indices)]

    if len(hidden) > FIRMWARE_MAX_HIDDEN_LAYERS:
        raise ValueError(
            f"Firmware supports at most {FIRMWARE_MAX_HIDDEN_LAYERS} hidden layers, got {len(hidden)}."
        )
    if any(size > FIRMWARE_MAX_WIDTH for size in hidden):
        raise ValueError(f"Firmware supports hidden widths up to {FIRMWARE_MAX_WIDTH}, got {hidden}.")
    if len(weights) > FIRMWARE_MAX_WEIGHTS:
        raise ValueError(f"Firmware supports up to {FIRMWARE_MAX_WEIGHTS} weights, got {len(weights)}.")
    if sizes[-1] != len(scheduled):
        raise ValueError(f"Output size {sizes[-1]} does not match scheduled_indices {scheduled}.")

    return {
        "kind": error_mlp.KIND,
        "sizes": sizes,
        "scheduled_indices": scheduled,
        "bound": max(float(params.bound), 1.0),
        "feature_scale": np.asarray(params.feature_scale, dtype=np.float32).tolist(),
        "weights": weights,
    }


def firmware_base_gains(controller_gains, max_wheel_speed: float) -> list:
    """Simulator gain vector -> firmware units (inner motor gains / motor gain).

    Takes the full simulator gain vector and keeps the firmware-known prefix
    (the firmware implements the Kanayama law only, so the dynamic-feedback
    gains have no firmware counterpart).
    """
    gains = [float(gain) for gain in controller_gains]
    if len(gains) != SIMULATOR_NUM_GAINS:
        raise ValueError(f"Expected {SIMULATOR_NUM_GAINS} controller gains, got {len(gains)}.")
    gains = gains[FIRMWARE_GAIN_SLICE]
    motor_gain = float(max_wheel_speed)
    if motor_gain <= 0.0:
        raise ValueError("max_wheel_speed must be positive for the inner-gain conversion.")
    return gains[0:3] + [gains[3] / motor_gain, gains[4] / motor_gain]


def reference_forward(payload: dict, ref: list, pose: list, twist: list) -> np.ndarray:
    """Numpy replica of the firmware forward pass; returns 5 factors.

    ``ref`` is [x_d, y_d, theta_d, v_d, omega_d] (the firmware setpoint),
    ``pose`` the estimated pose, ``twist`` the encoder body twist [v, omega].
    """
    x_d, y_d, theta_d, v_d, omega_d = (np.float32(value) for value in ref)
    x, y, theta = (np.float32(value) for value in pose)
    v, omega = (np.float32(value) for value in twist)

    dx = x_d - x
    dy = y_d - y
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    theta_e = np.mod(theta_d - theta + np.float32(np.pi), np.float32(2.0 * np.pi)) - np.float32(np.pi)
    raw = np.array(
        [
            dx * cos_t + dy * sin_t,
            -dx * sin_t + dy * cos_t,
            theta_e,
            v_d - v,
            omega_d - omega,
            v_d,
            np.abs(omega_d),
        ],
        dtype=np.float32,
    )
    h = raw / np.asarray(payload["feature_scale"], dtype=np.float32)

    sizes = payload["sizes"]
    weights = np.asarray(payload["weights"], dtype=np.float32)
    offset = 0
    for layer, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        w = weights[offset : offset + fan_out * fan_in].reshape(fan_out, fan_in)
        offset += fan_out * fan_in
        b = weights[offset : offset + fan_out]
        offset += fan_out
        h = w @ h + b
        if layer < len(sizes) - 2:
            h = np.tanh(h)
    if offset != weights.size:
        raise ValueError(f"weights length {weights.size} does not match sizes {sizes}.")

    factors = np.ones(NUM_GAINS, dtype=np.float32)
    factors[payload["scheduled_indices"]] = np.clip(
        np.float32(1.0) + h, np.float32(0.0), np.float32(payload["bound"])
    )
    return factors


def golden_payload(
    params: ErrorMlpParams,
    base_gains_firmware: list,
    num_cases: int = 32,
    seed: int = 0,
) -> dict:
    """Network payload plus JAX-evaluated input/output cases for the firmware test."""
    import jax.numpy as jnp

    payload = gain_mlp_payload(params)
    feature_scale = np.asarray(params.feature_scale, dtype=np.float32)
    rng = np.random.default_rng(seed)

    cases = []
    for _ in range(int(num_cases)):
        theta_d = rng.uniform(-np.pi, np.pi)
        ref_xy = rng.uniform(-1.0, 1.0, size=2)
        v_d = rng.uniform(0.0, 0.5 * feature_scale[5])
        omega_d = rng.uniform(-0.5, 0.5) * feature_scale[6]
        pose = [
            ref_xy[0] - rng.normal(0.0, 0.3 * feature_scale[0]),
            ref_xy[1] - rng.normal(0.0, 0.3 * feature_scale[1]),
            theta_d - rng.normal(0.0, 0.3 * feature_scale[2]),
        ]
        twist = [v_d - rng.normal(0.0, 0.2 * feature_scale[3]), omega_d - rng.normal(0.0, 0.2 * feature_scale[4])]

        ref_state = jnp.asarray(
            [ref_xy[0], ref_xy[1], theta_d, v_d, 0.0, omega_d, 0.0, 0.0], dtype=jnp.float32
        )
        factors = np.ones(NUM_GAINS, dtype=np.float32)
        factors[np.asarray(params.scheduled_indices)] = np.asarray(
            error_mlp.factors(
                params,
                ref_state,
                jnp.asarray(pose, dtype=jnp.float32),
                jnp.asarray(twist, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )
        gains = np.asarray(base_gains_firmware, dtype=np.float32) * factors
        cases.append(
            {
                "ref": [float(ref_xy[0]), float(ref_xy[1]), float(theta_d), float(v_d), float(omega_d)],
                "pose": [float(value) for value in pose],
                "twist": [float(value) for value in twist],
                "base_gains": [float(value) for value in base_gains_firmware],
                "expected_factors": factors.tolist(),
                "expected_gains": gains.tolist(),
            }
        )
    return {"network": payload, "cases": cases}


def export_gain_mlp(output_path: str | Path, params: ErrorMlpParams) -> Path:
    """Write the firmware network file (``GAINMLP.JSN``) and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gain_mlp_payload(params)), encoding="utf-8")
    return output_path


def export_gain_mlp_golden(
    output_path: str | Path,
    params: ErrorMlpParams,
    base_gains_firmware: list,
    num_cases: int = 32,
    seed: int = 0,
) -> Path:
    """Write the golden test file for the firmware crate and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = golden_payload(params, base_gains_firmware, num_cases=num_cases, seed=seed)
    output_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return output_path


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the error-MLP gain parametrization for the firmware.")
    parser.add_argument("output", type=str, help="Output path for the network file (e.g. GAINMLP.JSN).")
    parser.add_argument("--problem", type=str, default="problems/pololu_gains.yaml",
                        help="Problem yaml providing the parametrization config, gains and robot limits.")
    parser.add_argument("--tuned", type=str, default=None,
                        help="Tuning result yaml (run_gain_tuning --out); overrides gains and parametrization.")
    parser.add_argument("--golden", type=str, default=None,
                        help="Also write a golden test file with JAX-evaluated cases to this path.")
    parser.add_argument("--cases", type=int, default=32, help="Number of golden cases.")
    parser.add_argument("--seed", type=int, default=0, help="Golden case sampling seed.")
    args = parser.parse_args(argv)

    import yaml

    from wmr_simulator.controller import gains_from_cfg
    from wmr_simulator.gain_parametrization import params_from_cfg, parametrization_kind

    with open(args.problem, "r", encoding="utf-8") as file:
        problem_cfg = yaml.safe_load(file)
    robot_cfg = problem_cfg["robot"]
    feature_scale = [robot_cfg["v_max"], robot_cfg["omega_max"]]
    gains = gains_from_cfg(problem_cfg["controller"])
    parametrization_cfg = problem_cfg["controller"].get("gain_parametrization")

    if args.tuned is not None:
        with open(args.tuned, "r", encoding="utf-8") as file:
            tuned = yaml.safe_load(file)
        gains = tuned["gains"]
        if tuned.get("schedule") is not None:
            parametrization_cfg = tuned["schedule"]

    if parametrization_kind(parametrization_cfg) != error_mlp.KIND:
        raise SystemExit(
            f"Gain parametrization kind {parametrization_kind(parametrization_cfg)!r} is not "
            f"{error_mlp.KIND!r}; nothing to export."
        )
    params = params_from_cfg(parametrization_cfg, feature_scale)

    output_path = export_gain_mlp(args.output, params)
    print(output_path)
    if args.golden is not None:
        base_gains = firmware_base_gains(gains, robot_cfg["max_wheel_speed"])
        golden_path = export_gain_mlp_golden(
            args.golden, params, base_gains, num_cases=args.cases, seed=args.seed
        )
        print(golden_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
