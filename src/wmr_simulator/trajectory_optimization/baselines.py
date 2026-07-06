"""Analytic baseline reference trajectories for comparison with optimized ones.

All baselines share the [T, 8] reference-state layout of the Bezier synthesis
([x, y, theta, vx, vy, omega, ax, ay] sampled at geometry_controller_dt) and use
the same quintic s-curve time scaling, so start and end velocities go to zero
gracefully and the sample count is time / geometry_dt, just like the optimized
trajectories they are compared against.

Baseline types:
  - circle:          constant-radius circle, parametrized by radius and time.
  - lemniscate:      figure-eight (lemniscate of Gerono), parametrized by time
                     and peak speed (amplitude is solved for the peak speed).
  - lemniscate-ramp: figure-eight repeated for a number of cycles whose phase
                     rate grows by a constant factor per completed cycle. An
                     s-curve envelope ramps the rate in from zero and back out
                     to zero, so the robot is never hit with a step in speed.
  - spin:            rotation on the spot, parametrized by time and peak omega
                     (the total rotation is solved for the peak omega). Meant
                     for estimating the mocap delay from the IMU-gyro vs mocap
                     omega difference (identification/mocap_delay.py).

The CLI wrapper lives in scripts/generate_baseline_reference.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from wmr_simulator.trajectory_optimization.parametrization import sigma, sigma_dot, sigma_ddot

BASELINE_TYPES = ("circle", "lemniscate", "lemniscate-ramp", "spin")

# Peak of the quintic s-curve rate: sigma_dot(T/2) = 15 / (8 * T).
S_CURVE_PEAK_RATE_FACTOR = 15.0 / 8.0

TWO_PI = 2.0 * np.pi


def baseline_time_grid(total_time: float, dt: float) -> np.ndarray:
    steps = int(np.round(total_time / dt))
    if steps <= 0:
        raise ValueError("total_time must contain at least one dt interval.")
    return np.linspace(0.0, steps * dt, steps + 1)


def _assemble_reference(
    position: np.ndarray,
    dpos_du: np.ndarray,
    d2pos_du2: np.ndarray,
    u_dot: np.ndarray,
    u_ddot: np.ndarray,
) -> np.ndarray:
    """Assemble the [T, 8] reference-state matrix from path derivatives.

    Mirrors bezier._reference_from_derivatives: heading comes from the path
    tangent (well-defined even where the speed is zero at the endpoints).
    """
    velocity = dpos_du * u_dot[:, None]
    acceleration = dpos_du * u_ddot[:, None] + d2pos_du2 * (u_dot[:, None] ** 2)
    theta = np.arctan2(dpos_du[:, 1], dpos_du[:, 0])

    tangent_norm_sq = np.sum(dpos_du**2, axis=1)
    dtheta_du = (
        dpos_du[:, 0] * d2pos_du2[:, 1] - dpos_du[:, 1] * d2pos_du2[:, 0]
    ) / (tangent_norm_sq + 1e-12)
    omega = dtheta_du * u_dot

    return np.column_stack(
        [
            position[:, 0],
            position[:, 1],
            theta,
            velocity[:, 0],
            velocity[:, 1],
            omega,
            acceleration[:, 0],
            acceleration[:, 1],
        ]
    )


def _lemniscate_path(u: np.ndarray, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lemniscate of Gerono: gamma(u) = a * [sin(u), 0.5 * sin(2u)], u in [0, 2*pi] per cycle."""
    position = amplitude * np.column_stack([np.sin(u), 0.5 * np.sin(2.0 * u)])
    dpos_du = amplitude * np.column_stack([np.cos(u), np.cos(2.0 * u)])
    d2pos_du2 = amplitude * np.column_stack([-np.sin(u), -2.0 * np.sin(2.0 * u)])
    return position, dpos_du, d2pos_du2


def circle_reference(
    radius: float,
    total_time: float,
    dt: float,
    center: tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
    clockwise: bool = False,
) -> np.ndarray:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    time_grid = baseline_time_grid(total_time, dt)
    duration = time_grid[-1]
    direction = -1.0 if clockwise else 1.0
    phi = start_angle + direction * TWO_PI * sigma(time_grid, duration)
    phi_dot = direction * TWO_PI * sigma_dot(time_grid, duration)
    phi_ddot = direction * TWO_PI * sigma_ddot(time_grid, duration)

    center = np.asarray(center, dtype=float)
    position = center[None, :] + radius * np.column_stack([np.cos(phi), np.sin(phi)])
    dpos_dphi = radius * np.column_stack([-np.sin(phi), np.cos(phi)])
    d2pos_dphi2 = radius * np.column_stack([-np.cos(phi), -np.sin(phi)])
    return _assemble_reference(position, dpos_dphi, d2pos_dphi2, phi_dot, phi_ddot)


def lemniscate_reference(
    total_time: float,
    max_speed: float,
    dt: float,
    cycles: int = 1,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive.")
    if cycles < 1:
        raise ValueError("cycles must be at least 1.")
    time_grid = baseline_time_grid(total_time, dt)
    duration = time_grid[-1]
    total_phase = TWO_PI * cycles
    u = total_phase * sigma(time_grid, duration)
    u_dot = total_phase * sigma_dot(time_grid, duration)
    u_ddot = total_phase * sigma_ddot(time_grid, duration)

    # Speed scales linearly with the amplitude, so generate with a = 1 and
    # rescale so the peak speed matches max_speed exactly.
    _, unit_dpos_du, _ = _lemniscate_path(u, amplitude=1.0)
    peak_unit_speed = np.max(np.linalg.norm(unit_dpos_du, axis=1) * np.abs(u_dot))
    amplitude = max_speed / peak_unit_speed

    position, dpos_du, d2pos_du2 = _lemniscate_path(u, amplitude)
    position = position + np.asarray(center, dtype=float)[None, :]
    return _assemble_reference(position, dpos_du, d2pos_du2, u_dot, u_ddot)


def _ramp_envelope(
    t: np.ndarray,
    total_time: float,
    ramp_up_time: float,
    ramp_down_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """S-curve window 0 -> 1 -> 0 with analytic time derivative."""
    t = np.asarray(t, dtype=float)
    env = np.ones_like(t)
    env_dot = np.zeros_like(t)
    if ramp_up_time > 0.0:
        rising = t < ramp_up_time
        env = np.where(rising, sigma(np.minimum(t, ramp_up_time), ramp_up_time), env)
        env_dot = np.where(rising, sigma_dot(np.minimum(t, ramp_up_time), ramp_up_time), env_dot)
    if ramp_down_time > 0.0:
        falling = t > total_time - ramp_down_time
        remaining = np.clip(total_time - t, 0.0, ramp_down_time)
        env = np.where(falling, sigma(remaining, ramp_down_time), env)
        env_dot = np.where(falling, -sigma_dot(remaining, ramp_down_time), env_dot)
    return env, env_dot


def _integrate_ramp_phase(
    omega0: float,
    time_grid: np.ndarray,
    growth: float,
    ramp_up_time: float,
    ramp_down_time: float,
    substeps: int = 20,
) -> np.ndarray:
    """Integrate du/dt = omega0 * env(t) * growth^(u / 2pi) on the time grid."""
    duration = time_grid[-1]
    num_intervals = len(time_grid) - 1
    h = (time_grid[1] - time_grid[0]) / substeps
    t_sub = (time_grid[:-1, None] + h * np.arange(substeps)[None, :]).ravel()
    env_sub, _ = _ramp_envelope(t_sub, duration, ramp_up_time, ramp_down_time)
    log_growth = np.log(growth)

    u = np.zeros(len(time_grid))
    current = 0.0
    sub_index = 0
    for interval in range(num_intervals):
        for _ in range(substeps):
            # Clamp the exponent: over-large bracketing candidates would overflow.
            current += h * omega0 * env_sub[sub_index] * np.exp(min(log_growth * current / TWO_PI, 50.0))
            sub_index += 1
        u[interval + 1] = current
    return u


def lemniscate_ramp_reference(
    total_time: float,
    cycles: int,
    speed_rate: float,
    dt: float,
    amplitude: float = 0.5,
    max_speed: float | None = None,
    ramp_up_fraction: float = 0.2,
    ramp_down_fraction: float = 0.1,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Figure-eight whose phase rate grows by speed_rate per completed cycle.

    The phase obeys du/dt = omega0 * env(t) * speed_rate^(u / 2pi); omega0 is
    solved by bisection so exactly `cycles` cycles fit into total_time, and the
    s-curve envelope brings the rate in from zero and back to zero at the end.
    If max_speed is given, the amplitude is rescaled so the peak speed matches.
    """
    if cycles < 1:
        raise ValueError("cycles must be at least 1.")
    if speed_rate <= 0.0:
        raise ValueError("speed_rate must be positive.")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")
    if ramp_up_fraction < 0.0 or ramp_down_fraction < 0.0 or ramp_up_fraction + ramp_down_fraction >= 1.0:
        raise ValueError("ramp fractions must be non-negative and sum to less than 1.")

    time_grid = baseline_time_grid(total_time, dt)
    duration = time_grid[-1]
    ramp_up_time = ramp_up_fraction * duration
    ramp_down_time = ramp_down_fraction * duration
    target_phase = TWO_PI * cycles

    def final_phase(omega0: float) -> float:
        return _integrate_ramp_phase(omega0, time_grid, speed_rate, ramp_up_time, ramp_down_time)[-1]

    low, high = 0.0, target_phase / duration
    while final_phase(high) < target_phase:
        high *= 2.0
        if high > 1e9:
            raise RuntimeError("Failed to bracket the ramp phase rate; check speed_rate and cycles.")
    for _ in range(80):
        mid = 0.5 * (low + high)
        if final_phase(mid) < target_phase:
            low = mid
        else:
            high = mid
    omega0 = 0.5 * (low + high)

    u = _integrate_ramp_phase(omega0, time_grid, speed_rate, ramp_up_time, ramp_down_time)
    env, env_dot = _ramp_envelope(time_grid, duration, ramp_up_time, ramp_down_time)
    log_growth = np.log(speed_rate)
    growth_term = np.exp(log_growth * u / TWO_PI)
    u_dot = omega0 * env * growth_term
    u_ddot = omega0 * env_dot * growth_term + (log_growth / TWO_PI) * u_dot**2

    if max_speed is not None:
        if max_speed <= 0.0:
            raise ValueError("max_speed must be positive.")
        _, unit_dpos_du, _ = _lemniscate_path(u, amplitude=1.0)
        peak_unit_speed = np.max(np.linalg.norm(unit_dpos_du, axis=1) * np.abs(u_dot))
        amplitude = max_speed / peak_unit_speed

    position, dpos_du, d2pos_du2 = _lemniscate_path(u, amplitude)
    position = position + np.asarray(center, dtype=float)[None, :]
    return _assemble_reference(position, dpos_du, d2pos_du2, u_dot, u_ddot)


def spin_reference(
    total_time: float,
    max_omega: float,
    dt: float,
    center: tuple[float, float] = (0.0, 0.0),
    start_angle: float = 0.0,
    clockwise: bool = False,
) -> np.ndarray:
    """Rotation on the spot with s-curve eased heading, peaking at max_omega.

    The position stays fixed at `center` and the total rotation is chosen so
    the quintic s-curve's peak rate lands exactly on max_omega. Intended for
    mocap-delay identification: the delay shows up as the lag between the IMU
    gyro rate and the differentiated mocap heading.
    """
    if max_omega <= 0.0:
        raise ValueError("max_omega must be positive.")
    time_grid = baseline_time_grid(total_time, dt)
    duration = time_grid[-1]
    direction = -1.0 if clockwise else 1.0
    # Peak omega = total_angle * sigma_dot_peak = total_angle * 15 / (8 * T).
    total_angle = max_omega * duration / S_CURVE_PEAK_RATE_FACTOR

    theta = start_angle + direction * total_angle * sigma(time_grid, duration)
    omega = direction * total_angle * sigma_dot(time_grid, duration)

    num_samples = len(time_grid)
    states = np.zeros((num_samples, 8))
    states[:, 0] = center[0]
    states[:, 1] = center[1]
    states[:, 2] = theta
    states[:, 5] = omega
    return states


def summarize_motion(reference_states: np.ndarray, dt: float) -> dict[str, float]:
    """Peak/mean motion quantities for feasibility reporting against robot limits."""
    reference_states = np.asarray(reference_states, dtype=float)
    speed = np.linalg.norm(reference_states[:, 3:5], axis=1)
    acceleration = np.linalg.norm(reference_states[:, 6:8], axis=1)
    omega = reference_states[:, 5]
    alpha = np.gradient(omega, dt)
    return {
        "duration": float((reference_states.shape[0] - 1) * dt),
        "num_samples": float(reference_states.shape[0]),
        "v_peak": float(np.max(speed)),
        "v_mean": float(np.mean(speed)),
        "a_peak": float(np.max(acceleration)),
        "omega_peak": float(np.max(np.abs(omega))),
        "alpha_peak": float(np.max(np.abs(alpha))),
    }


def generate_baseline_reference(baseline_type: str, dt: float, **params) -> np.ndarray:
    baseline_type = baseline_type.strip().lower().replace("_", "-")
    if baseline_type == "circle":
        return circle_reference(dt=dt, **params)
    if baseline_type == "lemniscate":
        return lemniscate_reference(dt=dt, **params)
    if baseline_type == "lemniscate-ramp":
        return lemniscate_ramp_reference(dt=dt, **params)
    if baseline_type == "spin":
        return spin_reference(dt=dt, **params)
    raise ValueError(f"Unsupported baseline type '{baseline_type}'. Expected one of {BASELINE_TYPES}.")


def export_baseline_reference(
    reference_states: np.ndarray,
    dt: float,
    export_dir: str | Path,
    name: str,
    **metadata,
) -> tuple[Path, Path]:
    """Write <name>.pkl (pipeline format) and <name>.JSN (Pololu format); return both paths."""
    from wmr_simulator.pololu.reference_exporter import (
        ReferenceTrajectory,
        export_reference_trajectory,
    )
    from wmr_simulator.trajectory_optimization.pipeline import reference_states_export_payload

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = export_dir / f"{name}.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(reference_states_export_payload(reference_states, dt, **metadata), file)

    jsn_path = export_reference_trajectory(
        ReferenceTrajectory(path=pickle_path, reference_states=np.asarray(reference_states, dtype=float), dt=dt),
        export_dir,
        output_name=f"{name}.JSN",
    )
    return pickle_path, jsn_path


def run_baseline_closed_loop(problem_path: str, reference_states: np.ndarray, seed: int = 42):
    """Run the problem's closed loop (hidden robot, nominal gains) on the baseline reference."""
    import jax.numpy as jnp

    from wmr_simulator.simulation import SimulationPipeline

    simulation = SimulationPipeline(problem_path=problem_path, seed=seed, reference_trajectories_dir=None)
    _extend_time_grids(simulation, len(reference_states))
    log = simulation.run_closed_loop(
        simulation.hidden_params,
        use_hidden_robot=True,
        controller_gains=simulation.gains,
        reference_states=jnp.asarray(reference_states, dtype=jnp.float32),
    )
    return simulation, log


def _extend_time_grids(simulation, num_reference_samples: int) -> None:
    """Regrow the pipeline time grids when the baseline outlasts the problem sim_time.

    run_closed_loop slices its log time axes from these grids, so a reference
    longer than sim_time would otherwise produce time/state length mismatches.
    """
    if num_reference_samples <= len(simulation.reference_time_grid):
        return
    duration = (num_reference_samples - 1) * simulation.geometry_dt
    simulation.sim_time = duration
    simulation.reference_time_grid = simulation._time_grid(duration, simulation.geometry_dt)
    simulation.pose_time_grid = simulation._time_grid(duration, simulation.wheel_dt)
    simulation.sim_time_grid = simulation.pose_time_grid
    simulation.wheel_time_grid = simulation.pose_time_grid - simulation.wheel_dt
    simulation.command_time_grid = simulation.reference_time_grid[:-1] + 0.5 * simulation.wheel_dt
    simulation.reference_pose_indices = np.arange(
        0,
        len(simulation.reference_time_grid) * simulation.inner_steps_per_geometry_step,
        simulation.inner_steps_per_geometry_step,
        dtype=np.int32,
    )
