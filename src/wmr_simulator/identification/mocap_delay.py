"""Estimate the mocap measurement delay from IMU gyro cross-correlation.

The mocap pose reaches the logger through the network + radio/UART chain and
arrives a few milliseconds later than the onboard IMU sample of the same
motion. The delay is estimated by cross-correlating the finite-difference
mocap yaw rate with the IMU z gyro: both observe the identical body yaw rate,
so the lag that maximizes their correlation is the extra latency of the mocap
path relative to the IMU path.

Convention: ``delay_s`` > 0 means the mocap samples are logged *late* by that
amount, i.e. the pose logged at time t was actually assumed at t - delay_s.
Correct a log by shifting the mocap timestamps: t_corrected = t_logged -
delay_s (see pololu.log_loader.load_pololu_traj_control_log(mocap_delay_s=...)).

The estimator resamples both signals onto a common uniform grid, computes the
normalized cross-correlation over integer lags up to ``max_delay_s`` in both
directions, and refines the peak with parabolic interpolation for sub-sample
resolution. The lag search uses |correlation| so an inverted gyro axis still
locks on; a negative peak correlation is reported so the sign flip is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def estimate_delay_by_crosscorrelation(
    time_a: np.ndarray,
    signal_a: np.ndarray,
    time_b: np.ndarray,
    signal_b: np.ndarray,
    *,
    max_delay_s: float = 0.2,
    resample_dt: float | None = None,
) -> dict:
    """Delay of signal a relative to signal b (a(t) ~ b(t - delay)).

    Both signals are linearly resampled onto a shared uniform grid over their
    overlap. Returns a dict with ``delay_s`` (sub-sample, parabolic-refined),
    ``correlation`` (signed normalized correlation at the peak), ``lags_s`` and
    ``correlations`` (the full scan, for plotting/diagnostics), and
    ``resample_dt``.
    """
    time_a = np.asarray(time_a, dtype=float)
    time_b = np.asarray(time_b, dtype=float)
    signal_a = np.asarray(signal_a, dtype=float)
    signal_b = np.asarray(signal_b, dtype=float)
    if len(time_a) < 3 or len(time_b) < 3:
        raise ValueError("Need at least 3 samples per signal for delay estimation.")

    if resample_dt is None:
        resample_dt = min(_median_dt(time_a), _median_dt(time_b))
    if resample_dt <= 0.0:
        raise ValueError("Could not derive a positive resampling dt from the timestamps.")

    # Overlap window, padded so +-max_delay_s shifts stay inside both signals.
    start = max(time_a[0], time_b[0]) + max_delay_s
    end = min(time_a[-1], time_b[-1]) - max_delay_s
    if end - start < 10.0 * resample_dt:
        raise ValueError(
            f"Signals overlap for only {max(end - start, 0.0):.3f} s after padding by "
            f"max_delay_s={max_delay_s}; not enough for delay estimation."
        )
    grid = np.arange(start, end, resample_dt)
    a = np.interp(grid, time_a, signal_a)
    a = a - a.mean()

    max_lag_steps = max(int(np.ceil(max_delay_s / resample_dt)), 1)
    lags_s = resample_dt * np.arange(-max_lag_steps, max_lag_steps + 1)
    correlations = np.empty(len(lags_s), dtype=float)
    a_norm = float(np.linalg.norm(a))
    for index, lag in enumerate(lags_s):
        # a(t) ~ b(t - delay): evaluate b on the grid shifted back by the lag.
        b = np.interp(grid - lag, time_b, signal_b)
        b = b - b.mean()
        denominator = a_norm * float(np.linalg.norm(b))
        correlations[index] = float(a @ b) / denominator if denominator > 0.0 else 0.0

    peak_index = int(np.argmax(np.abs(correlations)))
    delay_s = float(lags_s[peak_index])
    peak_correlation = float(correlations[peak_index])
    if 0 < peak_index < len(lags_s) - 1:
        delay_s += _parabolic_offset(np.abs(correlations), peak_index) * resample_dt

    return {
        "delay_s": delay_s,
        "correlation": peak_correlation,
        "lags_s": lags_s,
        "correlations": correlations,
        "resample_dt": float(resample_dt),
        "num_samples": int(len(grid)),
    }


def mocap_yaw_rate(pose_time_s: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference yaw rate at the mocap interval midpoints (yaw unwrapped)."""
    pose_time_s = np.asarray(pose_time_s, dtype=float)
    yaw_unwrapped = np.unwrap(np.asarray(yaw, dtype=float))
    dt = np.diff(pose_time_s)
    valid = dt > 1e-9
    midpoints = 0.5 * (pose_time_s[:-1] + pose_time_s[1:])
    rate = np.zeros_like(dt)
    rate[valid] = np.diff(yaw_unwrapped)[valid] / dt[valid]
    return midpoints[valid], rate[valid]


def estimate_mocap_delay_from_log_file(
    log_path: str | Path,
    *,
    max_delay_s: float = 0.2,
) -> dict:
    """Estimate the mocap delay of one Pololu csv log against its IMU gyro z.

    Raises ValueError when the log carries no IMU samples (older logs have the
    gyro columns in the header but never fill them).
    """
    from wmr_simulator.pololu.log_loader import (
        POLOLU_TRAJ_CONTROL_COLUMNS,
        _read_csv,
        _sparse_stream,
    )

    log_path = Path(log_path)
    columns, data = _read_csv(log_path)
    if tuple(columns) != POLOLU_TRAJ_CONTROL_COLUMNS:
        raise ValueError(f"Unexpected columns in {log_path}: {tuple(columns)}")
    ts_index = columns.index("ts")
    data = data[np.isfinite(data[:, ts_index])]
    data = data[np.argsort(data[:, ts_index])]
    data = data.copy()
    data[:, ts_index] = data[:, ts_index] / 1000.0 - data[0, ts_index] / 1000.0

    pose_time, pose_states = _sparse_stream(columns, data, ("x_raw", "y_raw", "yaw_raw"))
    gyro_time, gyro = _sparse_stream(columns, data, ("gyro_z",), trigger_names=("gyro_z",))
    if len(gyro_time) == 0:
        raise ValueError(f"{log_path} contains no IMU gyro samples; cannot estimate the mocap delay.")
    if len(pose_time) < 3:
        raise ValueError(f"{log_path} contains too few mocap samples for delay estimation.")

    # Smooth yaw rate from the Savitzky-Golay filter derivative (pololu.measurement_smoothing);
    # finite differences remain the fallback for very short pose streams.
    from wmr_simulator.pololu.measurement_smoothing import smooth_pose_stream

    try:
        smoothed = smooth_pose_stream(pose_time, pose_states)
        omega_time = np.asarray(pose_time, dtype=float)
        omega_mocap = smoothed.world_velocity(omega_time)[:, 2]
    except ValueError:
        omega_time, omega_mocap = mocap_yaw_rate(pose_time, pose_states[:, 2])

    result = estimate_delay_by_crosscorrelation(
        omega_time,
        omega_mocap,
        gyro_time,
        gyro[:, 0],
        max_delay_s=max_delay_s,
    )
    result["log"] = str(log_path)
    return result


def print_delay_result(result: dict) -> None:
    print(
        f"Estimated mocap delay: {1000.0 * result['delay_s']:.2f} ms "
        f"(peak correlation {result['correlation']:.3f}, "
        f"{result['num_samples']} samples at dt={1000.0 * result['resample_dt']:.2f} ms)"
    )
    if result["correlation"] < 0.0:
        print("  warning: peak correlation is negative -- gyro z axis appears inverted "
              "relative to the mocap yaw convention.")
    if abs(result["correlation"]) < 0.5:
        print("  warning: weak correlation; treat the estimated delay with caution "
              "(little yaw excitation or noisy signals).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate the mocap latency of a Pololu log by cross-correlating "
        "the finite-difference mocap yaw rate with the IMU gyro z."
    )
    parser.add_argument("--log", type=str, required=True, help="Pololu csv log with IMU data.")
    parser.add_argument("--max-delay", type=float, default=0.05, help="Search range in seconds (both directions).")
    parser.add_argument("--plot", action="store_true", help="Save the correlation-vs-lag curve to visualize/.")
    parser.add_argument("--out-dir", type=str, default="visualize")
    args = parser.parse_args(argv)

    try:
        result = estimate_mocap_delay_from_log_file(
            args.log,
            max_delay_s=args.max_delay,
        )
    except ValueError as error:
        print(f"error: {error}")
        return 1
    print_delay_result(result)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(1000.0 * result["lags_s"], result["correlations"])
        axis.axvline(1000.0 * result["delay_s"], color="tab:red", linestyle="--",
                     label=f"delay = {1000.0 * result['delay_s']:.2f} ms")
        axis.set_xlabel("mocap lag [ms]")
        axis.set_ylabel("normalized correlation")
        axis.set_title(f"Mocap delay estimation ({Path(args.log).stem})")
        axis.legend()
        axis.grid(True, alpha=0.3)
        out_path = out_dir / f"mocap_delay_{Path(args.log).stem}.pdf"
        figure.savefig(out_path, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved correlation plot: {out_path}")
    return 0


def _median_dt(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    positive = dt[dt > 0.0]
    return float(np.median(positive)) if len(positive) else 0.0


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    """Sub-sample peak offset in [-0.5, 0.5] from a 3-point parabola fit."""
    left, center, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
