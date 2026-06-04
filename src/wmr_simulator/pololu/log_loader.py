from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wmr_simulator.estimator import EstimatorState
from wmr_simulator.robot import DiffDriveState
from wmr_simulator.types import SimulationLog


POLOLU_TRAJ_CONTROL_COLUMNS = (
    "ts",
    "target_x",
    "target_y",
    "target_theta",
    "actual_x",
    "actual_y",
    "actual_theta",
    "target_vx",
    "target_vy",
    "target_vz",
    "actual_vx",
    "actual_vy",
    "actual_vz",
    "target_qw",
    "target_qx",
    "target_qy",
    "target_qz",
    "actual_qw",
    "actual_qx",
    "actual_qy",
    "actual_qz",
    "xerror",
    "yerror",
    "thetaerror",
    "ul",
    "ur",
    "dutyl",
    "dutyr",
)

@dataclass(frozen=True)
class PololuTrajControlLog:
    columns: tuple[str, ...]
    data: np.ndarray
    raw_time_s: np.ndarray
    time_s: np.ndarray

    def column(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    @property
    def dt(self) -> float | None:
        if self.time_s.size < 2:
            return None
        return float(np.median(np.diff(self.time_s)))

    @property
    def measurement_dts(self) -> np.ndarray:
        return np.diff(self.time_s)

    def to_target_log(
        self,
        wheel_radius: float,
        base_diameter: float,
    ) -> SimulationLog:
        import jax.numpy as jnp

        pose = np.stack(
            [
                self.column("actual_x"),
                self.column("actual_y"),
                self.column("actual_theta"),
            ],
            axis=1,
        ).astype(np.float32)
        vel_omega = np.stack(
            [
                self.column("actual_vx"),
                self.column("actual_vz"),
            ],
            axis=1,
        ).astype(np.float32)
        wheel_speeds = self.reconstructed_wheel_speeds(
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        ).astype(np.float32)
        wheel_cmd = self.commanded_wheel_speeds().astype(np.float32)

        num_steps = pose.shape[0]
        keys = np.zeros((num_steps, 2), dtype=np.uint32)
        covariance = np.zeros((num_steps, 3, 3), dtype=np.float32)

        robot_states = DiffDriveState(
            pose=jnp.asarray(pose),
            wheel_speeds=jnp.asarray(wheel_speeds),
            key=jnp.asarray(keys),
            vel_omega=jnp.asarray(vel_omega),
            wheel_cmd=jnp.asarray(wheel_cmd),
        )
        estimator_states = EstimatorState(
            pose_hat=jnp.asarray(pose),
            pose_meas=jnp.asarray(pose),
            u_hat=jnp.asarray(wheel_speeds),
            u_true=jnp.asarray(wheel_speeds),
            P=jnp.asarray(covariance),
            key=jnp.asarray(keys),
        )
        return SimulationLog(robot_states=robot_states, estimator_states=estimator_states)

    def target_pose(self) -> np.ndarray:
        return np.stack(
            [self.column("target_x"), self.column("target_y"), self.column("target_theta")],
            axis=1,
        )

    def actual_pose(self) -> np.ndarray:
        return np.stack(
            [self.column("actual_x"), self.column("actual_y"), self.column("actual_theta")],
            axis=1,
        )

    def target_vel_omega(self) -> np.ndarray:
        return np.stack([self.column("target_vx"), self.column("target_vz")], axis=1)

    def target_reference_states(self) -> np.ndarray:
        zeros = np.zeros_like(self.time_s)
        return np.stack(
            [
                self.column("target_x"),
                self.column("target_y"),
                self.column("target_theta"),
                self.column("target_vx"),
                self.column("target_vy"),
                self.column("target_vz"),
                zeros,
                zeros,
            ],
            axis=1,
        ).astype(np.float32)

    def actual_vel_omega(self) -> np.ndarray:
        return np.stack([self.column("actual_vx"), self.column("actual_vz")], axis=1)

    def commanded_wheel_speeds(self) -> np.ndarray:
        return np.stack([self.column("ur"), self.column("ul")], axis=1)

    def reconstructed_wheel_speeds(self, wheel_radius: float, base_diameter: float) -> np.ndarray:
        actual = self.actual_vel_omega()
        return _vw_to_simulator_wheels(
            actual[:, 0],
            actual[:, 1],
            wheel_radius=wheel_radius,
            base_diameter=base_diameter,
        )

def load_pololu_traj_control_log(
    path: str | Path,
    *,
    trim_stationary: bool = True,
    start_time: float | None = None,
    stop_time: float | None = None,
) -> PololuTrajControlLog:
    path = Path(path)
    columns, data = _read_csv(path)
    if tuple(columns) != POLOLU_TRAJ_CONTROL_COLUMNS:
        raise ValueError(
            f"Expected Pololu trajectory-control columns {POLOLU_TRAJ_CONTROL_COLUMNS}, got {tuple(columns)}"
        )

    data = _drop_invalid_rows(data)
    if trim_stationary:
        data = _trim_to_motion(data, columns)
    if data.shape[0] < 2:
        raise ValueError(f"Log must contain at least two valid rows after trimming: {path}")

    raw_time_s = _normalized_time_s(data[:, columns.index("ts")])
    data, raw_time_s = _clip_existing_time_window(
        data,
        raw_time_s,
        start_time=start_time,
        stop_time=stop_time,
    )
    time_s = raw_time_s

    data = data.copy()
    data[:, columns.index("ts")] = time_s

    return PololuTrajControlLog(
        columns=tuple(columns),
        data=data.astype(np.float32),
        raw_time_s=raw_time_s.astype(np.float32),
        time_s=time_s.astype(np.float32),
    )


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    has_header = any(ch.isalpha() for ch in first_line)
    if has_header:
        loaded = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        columns = list(loaded.dtype.names or ())
        data = np.column_stack([loaded[name] for name in columns])
    else:
        columns = list(POLOLU_TRAJ_CONTROL_COLUMNS)
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)

    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} columns in {path}, got {data.shape[1]}")
    return columns, data


def _drop_invalid_rows(data: np.ndarray) -> np.ndarray:
    data = data[np.all(np.isfinite(data), axis=1)]
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    unique_indices.sort()
    return data[unique_indices]


def _trim_to_motion(data: np.ndarray, columns: list[str]) -> np.ndarray:
    wheel_cmds = data[:, [columns.index("ul"), columns.index("ur")]]
    actual_pose = data[:, [columns.index("actual_x"), columns.index("actual_y"), columns.index("actual_theta")]]
    active = np.linalg.norm(wheel_cmds, axis=1) > 1e-6
    active |= np.linalg.norm(actual_pose - actual_pose[0], axis=1) > 1e-6
    if not np.any(active):
        return data
    first = int(np.argmax(active))
    last = int(len(active) - np.argmax(active[::-1]))
    return data[first:last]


def _normalized_time_s(timestamp_ms: np.ndarray) -> np.ndarray:
    timestamp_s = np.asarray(timestamp_ms, dtype=np.float64) / 1000.0
    return timestamp_s - timestamp_s[0]


def _clip_existing_time_window(
    data: np.ndarray,
    time_s: np.ndarray,
    *,
    start_time: float | None,
    stop_time: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if start_time is None and stop_time is None:
        return data, time_s

    start = 0.0 if start_time is None else float(start_time)
    duration = float(time_s[-1])
    stop = duration if stop_time is None else min(float(stop_time), duration)

    if start < 0.0:
        raise ValueError("start_time must be non-negative.")
    if start >= duration:
        raise ValueError(f"start_time must be smaller than the log duration ({duration:.6f} s).")
    if stop <= start:
        raise ValueError("stop_time must be greater than start_time.")

    mask = (time_s >= start) & (time_s <= stop)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"Time window [{start:.6f}, {stop:.6f}] s must contain at least two samples."
        )

    clipped_data = data[mask]
    clipped_time_s = time_s[mask]
    rebased_time_s = clipped_time_s - clipped_time_s[0]
    return clipped_data, rebased_time_s


def _vw_to_simulator_wheels(
    v: np.ndarray,
    w: np.ndarray,
    *,
    wheel_radius: float,
    base_diameter: float,
) -> np.ndarray:
    ur = (2.0 * v + base_diameter * w) / (2.0 * wheel_radius)
    ul = (2.0 * v - base_diameter * w) / (2.0 * wheel_radius)
    return np.stack([ur, ul], axis=1)


if __name__ == "__main__":
    from wmr_simulator.visualization.pololu import plot_logged_multipage, plot_logged_trajectory

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="Pololu Data/Logs/10cp_constrained_scurve")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--stop-time", type=float, default=12.0)
    parser.add_argument("--wheel-radius", type=float, default=0.016, help="Wheel radius in meters.")
    parser.add_argument("--base-diameter", type=float, default=0.0842, help="Wheel base diameter in meters.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="visualize")
    parser.add_argument("--dt-histogram", action="store_true", default=True,
                        help="Add a measurement dt histogram page to the multi-page PDF.")
    parser.add_argument("--no-trim-stationary", action="store_true",
                        help="Do not trim stationary leading/trailing rows.")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_prefix = args.output if args.output is not None else f"pololu_{log_path.stem}"

    log = load_pololu_traj_control_log(
        log_path,
        trim_stationary=not args.no_trim_stationary,
        start_time=args.start_time,
        stop_time=args.stop_time,
    )
    print(f"Loaded {log_path}")
    print(f"Samples: {log.time_s.size}")
    print(f"Duration: {float(log.time_s[-1]):.3f} s")
    if log.dt is not None:
        print(f"Median dt: {log.dt:.3f} s")

    plot_logged_trajectory(log, out_prefix=out_prefix, out_dir=args.out_dir)
    plot_logged_multipage(
        log,
        wheel_radius=args.wheel_radius,
        base_diameter=args.base_diameter,
        out_prefix=out_prefix,
        out_dir=args.out_dir,
        include_dt_histogram=args.dt_histogram,
    )
