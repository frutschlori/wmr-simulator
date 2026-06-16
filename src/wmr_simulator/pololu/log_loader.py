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
    "x",
    "y",
    "yaw",
    "x_des",
    "y_des",
    "yaw_des",
    "v_ff",
    "w_ff",
    "v_actual",
    "w_actual",
    "omega_l_cmd",
    "omega_r_cmd",
    "omega_l_meas",
    "omega_r_meas",
    "duty_l",
    "duty_r",
    "x_err",
    "y_err",
    "yaw_err",
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

    def to_target_log(self) -> SimulationLog:
        import jax.numpy as jnp

        pose = self.actual_pose().astype(np.float32)
        vel_omega = self.actual_vel_omega().astype(np.float32)
        wheel_speeds = self.measured_wheel_speeds().astype(np.float32)
        duty_cycle = self.duty_cycles().astype(np.float32)
        wheel_speed_cmd = self.commanded_wheel_speeds().astype(np.float32)

        num_steps = pose.shape[0]
        keys = np.zeros((num_steps, 2), dtype=np.uint32)
        covariance = np.zeros((num_steps, 3, 3), dtype=np.float32)

        robot_states = DiffDriveState(
            pose=jnp.asarray(pose),
            wheel_speeds=jnp.asarray(wheel_speeds),
            key=jnp.asarray(keys),
            vel_omega=jnp.asarray(vel_omega),
            duty_cycle=jnp.asarray(duty_cycle),
            wheel_speed_cmd=jnp.asarray(wheel_speed_cmd),
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
            [self.column("x_des"), self.column("y_des"), self.column("yaw_des")],
            axis=1,
        )

    def actual_pose(self) -> np.ndarray:
        return np.stack([self.column("x"), self.column("y"), self.column("yaw")], axis=1)

    def target_vel_omega(self) -> np.ndarray:
        return np.stack([self.column("v_ff"), self.column("w_ff")], axis=1)

    def target_reference_states(self) -> np.ndarray:
        zeros = np.zeros_like(self.time_s)
        return np.stack(
            [
                self.column("x_des"),
                self.column("y_des"),
                self.column("yaw_des"),
                self.column("v_ff"),
                zeros,
                self.column("w_ff"),
                zeros,
                zeros,
            ],
            axis=1,
        ).astype(np.float32)

    def actual_vel_omega(self) -> np.ndarray:
        pose = self.actual_pose()
        interval_dt = np.diff(self.time_s)
        dx_dt = np.diff(pose[:, 0]) / interval_dt
        dy_dt = np.diff(pose[:, 1]) / interval_dt
        yaw = np.unwrap(pose[:, 2])
        omega = np.diff(yaw) / interval_dt
        linear_speed = dx_dt * np.cos(pose[1:, 2]) + dy_dt * np.sin(pose[1:, 2])
        return np.stack([linear_speed, omega], axis=1)

    def wheel_odometry_vel_omega(self) -> np.ndarray:
        return np.stack([self.column("v_actual"), self.column("w_actual")], axis=1)

    def commanded_wheel_speeds(self) -> np.ndarray:
        return np.stack([self.column("omega_r_cmd"), self.column("omega_l_cmd")], axis=1)

    def measured_wheel_speeds(self) -> np.ndarray:
        return np.stack([self.column("omega_r_meas"), self.column("omega_l_meas")], axis=1)

    def duty_cycles(self) -> np.ndarray:
        return np.stack([self.column("duty_r"), self.column("duty_l")], axis=1)

def load_pololu_traj_control_log(path: str | Path) -> PololuTrajControlLog:
    path = Path(path)
    columns, data = _read_csv(path)
    if tuple(columns) != POLOLU_TRAJ_CONTROL_COLUMNS:
        raise ValueError(
            f"Expected Pololu trajectory-control columns {POLOLU_TRAJ_CONTROL_COLUMNS}, got {tuple(columns)}"
        )

    data = _drop_invalid_rows(data)
    if data.shape[0] < 2:
        raise ValueError(f"Log must contain at least two valid rows: {path}")

    raw_time_s = _normalized_time_s(data[:, columns.index("ts")])
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
    if not any(ch.isalpha() for ch in first_line):
        raise ValueError(f"Expected a header row in {path}. Legacy headerless logs are no longer supported.")

    loaded = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    columns = list(loaded.dtype.names or ())
    data = np.column_stack([loaded[name] for name in columns])

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


def _normalized_time_s(timestamp_ms: np.ndarray) -> np.ndarray:
    timestamp_s = np.asarray(timestamp_ms, dtype=np.float64) / 1000.0
    return timestamp_s - timestamp_s[0]


def _finite_difference(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    edge_order = 2 if values.shape[0] >= 3 else 1
    return np.gradient(values, time_s, edge_order=edge_order)


if __name__ == "__main__":
    from wmr_simulator.visualization.pololu import plot_logged_summary, plot_velocity_difference

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="Pololu Data/Logs/20cp_constrained_scurve/TR47")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="visualize")
    parser.add_argument("--hide-reference-velocity", action="store_true", default=True)
    parser.add_argument("--show-markers", action="store_true", default=True)
    parser.add_argument("--hide-velocity-difference-plot", action="store_true", default=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    out_prefix = args.output if args.output is not None else f"pololu_{log_path.stem}"

    log = load_pololu_traj_control_log(log_path)
    print(f"Loaded {log_path}")
    print(f"Samples: {log.time_s.size}")
    print(f"Duration: {float(log.time_s[-1]):.3f} s")
    if log.dt is not None:
        print(f"Median dt: {log.dt:.3f} s")
    plot_logged_summary(
        log,
        out_prefix=out_prefix,
        out_dir=args.out_dir,
        show_reference_velocity=not args.hide_reference_velocity,
        show_markers=args.show_markers,
    )
    if not args.hide_velocity_difference_plot:
        plot_velocity_difference(
            log,
            out_prefix=out_prefix,
            out_dir=args.out_dir,
            show_markers=args.show_markers,
        )
