from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_logged_summary(
    log,
    *,
    out_prefix: str = "pololu_log",
    out_dir: str | Path = "visualize",
    show_reference_velocity: bool = True,
    show_markers: bool = False,
) -> Path:
    plt = _plot_module()
    marker_kwargs = _line_marker_kwargs(show_markers)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    time = log.time_s
    reference = log.target_pose()
    measured = log.actual_pose()
    reference_vel = log.target_vel_omega()
    measured_vel = log.actual_vel_omega()
    odom_vel = log.wheel_odometry_vel_omega()
    wheel_cmd = log.commanded_wheel_speeds()
    wheel_meas = log.measured_wheel_speeds()
    velocity_len = min(len(time) - 1, len(measured_vel), len(odom_vel) - 1, len(reference_vel) - 1)
    velocity_time = time[1 : velocity_len + 1]
    measured_vel = measured_vel[:velocity_len]
    odom_vel = odom_vel[1 : velocity_len + 1]
    reference_vel = reference_vel[1 : velocity_len + 1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Pololu Log Summary", fontsize=16)

    ax_traj = axes[0, 0]
    ax_traj.plot(
        reference[:, 0],
        reference[:, 1],
        color="tab:red",
        linestyle="--",
        linewidth=1.3,
        label="Reference",
        **marker_kwargs,
    )
    ax_traj.plot(
        measured[:, 0],
        measured[:, 1],
        color="tab:blue",
        linewidth=1.2,
        label="Measured",
        **marker_kwargs,
    )
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax_vel = axes[0, 1]
    ax_vel_omega = ax_vel.twinx()
    line_meas_v = ax_vel.plot(
        velocity_time,
        measured_vel[:, 0],
        color="tab:blue",
        linewidth=1.2,
        label="mocap v",
        **marker_kwargs,
    )[0]
    line_ref_v = None
    if show_reference_velocity:
        line_ref_v = ax_vel.plot(
            velocity_time,
            reference_vel[:, 0],
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            label="ref v",
            **marker_kwargs,
        )[0]
    line_odom_v = ax_vel.plot(
        velocity_time,
        odom_vel[:, 0],
        color="tab:blue",
        linestyle=":",
        linewidth=1.4,
        label="odom v",
        **marker_kwargs,
    )[0]
    line_meas_w = ax_vel_omega.plot(
        velocity_time,
        measured_vel[:, 1],
        color="tab:purple",
        linewidth=1.2,
        label=r"mocap $\omega$",
        **marker_kwargs,
    )[0]
    line_odom_w = ax_vel_omega.plot(
        velocity_time,
        odom_vel[:, 1],
        color="tab:purple",
        linestyle=":",
        linewidth=1.4,
        label=r"odom $\omega$",
        **marker_kwargs,
    )[0]
    line_ref_w = None
    if show_reference_velocity:
        line_ref_w = ax_vel_omega.plot(
            velocity_time,
            reference_vel[:, 1],
            color="tab:purple",
            linestyle="--",
            linewidth=1.2,
            label=r"ref $\omega$",
            **marker_kwargs,
        )[0]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity Tracking")
    ax_vel.grid(True)
    ax_vel.legend(
        handles=[
            handle
            for handle in [line_ref_v, line_meas_v, line_odom_v, line_ref_w, line_meas_w, line_odom_w]
            if handle is not None
        ],
        loc="best",
    )

    ax_wheels = axes[0, 2]
    line_cmd_right = ax_wheels.plot(
        time,
        wheel_cmd[:, 0],
        color="tab:green",
        linestyle="--",
        linewidth=1.1,
        label="cmd right",
        **marker_kwargs,
    )[0]
    line_cmd_left = ax_wheels.plot(
        time,
        wheel_cmd[:, 1],
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="cmd left",
        **marker_kwargs,
    )[0]
    line_meas_right = ax_wheels.plot(
        time,
        wheel_meas[:, 0],
        color="tab:green",
        linewidth=1.2,
        label="meas right",
        **marker_kwargs,
    )[0]
    line_meas_left = ax_wheels.plot(
        time,
        wheel_meas[:, 1],
        color="tab:orange",
        linewidth=1.2,
        label="meas left",
        **marker_kwargs,
    )[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    ax_wheels.legend(handles=[line_cmd_right, line_meas_right, line_cmd_left, line_meas_left])

    state_labels = ("x [m]", "y [m]", "theta [rad]")
    state_names = ("x", "y", "theta")
    for index, ax in enumerate(axes[1]):
        ax.plot(time, reference[:, index], "r--", linewidth=1.3, label="Reference", **marker_kwargs)
        ax.plot(time, measured[:, index], color="tab:blue", linewidth=1.2, label="Measured", **marker_kwargs)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(state_labels[index])
        ax.set_title(f"{state_names[index]} State")
        ax.grid(True)
        ax.legend()

    if show_markers:
        _set_line_widths(fig, 1.0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Log summary PDF saved at: {output_path}")
    return output_path

def _plot_module():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _line_marker_kwargs(show_markers: bool) -> dict[str, object]:
    if not show_markers:
        return {}
    return {
        "marker": "x",
        "markersize": 3.2,
        "markeredgewidth": 0.8,
    }


def _set_line_widths(fig, linewidth: float) -> None:
    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(linewidth)
