from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_logged_trajectory(
    log,
    *,
    out_prefix: str = "pololu_log",
    out_dir: str | Path = "visualize",
    theta_arrow_stride: int = 20,
    theta_arrow_length: float = 0.12,
) -> Path:
    plt, _ = _plot_modules()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}_trajectory.pdf"

    target = log.target_pose()
    actual = log.actual_pose()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(target[:, 0], target[:, 1], "r--", linewidth=1.2, label="Target")
    ax.plot(actual[:, 0], actual[:, 1], "b-", linewidth=1.0, label="Actual")
    _draw_pose_arrows(ax, target, color="red", stride=theta_arrow_stride, length=theta_arrow_length)
    _draw_pose_arrows(ax, actual, color="blue", stride=theta_arrow_stride, length=theta_arrow_length)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Pololu Logged Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Trajectory PDF saved at: {output_path}")
    return output_path


def plot_logged_multipage(
    log,
    *,
    wheel_radius: float,
    base_diameter: float,
    out_prefix: str = "pololu_log",
    out_dir: str | Path = "visualize",
    include_dt_histogram: bool = False,
) -> Path:
    plt, PdfPages = _plot_modules()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    time = log.time_s
    target_pose = log.target_pose()
    actual_pose = log.actual_pose()
    target_vel = log.target_vel_omega()
    actual_vel = log.actual_vel_omega()
    wheel_cmd = log.commanded_wheel_speeds()
    wheel_meas = log.reconstructed_wheel_speeds(
        wheel_radius=wheel_radius,
        base_diameter=base_diameter,
    )

    with PdfPages(output_path) as pdf:
        fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ("x [m]", "y [m]", "theta [rad]")
        names = ("x", "y", "theta")
        for index, ax in enumerate(axes1):
            ax.plot(time, target_pose[:, index], "r--", label=f"Target {names[index]}")
            ax.plot(time, actual_pose[:, index], "b-", label=f"Actual {names[index]}")
            ax.set_ylabel(labels[index])
            ax.grid(True)
            ax.legend()
        axes1[-1].set_xlabel("time [s]")
        fig1.suptitle("State Tracking", fontsize=14)
        fig1.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig1, bbox_inches="tight", transparent=False, facecolor="white")
        plt.close(fig1)

        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes2[0].plot(time, target_vel[:, 0], "r--", label="Target linear velocity")
        axes2[0].plot(time, actual_vel[:, 0], "b-", label="Actual linear velocity")
        axes2[0].set_ylabel("v [m/s]")
        axes2[0].grid(True)
        axes2[0].legend()
        axes2[1].plot(time, target_vel[:, 1], "r--", label="Target angular velocity")
        axes2[1].plot(time, actual_vel[:, 1], "b-", label="Actual angular velocity")
        axes2[1].set_ylabel("omega [rad/s]")
        axes2[1].set_xlabel("time [s]")
        axes2[1].grid(True)
        axes2[1].legend()
        fig2.suptitle("Velocity Tracking", fontsize=14)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig2, bbox_inches="tight", transparent=False, facecolor="white")
        plt.close(fig2)

        fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes3[0].plot(time, wheel_cmd[:, 0], "g-", label="Commanded right wheel")
        axes3[0].plot(time, wheel_meas[:, 0], "g--", label="Reconstructed right wheel")
        axes3[0].set_ylabel("right wheel [rad/s]")
        axes3[0].grid(True)
        axes3[0].legend()
        axes3[1].plot(time, wheel_cmd[:, 1], "m-", label="Commanded left wheel")
        axes3[1].plot(time, wheel_meas[:, 1], "m--", label="Reconstructed left wheel")
        axes3[1].set_ylabel("left wheel [rad/s]")
        axes3[1].set_xlabel("time [s]")
        axes3[1].grid(True)
        axes3[1].legend()
        fig3.suptitle("Wheel Speeds", fontsize=14)
        fig3.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig3, bbox_inches="tight", transparent=False, facecolor="white")
        plt.close(fig3)

        fig4, axes4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes4[0].plot(time, log.column("xerror"), label="x error")
        axes4[0].set_ylabel("x [m]")
        axes4[1].plot(time, log.column("yerror"), label="y error")
        axes4[1].set_ylabel("y [m]")
        axes4[2].plot(time, log.column("thetaerror"), label="theta error")
        axes4[2].set_ylabel("theta [rad]")
        axes4[2].set_xlabel("time [s]")
        for ax in axes4:
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.grid(True)
            ax.legend()
        fig4.suptitle("Logged Tracking Error", fontsize=14)
        fig4.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig4, bbox_inches="tight", transparent=False, facecolor="white")
        plt.close(fig4)

        raw_dt = np.diff(log.raw_time_s)
        if include_dt_histogram and raw_dt.size > 0:
            fig5, ax5 = plt.subplots(1, 1, figsize=(10, 6))
            raw_dt_ms = 1000.0 * raw_dt
            bins = min(50, max(10, int(np.sqrt(raw_dt.size))))
            ax5.hist(raw_dt_ms, bins=bins, color="tab:blue", edgecolor="black", alpha=0.8)
            ax5.axvline(float(np.mean(raw_dt_ms)), color="tab:red", linestyle="--", label="Mean")
            ax5.axvline(float(np.median(raw_dt_ms)), color="tab:green", linestyle="-.", label="Median")
            ax5.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax5.set_xlabel("measured dt [ms]")
            ax5.set_ylabel("count")
            ax5.set_title(
                "Measurement Timing "
                f"(mean={float(np.mean(raw_dt_ms)):.3f} ms, "
                f"median={float(np.median(raw_dt_ms)):.3f} ms, "
                f"std={float(np.std(raw_dt_ms)):.3f} ms)"
            )
            ax5.grid(True)
            ax5.legend()
            fig5.tight_layout()
            pdf.savefig(fig5, bbox_inches="tight", transparent=False, facecolor="white")
            plt.close(fig5)

        metadata = pdf.infodict()
        metadata["Title"] = "Pololu Log"
        metadata["Author"] = "Wheeled Robot Simulator"
        metadata["Subject"] = "Logged trajectory, velocity, and wheel-speed tracking"

    print(f"Multi-page PDF saved at: {output_path}")
    return output_path


def _draw_pose_arrows(
    ax,
    poses: np.ndarray,
    *,
    color: str,
    stride: int,
    length: float,
):
    stride = max(1, int(stride))
    arrow_poses = poses[::stride]
    ax.quiver(
        arrow_poses[:, 0],
        arrow_poses[:, 1],
        length * np.cos(arrow_poses[:, 2]),
        length * np.sin(arrow_poses[:, 2]),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        alpha=0.8,
        width=0.0025,
    )


def _plot_modules():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    return plt, PdfPages
