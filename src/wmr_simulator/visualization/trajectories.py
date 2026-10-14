import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np


# Maximum number of points drawn per path.
PLOT_PATH_MAX_POINTS = 1500


def decimate_path(points, max_points: int | None = PLOT_PATH_MAX_POINTS):
    """Stride a densely sampled path down to at most ``max_points`` rows,
    always keeping the final sample so the drawn path ends where it should."""
    points = np.asarray(points)
    if max_points is None or len(points) <= max_points:
        return points
    stride = int(np.ceil(len(points) / max_points))
    decimated = points[::stride]
    if not np.array_equal(decimated[-1], points[-1]):
        decimated = np.concatenate([decimated, points[-1:]], axis=0)
    return decimated


def _plot_trajectory_axes(
    ax,
    pipeline,
    reference_states,
    control_points,
    closed_loop_log,
    replay_actual,
    replay_estimates,
    window_length,
    axis_limits=None,
    title="Trajectory Plot",
):
    closed_loop_actual = np.asarray(closed_loop_log.pose.true_states)
    closed_loop_estimates = np.asarray(closed_loop_log.pose.states)
    show_replay = replay_actual is not None and replay_estimates is not None
    if show_replay:
        replay_actual = np.asarray(replay_actual)
        replay_estimates = np.asarray(replay_estimates)
        num_replay_intervals = max(len(replay_estimates) - 1, 1)
        resolved_window_length = pipeline.simulation.resolve_replay_window_length(window_length, num_replay_intervals)
        window_start_indices = np.arange(0, num_replay_intervals, resolved_window_length)
        # A single window is an open-loop replay of the whole log.
        is_windowed_replay = len(window_start_indices) > 1
    if axis_limits is not None:
        x_limits, y_limits = axis_limits
    elif control_points is not None:
        x_points = np.asarray(control_points[:, 0], dtype=float)
        y_points = np.asarray(control_points[:, 1], dtype=float)
        x_min = x_points.min()
        x_max = x_points.max()
        y_min = y_points.min()
        y_max = y_points.max()
        if x_min == x_max:
            x_margin = max(0.1, 0.2 * max(abs(x_min), 1.0))
        else:
            x_margin = 0.2 * (x_max - x_min)
        if y_min == y_max:
            y_margin = max(0.1, 0.2 * max(abs(y_min), 1.0))
        else:
            y_margin = 0.2 * (y_max - y_min)
        x_limits = (x_min - x_margin, x_max + x_margin)
        y_limits = (y_min - y_margin, y_max + y_margin)
    else:
        x_limits = None
        y_limits = None

    ax.plot(
        reference_states[:, 0],
        reference_states[:, 1],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Reference",
    )
    if control_points is not None:
        ax.plot(
            control_points[:, 0],
            control_points[:, 1],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        ax.scatter(
            control_points[:, 0],
            control_points[:, 1],
            marker="o",
            s=36,
            facecolors="none",
            edgecolors="black",
            linewidth=1.0,
        )
    actual_path = decimate_path(closed_loop_actual)
    estimate_path = decimate_path(closed_loop_estimates)
    ax.plot(
        actual_path[:, 0],
        actual_path[:, 1],
        color="blue",
        linestyle="-",
        linewidth=0.9,
        label="Closed-Loop Actual",
    )
    ax.scatter(
        estimate_path[:, 0],
        estimate_path[:, 1],
        color="blue",
        marker=".",
        s=3,
        alpha=1,
        linewidth=0.0,
    )
    if show_replay:
        for window_idx, start_idx in enumerate(window_start_indices):
            end_idx = min(start_idx + resolved_window_length, num_replay_intervals)
            if end_idx <= start_idx:
                continue
            if window_idx == 0:
                label = "Windowed Replay Actual" if is_windowed_replay else "Open-Loop Replay Actual (full sequence)"
            else:
                label = None
            window_start = closed_loop_estimates[start_idx : start_idx + 1]
            window_actual = np.concatenate([window_start, replay_actual[start_idx + 1 : end_idx + 1]], axis=0)
            window_actual = decimate_path(window_actual)
            ax.plot(
                window_actual[:, 0],
                window_actual[:, 1],
                color="orange",
                linestyle="-",
                linewidth=0.9,
                label=label,
            )
        replay_estimate_path = decimate_path(replay_estimates)
        ax.scatter(
            replay_estimate_path[:, 0],
            replay_estimate_path[:, 1],
            color="orange",
            s=3,
            marker=".",
            alpha=1,
            linewidth=0.0,
        )
        ax.scatter(
            closed_loop_estimates[window_start_indices, 0],
            closed_loop_estimates[window_start_indices, 1],
            marker="x",
            s=36,
            linewidths=1.0,
            color="black",
            label="Window Start" if is_windowed_replay else "Replay Start",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if x_limits is not None and y_limits is not None:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
    ax.grid(True)
    ax.legend()


def plot_trajectory(
    pipeline,
    window_length=None,
    out_prefix="trajectory_plot",
    out_path=None,
    reference_states=None,
    control_points=None,
    closed_loop_log=None,
    axis_limits=None,
    title="Trajectory Plot",
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    if reference_states is None:
        reference_states = np.asarray(pipeline.reference_states)
    else:
        reference_states = np.asarray(reference_states)
    if control_points is None:
        control_points = pipeline.current_control_points()
    else:
        control_points = np.asarray(control_points)
    if closed_loop_log is None:
        closed_loop_log = pipeline.closed_loop_log

    is_gain_tuning = getattr(pipeline, "objective_mode", None) == "gain-tuning"
    if is_gain_tuning:
        replay_actual = None
        replay_estimates = None
    else:
        replay_actual, replay_estimates = pipeline.replay_rollout(
            pipeline.nominal_parameters(),
            window_length=window_length,
            closed_loop_log=closed_loop_log,
        )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _plot_trajectory_axes(
        ax,
        pipeline,
        reference_states,
        control_points,
        closed_loop_log,
        replay_actual,
        replay_estimates,
        window_length,
        axis_limits=axis_limits,
        title=title,
    )
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)

    print(f"Trajectory PDF saved at: {output_filename}")


def draw_start_pose_arrow(ax, pose, color, length: float, alpha: float = 0.9) -> None:
    """Mark where a rollout started and which way it faced."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    ax.annotate(
        "",
        xy=(x + length * np.cos(theta), y + length * np.sin(theta)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9, alpha=alpha, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )


def start_arrow_length(paths) -> float:
    """Arrow length: 4% of the drawn extent."""
    paths = np.asarray(paths, dtype=float).reshape(-1, np.asarray(paths).shape[-1])
    if paths.size == 0:
        return 0.05
    extent = float(
        max(
            np.nanmax(paths[:, 0]) - np.nanmin(paths[:, 0]),
            np.nanmax(paths[:, 1]) - np.nanmin(paths[:, 1]),
        )
    )
    return max(0.04 * extent, 1e-3)


def plot_reference_set_overview(
    references,
    dt: float,
    out_path,
    title: str,
    environment=None,
    limits: dict | None = None,
):
    """One reference set at a glance: every path in the environment box, and
    every sample in the (v, |omega|) plane against the motion limits.

    ``references`` is a ``{name: [N, 8] reference states}`` mapping (differing
    lengths are fine). ``environment`` is ``(min, max)`` corners of the box,
    ``limits`` a fixed_sets.motion_limits dict; the lateral limit is the
    hyperbola ``v * |omega| = a_max_lateral`` and the total acceleration caps
    the same product at ``a_max`` where the speed peaks.
    """
    references = {name: np.asarray(states, dtype=float) for name, states in references.items()}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    many = len(references) > 20
    fig, (path_axis, twist_axis) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1.2]})

    all_paths = np.concatenate([states[:, :2] for states in references.values()], axis=0)
    arrow_length = start_arrow_length(all_paths)
    for index, states in enumerate(references.values()):
        color = colors[index % len(colors)]
        path_axis.plot(states[:, 0], states[:, 1], color=color, linewidth=0.5 if many else 1.0,
                       alpha=0.5 if many else 1.0)
        draw_start_pose_arrow(path_axis, states[0, :3], color, arrow_length, alpha=0.5 if many else 0.9)
        speed = np.hypot(states[:, 3], states[:, 4])
        twist_axis.plot(speed, np.abs(states[:, 5]), linestyle="none", marker=".", markersize=1.5,
                        color=color, alpha=0.4 if many else 0.7)
    if environment is not None:
        (x_min, y_min), (x_max, y_max) = environment
        path_axis.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
                       color="0.4", linestyle="--", linewidth=1.0)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.grid(True, alpha=0.3)
    path_axis.set_title(f"Paths ({len(references)} references, arrows at the start)")

    if limits is not None:
        v_grid = np.linspace(1e-3, limits["v_max"], 200)
        twist_axis.axvline(limits["v_max"], color="0.4", linestyle="--", linewidth=1.0)
        twist_axis.axhline(limits["omega_max"], color="0.4", linestyle="--", linewidth=1.0)
        twist_axis.plot(v_grid, limits["a_max_lateral"] / v_grid, color="0.4", linestyle=":", linewidth=1.0,
                        label=f"v|omega| = a_max_lateral {limits['a_max_lateral']:.1f}")
        twist_axis.plot(v_grid, limits["a_max"] / v_grid, color="0.4", linestyle="-.", linewidth=1.0,
                        label=f"v|omega| = a_max {limits['a_max']:.1f}")
        twist_axis.set_xlim(0.0, 1.05 * limits["v_max"])
        twist_axis.set_ylim(0.0, 1.05 * limits["omega_max"])
        twist_axis.legend(loc="upper right", fontsize=8)
    twist_axis.set_xlabel("v [m/s]")
    twist_axis.set_ylabel("|omega| [rad/s]")
    twist_axis.grid(True, alpha=0.3)
    twist_axis.set_title(f"Samples every {dt:g} s")

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Reference set overview saved at: {out_path}")
    return out_path


def plot_loss_history(
    loss_history,
    out_prefix="loss_history",
    out_path=None,
):
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    loss_history = np.asarray(loss_history, dtype=float)
    steps = np.arange(1, len(loss_history) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(steps, loss_history, color="C0", linewidth=1.2)
    ax.set_xlabel("Optimization Step")
    # log(A-optimality) + penalty: legitimately negative, so no log y-scale.
    ax.set_ylabel("Objective")
    ax.set_title("Loss History")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Loss history PDF saved at: {output_filename}")
