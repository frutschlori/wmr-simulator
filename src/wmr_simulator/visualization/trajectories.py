import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
    ax.plot(
        closed_loop_actual[:, 0],
        closed_loop_actual[:, 1],
        color="blue",
        linestyle="-",
        linewidth=0.9,
        label="Closed-Loop Actual",
    )
    ax.scatter(
        closed_loop_estimates[:, 0],
        closed_loop_estimates[:, 1],
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
            label = "Windowed Replay Actual" if window_idx == 0 else None
            window_start = closed_loop_estimates[start_idx : start_idx + 1]
            window_actual = np.concatenate([window_start, replay_actual[start_idx + 1 : end_idx + 1]], axis=0)
            ax.plot(
                window_actual[:, 0],
                window_actual[:, 1],
                color="orange",
                linestyle="-",
                linewidth=0.9,
                label=label,
            )
        ax.scatter(
            replay_estimates[:, 0],
            replay_estimates[:, 1],
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
            label="Window Start",
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
    ax.set_ylabel("Normalized Loss")
    ax.set_yscale("log")
    ax.set_title("Loss History")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_filename, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Loss history PDF saved at: {output_filename}")


def save_optimization_trace(
    pipeline,
    snapshots,
    window_length=None,
    out_prefix="traj_opt_trace",
    frame_duration=0.2,
    stop_duration=1.0,
    frames_dir=None,
    gif_path=None,
):
    os.makedirs("visualize", exist_ok=True)
    if frames_dir is None:
        frames_dir = os.path.join("visualize", f"{out_prefix}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for snapshot in snapshots:
        control_points = np.asarray(snapshot.control_points, dtype=float)
        x_points = control_points[:, 0]
        y_points = control_points[:, 1]
        x_mins.append(x_points.min())
        x_maxs.append(x_points.max())
        y_mins.append(y_points.min())
        y_maxs.append(y_points.max())

    global_x_min = min(x_mins)
    global_x_max = max(x_maxs)
    global_y_min = min(y_mins)
    global_y_max = max(y_maxs)

    if global_x_min == global_x_max:
        x_margin = max(0.1, 0.2 * max(abs(global_x_min), 1.0))
    else:
        x_margin = 0.2 * (global_x_max - global_x_min)
    if global_y_min == global_y_max:
        y_margin = max(0.1, 0.2 * max(abs(global_y_min), 1.0))
    else:
        y_margin = 0.2 * (global_y_max - global_y_min)

    axis_limits = (
        (global_x_min - x_margin, global_x_max + x_margin),
        (global_y_min - y_margin, global_y_max + y_margin),
    )

    frame_paths = []
    for snapshot in snapshots:
        frame_path = os.path.join(frames_dir, f"frame_{snapshot.step:05d}.png")
        plot_trajectory(
            pipeline,
            window_length=window_length,
            out_path=frame_path,
            reference_states=snapshot.reference_states,
            control_points=snapshot.control_points,
            closed_loop_log=snapshot.closed_loop_log,
            axis_limits=axis_limits,
            title=f"Trajectory Plot - Iteration {snapshot.step}",
        )
        frame_paths.append(frame_path)

    if gif_path is None:
        gif_path = os.path.join(frames_dir, f"{out_prefix}.gif")
    gif_parent = os.path.dirname(gif_path)
    if gif_parent:
        os.makedirs(gif_parent, exist_ok=True)
    frame_duration_ms = int(1000 * frame_duration)
    stop_duration_ms = int(1000 * stop_duration)

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGBA"))

    durations = [frame_duration_ms] * len(frames)
    durations[-1] = stop_duration_ms

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    for frame in frames:
        frame.close()
    print(f"Optimization trace GIF saved at: {gif_path}")
    return gif_path
