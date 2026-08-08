import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


# Pose logs run at the wheel dt, so a single rollout carries tens of thousands
# of samples; drawing them all is the slow part of a trajectory figure and is
# invisible at print resolution.
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
        # One window means the replay was never re-anchored: it is a single
        # open-loop integration of the whole log, which drifts for reasons that
        # have nothing to do with a window. Say so rather than calling it
        # "windowed" and marking a "window start" that is just the start.
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
    """Mark where a rollout started *and which way it faced*.

    The start offset is three numbers, and a dot only shows two of them: two
    rollouts starting from the same point with opposite headings are the same
    marker but very different runs, so the heading gets drawn.
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    ax.annotate(
        "",
        xy=(x + length * np.cos(theta), y + length * np.sin(theta)),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9, alpha=alpha, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )


def start_arrow_length(paths) -> float:
    """Arrow length scaled to the figure: 4% of the drawn extent, so the arrows
    stay readable whether the trajectories span 1 m or 10."""
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


def plot_trajectory_set(
    reference_trajectories,
    closed_loop_poses,
    out_prefix="trajectory_set",
    out_path=None,
    title="Optimized Trajectories",
    axis_limits=None,
    annotation=None,
    verbose=True,
):
    """Draw a whole batch of optimized trajectories in one figure: each
    trajectory gets a color, its reference dashed and its closed-loop pose
    solid. Replaces one PDF per trajectory, which is tedious to page through.

    ``closed_loop_poses`` may be ``[N, T, 3]`` (one rollout per trajectory) or
    ``[N, K, T, 3]`` (one per start-pose offset). The second form is what the
    gain-tuning design actually optimizes -- its FIM is an average over those K
    starts -- so drawing the whole family shows the spread the objective saw.

    ``axis_limits`` (``((x_min, x_max), (y_min, y_max))``) pins the view, which
    an animation needs: autoscaled frames make the curves appear to move
    whenever the extent changes. ``annotation`` is a corner text block, and
    ``verbose`` silences the per-file print when this is called once per frame.
    """
    os.makedirs("visualize", exist_ok=True)
    output_filename = out_path if out_path is not None else os.path.join("visualize", f"{out_prefix}.pdf")

    reference_trajectories = np.asarray(reference_trajectories, dtype=float)
    closed_loop_poses = np.asarray(closed_loop_poses, dtype=float)
    if closed_loop_poses.ndim == 3:
        closed_loop_poses = closed_loop_poses[:, None]
    num_trajectories = reference_trajectories.shape[0]
    num_realizations = closed_loop_poses.shape[1]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    arrow_length = start_arrow_length(closed_loop_poses[..., :2])
    fig, ax = plt.subplots(figsize=(8, 8))
    for index in range(num_trajectories):
        color = colors[index % len(colors)]
        reference_path = decimate_path(reference_trajectories[index][:, :2])
        ax.plot(reference_path[:, 0], reference_path[:, 1], color=color, linestyle="--", linewidth=0.8)
        for realization in range(num_realizations):
            actual_path = decimate_path(closed_loop_poses[index, realization][:, :2])
            ax.plot(actual_path[:, 0], actual_path[:, 1], color=color, linestyle="-",
                    linewidth=0.9 if num_realizations == 1 else 0.6,
                    alpha=1.0 if num_realizations == 1 else 0.75)
            # Mark where the robot actually started, which is the offset, and
            # which way it faced -- the heading is a design variable too.
            ax.plot(actual_path[0, 0], actual_path[0, 1], marker=".", color=color,
                    markersize=4, linestyle="none")
            draw_start_pose_arrow(ax, closed_loop_poses[index, realization][0, :3], color, arrow_length)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Reference"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=0.9,
               label="Closed-Loop Actual" if num_realizations == 1
               else f"Closed-Loop Actual ({num_realizations} start offsets)"),
    ]
    # "best" puts the legend top-left, which is where the annotation box goes.
    ax.legend(handles=legend_handles, loc="lower right" if annotation is not None else "best")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title} ({num_trajectories} shown)")
    if axis_limits is not None:
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
    if annotation is not None:
        ax.text(
            0.015, 0.985, annotation, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace", zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.9),
        )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    fig.tight_layout()
    # bbox_inches="tight" crops to the drawn content, which changes the pixel
    # size from frame to frame and makes a GIF jitter; with pinned axes the
    # frames must all be the same canvas.
    fig.savefig(
        output_filename,
        bbox_inches=None if axis_limits is not None else "tight",
        transparent=False,
        facecolor="white",
    )
    plt.close(fig)

    if verbose:
        print(f"Trajectory set PDF saved at: {output_filename}")
    return output_filename


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
