"""Histograms of the motion (v, omega, a, alpha) of a reference set or a
recorded run, one overlaid step histogram per dataset. Accelerations are
``np.gradient`` of the velocities on their own time base for every source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# (key, axis label, unit, robot-config key holding the limit)
MOTION_CHANNELS = (
    ("v", "Linear velocity", "m/s", "v_max"),
    ("omega", "Angular velocity", "rad/s", "omega_max"),
    ("a", "Linear acceleration", "m/s^2", "a_max"),
    ("alpha", "Angular acceleration", "rad/s^2", "alpha_max"),
)


def motion_channels(time_s, v, omega) -> dict[str, np.ndarray]:
    """``{v, omega, a, alpha}`` from a forward-speed / yaw-rate pair;
    ``time_s`` is a sample time or the timestamps (repeated ones are dropped)."""
    v = np.asarray(v, dtype=float).reshape(-1)
    omega = np.asarray(omega, dtype=float).reshape(-1)
    if len(v) == 0:
        return {name: v for name, _, _, _ in MOTION_CHANNELS}
    if np.ndim(time_s) == 0:
        time = float(time_s) * np.arange(len(v), dtype=float)
    else:
        time = np.asarray(time_s, dtype=float).reshape(-1)
        keep = np.concatenate([[True], np.diff(time) > 0.0])
        time, v, omega = time[keep], v[keep], omega[keep]
    if len(v) < 2:
        zeros = np.zeros_like(v)
        return {"v": v, "omega": omega, "a": zeros, "alpha": zeros}
    return {
        "v": v,
        "omega": omega,
        "a": np.gradient(v, time),
        "alpha": np.gradient(omega, time),
    }


def motion_channels_from_reference_states(reference_states, dt: float) -> dict[str, np.ndarray]:
    """Channels of one ``(S, 8)`` or several ``(T, S, 8)`` designed references
    on a uniform ``dt`` grid (each differentiated on its own, then pooled)."""
    states = np.asarray(reference_states, dtype=float)
    if states.ndim == 2:
        states = states[None, ...]
    pooled: dict[str, list[np.ndarray]] = {name: [] for name, _, _, _ in MOTION_CHANNELS}
    for trajectory in states:
        heading = trajectory[:, 2]
        velocity = trajectory[:, 3:5]
        # Forward speed: the velocity norm, signed by its projection on the heading.
        forward = velocity[:, 0] * np.cos(heading) + velocity[:, 1] * np.sin(heading)
        speed = np.hypot(velocity[:, 0], velocity[:, 1]) * np.where(forward < 0.0, -1.0, 1.0)
        channels = motion_channels(dt, speed, trajectory[:, 5])
        for name in pooled:
            pooled[name].append(channels[name])
    return {name: np.concatenate(values) for name, values in pooled.items()}


def reference_motion_channels_from_log(log) -> dict[str, np.ndarray]:
    """Channels of the reference a recorded run was driving (``v_ff``/``w_ff``)."""
    states = np.asarray(log.reference.states, dtype=float)
    return motion_channels(np.asarray(log.reference.time_s, dtype=float), states[:, 3], states[:, 5])


def measured_motion_channels_from_log(log) -> dict[str, np.ndarray]:
    """Channels of what a recorded run drove (smoothed mocap twist, else the
    encoder-derived ``(v, omega)``)."""
    twists = getattr(log.pose, "twists", None)
    if twists is not None and len(twists):
        twists = np.asarray(twists, dtype=float)
        return motion_channels(np.asarray(log.pose.time_s, dtype=float), twists[:, 0], twists[:, 2])
    vel_omega = np.asarray(log.wheel.vel_omega, dtype=float)
    return motion_channels(np.asarray(log.wheel.time_s, dtype=float), vel_omega[:, 0], vel_omega[:, 1])


def pool_motion_channels(channel_sets) -> dict[str, np.ndarray]:
    """Concatenate several channel dicts (the runs of one recording set)."""
    channel_sets = [channels for channels in channel_sets if len(channels["v"])]
    if not channel_sets:
        return {name: np.zeros(0) for name, _, _, _ in MOTION_CHANNELS}
    return {
        name: np.concatenate([channels[name] for channels in channel_sets])
        for name, _, _, _ in MOTION_CHANNELS
    }


def _bin_edges(values, bins: int, limit=None) -> np.ndarray:
    """Shared bin edges over every dataset's 0.5-99.5 percentile range,
    including zero and the motion limit."""
    low = min(float(np.percentile(value, 0.5)) for value in values)
    high = max(float(np.percentile(value, 99.5)) for value in values)
    low, high = min(low, 0.0), max(high, 0.0)
    if limit:
        low, high = min(low, -abs(float(limit))), max(high, abs(float(limit)))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = low - 0.5, high + 0.5
    margin = 0.02 * (high - low)
    return np.linspace(low - margin, high + margin, bins + 1)


def plot_motion_histograms(
    datasets,
    *,
    out_prefix: str = "motion_histograms",
    out_dir: str | Path = "visualize",
    title: str = "Motion distributions",
    robot_cfg: dict | None = None,
    bins: int = 60,
) -> Path | None:
    """2x2 of linear/angular velocity and acceleration histograms of
    ``datasets`` (``(label, channels)`` pairs), as densities; ``robot_cfg``
    adds the motion limits as dashed lines."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    datasets = [(label, channels) for label, channels in datasets if len(channels["v"])]
    if not datasets:
        print(f"No motion samples for {title}; skipping histogram plot.")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{out_prefix}.pdf"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (name, channel_label, unit, limit_key) in zip(axes.ravel(), MOTION_CHANNELS):
        values = [np.asarray(channels[name], dtype=float) for _, channels in datasets]
        limit = None if robot_cfg is None else robot_cfg.get(limit_key)
        edges = _bin_edges(values, bins, limit)
        outside = 0
        for index, ((label, _), value) in enumerate(zip(datasets, values)):
            magnitude = np.abs(value)
            outside += int(np.count_nonzero((value < edges[0]) | (value > edges[-1])))
            axis.hist(
                # Pile the far tail into the edge bins.
                np.clip(value, edges[0], edges[-1]),
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.4,
                color=colors[index % len(colors)],
                # Percentiles of the magnitude.
                label=(
                    f"{label} (|p50| {np.median(magnitude):.2f}, "
                    f"|p95| {np.percentile(magnitude, 95):.2f})"
                ),
            )
        if limit:
            for sign in (-1.0, 1.0):
                bound = sign * float(limit)
                if edges[0] <= bound <= edges[-1]:
                    axis.axvline(bound, color="0.4", linestyle="--", linewidth=1.0)
            axis.plot([], [], color="0.4", linestyle="--", linewidth=1.0, label=f"{limit_key} {float(limit):.2f}")
        xlabel = f"{channel_label} [{unit}]"
        if outside:
            xlabel += f" ({outside} off-axis)"
        axis.set_xlabel(xlabel)
        axis.set_ylabel("density")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Motion histogram PDF saved at: {output_path}")
    return output_path
