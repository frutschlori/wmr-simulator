"""Histograms of the motion a reference set or a recorded run actually asks for.

The tuning trajectories and the fixed benchmark reference are optimized and
scored on completely different criteria, so the gains that come out of the
tuning set only transfer to the benchmark if the two cover the same part of the
(v, omega, a, alpha) envelope. The 2x2 figure here is that comparison: linear
and angular velocity on the top row, their time derivatives on the bottom, one
overlaid step-histogram per dataset.

Every channel is derived the same way for a designed reference and for a log --
the velocities are read off, the accelerations are ``np.gradient`` of them on
their own time base -- so two figures from different sources are comparable.
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
    """``{v, omega, a, alpha}`` from a forward-speed / yaw-rate pair.

    ``time_s`` is either a scalar sample time or the sample timestamps; a log's
    stream can carry repeated timestamps, and a zero interval would put an
    infinity in the derivative, so those samples are dropped first.
    """
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
    """Channels of one or many designed reference trajectories.

    ``reference_states`` is ``(S, 8)`` or ``(T, S, 8)`` of
    ``[x, y, theta, vx, vy, omega, ax, ay]`` on a uniform ``dt`` grid; a stacked
    set is pooled, each trajectory differentiated on its own before pooling so
    no derivative is taken across the seam between two trajectories.
    """
    states = np.asarray(reference_states, dtype=float)
    if states.ndim == 2:
        states = states[None, ...]
    pooled: dict[str, list[np.ndarray]] = {name: [] for name, _, _, _ in MOTION_CHANNELS}
    for trajectory in states:
        heading = trajectory[:, 2]
        velocity = trajectory[:, 3:5]
        # The reference is nonholonomic, so the velocity lies along the heading
        # and its norm is the forward speed; the projection only supplies the
        # sign (a reversing reference).
        forward = velocity[:, 0] * np.cos(heading) + velocity[:, 1] * np.sin(heading)
        speed = np.hypot(velocity[:, 0], velocity[:, 1]) * np.where(forward < 0.0, -1.0, 1.0)
        channels = motion_channels(dt, speed, trajectory[:, 5])
        for name in pooled:
            pooled[name].append(channels[name])
    return {name: np.concatenate(values) for name, values in pooled.items()}


def reference_motion_channels_from_log(log) -> dict[str, np.ndarray]:
    """Channels of the reference a recorded run was driving.

    A Pololu log's reference carries the commanded forward speed and yaw rate
    (``v_ff``/``w_ff``) rather than a global velocity vector, so the forward
    speed is read straight off instead of being projected.
    """
    states = np.asarray(log.reference.states, dtype=float)
    return motion_channels(np.asarray(log.reference.time_s, dtype=float), states[:, 3], states[:, 5])


def measured_motion_channels_from_log(log) -> dict[str, np.ndarray]:
    """Channels of what a recorded run actually drove.

    Prefers the smoothed mocap body twist (the canonical measured velocity, see
    pololu.measurement_smoothing) and falls back to the firmware's own
    encoder-derived ``(v, omega)`` for logs without a pose twist.
    """
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
    """Shared bin edges over the datasets' robust range.

    The range spans every dataset's 0.5-99.5 percentile, always includes zero
    (so the panels stay centred on standing still) and the motion limit when
    there is one, so the dashed limit line is always in view.
    """
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
    """2x2 of linear/angular velocity and acceleration histograms.

    ``datasets`` is a sequence of ``(label, channels)`` pairs as returned by the
    ``*_motion_channels*`` helpers above; each is drawn as its own step
    histogram, normalized to a density so sets of different length compare.
    ``robot_cfg`` (a problem yaml ``robot`` block) adds the motion limits as
    dashed lines -- the bar every one of these quantities is designed under.
    """
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
                # A diverged run puts a handful of samples decades out; binning
                # over the full range would leave every panel a single spike, so
                # the far tail is piled into the edge bins instead. The
                # percentiles below are of the unclipped values.
                np.clip(value, edges[0], edges[-1]),
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.4,
                color=colors[index % len(colors)],
                # The percentiles are of the *magnitude*: what is being compared
                # is how hard a set drives each channel, not which way it turns.
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
