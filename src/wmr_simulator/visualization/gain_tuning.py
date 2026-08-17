import os

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# The summary figures reproduce the objective's noise keys exactly, so they take
# the namespaces from the objective rather than restating them.
from wmr_simulator.gain_tuning.objectives import (
    TRAINING_KEY_NAMESPACE,
    VALIDATION_KEY_NAMESPACE,
)
from wmr_simulator.visualization.trajectories import (
    decimate_path,
    draw_start_pose_arrow,
    start_arrow_length,
)


def realization_keys_for_set(realizations, num_trajectories, key_namespace):
    """The ``(T, R, 2)`` noise-key pairs the objective scored a trajectory set
    under, as ``(robot_keys, estimator_keys)``.

    Exactly the split ``optimizers._make_terms_for_values`` does, so a summary
    figure draws the rollouts that were actually scored rather than a fresh
    draw: every (trajectory, realization) pair gets its own key, and
    ``key_namespace`` keeps the training set (0) independent of the validation
    set (1).
    """
    from wmr_simulator.gain_tuning.objectives import split_realization_keys_by_trajectory

    return (
        split_realization_keys_by_trajectory(
            realizations.robot_keys, num_trajectories, namespace=key_namespace
        ),
        split_realization_keys_by_trajectory(
            realizations.estimator_keys, num_trajectories, namespace=key_namespace
        ),
    )


def rollout_realizations(
    pipeline,
    robot_params,
    reference_trajectories,
    start_offsets,
    robot_keys,
    estimator_keys,
    controller_gains=None,
    schedule_params=None,
):
    """Roll out every (trajectory, start offset) pair: ``(T, R, S, 3)`` poses.

    ``robot_keys`` / ``estimator_keys`` are the ``(T, R, 2)`` noise draws the
    objective scored these rollouts under (see ``realization_keys_for_set``).
    They are required: letting ``run_closed_loop`` fall back to the pipeline's
    single ``target_*_key`` puts the whole family on *one* noise realization,
    so the figure shows a draw the tuner never scored and the realizations
    differ only in their start pose. Measured on an active-learning tuning set
    at the stock gains, one shared key against the objective's own keys: pose
    RMSE 0.0352 vs 0.0386 m, max deviation 0.103 vs 0.161 m, max heading error
    0.285 vs 0.476 rad -- i.e. the shared-key figure reads as a controller that
    tracks visibly better than the one being tuned.

    The tuning objective is an average over the start offsets, so a figure that
    draws one rollout per trajectory shows a single sample of what was scored.
    Rolling out the whole family under one jitted double vmap costs one compile
    instead of T*R eager ones.
    """
    import jax
    import jax.numpy as jnp

    reference_trajectories = jnp.asarray(reference_trajectories, dtype=jnp.float32)
    start_offsets = jnp.asarray(start_offsets, dtype=jnp.float32)
    robot_keys = jnp.asarray(robot_keys)
    estimator_keys = jnp.asarray(estimator_keys)

    def rollout(reference_states, start_offset, robot_key, estimator_key):
        return pipeline.run_closed_loop(
            robot_params,
            use_hidden_robot=True,
            controller_gains=controller_gains,
            schedule_params=schedule_params,
            robot_key=robot_key,
            estimator_key=estimator_key,
            reference_states=reference_states,
            initial_pose=reference_states[0, :3] + start_offset,
        ).pose.true_states

    batched = jax.jit(
        jax.vmap(jax.vmap(rollout, in_axes=(None, 0, 0, 0)), in_axes=(0, 0, 0, 0))
    )
    return np.asarray(
        batched(reference_trajectories, start_offsets, robot_keys, estimator_keys), dtype=float
    )


def plot_gain_tuning_summary(
    pipeline,
    init_log,
    tuned_log,
    static_log=None,
    init_realization_poses=None,
    tuned_realization_poses=None,
    static_realization_poses=None,
    out_prefix="gain_tuning_summary",
):
    """The one-run summary. The logs are realization 0 of trajectory 0, started
    at *its* offset rather than on the reference; the optional
    ``*_realization_poses`` ``(R, S, 3)`` batches add the remaining realizations
    to the trajectory panel, which is the spread the loss was averaged over."""
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize", f"{out_prefix}.pdf")

    # When a static-tune rollout is supplied, the main tuned line is the
    # parametrized (MLP) result; label it "tuned (param)" and overlay the
    # static-gain rollout as "tuned (static)". Without one, keep the plain "Tuned".
    tuned_label = "tuned (param)" if static_log is not None else "Tuned"
    static_color = "tab:green"

    reference_time = np.asarray(tuned_log.reference.time_s, dtype=float)
    reference = np.asarray(tuned_log.reference.states, dtype=float)
    init_time = np.asarray(init_log.pose.time_s, dtype=float)
    init_pose = np.asarray(init_log.pose.true_states, dtype=float)
    tuned_time = np.asarray(tuned_log.pose.time_s, dtype=float)
    tuned_pose = np.asarray(tuned_log.pose.true_states, dtype=float)
    static_time = None if static_log is None else np.asarray(static_log.pose.time_s, dtype=float)
    static_pose = None if static_log is None else np.asarray(static_log.pose.true_states, dtype=float)
    wheel_time = np.asarray(tuned_log.wheel.time_s, dtype=float)
    wheel_speeds = np.asarray(tuned_log.wheel.speeds, dtype=float)
    duty_cycle = np.asarray(tuned_log.wheel.duty_cycle, dtype=float)
    command_time = np.asarray(tuned_log.pose.command_time_s, dtype=float)
    wheel_cmd = np.asarray(tuned_log.pose.wheel_cmd, dtype=float)

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f"Gain Tuning Summary ({out_prefix})", fontsize=16)
    # The trajectory panel gets two columns: it now carries every realization
    # and its start arrow, which is unreadable in one.
    ax_traj = plt.subplot2grid((2, 12), (0, 0), colspan=2, fig=fig)
    ax_vel = plt.subplot2grid((2, 12), (0, 2), colspan=4, fig=fig)
    ax_wheels = plt.subplot2grid((2, 12), (0, 6), colspan=3, fig=fig)
    ax_motor = plt.subplot2grid((2, 12), (0, 9), colspan=3, fig=fig)
    state_axes = [
        plt.subplot2grid((2, 12), (1, 0), colspan=4, fig=fig),
        plt.subplot2grid((2, 12), (1, 4), colspan=4, fig=fig),
        plt.subplot2grid((2, 12), (1, 8), colspan=4, fig=fig),
    ]

    ax_traj.plot(reference[:, 0], reference[:, 1], color="tab:red", linestyle="--", linewidth=1.0, label="Reference")
    # The remaining realizations first, so the highlighted run stays on top.
    realization_sets = (
        (init_realization_poses, "tab:blue"),
        (static_realization_poses, static_color),
        (tuned_realization_poses, "tab:orange"),
    )
    drawn = [poses for poses, _ in realization_sets if poses is not None]
    if drawn:
        for poses, color in realization_sets:
            if poses is None:
                continue
            for realization_poses in np.asarray(poses, dtype=float):
                ax_traj.plot(
                    realization_poses[:, 0], realization_poses[:, 1],
                    color=color, linewidth=0.6, alpha=0.45,
                )
    ax_traj.plot(init_pose[:, 0], init_pose[:, 1], color="tab:blue", linewidth=1.0, label="Initial")
    if static_pose is not None:
        ax_traj.plot(static_pose[:, 0], static_pose[:, 1], color=static_color, linewidth=1.0, label="tuned (static)")
    ax_traj.plot(tuned_pose[:, 0], tuned_pose[:, 1], color="tab:orange", linewidth=1.2, label=tuned_label)
    # Only the highlighted run gets an arrow: the panel's time series are that
    # one realization, and an arrow per realization reads as several starts for
    # a single plotted run.
    draw_start_pose_arrow(
        ax_traj,
        tuned_pose[0, :3],
        "tab:orange",
        start_arrow_length(np.concatenate([init_pose, tuned_pose], axis=0)[:, :2]),
    )
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.set_title("Trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True)
    ax_traj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize="small")

    ax_vel_omega = ax_vel.twinx()
    reference_v = reference[:, 3] * np.cos(reference[:, 2]) + reference[:, 4] * np.sin(reference[:, 2])
    line_ref_v = ax_vel.step(
        reference_time,
        reference_v,
        where="post",
        color="tab:cyan",
        linestyle="--",
        linewidth=0.8,
        label="ref v",
    )[0]
    line_ref_w = ax_vel_omega.step(
        reference_time,
        reference[:, 5],
        where="post",
        color="tab:pink",
        linestyle="--",
        linewidth=0.8,
        label=r"ref $\omega$",
    )[0]
    velocity_prefix = "param" if static_log is not None else "tuned"
    tuned_vel_time, tuned_vel = _pose_vel_omega(tuned_time, tuned_pose)
    line_tuned_v = ax_vel.plot(tuned_vel_time, tuned_vel[:, 0], color="tab:blue", linewidth=1.0, label=f"{velocity_prefix} v")[0]
    line_tuned_w = ax_vel_omega.plot(
        tuned_vel_time,
        tuned_vel[:, 1],
        color="tab:purple",
        linewidth=1.0,
        label=rf"{velocity_prefix} $\omega$",
    )[0]
    velocity_handles = [line_ref_v, line_tuned_v, line_ref_w, line_tuned_w]
    if static_pose is not None:
        static_vel_time, static_vel = _pose_vel_omega(static_time, static_pose)
        line_static_v = ax_vel.plot(
            static_vel_time, static_vel[:, 0], color=static_color, linewidth=1.0, label="static v"
        )[0]
        line_static_w = ax_vel_omega.plot(
            static_vel_time, static_vel[:, 1], color="tab:olive", linewidth=1.0, label=r"static $\omega$"
        )[0]
        velocity_handles = [line_ref_v, line_tuned_v, line_static_v, line_ref_w, line_tuned_w, line_static_w]
    ax_vel.set_xlabel("time [s]")
    ax_vel.set_ylabel("linear velocity [m/s]")
    ax_vel_omega.set_ylabel("angular velocity [rad/s]")
    ax_vel.set_title("Velocity")
    ax_vel.grid(True)
    ax_vel.legend(handles=velocity_handles, loc="best")

    cmd_time, cmd_right = _stair_series(command_time, wheel_cmd[:, 0], wheel_time[-1] if len(wheel_time) else None)
    _, cmd_left = _stair_series(command_time, wheel_cmd[:, 1], wheel_time[-1] if len(wheel_time) else None)
    line_cmd_right = ax_wheels.step(
        cmd_time,
        cmd_right,
        where="post",
        color="tab:green",
        linestyle="--",
        linewidth=0.7,
        label="cmd right",
    )[0]
    line_meas_right = ax_wheels.plot(
        wheel_time,
        wheel_speeds[:, 0],
        color="tab:green",
        linewidth=0.9,
        label="meas right",
    )[0]
    line_cmd_left = ax_wheels.step(
        cmd_time,
        cmd_left,
        where="post",
        color="tab:orange",
        linestyle="--",
        linewidth=0.7,
        label="cmd left",
    )[0]
    line_meas_left = ax_wheels.plot(
        wheel_time,
        wheel_speeds[:, 1],
        color="tab:orange",
        linewidth=0.9,
        label="meas left",
    )[0]
    ax_wheels.set_xlabel("time [s]")
    ax_wheels.set_ylabel("wheel speed [rad/s]")
    ax_wheels.set_title("Wheel Speeds")
    ax_wheels.grid(True)
    wheel_legend_handles = [line_cmd_right, line_meas_right, line_cmd_left, line_meas_left]

    # Applied controller gains along the tuned rollout (time-varying under a
    # gain parametrization): motor PI gains on the wheel-speeds subplot, the
    # outer gains on their state subplots below.
    tuned_gains_log = None if tuned_log.pose.gains is None else np.asarray(tuned_log.pose.gains, dtype=float)
    if tuned_gains_log is not None:
        ax_wheel_gains = ax_wheels.twinx()
        for column, label, style in ((3, r"$k_p$ motor", "-"), (4, r"$k_i$ motor", "--")):
            wheel_legend_handles.append(
                ax_wheel_gains.step(
                    command_time,
                    tuned_gains_log[:, column],
                    where="post",
                    color="black",
                    linestyle=style,
                    linewidth=0.8,
                    label=label,
                )[0]
            )
        ax_wheel_gains.set_ylabel("motor gains [-]")
    ax_wheels.legend(handles=wheel_legend_handles)

    _plot_tuned_motor_axes(ax_motor, wheel_time, duty_cycle, wheel_speeds)

    state_labels = ("x [m]", "y [m]", "theta [rad]")
    state_names = ("x", "y", "theta")
    state_gain_labels = (r"$k_x$", r"$k_y$", r"$k_\theta$")
    for index, ax in enumerate(state_axes):
        ax.step(
            reference_time,
            reference[:, index],
            where="post",
            color="tab:red",
            linestyle="--",
            linewidth=0.9,
            label="Reference",
        )
        ax.plot(init_time, init_pose[:, index], color="tab:blue", linewidth=1.0, label="Initial")
        if static_pose is not None:
            ax.plot(static_time, static_pose[:, index], color=static_color, linewidth=1.0, label="tuned (static)")
        ax.plot(tuned_time, tuned_pose[:, index], color="tab:orange", linewidth=1.2, label=tuned_label)
        legend_handles = ax.get_lines()[:]
        if tuned_gains_log is not None:
            gain_ax = ax.twinx()
            legend_handles.append(
                gain_ax.step(
                    command_time,
                    tuned_gains_log[:, index],
                    where="post",
                    color="black",
                    linewidth=0.8,
                    label=state_gain_labels[index],
                )[0]
            )
            gain_ax.set_ylabel(f"{state_gain_labels[index]} [-]")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(state_labels[index])
        ax.set_title(f"{state_names[index]} State")
        ax.grid(True)
        ax.legend(handles=legend_handles)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf_filename, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Gain tuning summary PDF saved at: {pdf_filename}")


def plot_trajectory_set_summary(
    pipeline,
    robot_params,
    tuned_gains,
    reference_trajectories,
    start_offsets,
    realizations,
    key_namespace,
    schedule_params=None,
    static_gains=None,
    max_trajectories: int | None = None,
    title: str = "Trajectories",
    out_prefix="trajectory_summary",
):
    """Every trajectory rolled out under *every* realization it was tuned on --
    its own start offset (``start_offsets`` is (T, R, 3)) *and* its own noise
    draw (from ``realizations`` + ``key_namespace``) -- initial gains against
    tuned.

    Drawing one rollout per trajectory would show one draw out of the R the loss
    averages over, and would start it on the reference -- where the tracking
    gains have almost nothing to act on and the run is not the one that was
    scored."""
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize", f"{out_prefix}.pdf")

    reference_trajectories = np.asarray(reference_trajectories, dtype=float)
    if reference_trajectories.shape[0] == 0:
        print(f"No trajectories available for {title}; skipping plot.")
        return None
    num_trajectories = reference_trajectories.shape[0]
    if max_trajectories is not None:
        num_trajectories = min(max_trajectories, num_trajectories)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # The main tuned rollout applies the parametrization (MLP) schedule; when a
    # static-tune result is supplied it is overlaid (dash-dot) as
    # "tuned (static)" and the main line is relabeled "tuned (param)".
    tuned_label = "tuned (param)" if static_gains is not None else "Tuned"

    # All rollouts for the figure run as one vmapped, jitted batch per gain set
    # -- eager per-trajectory calls pay a fresh XLA compile each.
    plotted_references = reference_trajectories[:num_trajectories]
    plotted_offsets = np.asarray(start_offsets, dtype=float)[:num_trajectories]
    # Split over the whole set before slicing: the keys a trajectory was scored
    # under depend on how many trajectories the set has, so a `max_trajectories`
    # figure has to take the first few of the full split, not a fresh split of
    # the first few.
    robot_keys, estimator_keys = realization_keys_for_set(
        realizations, reference_trajectories.shape[0], key_namespace
    )
    plotted_robot_keys = robot_keys[:num_trajectories]
    plotted_estimator_keys = estimator_keys[:num_trajectories]
    init_poses = rollout_realizations(
        pipeline,
        robot_params,
        plotted_references,
        plotted_offsets,
        plotted_robot_keys,
        plotted_estimator_keys,
    )
    tuned_poses = rollout_realizations(
        pipeline,
        robot_params,
        plotted_references,
        plotted_offsets,
        plotted_robot_keys,
        plotted_estimator_keys,
        controller_gains=tuned_gains,
        schedule_params=schedule_params,
    )
    static_poses = (
        None
        if static_gains is None
        else rollout_realizations(
            pipeline,
            robot_params,
            plotted_references,
            plotted_offsets,
            plotted_robot_keys,
            plotted_estimator_keys,
            controller_gains=static_gains,
        )
    )
    num_realizations = init_poses.shape[1]
    arrow_length = start_arrow_length(tuned_poses[..., :2])
    # Same rollout weight as the trajectory designer's batch summary
    # (visualization.trajectories.plot_trajectory_set): with several
    # realizations per trajectory the lines overlap, and thin, slightly
    # transparent strokes read far better than solid ones.
    rollout_width = 0.9 if num_realizations == 1 else 0.6
    rollout_alpha = 1.0 if num_realizations == 1 else 0.75

    fig, ax = plt.subplots(figsize=(8, 8))
    for index, reference_states in enumerate(plotted_references):
        color = colors[index % len(colors)]
        reference_path = decimate_path(reference_states[:, :2])
        ax.plot(reference_path[:, 0], reference_path[:, 1], color=color, linestyle="--", linewidth=0.8)
        for realization in range(num_realizations):
            init_pose = decimate_path(init_poses[index, realization])
            tuned_pose = decimate_path(tuned_poses[index, realization])
            ax.plot(init_pose[:, 0], init_pose[:, 1], color=color, linestyle=":",
                    linewidth=rollout_width, alpha=rollout_alpha)
            if static_poses is not None:
                static_pose = decimate_path(static_poses[index, realization])
                ax.plot(static_pose[:, 0], static_pose[:, 1], color=color, linestyle="-.",
                        linewidth=rollout_width, alpha=rollout_alpha)
            ax.plot(tuned_pose[:, 0], tuned_pose[:, 1], color=color, linestyle="-",
                    linewidth=rollout_width, alpha=rollout_alpha)
            # The start pose is shared by all three rollouts of this realization.
            draw_start_pose_arrow(ax, tuned_poses[index, realization][0, :3], color, arrow_length)

    legend_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Reference"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=rollout_width, label="Initial"),
    ]
    if static_gains is not None:
        legend_handles.append(
            Line2D([0], [0], color="black", linestyle="-.", linewidth=rollout_width, label="tuned (static)")
        )
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="-", linewidth=rollout_width, label=tuned_label)
    )
    ax.legend(handles=legend_handles, loc="best")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title} ({num_trajectories} shown, {num_realizations} start offsets each)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(pdf_filename, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"{title} summary PDF saved at: {pdf_filename}")
    return pdf_filename


def plot_training_trajectory_summary(
    pipeline,
    robot_params,
    tuned_gains,
    start_offsets,
    realizations,
    schedule_params=None,
    static_gains=None,
    max_trajectories: int | None = 5,
    out_prefix="summary_training",
):
    return plot_trajectory_set_summary(
        pipeline=pipeline,
        robot_params=robot_params,
        tuned_gains=tuned_gains,
        reference_trajectories=pipeline.training_reference_trajectories,
        start_offsets=start_offsets,
        realizations=realizations,
        key_namespace=TRAINING_KEY_NAMESPACE,
        schedule_params=schedule_params,
        static_gains=static_gains,
        max_trajectories=max_trajectories,
        title="Training Trajectories",
        out_prefix=out_prefix,
    )


def plot_validation_trajectory_summary(
    pipeline,
    robot_params,
    tuned_gains,
    start_offsets,
    realizations,
    schedule_params=None,
    static_gains=None,
    out_prefix="summary_validation",
):
    return plot_trajectory_set_summary(
        pipeline=pipeline,
        robot_params=robot_params,
        tuned_gains=tuned_gains,
        reference_trajectories=pipeline.validation_reference_trajectories,
        start_offsets=start_offsets,
        realizations=realizations,
        key_namespace=VALIDATION_KEY_NAMESPACE,
        schedule_params=schedule_params,
        static_gains=static_gains,
        max_trajectories=None,
        title="Validation Trajectories",
        out_prefix=out_prefix,
    )


def plot_controller_tuning_errors(pipeline, init_log, tuned_log, static_log=None, out_prefix="ctrl_tuning"):
    os.makedirs("visualize", exist_ok=True)
    pdf_filename = os.path.join("visualize", f"{out_prefix}_tracking_errors.pdf")

    reference_poses = np.asarray(pipeline.reference_states[:, :3])
    reference_pose_indices = np.arange(
        0,
        len(reference_poses) * pipeline.inner_steps_per_geometry_step,
        pipeline.inner_steps_per_geometry_step,
        dtype=int,
    )
    init_poses = np.asarray(init_log.pose.states)[reference_pose_indices]
    tuned_poses = np.asarray(tuned_log.pose.states)[reference_pose_indices]
    lengths = [len(reference_poses), len(init_poses), len(tuned_poses), len(pipeline.reference_time_grid)]
    static_errors = None
    if static_log is not None:
        static_poses = np.asarray(static_log.pose.states)[reference_pose_indices]
        lengths.append(len(static_poses))
    plot_len = min(lengths)
    plot_time = np.asarray(pipeline.reference_time_grid[:plot_len])

    init_errors = init_poses[:plot_len] - reference_poses[:plot_len]
    tuned_errors = tuned_poses[:plot_len] - reference_poses[:plot_len]
    init_errors[:, 2] = (init_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    tuned_errors[:, 2] = (tuned_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    if static_log is not None:
        static_errors = static_poses[:plot_len] - reference_poses[:plot_len]
        static_errors[:, 2] = (static_errors[:, 2] + np.pi) % (2.0 * np.pi) - np.pi

    tuned_label = "tuned (param)" if static_log is not None else "Tuned gains"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x error [m]", "y error [m]", "theta error [rad]"]
    for idx, label in enumerate(labels):
        axes[idx].plot(plot_time, init_errors[:, idx], label="Initial gains", linewidth=1.2)
        if static_errors is not None:
            axes[idx].plot(plot_time, static_errors[:, idx], label="tuned (static)", color="tab:green", linewidth=1.2)
        axes[idx].plot(plot_time, tuned_errors[:, idx], label=tuned_label, color="tab:orange", linewidth=1.2)
        axes[idx].set_ylabel(label)
        axes[idx].grid(True)
        axes[idx].legend()

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Pose Tracking Errors", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(pdf_filename, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"Controller tuning error PDF saved at: {pdf_filename}")


def _stair_series(time: np.ndarray, values: np.ndarray, end_time: float | None) -> tuple[np.ndarray, np.ndarray]:
    if end_time is None or len(time) == 0 or end_time <= time[-1]:
        return time, values
    return np.concatenate([time, [end_time]]), np.concatenate([values, values[-1:]])


def _pose_vel_omega(time_s: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(time_s) < 2:
        return time_s, np.zeros((len(time_s), 2), dtype=float)

    dt = np.diff(time_s)
    delta_xy = np.diff(pose[:, :2], axis=0)
    theta = np.unwrap(pose[:, 2])
    heading = theta[1:]
    linear_velocity = (delta_xy[:, 0] * np.cos(heading) + delta_xy[:, 1] * np.sin(heading)) / dt
    angular_velocity = np.diff(theta) / dt
    return time_s[1:], np.column_stack([linear_velocity, angular_velocity])


def _plot_tuned_motor_axes(
    ax_duty,
    time: np.ndarray,
    duty_cycle: np.ndarray,
    wheel_speeds: np.ndarray,
):
    ax_speed = ax_duty.twinx()
    line_dc_l = ax_duty.plot(
        time,
        duty_cycle[:, 1],
        color="#f2c14e",
        linewidth=0.7,
        label="DC left",
    )[0]
    line_speed_l = ax_speed.plot(
        time,
        wheel_speeds[:, 1],
        color="#f28e2b",
        linewidth=0.9,
        label="speed left",
    )[0]
    line_dc_r = ax_duty.plot(
        time,
        duty_cycle[:, 0],
        color="#59a14f",
        linewidth=0.7,
        label="DC right",
    )[0]
    line_speed_r = ax_speed.plot(
        time,
        wheel_speeds[:, 0],
        color="#1b7f3a",
        linewidth=0.9,
        label="speed right",
    )[0]

    _set_symmetric_ylim(ax_duty, duty_cycle)
    _set_symmetric_ylim(ax_speed, wheel_speeds)
    ax_duty.set_xlabel("time [s]")
    ax_duty.set_ylabel("duty cycle")
    ax_speed.set_ylabel("wheel speed [rad/s]")
    ax_duty.set_title("Motor")
    ax_duty.grid(True)
    ax_duty.legend(handles=[line_dc_l, line_speed_l, line_dc_r, line_speed_r], loc="best")


def _set_symmetric_ylim(ax, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    limit = np.nanmax(np.abs(values)) if values.size else 0.0
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    ax.set_ylim(-1.05 * limit, 1.05 * limit)
