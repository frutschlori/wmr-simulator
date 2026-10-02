"""Closed-loop training of the residual on recorded runs.

The one-step fit (``residual.train_residual_ensemble``) predicts each step's
twist error from that step's nominal twist. It lowers the one-step error but,
measured on held-out MuJoCo runs (2026-10-02, 5 seeds), makes the *closed loop*
predict worse (+20 % sim-vs-MuJoCo position error at gate 1.0), and a
teacher-forced multi-step pose loss is worse still. What the tuner needs is the
closed loop, so this module trains the residual on it directly: short windows
cut from the recorded runs are re-simulated in closed loop -- from the measured
pose and wheel speeds at the window start, under the controller that recorded
the run (its gains and its parameter belief) -- and the residual's expert
weights are fitted to the mocap track by backpropagating through the rollouts.
One-second windows from a zero residual cut the held-out closed-loop error by
about 20 % where the data covers the regime; 0.5 s windows are too short to
see the error build up, 2 s ones get harder to fit.

Time alignment: a log's reference starts later than its pose stream (40 ms on
the MuJoCo logs), and the controller takes reference sample k at the log time
of that sample. Windows therefore start at reference samples and are scored
against the mocap track on that clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wmr_simulator.residual_model.residual import ResidualEnsemble, _normalized_weight


@dataclass
class ClosedLoopWindows:
    """Stacked closed-loop windows (leading axis: window)."""

    references: jax.Array      # (W, n_geo + 1, 8) reference samples
    initial_poses: jax.Array   # (W, 3) mocap pose at the window start
    initial_wheels: jax.Array  # (W, 2) encoder wheel speeds at the window start
    targets: jax.Array         # (W, n_geo * steps, 3) mocap pose after every wheel step
    controller_params: object  # PhysicalParams of the recording controller's belief, stacked
    controller_gains: jax.Array  # (W, 6)
    log_index: np.ndarray      # (W,) which log a window came from

    def __len__(self) -> int:
        return int(self.log_index.shape[0])


def _controller_of(problem_path: str):
    from wmr_simulator.controller import controller_gains_array
    from wmr_simulator.types import PhysicalParams

    config = yaml.safe_load(Path(problem_path).read_text(encoding="utf-8"))
    robot = config["robot"]
    params = PhysicalParams(
        wheel_radius=np.float32(robot["wheel_radius"]),
        base_diameter=np.float32(robot["base_diameter"]),
        max_wheel_speed=np.float32(robot["max_wheel_speed"]),
        time_constant=np.float32(robot["time_constant"]),
    )
    return params, np.asarray(controller_gains_array(config["controller"]["gains"]), dtype=np.float32)


def build_windows(
    logs: list,
    controller_problems: list[str],
    *,
    window_s: float,
    stride_s: float,
    geometry_dt: float,
    wheel_dt: float,
) -> ClosedLoopWindows | None:
    """Cut closed-loop windows from decoded logs (``PololuLog``), each with the
    controller problem that recorded it. Returns None when no log is long enough."""
    n_geo = int(round(window_s / geometry_dt))
    stride = max(int(round(stride_s / geometry_dt)), 1)
    steps = int(round(geometry_dt / wheel_dt))
    references, poses0, wheels0, targets, params, gains, owner = [], [], [], [], [], [], []
    controllers: dict[str, tuple] = {}
    for index, (log, problem) in enumerate(zip(logs, controller_problems)):
        if problem not in controllers:
            controllers[problem] = _controller_of(problem)
        controller_params, controller_gains = controllers[problem]
        reference_time = np.asarray(log.reference.time_s, dtype=float)
        reference = np.asarray(log.reference.states, dtype=np.float32)
        pose_time = np.asarray(log.pose.time_s, dtype=float)
        pose = np.asarray(log.pose.states, dtype=float)
        heading = np.unwrap(pose[:, 2])
        wheel_time = np.asarray(log.wheel.time_s, dtype=float)
        wheel = np.asarray(log.wheel.speeds, dtype=float)
        for start in range(0, len(reference) - n_geo - 1, stride):
            t0 = reference_time[start]
            tau = t0 + wheel_dt * np.arange(1, n_geo * steps + 1)
            if t0 < pose_time[0] or tau[-1] > pose_time[-1]:
                continue
            references.append(reference[start : start + n_geo + 1])
            poses0.append([np.interp(t0, pose_time, pose[:, 0]), np.interp(t0, pose_time, pose[:, 1]),
                           np.interp(t0, pose_time, heading)])
            wheels0.append([np.interp(t0, wheel_time, wheel[:, 0]), np.interp(t0, wheel_time, wheel[:, 1])])
            targets.append(np.column_stack([np.interp(tau, pose_time, pose[:, 0]), np.interp(tau, pose_time, pose[:, 1]),
                                            np.interp(tau, pose_time, heading)]))
            params.append(controller_params)
            gains.append(controller_gains)
            owner.append(index)
    if not owner:
        return None
    return ClosedLoopWindows(
        references=jnp.asarray(np.stack(references), dtype=jnp.float32),
        initial_poses=jnp.asarray(np.asarray(poses0), dtype=jnp.float32),
        initial_wheels=jnp.asarray(np.asarray(wheels0), dtype=jnp.float32),
        targets=jnp.asarray(np.stack(targets), dtype=jnp.float32),
        controller_params=jax.tree_util.tree_map(lambda *x: jnp.asarray(np.stack(x)), *params),
        controller_gains=jnp.asarray(np.stack(gains)),
        log_index=np.asarray(owner),
    )


def window_pipeline(problem_path: str, plant_robot: dict, window_s: float):
    """A simulator for the windows: the problem's controller/estimator config,
    the residual's nominal plant, a horizon one window long."""
    from wmr_simulator.robot import DiffDrive
    from wmr_simulator.simulation import SimulationPipeline

    pipeline = SimulationPipeline(problem_path=problem_path, seed=0, sim_time=float(window_s) + 0.1)
    pipeline.robot = DiffDrive(robot_cfg=plant_robot, dt=pipeline.wheel_dt)
    return pipeline


def with_normalized_weights(model: ResidualEnsemble) -> ResidualEnsemble:
    """The same function with the spectral normalization applied once to the
    weights (cap disabled), so a rollout does not repeat the power iteration at
    every step. Differentiable in the raw weights."""
    cap = model.spectral_norm_cap
    layers = tuple(
        jax.vmap(lambda weight: _normalized_weight(weight, cap))(layer) if layer.ndim == 3 else layer
        for layer in model.layers
    )
    return ResidualEnsemble(
        layers=layers, centers=model.centers, scales=model.scales, ood_sigma=model.ood_sigma,
        input_mean=model.input_mean, input_std=model.input_std, target_mean=model.target_mean,
        target_std=model.target_std, spectral_norm_cap=0.0,
    )


def window_loss_fn(pipeline, windows: ClosedLoopWindows, *, position_scale: float = 0.01, heading_scale: float = 0.02):
    """``loss(model) -> (mean loss, per-window loss)``: the squared mocap error
    of every closed-loop window (position over ``position_scale``, wrapped
    heading over ``heading_scale``), averaged over its steps."""

    def rollout(model, reference, pose0, wheel0, params, gains):
        log = pipeline.run_closed_loop(
            params, use_hidden_robot=True, controller_gains=gains, reference_states=reference,
            residual_model=model, initial_pose=pose0, initial_wheel_speeds=wheel0,
        )
        return log.pose.true_states[1:]

    batched = jax.vmap(rollout, in_axes=(None, 0, 0, 0, 0, 0))

    def loss(model):
        poses = batched(with_normalized_weights(model), windows.references, windows.initial_poses,
                        windows.initial_wheels, windows.controller_params, windows.controller_gains)
        position = jnp.sum((poses[..., :2] - windows.targets[..., :2]) ** 2, axis=-1) / position_scale**2
        heading = poses[..., 2] - windows.targets[..., 2]
        heading = jnp.arctan2(jnp.sin(heading), jnp.cos(heading)) ** 2 / heading_scale**2
        per_window = jnp.mean(position + heading, axis=1)
        return jnp.mean(per_window), per_window

    return loss


def zero_residual(model: ResidualEnsemble) -> ResidualEnsemble:
    """The same ensemble with a zero output layer: exactly the nominal plant."""
    return eqx.tree_at(
        lambda m: (m.layers[-2], m.layers[-1]),
        model,
        replace=(jnp.zeros_like(model.layers[-2]), jnp.zeros_like(model.layers[-1])),
    )


def train_closed_loop(model: ResidualEnsemble, loss, *, steps: int, learning_rate: float, log_every: int = 50):
    """Adam on the closed-loop window loss; only the expert weights move (the
    gate and the normalization stay as fitted). Returns (model, loss history)."""
    import optax

    spec = jax.tree_util.tree_map(lambda _: False, model)
    spec = eqx.tree_at(lambda m: m.layers, spec, replace=jax.tree_util.tree_map(lambda _: True, spec.layers))
    trainable, frozen = eqx.partition(model, spec)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(trainable)

    @jax.jit
    def step(trainable, state):
        value, grads = jax.value_and_grad(lambda t: loss(eqx.combine(t, frozen))[0])(trainable)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(trainable, updates), state, value

    history = []
    for index in range(int(steps)):
        trainable, state, value = step(trainable, state)
        history.append(float(value))
        if log_every and (index % log_every == 0 or index == steps - 1):
            print(f"closed-loop step {index + 1:4d}/{steps}: window loss {history[-1]:.4f}", flush=True)
    return eqx.combine(trainable, frozen), history
