import jax
import jax.numpy as jnp


DEFAULT_CONSTRAINT_WEIGHTS = {
    "v": 1.0,
    "a": 1.0,
    "lateral": 1.0,
    "omega": 1.0,
    "alpha": 1.0,
    # Lower bound on the trajectory's *mean* speed; see min_speed_loss.
    "v_min": 1.0,
    # Lower bound on its mean lateral acceleration; see
    # min_lateral_acceleration_loss.
    "a_lat_min": 1.0,
}


def smooth_max(x: jnp.ndarray, beta: float = 20.0) -> jnp.ndarray:
    return jax.nn.logsumexp(beta * x) / beta


def smooth_positive_max(x: jnp.ndarray, beta: float = 20.0) -> jnp.ndarray:
    zero = jnp.zeros((1,), dtype=x.dtype)
    return smooth_max(jnp.concatenate([zero, x], axis=0), beta=beta)


def smooth_norm(x: jnp.ndarray, axis: int = -1, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)


def motion_limits_from_robot_config(robot_cfg: dict) -> dict[str, jnp.ndarray]:
    return {
        "v_max": jnp.asarray(robot_cfg["v_max"], dtype=jnp.float32),
        "a_max": jnp.asarray(robot_cfg["a_max"], dtype=jnp.float32),
        "a_lat_max": jnp.asarray(robot_cfg.get("a_max_lateral", robot_cfg["a_max"]), dtype=jnp.float32),
        "omega_max": jnp.asarray(robot_cfg["omega_max"], dtype=jnp.float32),
        "alpha_max": jnp.asarray(robot_cfg["alpha_max"], dtype=jnp.float32),
        # Lower bound, 0 = off. Not a property of the robot -- it is a property
        # of the *experiment being designed*, so the pipeline substitutes a
        # per-trajectory value; the robot config only supplies a fallback.
        "v_min": jnp.asarray(robot_cfg.get("v_min", 0.0), dtype=jnp.float32),
        # Same story for the lower bound on mean lateral acceleration.
        "a_lat_min": jnp.asarray(robot_cfg.get("a_lat_min", 0.0), dtype=jnp.float32),
    }


def min_speeds_for_batch(num_trajectories: int, min_speed: float, fraction: float = 0.5):
    """Per-trajectory ``v_min`` for a batch design: some fast, some left free.

    A design set exists to be informative *and* to cover the regimes the
    controller is judged in, and those pull in opposite directions -- an
    unconstrained FIM buys slow tight wiggles, and a set that is uniformly fast
    stops covering the slow regime at all. So ``fraction`` of the batch carries
    a minimum and the rest carries none, and the constrained ones are spread
    linearly over ``[0.5 * min_speed, min_speed]`` so the fast group is itself a
    range rather than one operating point.

    ``min_speed <= 0`` or ``fraction <= 0`` returns all zeros, which the
    constraint reads as "off".
    """
    import numpy as np

    speeds = np.zeros(int(num_trajectories), dtype=float)
    if float(min_speed) <= 0.0 or float(fraction) <= 0.0 or num_trajectories <= 0:
        return speeds
    count = int(np.clip(round(float(fraction) * num_trajectories), 1, num_trajectories))
    if count == 1:
        speeds[0] = float(min_speed)
        return speeds
    speeds[:count] = float(min_speed) * np.linspace(0.5, 1.0, count)
    return speeds


def motion_floors_for_batch(
    num_trajectories: int,
    min_speed: float,
    min_speed_fraction: float = 0.5,
    min_lateral_acceleration: float = 0.0,
    min_lateral_acceleration_fraction: float = 0.25,
):
    """Per-trajectory ``[v_min, a_lat_min]`` for a batch design, shape ``(T, 2)``.

    Each column is spread like :func:`min_speeds_for_batch`, rising with the
    index. The speed floors sit on the leading trajectories; the turning floors
    are aligned to *end* where the speed floors end, so with
    ``min_lateral_acceleration_fraction <= min_speed_fraction`` every trajectory
    that has to turn hard also has to be fast, and the hardest turn goes to the
    fastest one: the regime the fast circle drives (v ~ 2 m/s at omega ~ 2
    rad/s), which a mean-speed floor alone satisfies with fast straight sweeps.
    """
    import numpy as np

    speeds = min_speeds_for_batch(num_trajectories, min_speed, min_speed_fraction)
    lateral_head = min_speeds_for_batch(num_trajectories, min_lateral_acceleration, min_lateral_acceleration_fraction)
    num_lateral = int(np.count_nonzero(lateral_head))
    num_speed = int(np.count_nonzero(speeds))
    end = max(num_speed, num_lateral)
    lateral = np.zeros(int(num_trajectories), dtype=float)
    lateral[end - num_lateral : end] = lateral_head[:num_lateral]
    return np.stack([speeds, lateral], axis=1)


def constraint_weights(scale: float = 1.0, component_weights: dict | None = None) -> dict[str, jnp.ndarray]:
    weights = dict(DEFAULT_CONSTRAINT_WEIGHTS)
    if component_weights is not None:
        weights.update(component_weights)
    return {
        name: jnp.asarray(scale * float(weights[name]), dtype=jnp.float32)
        for name in DEFAULT_CONSTRAINT_WEIGHTS
    }


def finite_difference(values: jnp.ndarray, dt: float) -> jnp.ndarray:
    dt = jnp.asarray(dt, dtype=values.dtype)
    if values.shape[0] < 2:
        return jnp.zeros_like(values)

    first = (values[1:2] - values[0:1]) / dt
    last = (values[-1:] - values[-2:-1]) / dt
    if values.shape[0] == 2:
        return jnp.concatenate([first, last], axis=0)

    middle = (values[2:] - values[:-2]) / (2.0 * dt)
    return jnp.concatenate([first, middle, last], axis=0)


def constraint_loss(
    v: jnp.ndarray,
    a: jnp.ndarray,
    omega: jnp.ndarray,
    alpha_ang: jnp.ndarray,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    components = constraint_loss_components(
        v=v,
        a=a,
        omega=omega,
        alpha_ang=alpha_ang,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
    return sum(components.values())


def min_speed_loss(v_norm: jnp.ndarray, v_min, smooth_max_beta: float = 20.0) -> jnp.ndarray:
    """Squared fractional shortfall of the *mean* speed below ``v_min``.

    Every other limit here is an upper bound read off the worst sample, but a
    minimum speed cannot be: under the s-curve time scaling the reference is at
    rest at both ends by construction, so a worst-sample rule would be violated
    on every feasible design and would only fight the time scaling. The mean is
    the statistic that expresses "this trajectory is driven fast" without
    saying anything about how it starts and stops.

    ``sim_time`` is fixed and not a decision variable, so mean speed is
    arc length / sim_time: the only way the optimizer can satisfy this is to
    lay out a *longer* path, which is exactly the intent -- long sustained
    sweeps instead of the tight slow wiggles an unconstrained FIM prefers.
    It is a quadratic penalty, not a barrier, so an infeasible ``v_min`` (one
    that would need more arc than the environment box holds) trades off against
    the FIM rather than breaking the run.

    ``v_min <= 0`` disables it and returns exactly 0, so a design that does not
    ask for a minimum speed is bit-identical to one from before this existed.
    """
    v_min = jnp.asarray(v_min, dtype=v_norm.dtype)
    shortfall = 1.0 - jnp.mean(v_norm) / jnp.maximum(v_min, 1e-6)
    loss = smooth_positive_max(jnp.reshape(shortfall, (1,)), beta=smooth_max_beta) ** 2
    return jnp.where(v_min > 0.0, loss, jnp.zeros_like(loss))


def min_lateral_acceleration_loss(a_lat: jnp.ndarray, a_lat_min, smooth_max_beta: float = 20.0) -> jnp.ndarray:
    """Squared fractional shortfall of the *mean* |v * omega| below ``a_lat_min``.

    The turning counterpart of :func:`min_speed_loss`, and a mean for the same
    reason (the s-curve reference rests at both ends). A mean speed floor is met
    by long straight sweeps -- measured 2026-09-13, at v > 1.8 m/s the designs
    kept |omega| <= 1.06 rad/s and lateral acceleration <= 2 m/s^2 -- while the
    benchmark's fast shapes turn at 3.4-5.7 m/s^2, the regime that rings and
    loses traction on the robot. Holding the mean |a_lat| up is only satisfiable
    by turning *while* fast, and it is a quadratic penalty, so it trades off
    against the FIM and the upper limits (a_lat_max, omega_max, alpha_max)
    rather than overriding them.

    ``a_lat_min <= 0`` disables it and returns exactly 0.
    """
    a_lat_min = jnp.asarray(a_lat_min, dtype=a_lat.dtype)
    shortfall = 1.0 - jnp.mean(a_lat) / jnp.maximum(a_lat_min, 1e-6)
    loss = smooth_positive_max(jnp.reshape(shortfall, (1,)), beta=smooth_max_beta) ** 2
    return jnp.where(a_lat_min > 0.0, loss, jnp.zeros_like(loss))


def constraint_loss_components(
    v: jnp.ndarray,
    a: jnp.ndarray,
    omega: jnp.ndarray,
    alpha_ang: jnp.ndarray,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> dict[str, jnp.ndarray]:
    v_norm = smooth_norm(v, axis=-1)
    a_norm = smooth_norm(a, axis=-1)
    a_lat = jnp.abs(v_norm * omega)

    g_v_samples = v_norm / limits["v_max"] - 1.0
    g_a_samples = a_norm / limits["a_max"] - 1.0
    g_lat_samples = a_lat / limits["a_lat_max"] - 1.0
    g_w_samples = jnp.abs(omega) / limits["omega_max"] - 1.0
    g_alpha_samples = jnp.abs(alpha_ang) / limits["alpha_max"] - 1.0

    v_loss = smooth_positive_max(g_v_samples, beta=smooth_max_beta) ** 2
    a_loss = smooth_positive_max(g_a_samples, beta=smooth_max_beta) ** 2
    lateral_loss = smooth_positive_max(g_lat_samples, beta=smooth_max_beta) ** 2
    omega_loss = smooth_positive_max(g_w_samples, beta=smooth_max_beta) ** 2
    alpha_loss = smooth_positive_max(g_alpha_samples, beta=smooth_max_beta) ** 2

    return {
        "v": weights["v"] * v_loss,
        "a": weights["a"] * a_loss,
        "lateral": weights["lateral"] * lateral_loss,
        "omega": weights["omega"] * omega_loss,
        "alpha": weights["alpha"] * alpha_loss,
        "v_min": weights["v_min"]
        * min_speed_loss(v_norm, limits.get("v_min", 0.0), smooth_max_beta=smooth_max_beta),
        "a_lat_min": weights["a_lat_min"]
        * min_lateral_acceleration_loss(a_lat, limits.get("a_lat_min", 0.0), smooth_max_beta=smooth_max_beta),
    }


def constraint_loss_from_reference_states(
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    v = reference_states[:, 3:5]
    omega = reference_states[:, 5]
    a = reference_states[:, 6:8]
    alpha_ang = finite_difference(omega, dt)
    return constraint_loss(
        v=v,
        a=a,
        omega=omega,
        alpha_ang=alpha_ang,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )


def constraint_loss_components_from_reference_states(
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> dict[str, jnp.ndarray]:
    v = reference_states[:, 3:5]
    omega = reference_states[:, 5]
    a = reference_states[:, 6:8]
    alpha_ang = finite_difference(omega, dt)
    return constraint_loss_components(
        v=v,
        a=a,
        omega=omega,
        alpha_ang=alpha_ang,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )


# Where the motion limits come from -----------------------------------------

GRAVITY = 9.81


def derive_motion_limits(
    robot_cfg: dict,
    mass: float,
    stall_torque: float,
    yaw_inertia: float | None = None,
    chassis_radius: float = 0.048,
    speed_headroom: float = 0.6,
) -> dict[str, dict]:
    """Motion limits derived from the robot rather than chosen by hand.

    Every value comes back with the quantity that produced it, because the four
    limits are set by three *different* physical mechanisms and knowing which
    one binds is most of the value:

    ``v_max``   wheel-speed budget. ``r * max_wheel_speed`` is what the wheels
                can do; the reference must leave the controller room to correct
                on top of it *and* room for the yaw differential, since the real
                constraint is per wheel: ``|v| + |omega| * L/2 <= r * w_max``.
                ``speed_headroom`` is the share of that budget the reference is
                allowed to claim.

    ``a_max``   two ceilings, and they mean different things. Motor torque
                (``2 * tau(omega) / (r * m)``, ``tau`` falling linearly with
                wheel speed) is a hard one: the robot cannot exceed it at all.
                Traction (``a_slip_max``) is not -- past it the tires slip, and
                the burnout model in ``robot.py`` is there precisely to
                represent that. On this robot torque is the larger of the two
                everywhere below ~3.3 m/s, so ``grip`` is what a reference meets
                first.

                **Setting ``a_max`` above ``a_slip_max`` is therefore a
                deliberate choice, not an inconsistency.** The robot does break
                traction on the real benchmark circles, so a design envelope
                that stops at the grip limit would never produce a tuning or
                identification run in the regime the controller actually has to
                survive -- and the residual model cannot learn drift dynamics
                from logs that never drift. What ``a_slip_max`` marks is where
                the plant stops being kinematic, not where the design has to
                stop. The value to keep well clear of is the *torque* ceiling.

    ``a_max_lateral``  the same argument. Longitudinal and lateral draw on one
                friction budget (``sqrt(a_long^2 + a_lat^2) <= a_slip_max``), so
                this is where the tires let go laterally; exceeding it buys
                cornering slip on purpose.

    ``omega_max``  the wheel-speed budget again, and it is never the binding
                constraint in practice: lateral traction caps ``omega`` at
                ``a_slip_max / v``, which is 1.2-3.0 rad/s over the speeds these
                references run at, against the tens of rad/s the wheels allow.
                It is reported for completeness.

    ``alpha_max``  **not derivable, and this function says so.** The traction
                bound is ``a_slip_max * m * L / (2 * I_zz)`` -- 80-110 rad/s^2
                here -- which is 5-7x above any value that has ever worked.
                What actually limits it is whether the closed loop can *track*
                the yaw acceleration, measured rather than derived (designs past
                ~1.0x the configured value stop being trackable). So the
                returned entry carries the physical ceiling and a note, not a
                recommendation: it is a tuning constant, and it should be
                labelled as one rather than sitting in the robot block looking
                like a property of the hardware.

    ``yaw_inertia`` defaults to a uniform disc of ``chassis_radius``, which is
    the weakest input here; a measured value would tighten only ``alpha_max``,
    which is the one number this function declines to set anyway.
    """
    r = float(robot_cfg["wheel_radius"])
    wheelbase = float(robot_cfg["base_diameter"])
    wheel_speed = float(robot_cfg["max_wheel_speed"])
    a_slip = float(robot_cfg.get("a_slip_max", 0.0)) or GRAVITY
    inertia = float(yaw_inertia) if yaw_inertia else 0.5 * mass * chassis_radius**2

    wheel_budget = r * wheel_speed
    torque_at_rest = 2.0 * stall_torque / (r * mass)
    # Linear DC-motor torque falloff, evaluated where the reference actually
    # cruises rather than at stall.
    cruise = speed_headroom * wheel_budget
    torque_at_cruise = 2.0 * stall_torque * (1.0 - cruise / wheel_budget) / (r * mass)
    yaw_traction = a_slip * mass * wheelbase / (2.0 * inertia)

    return {
        "v_max": {
            "value": speed_headroom * wheel_budget,
            "binds": "wheel-speed budget",
            "detail": f"r*w_max = {wheel_budget:.2f} m/s, {speed_headroom:.0%} to the reference",
        },
        "a_max": {
            "value": torque_at_cruise,
            "binds": "motor torque (hard); traction is a choice",
            "detail": f"torque {torque_at_rest:.1f} at rest / {torque_at_cruise:.1f} at "
                      f"{cruise:.2f} m/s is the hard ceiling; grip ends at {a_slip:.2f} and "
                      f"going past it buys slip on purpose",
        },
        "a_max_lateral": {
            "value": torque_at_cruise,
            "binds": "motor torque (hard); traction is a choice",
            "detail": f"grip ends at {a_slip:.2f}, shared with a_max via "
                      f"sqrt(a_long^2 + a_lat^2) <= {a_slip:.2f}",
        },
        "omega_max": {
            "value": 2.0 * (wheel_budget - speed_headroom * wheel_budget) / wheelbase,
            "binds": "wheel-speed budget, but lateral traction binds first",
            "detail": f"traction caps omega at a_slip/v = {a_slip / max(speed_headroom * wheel_budget, 1e-9):.2f}"
                      f" rad/s at v_max",
        },
        "alpha_max": {
            "value": None,
            "binds": "closed-loop trackability, not physics",
            "detail": f"traction ceiling is {yaw_traction:.0f} rad/s^2 (I_zz = {inertia:.2e} kg m^2); "
                      f"the working value is 5-7x below it and has to be measured",
        },
    }
