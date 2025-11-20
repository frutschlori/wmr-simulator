# 5. Planner (`planner.py`)

The planner produces the reference trajectory:

$$
[x_d, y_d, \theta_d, \dot{x}_d, \dot{y}_d, \omega_d, \ddot{x}_d, \ddot{y}_d]
$$

It performs:

1. Quintic spline fitting in space
2. Time-scaling to match total duration
3. Trajectory sampling

## YAML fields

```yaml
planner:
  waypoints:
    - [...]
    - [...]
  time: 4.0
```

---

## Waypoints (position-only or pose)

Waypoints can be:

### Position-only (θ free)

```yaml
[x, y]
```

### Pose waypoint (θ constrained)

```yaml
[x, y, theta]
```

Length of the waypoint determines whether heading is enforced.

---

## Tangent constraint (when θ is provided)

$$
\frac{1}{\|\mathbf{t}(u_i)\|}\mathbf{t}(u_i)=
\begin{bmatrix}
\cos\theta_i\\
\sin\theta_i
\end{bmatrix}
$$

where

$$
\mathbf{t}(u_i)=
\begin{bmatrix}
x'(u_i)\\
y'(u_i)
\end{bmatrix}
$$

---

## Time parameterization

$$
x_d(t)=x(u(t)),\qquad y_d(t)=y(u(t))
$$

Velocities:

$$
\dot{x}_d = x'(u)\dot{u}, \qquad
\dot{y}_d = y'(u)\dot{u}
$$

Heading:

$$
\theta_d = \operatorname{atan2}(\dot{y}_d,\dot{x}_d)
$$

Curvature:

$$
\kappa(u) =
\frac{x'(u) y''(u) - y'(u) x''(u)}
{(x'(u)^2 + y'(u)^2)^{3/2}}
$$

Angular velocity:

$$
\omega_d = \kappa(u)\, v_d
$$

---

## Running the Planner Directly

To debug waypoints without running the simulator:

```bash
python3 planner.py
```

Modify the `__main__` block inside `planner.py`:

```python
intermediate_waypoints = [
    [x1, y1],           # no θ constraint
    [x2, y2, theta2],   # θ constrained
]
```

Running the module plots:

- spline path
- heading
- curvature
- velocities

---
