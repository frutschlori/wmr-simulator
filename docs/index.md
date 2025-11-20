---
layout: default
title: Wheeled Robot Simulator
---

# Wheeled Robot Simulator

A modular differential-drive robot simulator designed for **control**, **estimation**,  
and **active-learning gain tuning** research.

This README explains:

1. What each subsystem does  
2. Which YAML fields configure it  
3. The equations used in every module  
4. How all components interact in the simulation loop  
5. The planner, controller, robot model, estimator, and visualization

---

# 1. Problem YAML Overview

The simulator is driven by a YAML file such as **`empty.yaml`**.

```yaml
sim_time: 5.0
time_step: 0.01

environment:
  min: [-5, -5]
  max: [5, 5]
  obstacles: []

start: [0.0, 0.0, 0.0]
goal:  [2.0, 1.0, 1.57]

planner:
  waypoints:
    - [0.5, 0.0, -1.57]
    - [1.0, 0.5,  1.57]
  time: 4.0

controller:
  gains: [5.0, 5.0, 3.0, 0.4, 0.4, 0.2, 0.2]

robot:
  wheel_radius: 0.016
  base_diameter: 0.089
  max_wheel_speed: 40.0
  slip_r: 0.4
  slip_l: 0.5

estimator:
  type: "kf"
  wheel_radius: 0.015
  base_diameter: 0.09
  noise_pos: 0.0001
  noise_angle: 0.07
  enc_angle_noise: 0.01
  proc_pos_std: 0.7
  proc_theta_std: 0.7
  start: [0.0, 0.0, 0.0]
```

### YAML → Subsystems

| Subsystem | YAML fields | What it configures |
|----------|-------------|--------------------|
| Planner | `planner.waypoints`, `planner.time` | Path & time scaling |
| Controller | `controller.gains` | Geometric gains + PI wheel controllers |
| Robot | `robot.wheel_radius`, `robot.base_diameter`, `robot.slip_r`, `robot.slip_l`, `robot.max_wheel_speed` | True simulation dynamics |
| Estimator | `estimator.type`, noise params | DR or EKF |
| Simulation | `sim_time`, `time_step`, start/goal | Integration horizon & boundary conditions |

---


# 2. Running the Simulator

To run a complete simulation:

```bash
python3 scripts/simulator.py --problem problems/empty.yaml --output simulation_visualization
```


Outputs:

- PDF of plots  
- MeshCat 3D animation  
- Logs in `results/`  

---

# 3. Visualization (`visualize.py`)

Produces:

- Multi-page PDF  
- MeshCat 3D animation  
- True, estimated, and reference trajectories  

---

# 4. Robot Model (`robot.py`)

## YAML fields

- `wheel_radius`
- `base_diameter`
- `slip_r`, `slip_l`
- `max_wheel_speed`

## State

$$

\mathbf{x} = (x,\;y,\;\theta)

$$

## Inputs

$$

u_r,\; u_l

$$

## Kinematics

$$

v = \frac{r}{2}(u_r + u_l),
\qquad
\omega = \frac{r}{L}(u_r - u_l)

$$

State evolution:

$$

\begin{aligned}
x_{k+1} &= x_k + v\cos\theta_k \,\Delta t \\
y_{k+1} &= y_k + v\sin\theta_k \,\Delta t \\
\theta_{k+1} &= \theta_k + \omega \,\Delta t
\end{aligned}

$$

## Slip

$$

u_r^{\text{eff}} = u_r(1-\varepsilon_r),\qquad
u_l^{\text{eff}} = u_l(1-\varepsilon_l)

$$

$$

\varepsilon_r \sim U[-s_r,s_r],\qquad
\varepsilon_l \sim U[-s_l,s_l]

$$

Robot state update uses the **effective** speeds.


## Motor dynamics and time constant

In addition to the kinematic mapping from wheel speeds to \((v,\omega)\), the robot model includes a simple **first-order motor dynamics** model that smooths the wheel commands and approximates actuator delay.

The relevant YAML field is:

```yaml
robot:
  wheel_radius: 0.016
  base_diameter: 0.089
  max_wheel_speed: 40.0
  time_constant: 0.05   # [s] motor time constant
  slip_r: 0.4
  slip_l: 0.5
```

The continuous-time motor dynamics for each wheel are modeled as:

$$

\tau \,\dot{u}_r^{eff}(t) = -u_r^{eff}(t) + u_r^{cmd}(t),
\qquad
\tau \,\dot{u}_l^{eff}(t) = -u_l^{eff}(t) + u_l^{cmd}(t),

$$

where

- \(u_r^{cmd}, u_l^{cmd}\) are the **commanded** wheel speeds from the PI controllers,
- \(u_r^{eff}, u_l^{eff}\) are the **effective** wheel speeds seen by the kinematics,
- \(\tau = \texttt{robot.time\_constant}\) is the motor time constant.

In discrete time with simulator timestep \(\Delta t\), the implementation uses:

$$

\alpha = e^{-\Delta t / \tau},

$$

and updates:

$$

u_r^{eff}[k+1] = \alpha\,u_r^{eff}[k] + (1-\alpha)\,u_r^{cmd}[k],

$$
$$

u_l^{eff}[k+1] = \alpha\,u_l^{eff}[k] + (1-\alpha)\,u_l^{cmd}[k].

$$

If \(\tau < 10^{-3}\) (as a special case), the code sets

$$

\alpha = 0,

$$

which reduces to

$$

u_r^{eff}[k+1] = u_r^{cmd}[k],\qquad
u_l^{eff}[k+1] = u_l^{cmd}[k],

$$

i.e. **no motor lag** (instantaneous response).

These effective wheel speeds are then used in the kinematics (and slip model):

1. First apply motor dynamics to get \(u_r^{eff}, u_l^{eff}\).  
2. Then apply slip:
   $$

   u_r^{slip} = u_r^{eff}(1-\varepsilon_r),\qquad
   u_l^{slip} = u_l^{eff}(1-\varepsilon_l),
   
$$
3. Finally compute
   $$

   v = \frac{r}{2}(u_r^{slip}+u_l^{slip}),
   \qquad
   \omega = \frac{r}{L}(u_r^{slip}-u_l^{slip}).
   
$$

---


# 5. Planner (`planner.py`)

The planner produces the reference trajectory:

$$

[x_d, y_d, \theta_d, \dot{x}_d, \dot{y}_d, \omega_d,\ddot{x}_d,\ddot{y}_d]

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

## Tangent constraint (when θ is provided)

$$

\frac{1}{\|\mathbf{t}(u_i)\|}\mathbf{t}(u_i)=
\begin{bmatrix}
\cos\theta_i\\ \sin\theta_i
\end{bmatrix}

$$

where

$$

\mathbf{t}(u_i)=\begin{bmatrix}x'(u_i)\\y'(u_i)\end{bmatrix}

$$

---

## Time parameterization

$$

x_d(t)=x(u(t)),\qquad y_d(t)=y(u(t))

$$

Velocities:

$$

\dot{x}_d=x'(u)\dot{u},\qquad
\dot{y}_d=y'(u)\dot{u}

$$

Heading:

$$

\theta_d=\operatorname{atan2}(\dot{y}_d,\dot{x}_d)

$$

Curvature:

$$

\kappa(u)=
\frac{x'(u)y''(u)-y'(u)x''(u)}
{(x'(u)^2+y'(u)^2)^{3/2}}

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


# 6. Controller (`controller.py`)

## YAML field

```yaml
controller:
  gains: [k_x, k_y, k_θ, k_pr, k_pl, k_ir, k_il]
```

---

## 6.1 Geometric Pose Control

Inputs:

- Reference: \((x_d,y_d,\theta_d,v_d,\omega_d)\)  
- Estimate: \((\hat{x},\hat{y},\hat{\theta})\)

Pose errors in robot frame:

$$

\begin{aligned}
x_e &= (x_d-x)\cos\theta + (y_d-y)\sin\theta,\\  
y_e &= -(x_d-x)\sin\theta + (y_d-y)\cos\theta,\\  
\theta_e &= \operatorname{wrap}(\theta_d - \theta)
\end{aligned}

$$

Control law:

$$

v^{ref} = v_d \cos\theta_e + k_x x_e

$$

$$

\omega^{ref} = \omega_d + v_d(k_y y_e + k_\theta\sin\theta_e) + k_\theta\theta_e

$$

---

## 6.2 Wheel-Speed Mapping

$$

u_r^{ref} = \frac{2v^{ref} + L\omega^{ref}}{2r},
\qquad
u_l^{ref} = \frac{2v^{ref} - L\omega^{ref}}{2r}

$$

---

## 6.3 PI Wheel-Speed Control

Errors:

$$

e_r = u_r^{ref} - u_r^{meas},\qquad
e_l = u_l^{ref} - u_l^{meas}

$$

Commands:

$$

\tilde{u}_r = u_r^{ref} + k_{pr}e_r + k_{ir}\!\int e_r dt

$$

$$

\tilde{u}_l = u_l^{ref} + k_{pl}e_l + k_{il}\!\int e_l dt

$$

Outputs \(\tilde{u}_r,\tilde{u}_l\) go to the robot.

---

# 7. Estimator (`estimator.py`)

## YAML fields

- `type`: `"dr"` or `"kf"`
- `wheel_radius`, `base_diameter`
- sensor noise: `noise_pos`, `noise_angle`
- encoder noise: `enc_angle_noise`
- process noise: `proc_pos_std`, `proc_theta_std`
- initial estimate: `start`

---

## 7.1 Wheel-speed measurement model

$$

\hat{u}_r = u_r^{eff} + n_r,\qquad
\hat{u}_l = u_l^{eff} + n_l

$$

---

## 7.2 Prediction

Speeds:

$$

v_{hat}=\frac{r}{2}(\hat{u}_r+\hat{u}_l),\qquad
\omega_{hat}=\frac{r}{L}(\hat{u}_r-\hat{u}_l)

$$

State prediction:

$$

\begin{aligned}
\hat{x}_{k+1}^- &= \hat{x}_k + v_{hat}\cos\hat{\theta}_k\,\Delta t \\
\hat{y}_{k+1}^- &= \hat{y}_k + v_{hat}\sin\hat{\theta}_k\,\Delta t \\
\hat{\theta}_{k+1}^- &= \hat{\theta}_k + \omega_{hat}\Delta t
\end{aligned}

$$

Jacobian:

$$

F_k=
\begin{bmatrix}
1 & 0 & -v_{hat}\sin\hat{\theta}\Delta t\\
0 & 1 & v_{hat}\cos\hat{\theta}\Delta t\\
0 & 0 & 1
\end{bmatrix}

$$

Covariance:

$$

P^- = FP F^\top + Q

$$

---

## 7.3 Measurement model

$$

h(x)=\begin{bmatrix}x\\y\\\theta\end{bmatrix},\qquad H=I_3

$$

Measurements:

$$

z_k=
\begin{bmatrix}
p_x\\p_y\\\theta_m
\end{bmatrix}

$$

Angle wrapping:

$$

\operatorname{wrap}(\alpha)=\operatorname{atan2}(\sin\alpha,\cos\alpha)

$$

Residual:

$$

y_k=
\begin{bmatrix}
p_x-\hat{x}^-\\
p_y-\hat{y}^-\\
\operatorname{wrap}(\theta_m-\hat{\theta}^-)
\end{bmatrix}

$$

Update:

$$

S=HP^-H^\top + R

$$
$$

K=P^-H^\top S^{-1}

$$
$$

\hat{x}=\hat{x}^- + Ky

$$
$$

P=(I-KH)P^-

$$

---

# 8. Full System Architecture

Subsystems:

1. Planner → smooth reference trajectory  
2. Trajectory sampler  
3. Estimator → produces \(\hat{x},\hat{y},\hat{\theta}\)  
4. Controller  
   - geometric pose control  
   - wheel-speed PI loops  
5. Robot  
   - slip  
   - differential-drive kinematics  
6. Visualization and logging  

