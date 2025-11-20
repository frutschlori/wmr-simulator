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
v = \frac{r}{2}(u_r + u_l), \qquad \omega = \frac{r}{L}(u_r - u_l)
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
u_r^{\text{eff}} = u_r(1-\varepsilon_r), \qquad
u_l^{\text{eff}} = u_l(1-\varepsilon_l)
$$

$$
\varepsilon_r \sim U[-s_r,s_r], \qquad
\varepsilon_l \sim U[-s_l,s_l]
$$

Robot state update uses the effective speeds.

## Motor dynamics and time constant

In addition to the kinematic mapping from wheel speeds to $(v,\omega)$, the robot model includes a simple first-order motor dynamics model that smooths the wheel commands and approximates actuator delay.

The relevant YAML field is:

\`\`\`yaml
robot:
  wheel_radius: 0.016
  base_diameter: 0.089
  max_wheel_speed: 40.0
  time_constant: 0.05   # [s] motor time constant
  slip_r: 0.4
  slip_l: 0.5
\`\`\`

The continuous-time motor dynamics for each wheel are:

$$
\tau \,\dot{u}_r^{eff}(t) = -u_r^{eff}(t) + u_r^{cmd}(t), \qquad
\tau \,\dot{u}_l^{eff}(t) = -u_l^{eff}(t) + u_l^{cmd}(t)
$$

where:

- $u_r^{cmd}, u_l^{cmd}$ are the commanded wheel speeds from the PI controllers,
- $u_r^{eff}, u_l^{eff}$ are the effective wheel speeds seen by the kinematics,
- $\tau = \texttt{robot.time\_constant}$ is the motor time constant.

In discrete time with simulator timestep $\Delta t$, the implementation uses:

$$
\alpha = e^{-\Delta t / \tau}
$$

and updates:

$$
u_r^{eff}[k+1] = \alpha u_r^{eff}[k] + (1-\alpha)u_r^{cmd}[k]
$$

$$
u_l^{eff}[k+1] = \alpha u_l^{eff}[k] + (1-\alpha)u_l^{cmd}[k]
$$

If $\tau < 10^{-3}$ (special case), the code sets:

$$
\alpha = 0
$$

which reduces to:

$$
u_r^{eff}[k+1] = u_r^{cmd}[k], \qquad
u_l^{eff}[k+1] = u_l^{cmd}[k]
$$

i.e., no motor lag (instantaneous response).

These effective wheel speeds are then used in the kinematics (and slip model):

1. Apply motor dynamics to get $u_r^{eff}, u_l^{eff}$
2. Apply slip:

$$
u_r^{slip} = u_r^{eff}(1-\varepsilon_r), \qquad
u_l^{slip} = u_l^{eff}(1-\varepsilon_l)
$$

3. Compute velocities:

$$
v = \frac{r}{2}(u_r^{slip}+u_l^{slip}), \qquad
\omega = \frac{r}{L}(u_r^{slip}-u_l^{slip})
$$

---
