# 6. Controller (`controller.py`)

## YAML field

```yaml
controller:
  gains: [k_x, k_y, k_θ, k_pr, k_pl, k_ir, k_il]
```

---

## 6.1 Geometric Pose Control

Inputs:

- Reference: $(x_d, y_d, \theta_d, v_d, \omega_d)$
- Estimate: $(\hat{x}, \hat{y}, \hat{\theta})$

Pose errors in robot frame:

$$
\begin{aligned}
x_e &= (x_d - x)\cos\theta + (y_d - y)\sin\theta, \\
y_e &= -(x_d - x)\sin\theta + (y_d - y)\cos\theta, \\
\theta_e &= \operatorname{wrap}(\theta_d - \theta)
\end{aligned}
$$

Control law:

$$
v^{ref} = v_d \cos\theta_e + k_x x_e
$$

$$
\omega^{ref} = \omega_d 
  + v_d(k_y y_e + k_\theta \sin\theta_e)
  + k_\theta \theta_e
$$

---

## 6.2 Wheel-Speed Mapping

$$
u_r^{ref} = \frac{2v^{ref} + L\omega^{ref}}{2r}
$$

$$
u_l^{ref} = \frac{2v^{ref} - L\omega^{ref}}{2r}
$$

---

## 6.3 PI Wheel-Speed Control

Errors:

$$
e_r = u_r^{ref} - u_r^{meas}
$$

$$
e_l = u_l^{ref} - u_l^{meas}
$$

Commands:

$$
\tilde{u}_r = u_r^{ref} + k_{pr} e_r + k_{ir} \int e_r \, dt
$$

$$
\tilde{u}_l = u_l^{ref} + k_{pl} e_l + k_{il} \int e_l \, dt
$$

Outputs $(\tilde{u}_r, \tilde{u}_l)$ go to the robot.

---
