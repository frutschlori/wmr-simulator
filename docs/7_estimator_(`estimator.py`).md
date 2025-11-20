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
v_{\hat{}}=\frac{r}{2}(\hat{u}_r+\hat{u}_l),\qquad
\omega_{\hat{}}=\frac{r}{L}(\hat{u}_r-\hat{u}_l)
$$

State prediction:

$$
\begin{aligned}
\hat{x}_{k+1}^- &= \hat{x}_k + v_{\hat{}}\cos\hat{\theta}_k\,\Delta t \\
\hat{y}_{k+1}^- &= \hat{y}_k + v_{\hat{}}\sin\hat{\theta}_k\,\Delta t \\
\hat{\theta}_{k+1}^- &= \hat{\theta}_k + \omega_{\hat{}}\,\Delta t
\end{aligned}
$$

Jacobian:

$$
F_k=
\begin{bmatrix}
1 & 0 & -v_{\hat{}}\sin\hat{\theta}\Delta t\\
0 & 1 & v_{\hat{}}\cos\hat{\theta}\Delta t\\
0 & 0 & 1
\end{bmatrix}
$$

Covariance:

$$
P^- = F P F^\top + Q
$$

---

## 7.3 Measurement model

$$
h(x)=\begin{bmatrix}x\\y\\\theta\end{bmatrix},\qquad 
H = I_3
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

Update equations:

$$
S = H P^- H^\top + R
$$

$$
K = P^- H^\top S^{-1}
$$

$$
\hat{x} = \hat{x}^- + K y_k
$$

$$
P = (I - K H) P^-
$$

---
