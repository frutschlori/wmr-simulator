"""Fisher information for the trajectory design criteria.

Everything downstream of :func:`compute_fim_factor` works on the *factor*
``J~`` (the noise-weighted, relatively-scaled measurement sensitivity) rather
than on ``FIM = J~^T J~`` itself, because forming the FIM squares its condition
number and this pipeline cannot afford that in float32.

Measured on the gain-tuning objective: a constraint-violating reference
destabilizes the closed loop and the 5-gain FIM eigenvalues spread to
``[1.5e4 ... 2.9e13]``, i.e. cond ~2e9. Accumulating ``J^T (J w)`` in float32
then loses positive-definiteness outright (an eigenvalue of ``-1.4e5`` was
observed, a relative error of 3e-9 ~ float32 eps), the smallest eigenvalue ends
up *below* the float32 noise floor of the product, and the old explicit
``jnp.linalg.inv`` returned NaN where float64 gives a perfectly ordinary
1.07e-4. A single NaN then poisons Adam's moment estimates permanently, which
is what made the loss history choppy.

Working from ``J~`` halves the exponent: an f32 QR of ``J~`` is backward stable
relative to ``||J~|| = sqrt(lambda_max)``, so cond(FIM) ~ 1e12 is still
comfortably resolvable. ``J~ = Q R`` gives ``FIM = R^T R`` with ``R`` a tiny
``[P, P]`` triangle, and every criterion below is a closed form in ``R`` -- no
inverse of an ill-conditioned matrix is ever formed. Tikhonov regularization is
applied by *appending* ``sqrt(reg) I`` rows to the factor, which is exactly
equivalent to ``FIM + reg I`` and keeps the whole path in factored form.
"""

import numpy as np
import jax
import jax.numpy as jnp


def default_measurement_variances(estimator_cfg: dict) -> np.ndarray:
    noise_pos = float(estimator_cfg.get("noise_pos", 1.0))
    noise_angle = float(estimator_cfg.get("noise_angle", 1.0))

    pos_var = max(noise_pos ** 2, 1e-6)
    angle_var = max(noise_angle ** 2, 1e-6)
    return np.array([pos_var, pos_var, angle_var], dtype=float)


def compute_fim_factor(
    measurement_vector_fn,
    params: jnp.ndarray,
    measurement_variances,
    parameter_scaling: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Noise-weighted relative measurement sensitivity ``J~``, ``FIM = J~^T J~``.

    Shape ``[3N, P]``. This, not the FIM, is the quantity to pass around: see the
    module docstring on why squaring it in float32 destroys the criteria.
    """
    measurement_variances = np.asarray(measurement_variances, dtype=float)
    inverse_variances = jnp.asarray(1.0 / measurement_variances, dtype=jnp.float32)

    params = jnp.asarray(params, dtype=jnp.float32)
    # jacfwd evaluates the primal internally; a separate measurement_vector_fn(params)
    # call would be a second, redundant rollout whose only use was its (static) shape.
    measurement_sensitivity = jax.jacfwd(measurement_vector_fn)(params)
    if parameter_scaling is None:
        parameter_scaling = params
    parameter_scaling = jnp.diag(jnp.asarray(parameter_scaling, dtype=jnp.float32))
    relative_measurement_sensitivity = measurement_sensitivity @ parameter_scaling

    num_measurements = measurement_sensitivity.shape[0] // 3
    stacked_inverse_variances = jnp.tile(inverse_variances, num_measurements)
    return relative_measurement_sensitivity * jnp.sqrt(stacked_inverse_variances)[:, None]


def compute_fim_matrix(
    measurement_vector_fn,
    params: jnp.ndarray,
    measurement_variances,
    parameter_scaling: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """The FIM itself. For *reporting* -- the criteria take the factor instead."""
    return fim_from_factor(
        compute_fim_factor(
            measurement_vector_fn,
            params=params,
            measurement_variances=measurement_variances,
            parameter_scaling=parameter_scaling,
        )
    )


def fim_from_factor(fim_factor: jnp.ndarray) -> jnp.ndarray:
    return fim_factor.T @ fim_factor


def _triangular_factor(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    """Upper triangular ``R`` with ``R^T R = FIM + regularization * I``.

    Appending ``sqrt(reg) I`` rows to the factor is the factored form of the
    Tikhonov shift, so the regularization never requires forming the FIM.
    """
    num_parameters = fim_factor.shape[1]
    regularization_rows = jnp.sqrt(
        jnp.asarray(regularization, dtype=fim_factor.dtype)
    ) * jnp.eye(num_parameters, dtype=fim_factor.dtype)
    return jnp.linalg.qr(
        jnp.concatenate([fim_factor, regularization_rows], axis=0), mode="r"
    )


def _fim_singular_values(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    """Singular values of ``R``; the FIM eigenvalues are their squares."""
    return jnp.linalg.svd(
        _triangular_factor(fim_factor, regularization=regularization), compute_uv=False
    )


def max_inverse_eigenvalue(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    return 1.0 / _fim_singular_values(fim_factor, regularization=regularization)[-1] ** 2


def logdet_criterion(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    triangular_factor = _triangular_factor(fim_factor, regularization=regularization)
    # det(FIM) = det(R)^2 and R is triangular, so this is exact and always finite
    # for a full-rank R -- no sign check needed, unlike slogdet on the FIM.
    return -2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(triangular_factor))))


def trace_inverse_criterion(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    # FIM^-1 = R^-1 R^-T, so trace(FIM^-1) = ||R^-1||_F^2; one triangular solve
    # replaces inverting an ill-conditioned symmetric matrix.
    triangular_factor = _triangular_factor(fim_factor, regularization=regularization)
    inverse_triangular_factor = jax.scipy.linalg.solve_triangular(
        triangular_factor,
        jnp.eye(triangular_factor.shape[0], dtype=triangular_factor.dtype),
        lower=False,
    )
    return jnp.sum(inverse_triangular_factor**2)


def condition_number_criterion(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    singular_values = _fim_singular_values(fim_factor, regularization=regularization)
    return (singular_values[0] / jnp.maximum(singular_values[-1], 1e-12)) ** 2
