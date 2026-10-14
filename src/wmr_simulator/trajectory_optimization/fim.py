"""Fisher information for the trajectory design criteria.

Everything works on the factor ``J~`` (the noise-weighted, relatively scaled
measurement sensitivity) rather than on ``FIM = J~^T J~``, because forming the
FIM squares its condition number, which float32 cannot resolve. ``J~ = Q R``
gives ``FIM = R^T R`` with a small triangular ``R``; Tikhonov regularization is
applied by appending ``sqrt(reg) I`` rows to the factor.

:func:`trace_inverse_criterion` (A-optimality, in the optimizer's inner loop)
uses the Cholesky of the small Gram matrix instead, whose backward pass is
much cheaper than the QR's.
"""

import numpy as np
import jax
import jax.numpy as jnp


# Relative component of the Tikhonov shift (times the trace), on top of the
# absolute one; keeps the float32 Gram positive definite.
_RELATIVE_REGULARIZATION = 1e-6


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
    """Noise-weighted relative measurement sensitivity ``J~`` ``[3N, P]``,
    ``FIM = J~^T J~``."""
    measurement_variances = np.asarray(measurement_variances, dtype=float)
    inverse_variances = jnp.asarray(1.0 / measurement_variances, dtype=jnp.float32)

    params = jnp.asarray(params, dtype=jnp.float32)
    measurement_sensitivity = jax.jacfwd(measurement_vector_fn)(params)
    if parameter_scaling is None:
        parameter_scaling = params
    parameter_scaling = jnp.diag(jnp.asarray(parameter_scaling, dtype=jnp.float32))
    relative_measurement_sensitivity = measurement_sensitivity @ parameter_scaling

    num_measurements = measurement_sensitivity.shape[0] // 3
    stacked_inverse_variances = jnp.tile(inverse_variances, num_measurements)
    return relative_measurement_sensitivity * jnp.sqrt(stacked_inverse_variances)[:, None]


def _triangular_factor(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    """Upper triangular ``R`` with ``R^T R = FIM + regularization * I``."""
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
    # det(FIM) = det(R)^2 with R triangular.
    return -2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(triangular_factor))))


def _regularized_gram(fim_factor: jnp.ndarray, regularization: float) -> jnp.ndarray:
    """``FIM + shift I`` as a symmetrized ``[P, P]`` matrix (absolute plus
    relative shift)."""
    gram = fim_factor.T @ fim_factor
    gram = 0.5 * (gram + gram.T)
    num_parameters = gram.shape[0]
    shift = regularization + _RELATIVE_REGULARIZATION * jnp.trace(gram)
    return gram + shift * jnp.eye(num_parameters, dtype=gram.dtype)


def trace_inverse_criterion(fim_factor: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    """A-optimality, ``trace(FIM^-1) = ||L^-1||_F^2`` with ``FIM = L L^T``
    (Cholesky of the small Gram matrix)."""
    lower = jnp.linalg.cholesky(_regularized_gram(fim_factor, regularization))
    inverse_lower = jax.scipy.linalg.solve_triangular(
        lower, jnp.eye(lower.shape[0], dtype=lower.dtype), lower=True
    )
    return jnp.sum(inverse_lower**2)
