import numpy as np
import jax
import jax.numpy as jnp


def default_measurement_variances(estimator_cfg: dict) -> np.ndarray:
    noise_pos = float(estimator_cfg.get("noise_pos", 1.0))
    noise_angle = float(estimator_cfg.get("noise_angle", 1.0))

    pos_var = max(noise_pos ** 2, 1e-6)
    angle_var = max(noise_angle ** 2, 1e-6)
    return np.array([pos_var, pos_var, angle_var], dtype=float)


def compute_fim_matrix(
    measurement_vector_fn,
    params: jnp.ndarray,
    measurement_variances,
) -> jnp.ndarray:
    measurement_variances = np.asarray(measurement_variances, dtype=float)
    inverse_variances = jnp.asarray(1.0 / measurement_variances, dtype=jnp.float32)

    measurement_vector = measurement_vector_fn(params)
    measurement_sensitivity = jax.jacfwd(measurement_vector_fn)(params)
    parameter_scaling = jnp.diag(params)
    # parameter_scaling = jnp.eye(2)
    weighted_measurement_sensitivity = measurement_sensitivity @ parameter_scaling

    num_measurements = measurement_vector.shape[0] // 3
    stacked_inverse_variances = jnp.tile(inverse_variances, num_measurements)
    weights = stacked_inverse_variances.reshape(-1, 1)

    return weighted_measurement_sensitivity.T @ (weighted_measurement_sensitivity * weights)


def regularize_fim(fim: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    return fim + regularization * jnp.eye(fim.shape[0], dtype=fim.dtype)


def max_inverse_eigenvalue(fim: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    regularized_fim = regularize_fim(fim, regularization=regularization)
    inverse_eigenvalues = jnp.linalg.eigvalsh(jnp.linalg.inv(regularized_fim))
    return jnp.max(inverse_eigenvalues)


def logdet_criterion(fim: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    sign, logabsdet = jnp.linalg.slogdet(regularize_fim(fim, regularization=regularization))
    return jnp.where(sign > 0, -logabsdet, jnp.inf)


def trace_inverse_criterion(fim: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    return jnp.trace(jnp.linalg.inv(regularize_fim(fim, regularization=regularization)))


def condition_number_criterion(fim: jnp.ndarray, regularization: float = 1e-6) -> jnp.ndarray:
    eigenvalues = jnp.linalg.eigvalsh(regularize_fim(fim, regularization=regularization))
    return jnp.max(eigenvalues) / jnp.maximum(jnp.min(eigenvalues), 1e-12)
