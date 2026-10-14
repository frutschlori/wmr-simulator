import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.constraints import constraint_loss_from_reference_states
from wmr_simulator.trajectory_optimization.fim import max_inverse_eigenvalue, trace_inverse_criterion, logdet_criterion


# The constraint penalty is divided by this tolerance, so a fractional
# violation of this size costs as much as the same relative loss of information
# (the FIM term is in log units).
DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE = 0.03


CRITERION_A_OPTIMALITY = "a-optimality"
CRITERION_D_OPTIMALITY = "d-optimality"
CRITERION_E_OPTIMALITY = "e-optimality"
CRITERIA = (CRITERION_A_OPTIMALITY, CRITERION_D_OPTIMALITY, CRITERION_E_OPTIMALITY)
DEFAULT_CRITERION = CRITERION_A_OPTIMALITY


def normalize_criterion(criterion: str) -> str:
    criterion = criterion.strip().lower().replace("_", "-")
    if criterion not in CRITERIA:
        raise ValueError(
            f"Unsupported design criterion '{criterion}'. Expected one of {list(CRITERIA)}."
        )
    return criterion


def fim_loss(fim_factor: jnp.ndarray, criterion: str = DEFAULT_CRITERION) -> jnp.ndarray:
    """Design criterion on the FIM factor (``FIM = fim_factor^T fim_factor``),
    smaller is better: trace of the inverse (A), negative log-determinant (D),
    largest eigenvalue of the inverse (E)."""
    if criterion == CRITERION_D_OPTIMALITY:
        return logdet_criterion(fim_factor)
    if criterion == CRITERION_E_OPTIMALITY:
        return max_inverse_eigenvalue(fim_factor)
    return trace_inverse_criterion(fim_factor)


def fim_objective_term(fim_factor: jnp.ndarray, criterion: str = DEFAULT_CRITERION) -> jnp.ndarray:
    """The FIM term of the objective in log units (``log`` of A and E,
    ``-logdet`` for D)."""
    if criterion == CRITERION_D_OPTIMALITY:
        return fim_loss(fim_factor, criterion)
    return jnp.log(fim_loss(fim_factor, criterion))


def trajectory_objective(
    fim_factor: jnp.ndarray,
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
    constraint_violation_tolerance: float = DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
    criterion: str = DEFAULT_CRITERION,
) -> jnp.ndarray:
    """``fim_objective_term + constraint_loss / constraint_violation_tolerance``
    (for A-optimality the first term is ``log(trace(FIM^-1))``)."""
    return fim_objective_term(fim_factor, criterion) + (
        1.0 / constraint_violation_tolerance
    ) * constraint_loss_from_reference_states(
        reference_states=reference_states,
        dt=dt,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
