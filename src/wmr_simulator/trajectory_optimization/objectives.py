import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.constraints import constraint_loss_from_reference_states
from wmr_simulator.trajectory_optimization.fim import max_inverse_eigenvalue, trace_inverse_criterion, logdet_criterion


def fim_loss(fim_factor: jnp.ndarray) -> jnp.ndarray:
    """Design criterion, on the FIM *factor* (``FIM = fim_factor^T fim_factor``);
    see ``fim.py`` on why the criteria never take the assembled FIM."""
    # return max_inverse_eigenvalue(fim_factor)
    return trace_inverse_criterion(fim_factor)
    # return logdet_criterion(fim_factor)

def trajectory_objective(
    fim_factor: jnp.ndarray,
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    return fim_loss(fim_factor) + constraint_loss_from_reference_states(
        reference_states=reference_states,
        dt=dt,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
