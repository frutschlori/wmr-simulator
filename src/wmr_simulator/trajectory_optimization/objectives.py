import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.constraints import constraint_loss_from_reference_states
from wmr_simulator.trajectory_optimization.fim import max_inverse_eigenvalue, trace_inverse_criterion, logdet_criterion


def fim_loss(fim: jnp.ndarray) -> jnp.ndarray:
    return max_inverse_eigenvalue(fim)
    # return trace_inverse_criterion(fim)
    # return logdet_criterion(fim)

def trajectory_objective(
    fim: jnp.ndarray,
    reference_states: jnp.ndarray,
    dt: float,
    limits: dict,
    weights: dict,
    smooth_max_beta: float = 20.0,
) -> jnp.ndarray:
    return fim_loss(fim) + constraint_loss_from_reference_states(
        reference_states=reference_states,
        dt=dt,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
