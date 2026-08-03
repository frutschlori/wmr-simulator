import jax.numpy as jnp

from wmr_simulator.trajectory_optimization.constraints import constraint_loss_from_reference_states
from wmr_simulator.trajectory_optimization.fim import max_inverse_eigenvalue, trace_inverse_criterion, logdet_criterion


# Fractional over-limit that is allowed to cost as much as an equal fractional
# loss of A-optimality. The constraint penalty is ``g**2`` with ``g`` the
# smooth-max fractional violation (constraints.py), and the FIM term below is in
# log units, where a fractional change ``d`` costs ``~d``; weighting the penalty
# by ``1 / g_tol`` therefore makes a ``g_tol`` violation cost exactly a ``g_tol``
# relative loss of information. 0.05 -> weight 20.
#
# Why this rather than the raw sum the objective used to be: measured on the
# gain-tuning objective, ``trace(FIM^-1)`` sits at ~4.7e-4 and moves ~2% per
# Adam step, so an unweighted penalty had an effective weight of ~2100 in units
# of the FIM term's own scale -- a 2% limit violation doubled the total loss and
# essentially all of the loss history's choppiness was Adam bouncing off that
# wall rather than anything about the information content.
DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE = 0.05


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
    constraint_violation_tolerance: float = DEFAULT_CONSTRAINT_VIOLATION_TOLERANCE,
) -> jnp.ndarray:
    """``log(fim_loss) + constraint_loss / constraint_violation_tolerance``.

    The log is monotone, so at a fixed constraint level it does not move the
    minimizer; what it changes is that the FIM term's gradient becomes the
    *relative* change in the criterion, which makes it scale-free with no state
    to carry. That matters for alternating gain/trajectory optimization, where
    the FIM's absolute scale changes every round because its design point (the
    gains) moves: a round-0 normalization constant would either go stale or, if
    refreshed, make the objective discontinuous between rounds and corrupt
    Adam's moments.
    """
    return jnp.log(fim_loss(fim_factor)) + (
        1.0 / constraint_violation_tolerance
    ) * constraint_loss_from_reference_states(
        reference_states=reference_states,
        dt=dt,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
