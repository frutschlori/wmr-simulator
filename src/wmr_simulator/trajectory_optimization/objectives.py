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
    """Design criterion, on the FIM *factor* (``FIM = fim_factor^T fim_factor``);
    see ``fim.py`` on why the criteria never take the assembled FIM.

    Smaller is better for all three: a sum of relative variances (A), the
    negative log-determinant of the information (D), the largest relative
    variance (E).
    """
    if criterion == CRITERION_D_OPTIMALITY:
        return logdet_criterion(fim_factor)
    if criterion == CRITERION_E_OPTIMALITY:
        return max_inverse_eigenvalue(fim_factor)
    return trace_inverse_criterion(fim_factor)


def fim_objective_term(fim_factor: jnp.ndarray, criterion: str = DEFAULT_CRITERION) -> jnp.ndarray:
    """The FIM half of the objective, in log-information units for every
    criterion.

    A and E are variances, so they get the ``log`` that makes their gradient the
    *relative* change in the criterion. D is ``-logdet(FIM)``, which is already
    in exactly those units -- and is signed, so a ``log`` of it would be NaN
    half the time. Putting all criteria on the same scale is what lets the one
    ``constraint_violation_tolerance`` keep its meaning when the criterion is
    swapped: a ``g_tol`` fractional over-limit costs a ``g_tol`` relative loss
    of information either way.
    """
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
    """``fim_objective_term + constraint_loss / constraint_violation_tolerance``.

    For the default A-optimality the first term is ``log(trace(FIM^-1))``. The
    log is monotone, so at a fixed constraint level it does not move the
    minimizer; what it changes is that the FIM term's gradient becomes the
    *relative* change in the criterion, which makes it scale-free with no state
    to carry. That matters for alternating gain/trajectory optimization, where
    the FIM's absolute scale changes every round because its design point (the
    gains) moves: a round-0 normalization constant would either go stale or, if
    refreshed, make the objective discontinuous between rounds and corrupt
    Adam's moments.
    """
    return fim_objective_term(fim_factor, criterion) + (
        1.0 / constraint_violation_tolerance
    ) * constraint_loss_from_reference_states(
        reference_states=reference_states,
        dt=dt,
        limits=limits,
        weights=weights,
        smooth_max_beta=smooth_max_beta,
    )
