"""The gain block's inner solver: BFGS via optimistix.

The gain block is 5-dimensional, which is the regime where a quasi-Newton method
is at its best -- a dense Hessian approximation costs nothing to store or
factor, and Adam's per-coordinate scaling is a poor substitute for real
curvature when the coordinates are as differently conditioned as
``[kx, ky, kth, kpmotor, kimotor]`` are.

**Adam was removed 2026-08-06** and should not come back. Measured on the gain
block from the stock gains with the trajectories held fixed (initial loss
0.0700): Adam reaches 0.01532 in 52 s of 2000 steps, ``BFGS`` reaches 0.01310 in
0.7 s, and ``cg-fr`` 0.00994 in 4.5 s. Worse than slow, Adam's step is
size-limited by its learning rate, so it cannot reach ``kimotor`` = 0 -- which
sits on the box boundary half an optimizer-space unit from the nominal value --
inside any sane budget. That is what the deleted presearch probe existed to
paper over. A quasi-Newton step has no such limit and gets there by gradient in
a single round.

``lbfgs`` and ``cg-pr`` were removed at the same time and for the same reason:
on this objective at a 200-step budget they reach 0.04807 and 0.04400, i.e. they
barely leave the starting point. ``cg-fr`` (nonlinear CG, Fletcher-Reeves) went
last: it reaches 0.00994, but so does BFGS once you measure it the way the loop
actually uses it -- a fresh bounded solve every round. 5 restarts x 40 steps
gives 0.009939 and the same gains to 5 s.f., in 3.2 s against CG's 4.5 s. At
5-D a quasi-Newton method is the better fit anyway: the dense inverse-Hessian
approximation costs nothing to store, while CG's matrix-free design buys
nothing.

Two things about the alternating setting shape this module.

**One optimistix ``solver.step`` is not one parameter update.** It is one
iteration of the solver's internal line search, so a step can (and usually does)
leave ``y`` untouched while the search backtracks -- measured on a 5-D quadratic,
``BFGS`` moved on step 9 of 12 and on no other. Alternating after each such step
would let the trajectory block move on every round while the gain block moved on
one in nine. So a round here is a bounded *solve* (:func:`optimistix.minimise`
with ``max_steps``), not a step: the stock, documented entry point, and it makes
each round a proper block-coordinate inner minimization.

**The Hessian approximation is rebuilt per round.** ``minimise`` starts fresh
each time, so curvature learned in round N is not carried into round N+1. That
is the cost of using the stock API rather than hand-rolling the state threading,
and it is the honest form of the open question: if the trajectories really do
change little from round to round, a carried Hessian would pay, and this
implementation deliberately does not assume that -- it measures the version that
cannot be wrong instead.

Box constraints are handled by clipping inside the objective. Note the
consequence for ``kimotor``, which is the one gain whose optimum sits *on* the
boundary at 0: clipping makes that direction locally flat, so the Hessian is
near-singular there and the quasi-Newton step in that coordinate is meaningless.
Adam's projection has the same blind spot but degrades more gracefully.
"""

import jax
import jax.numpy as jnp
import optimistix as optx

from wmr_simulator.gain_tuning.optimizers import clip_optimizer_values


# Loose: the block is re-solved every round against a changed objective, so
# solving it tightly is wasted work -- the answer moves before it is used.
_QUASI_NEWTON_RTOL = 1e-4
_QUASI_NEWTON_ATOL = 1e-8

# The inner budget is a cap on a bounded solve whose steps are *line-search
# trials*, not accepted updates, so it does not mean what a step count means.
# Measured on the gain block from stock gains: at a budget of 5 BFGS does not
# leave the initial point at all, at 15 it crawls, at 40 it reaches the
# conditional optimum -- which is where the default comes from.
DEFAULT_GAIN_STEPS_PER_ROUND = 40
# Below this a bounded solve is all line search and no accepted update, i.e. a
# silent no-op. Refuse rather than run a loop that cannot move -- shipping this
# footgun once already cost a full run that returned the stock gains unchanged.
MIN_STEPS_PER_ROUND = 15


def resolve_steps_per_round(steps_per_round: int | None) -> int:
    """The inner budget, with the silent-no-op case refused outright."""
    if steps_per_round is None:
        return DEFAULT_GAIN_STEPS_PER_ROUND
    if steps_per_round < MIN_STEPS_PER_ROUND:
        raise ValueError(
            f"gain_steps_per_round={steps_per_round} is too small: one optimistix step is a "
            "line-search trial, not an accepted update, so a budget this low leaves the gains "
            f"exactly where they started. Use at least {MIN_STEPS_PER_ROUND} "
            f"(default {DEFAULT_GAIN_STEPS_PER_ROUND})."
        )
    return steps_per_round


def make_gain_stepper(gain_loss, steps_per_round: int):
    """Build ``(init_state, step)`` for the gain block.

    ``step(values, state, decision_variables, free_offsets)`` returns
    ``(next_values, next_state, loss_pre, loss_post)``.
    """
    return _make_stepper(gain_loss, steps_per_round)


def _make_stepper(gain_loss, steps_per_round: int):
    # Wrapped so a round can never end on a worse iterate than one it visited:
    # the line search takes uphill trial steps by construction, and the
    # alternation hands whatever comes out straight to the trajectory block.
    solver = optx.BestSoFarMinimiser(
        optx.BFGS(rtol=_QUASI_NEWTON_RTOL, atol=_QUASI_NEWTON_ATOL)
    )

    def objective(values, args):
        """Bare scalar, not ``(value, aux)``: that form is for ``solver.step``,
        while ``minimise`` wants the scalar unless told ``has_aux``."""
        decision_variables, free_offsets = args
        # Clip inside, so the objective is defined off the box and the solver
        # never has to be told about the constraint. See the module docstring
        # for what this costs at the kimotor boundary.
        return gain_loss(clip_optimizer_values(values), decision_variables, free_offsets)

    @jax.jit
    def step(values, state, decision_variables, free_offsets):
        args = (decision_variables, free_offsets)
        loss_pre = gain_loss(values, decision_variables, free_offsets)
        solution = optx.minimise(
            objective,
            solver,
            values,
            args=args,
            max_steps=steps_per_round,
            # A bounded inner solve is expected to hit the step cap without
            # converging -- that is the budget doing its job, not a failure.
            throw=False,
        )
        next_values = clip_optimizer_values(solution.value)
        loss_post = gain_loss(next_values, decision_variables, free_offsets)
        # A non-finite or uphill solve leaves the block where it was: a
        # quasi-Newton step on a near-singular Hessian can return garbage, and
        # the alternation has no way to recover from a poisoned iterate.
        keep = jnp.isfinite(loss_post) & (loss_post <= loss_pre)
        return (
            jnp.where(keep, next_values, values),
            state,
            loss_pre,
            jnp.where(keep, loss_post, loss_pre),
        )

    # The solver carries no state across rounds (see the module docstring), so
    # the "state" is a placeholder that keeps the loop's plumbing uniform.
    return (lambda values: None), step
