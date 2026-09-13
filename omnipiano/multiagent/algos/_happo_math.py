"""Pure-tensor HAPPO mathematics, deliberately free of RLlib imports.

The part of an algorithm that can be *wrong in a way no smoke test catches* is
the loss. Keeping it here -- and the RLlib plumbing in ``_happo_learner.py`` --
means it is unit tested on synthetic tensors in milliseconds, on CPU, with no
MuJoCo, no Ray and no GPU. It is also what an oracle test can compare against
the reference implementation (see ``external/README.md``).

Reference: Kuba et al. (2022), "Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning", ICLR (arXiv:2109.11251), Sec. 4 / Eq. (10).

ORDERING MATTERS IN THIS FILE. Python evaluates default-argument expressions at
``def`` time, so every constant used as a default must be defined ABOVE the
functions that reference it. Moving DEFAULT_MAX_ABS_LOG_FACTOR below
clamp_log_factor raises NameError at import, taking the whole registry with it.
"""

from __future__ import annotations

import torch

# The compound ratio is a PRODUCT over already-updated agents, so it explodes
# geometrically in the number of agents. HARL clamps it; we clamp in log space,
# which keeps the bound symmetric (a factor of 10 and of 1/10 are equidistant)
# and makes the clamp hit rate a directly interpretable trust-region diagnostic.
DEFAULT_MAX_ABS_LOG_FACTOR = 2.302585092994046  # ln(10) -> factor in [0.1, 10]


def clamp_log_factor(
    log_factor: torch.Tensor,
    max_abs_log_factor: float = DEFAULT_MAX_ABS_LOG_FACTOR,
) -> torch.Tensor:
    """Bound the compound factor symmetrically in log space, detached."""
    return torch.clamp(log_factor.detach(),
                       min=-max_abs_log_factor, max=max_abs_log_factor)


def compound_log_factor_update(
    prev_log_factor: torch.Tensor,
    logp_after_update: torch.Tensor,
    logp_before_update: torch.Tensor,
    max_abs_log_factor: float = DEFAULT_MAX_ABS_LOG_FACTOR,
) -> torch.Tensor:
    """Fold one just-updated agent into the running compound factor M.

    Everything stays in log space: M = exp(sum_k log pi_new^k - log pi_old^k).
    The result is DETACHED -- M measures how much the world already moved, it is
    not a quantity we optimise. Letting gradients flow through it would make
    agent i_m responsible for agent i_1's parameters, which is exactly the
    cross-agent coupling the sequential scheme exists to avoid.
    """
    delta = (logp_after_update - logp_before_update).detach()
    return clamp_log_factor(prev_log_factor + delta, max_abs_log_factor)


def happo_surrogate(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    compound_log_factor: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """HAPPO's clipped surrogate: PPO's, with the advantage reweighted by M.

    Returns the quantity to be MAXIMISED (a loss is its negation), matching
    RLlib's ``surrogate_loss`` sign convention before negation.

    With ``compound_log_factor == 0`` (M == 1: the first agent in the
    permutation, or a single-agent problem) this reduces EXACTLY to PPO. That
    identity is the cheapest available test of this function and is asserted in
    ``algos_test.py`` -- but note it holds BY CONSTRUCTION and therefore proves
    nothing about the learner that calls it. See happo-m1-control for the test
    that does.
    """
    ratio = torch.exp(logp_new - logp_old)
    weighted_adv = torch.exp(compound_log_factor) * advantages
    return torch.min(ratio * weighted_adv,
                     torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * weighted_adv)


def ppo_surrogate(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """PPO's clipped surrogate == happo_surrogate with M == 1."""
    return happo_surrogate(logp_new, logp_old, advantages,
                           torch.zeros_like(advantages), clip_param)