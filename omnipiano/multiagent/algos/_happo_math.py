"""Pure-tensor HAPPO mathematics, deliberately free of framework imports.

The part of an algorithm that can be *wrong in a way no smoke test catches* is
the loss. Keeping it here means it is unit tested on synthetic tensors in
milliseconds, on CPU, with no MuJoCo, no Ray and no GPU. It is also what an
oracle test can compare against the reference implementation.

Reference: Kuba et al. (2022), "Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning", ICLR (arXiv:2109.11251), Sec. 4 / Eq. (10).

"""

from __future__ import annotations

import torch

def compound_log_factor_update(
    prev_log_factor: torch.Tensor,
    logp_after_update: torch.Tensor,
    logp_before_update: torch.Tensor,
) -> torch.Tensor:
    """Fold one just-updated agent into the running compound factor M.

    Everything stays in log space: M = exp(sum_k log pi_new^k - log pi_old^k).
    The result is DETACHED -- M measures how much the world already moved, it is
    not a quantity we optimise. Letting gradients flow through it would make
    agent i_m responsible for agent i_1's parameters, which is exactly the
    cross-agent coupling the sequential scheme exists to avoid.
    """
    updated = (
        prev_log_factor
        + logp_after_update.detach()
        - logp_before_update.detach()
    ).detach()
    # The official HAPPO implementations do not add a second clamp to M.  A
    # bespoke clamp changes the objective, so fail loudly instead of silently
    # training a different algorithm if the product overflows.
    if not bool(torch.isfinite(updated).all()) \
            or not bool(torch.isfinite(updated.exp()).all()):
        raise FloatingPointError("Non-finite HAPPO compound factor")
    return updated


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
    the unit tests -- but note it holds BY CONSTRUCTION and therefore does not
    by itself prove that the learner performs strict sequential optimizer steps.
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
