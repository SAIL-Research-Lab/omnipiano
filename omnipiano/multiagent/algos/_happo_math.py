"""Pure-tensor HAPPO mathematics, deliberately free of RLlib imports.

The part of an algorithm that can be *wrong in a way no smoke test catches* is
the loss. Keeping it here -- and the RLlib plumbing in ``_happo_learner.py`` --
means it is unit tested on synthetic tensors in milliseconds, on CPU, with no
MuJoCo, no Ray and no GPU.

Reference: Kuba et al. (2022), "Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning", ICLR (arXiv:2109.11251), Sec. 4 / Eq. (10).
"""

from __future__ import annotations

import torch

# The compound ratio is a PRODUCT over already-updated agents, so it explodes
# geometrically in n. HARL clamps it; we clamp in log space, which is both
# numerically safer and easier to monitor (we log the clamp hit rate).
DEFAULT_MAX_ABS_LOG_FACTOR = 2.302585092994046  # ln(10) -> factor in [0.1, 10]


def compound_log_factor_update(
    prev_log_factor: torch.Tensor,
    logp_after_update: torch.Tensor,
    logp_before_update: torch.Tensor,
    max_abs_log_factor: float = DEFAULT_MAX_ABS_LOG_FACTOR,
) -> torch.Tensor:
    """Fold one just-updated agent into the running compound factor M.

    Everything is kept in log space: M = exp(sum_k log pi_new^k - log pi_old^k).
    The result is DETACHED -- M measures how much the world already moved, it is
    not a quantity we optimise. Letting gradients flow through it would make
    agent i_m responsible for agent i_1's parameters, which is exactly the
    cross-agent coupling the sequential scheme exists to avoid.
    """
    delta = (logp_after_update - logp_before_update).detach()
    return torch.clamp(prev_log_factor + delta,
                       min=-max_abs_log_factor, max=max_abs_log_factor)


def happo_surrogate(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    compound_log_factor: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """HAPPO's clipped surrogate: PPO's, with the advantage reweighted by M.

    Returns the quantity to be MAXIMISED (so a loss is its negation), matching
    RLlib's ``surrogate_loss`` sign convention before negation.

    With ``compound_log_factor == 0`` (i.e. M == 1, the first agent in the
    permutation, or a single-agent problem) this reduces EXACTLY to PPO. That
    identity is the strongest available test of this function and is asserted in
    ``algos_test.py``.
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