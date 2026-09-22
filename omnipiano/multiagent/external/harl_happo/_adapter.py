"""Pinned tensor transcription of HARL's HAPPO actor objective.

Source: ``harl/algorithms/actors/happo.py`` plus the ``prod`` action
aggregation in ``harl/runners/on_policy_ha_runner.py`` at the commit below.
OmniPiano stores a summed diagonal-Gaussian log-probability for each agent, so
``exp(logp_new - logp_old)`` is exactly that product over action dimensions.
"""

from __future__ import annotations

import torch


UPSTREAM_URL = "https://github.com/PKU-MARL/HARL"
UPSTREAM_COMMIT = "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"


def happo_surrogate_reference(
    *,
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    compound_log_factor: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """Return HARL's unreduced actor surrogate before loss negation."""
    importance_weight = torch.exp(logp_new - logp_old)
    surrogate_1 = importance_weight * advantages
    surrogate_2 = (
        torch.clamp(importance_weight, 1.0 - clip_param, 1.0 + clip_param)
        * advantages
    )
    return torch.exp(compound_log_factor) * torch.minimum(
        surrogate_1, surrogate_2
    )
