"""QMIX factorisation used by FACMAC's centralised critic.

Reference: Peng et al. (2021), "FACMAC: Factored Multi-Agent Centralised Policy
Gradients", NeurIPS (arXiv:2003.06709); mixer design after Rashid et al. (QMIX).

FACMAC differs from MADDPG in two ways, both captured here and in the AlgoSpec:
  1. the centralised critic is FACTORED -- per-agent utilities combined by a
     state-conditioned network -- instead of monolithic;
  2. the policy gradient is taken over the ENTIRE JOINT action, not per agent
     with the others frozen, which is what lets policies move together.

The layer layout follows ``oxwhirl/facmac`` commit
``d7e62b8c51a5a77330de85f83c10553d0bd18fe5``, file
``src/modules/mixers/qmix.py``: two-layer hypernetworks, ELU mixer hidden
activation, absolute hypernetwork weights, and a state-conditioned ``V(s)``.
That repository has no redistributable licence, so its source is not vendored;
the implementation below is a clean-room transcription of the published
equations and is guarded by numerical layout tests.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactoredMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64,
                 monotonic: bool = True, hypernet_embed: int = 64) -> None:
        super().__init__()
        self.n_agents, self.embed_dim, self.monotonic = n_agents, embed_dim, monotonic
        self.state_dim = state_dim
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * embed_dim),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.state_value = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """agent_qs: (B, n_agents); state: (B, state_dim) -> q_tot: (B,)."""
        if agent_qs.ndim != 2 or agent_qs.shape[1] != self.n_agents:
            raise ValueError(
                f"agent_qs must have shape (B, {self.n_agents}), got "
                f"{tuple(agent_qs.shape)}"
            )
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape (B, {self.state_dim}), got "
                f"{tuple(state.shape)}"
            )
        b = agent_qs.shape[0]
        w1 = self.hyper_w1(state).view(b, self.n_agents, self.embed_dim)
        w2 = self.hyper_w2(state).view(b, self.embed_dim, 1)
        if self.monotonic:
            # Non-negative weights => dQ_tot/dQ_i >= 0 (QMIX's IGM condition).
            w1, w2 = w1.abs(), w2.abs()
        h = torch.nn.functional.elu(
            torch.bmm(agent_qs.unsqueeze(1), w1)
            + self.hyper_b1(state).unsqueeze(1)
        )
        return (
            torch.bmm(h, w2).squeeze(-1).squeeze(-1)
            + self.state_value(state).squeeze(-1)
        )


# Descriptive alias used in new code; the historical name remains public for
# checkpoint/module-path compatibility.
QMixer = FactoredMixer
