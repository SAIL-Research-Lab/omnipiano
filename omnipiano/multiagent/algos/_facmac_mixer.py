"""QMIX-style factorisation of a joint Q used by FACMAC's centralised critic.

Reference: Peng et al. (2021), "FACMAC: Factored Multi-Agent Centralised Policy
Gradients", NeurIPS (arXiv:2003.06709); mixer design after Rashid et al. (QMIX).

FACMAC differs from MADDPG in two ways, both captured here and in the AlgoSpec:
  1. the centralised critic is FACTORED -- per-agent utilities combined by a
     state-conditioned network -- instead of monolithic;
  2. the policy gradient is taken over the ENTIRE JOINT action, not per agent
     with the others frozen, which is what lets policies move together.

Monotonic mode reproduces QMIX's constraint (dQ_tot/dQ_i >= 0), guaranteeing
that the per-agent argmax is the joint argmax. Non-monotonic mode drops the
absolute value, buying representational capacity at the cost of that guarantee;
the FACMAC paper shows tasks solvable only in the latter mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactoredMixer(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 64,
                 monotonic: bool = True) -> None:
        super().__init__()
        self.n_agents, self.embed_dim, self.monotonic = n_agents, embed_dim, monotonic
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(),
                                      nn.Linear(embed_dim, n_agents * embed_dim))
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(),
                                      nn.Linear(embed_dim, embed_dim))
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(),
                                      nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """agent_qs: (B, n_agents); state: (B, state_dim) -> q_tot: (B,)."""
        b = agent_qs.shape[0]
        w1 = self.hyper_w1(state).view(b, self.n_agents, self.embed_dim)
        w2 = self.hyper_w2(state).view(b, self.embed_dim, 1)
        if self.monotonic:
            # Non-negative weights => dQ_tot/dQ_i >= 0 (QMIX's IGM condition).
            w1, w2 = w1.abs(), w2.abs()
        h = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1)
                       + self.hyper_b1(state).unsqueeze(1))
        return (torch.bmm(h, w2).squeeze(-1).squeeze(-1) + self.hyper_b2(state).squeeze(-1))