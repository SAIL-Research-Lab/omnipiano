"""Encoder-decoder core of the Multi-Agent Transformer (MAT).

Reference: Wen et al. (2022), "Multi-Agent Reinforcement Learning is a Sequence
Modeling Problem", NeurIPS (arXiv:2205.14953).

Scope of this file
------------------
The NETWORK is here and is unit tested (shapes + the causal-mask property that
agent m cannot see agent m+1's action). What is NOT here, and is the actual
blocking work, is the RLlib integration: MAT samples actions AUTOREGRESSIVELY
across agents within a single environment step, which RLlib's per-module
``forward_exploration`` cannot express. That needs a MultiRLModule plus a
ConnectorV2 that runs n sequential decoder passes per env step.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class _SelfAttention(nn.Module):
    """Multi-head self-attention over the AGENT axis, optionally causal."""

    def __init__(self, dim: int, n_heads: int, causal: bool) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads, self.head_dim, self.causal = n_heads, dim // n_heads, causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, n, self.n_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            # Agent m may attend to 1..m only. This mask IS the multi-agent
            # advantage decomposition: it is what makes the autoregressive
            # factorisation of the joint policy valid.
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
        out = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, causal: bool) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, n_heads, causal)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MATEncoder(nn.Module):
    """Observations of all agents -> latent tokens + per-agent state value."""

    def __init__(self, obs_dim: int, dim: int = 128, n_heads: int = 4,
                 n_blocks: int = 2) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(obs_dim), nn.Linear(obs_dim, dim), nn.GELU())
        self.blocks = nn.ModuleList([_Block(dim, n_heads, causal=False)
                                     for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(dim)
        self.v_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, n_agents, obs_dim) -> latent (B, n, dim), values (B, n)."""
        h = self.embed(obs)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return h, self.v_head(h).squeeze(-1)


class MATDecoder(nn.Module):
    """Latent tokens + shifted actions -> per-agent action distribution params.

    Continuous actions: emits mean and (state-independent) log-std, matching the
    Gaussian head used by the CTDE baselines so the comparison is about the
    ARCHITECTURE, not about the action distribution family.
    """

    def __init__(self, act_dim: int, dim: int = 128, n_heads: int = 4,
                 n_blocks: int = 2) -> None:
        super().__init__()
        self.act_embed = nn.Sequential(nn.Linear(act_dim, dim), nn.GELU())
        self.blocks = nn.ModuleList([_Block(dim, n_heads, causal=True)
                                     for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(dim)
        self.mean_head = nn.Linear(dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, latent: torch.Tensor,
                shifted_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """shifted_actions[:, m] must be action of agent m-1 (zeros for m=0)."""
        h = latent + self.act_embed(shifted_actions)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        mean = self.mean_head(h)
        return mean, self.log_std.expand_as(mean)