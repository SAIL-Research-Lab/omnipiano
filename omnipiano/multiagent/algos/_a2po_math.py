"""Pure A2PO mathematics with no environment or training-framework imports.

Reference: Wang et al. (2023), "Order Matters: Agent-by-agent Policy
Optimization", ICLR, Eq. (2), Eq. (6), Algorithm 1.

The official repository contains many switches for A2PO and its ablations.
These functions intentionally implement only the canonical combination:
PreOPC, clipping the *product* of preceding-agent ratios, joint-ratio PPO
clipping, semi-greedy ordering, and the near-linear adaptive clip schedule.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch


def adaptive_clip(
    base_clip: float,
    update_position: int,
    num_agents: int,
    weight: float = 0.5,
) -> float:
    """Official ``near_linear`` schedule for a one-based update position.

    The paper writes ``eps*c + eps*(1-c)*k/n``. The public implementation
    calls ``1-c`` the weight, hence the equivalent expression below.
    """
    if not 0.0 < base_clip < 1.0:
        raise ValueError("base_clip must be in (0, 1)")
    if type(update_position) is not int or type(num_agents) is not int \
            or not 1 <= update_position <= num_agents:
        raise ValueError("update_position must be one-based and within num_agents")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("adaptive clip weight must be in [0, 1]")
    return base_clip * (
        (1.0 - weight) + weight * update_position / num_agents
    )


def clipped_preceding_ratio(
    preceding_log_ratio: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """Clip a preceding-agent *joint* density ratio without ``exp`` overflow."""
    if not 0.0 < clip_param < 1.0:
        raise ValueError("preceding ratio clip must be in (0, 1)")
    if not bool(torch.isfinite(preceding_log_ratio).all()):
        raise FloatingPointError("Non-finite preceding-agent log ratio")
    lower = math.log(1.0 - clip_param)
    upper = math.log(1.0 + clip_param)
    return preceding_log_ratio.clamp(lower, upper).exp()


def a2po_surrogate(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    preceding_log_ratio: torch.Tensor,
    clip_param: float,
    preceding_clip_param: float,
) -> torch.Tensor:
    """A2PO Eq. (6), returned as an unreduced quantity to MAXIMISE.

    Gradients flow only through the current agent's probability ratio. The
    preceding policies have already completed their optimizer steps and their
    exact ratios are therefore detached context, not joint optimization terms.
    """
    preceding = clipped_preceding_ratio(
        preceding_log_ratio.detach(), preceding_clip_param)
    current = torch.exp(logp_new - logp_old)
    joint = current * preceding
    if not bool(torch.isfinite(joint).all()):
        raise FloatingPointError("Non-finite A2PO joint policy ratio")
    return torch.minimum(
        joint * advantages,
        joint.clamp(1.0 - clip_param, 1.0 + clip_param) * advantages,
    )


def official_order_scores(
    advantages: torch.Tensor,
    value_predictions: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Score agents as the public implementation does: mean ``|A_norm / V|``.

    The paper describes expected absolute advantage. The public code first
    standardizes advantages over the complete joint rollout and then divides
    by the (ValueNorm-space) rollout value prediction before taking the
    absolute value. We retain that implementation choice but add an explicit
    denominator floor; without it an exactly-zero initialized critic makes the
    selection rule undefined.
    """
    if advantages.ndim != 2 or advantages.shape != value_predictions.shape:
        raise ValueError("advantages and values must have shape [rows, agents]")
    if epsilon <= 0.0:
        raise ValueError("order score epsilon must be positive")
    normalized = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-5)
    denominator = value_predictions.abs().clamp_min(epsilon)
    scores = (normalized.abs() / denominator).mean(dim=0)
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Non-finite A2PO order score")
    return scores


def semi_greedy_order(
    scores: Sequence[float] | np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """Alternate highest-score and uniformly random remaining agents.

    Sorting is stable, matching the official implementation's agent-index
    tie-break. Passing ``rng=None`` uses NumPy's checkpointed global RNG.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    remaining = sorted(
        [(i, float(score)) for i, score in enumerate(values)],
        key=lambda pair: pair[1],
        reverse=True,
    )
    order: list[int] = []
    while remaining:
        order.append(remaining.pop(0)[0])
        if remaining:
            choice = (
                int(rng.integers(len(remaining)))
                if rng is not None
                else int(np.random.choice(len(remaining)))
            )
            order.append(remaining.pop(choice)[0])
    return order


def preopc_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    preceding_log_ratio: torch.Tensor,
    keys: torch.Tensor,
    *,
    gamma: float,
    trace_lambda: float,
    trace_clip_param: float = 1.0,
) -> torch.Tensor:
    """Compute Eq. (2) on interleaved vector-environment trajectories.

    ``keys`` rows are ``[stream, episode, timestep]``. The weight multiplying
    ``delta[t+k]`` starts at the ratio at ``t+1`` exactly as Eq. (2) requires.
    True terminations do not bootstrap. Time-limit truncations bootstrap from
    their final observation but never connect GAE to the reset episode.

    The corrected target is detached by definition. A small CPU scan avoids
    thousands of scalar CUDA synchronizations while moving only O(batch) scalar
    data; MuJoCo sampling dominates this negligible transfer in OmniPiano.
    """
    tensors = (
        rewards, values, next_values, terminated, truncated,
        preceding_log_ratio,
    )
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("PreOPC scalar fields must be one-dimensional")
    size = rewards.shape[0]
    if any(t.shape[0] != size for t in tensors):
        raise ValueError("PreOPC fields have different row counts")
    if keys.ndim != 2 or keys.shape != (size, 3):
        raise ValueError("keys must have shape [rows, 3]")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= trace_lambda <= 1.0:
        raise ValueError("gamma and trace_lambda must be in [0, 1]")
    if trace_clip_param <= 0.0 or not math.isfinite(trace_clip_param):
        raise ValueError("trace_clip_param must be finite and positive")
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise FloatingPointError("Non-finite PreOPC input")

    device, dtype = rewards.device, rewards.dtype
    arrays = [t.detach().cpu().numpy() for t in tensors]
    r, v, nv, term, trunc, log_ratio = arrays
    key_array = keys.detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.isin(term, (0.0, 1.0)).all() \
            or not np.isin(trunc, (0.0, 1.0)).all():
        raise ValueError("termination and truncation masks must be binary")

    delta = r + gamma * (1.0 - term) * nv - v
    corrected = np.empty_like(delta)
    next_by_stream: dict[int, int] = {}
    log_trace_clip = math.log(trace_clip_param)

    for row in range(size - 1, -1, -1):
        stream, episode, timestep = (int(x) for x in key_array[row])
        result = delta[row]
        successor = next_by_stream.get(stream)
        done = bool(term[row] or trunc[row])
        if successor is not None and not done:
            next_stream, next_episode, next_timestep = (
                int(x) for x in key_array[successor]
            )
            if next_stream != stream or next_episode != episode \
                    or next_timestep != timestep + 1:
                raise ValueError(
                    "Non-consecutive PreOPC trajectory without a done boundary"
                )
            trace_ratio = math.exp(min(float(log_ratio[successor]), log_trace_clip))
            result = (
                result
                + gamma * trace_lambda * trace_ratio * corrected[successor]
            )
        corrected[row] = result
        next_by_stream[stream] = row

    return torch.as_tensor(corrected, device=device, dtype=dtype)
