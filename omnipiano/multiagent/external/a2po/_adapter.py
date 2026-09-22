"""Pinned equation oracle transcribed from the official A2PO repository.

The upstream learner is entangled with its runner and buffer. These small
functions reproduce the relevant equations independently so tests can detect
clipping/order mistakes without importing the upstream framework.
"""

from __future__ import annotations

import numpy as np
import torch


UPSTREAM_URL = "https://github.com/xihuai18/A2PO-ICLR2023"
UPSTREAM_COMMIT = "28c11e6063bcf80caffc53a791c99dd7be1003b5"


def near_linear_reference(agent_order, num_agents, clip_param, weight=0.5):
    return clip_param * (
        (agent_order / num_agents) * weight + (1.0 - weight)
    )


def joint_surrogate_reference(
    *, logp_new, logp_old, advantages, preceding_log_ratio,
    clip_param, preceding_clip_param,
):
    preceding = torch.exp(preceding_log_ratio)
    preceding = torch.clamp(
        preceding, 1.0 - preceding_clip_param, 1.0 + preceding_clip_param)
    ratio = torch.exp(logp_new - logp_old) * preceding
    return torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages,
    )


def semi_greedy_reference(scores, random_choices):
    remaining = sorted(
        list(enumerate(np.asarray(scores).tolist())),
        key=lambda pair: pair[1], reverse=True)
    sequence = []
    choices = iter(random_choices)
    while remaining:
        sequence.append(remaining.pop(0)[0])
        if remaining:
            sequence.append(remaining.pop(int(next(choices)))[0])
    return sequence
