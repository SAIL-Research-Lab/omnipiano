"""Equation and trajectory-boundary tests for A2PO."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omnipiano.multiagent.algos._a2po_math import (
    adaptive_clip,
    a2po_surrogate,
    official_order_scores,
    preopc_advantages,
    semi_greedy_order,
)
from omnipiano.multiagent.external.a2po import _adapter as REFERENCE
from omnipiano.multiagent.training.native_io import add_gae


@pytest.mark.parametrize("agents", [2, 4, 7])
@pytest.mark.parametrize("position", [1, 2])
def test_adaptive_clip_matches_official_near_linear(agents, position):
    if position > agents:
        pytest.skip()
    ours = adaptive_clip(0.2, position, agents, 0.5)
    theirs = REFERENCE.near_linear_reference(position, agents, 0.2, 0.5)
    assert ours == pytest.approx(theirs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_joint_surrogate_matches_official_equation(seed):
    generator = torch.Generator().manual_seed(seed)
    values = {
        "logp_new": torch.randn(512, generator=generator) * 0.3,
        "logp_old": torch.randn(512, generator=generator) * 0.3,
        "advantages": torch.randn(512, generator=generator),
        "preceding_log_ratio": torch.randn(512, generator=generator) * 0.4,
    }
    ours = a2po_surrogate(
        **values, clip_param=0.15, preceding_clip_param=0.1)
    theirs = REFERENCE.joint_surrogate_reference(
        **values, clip_param=0.15, preceding_clip_param=0.1)
    torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6)


def test_preceding_ratio_is_context_not_a_gradient_target():
    current = torch.tensor([0.1, -0.2], requires_grad=True)
    old = torch.zeros(2)
    preceding = torch.tensor([0.3, -0.3], requires_grad=True)
    output = a2po_surrogate(
        current, old, torch.ones(2), preceding, 0.2, 0.1)
    output.sum().backward()
    assert current.grad is not None
    assert preceding.grad is None


def test_semi_greedy_order_matches_official_control_flow():
    class FixedRng:
        def __init__(self):
            self.values = iter([2, 0])

        def integers(self, high):
            value = next(self.values)
            assert 0 <= value < high
            return value

    scores = [0.1, 0.9, 0.5, 0.4, 0.2]
    ours = semi_greedy_order(scores, rng=FixedRng())
    theirs = REFERENCE.semi_greedy_reference(scores, [2, 0])
    assert ours == theirs
    assert sorted(ours) == list(range(len(scores)))


def test_order_score_matches_official_normalized_advantage_rule():
    advantage = torch.tensor([[2.0, 4.0], [4.0, 3.0]])
    value = torch.tensor([[2.0, -2.0], [4.0, 1.0]])
    score = official_order_scores(advantage, value)
    normalized = (
        advantage - advantage.mean()
    ) / (advantage.std(unbiased=False) + 1e-5)
    expected = (normalized.abs() / value.abs()).mean(dim=0)
    torch.testing.assert_close(score, expected)


def test_preopc_with_no_preceding_agents_reduces_exactly_to_existing_gae():
    # Two interleaved environment streams, exactly as NativeAlgorithm.collect
    # emits them. The final row of stream 0 is truncated; stream 1 merely ends
    # at the rollout boundary.
    keys = np.asarray([
        [0, 0, 0], [1, 0, 0],
        [0, 0, 1], [1, 0, 1],
        [0, 0, 2], [1, 0, 2],
    ], dtype=np.int64)
    batch = {
        "keys": keys,
        "r": np.asarray([1, 2, 3, 4, 5, 6], np.float32),
        "term": np.asarray([0, 0, 0, 0, 0, 0], np.float32),
        "trunc": np.asarray([0, 0, 0, 0, 1, 0], np.float32),
        "v": np.asarray([[0.2], [0.4], [0.3], [0.5], [0.6], [0.7]], np.float32),
        "nv": np.asarray([[0.3], [0.5], [0.6], [0.7], [0.9], [1.0]], np.float32),
    }
    add_gae(batch, gamma=0.8, gae_lambda=0.95, streams=2)
    corrected = preopc_advantages(
        *[torch.from_numpy(batch[name][:, 0] if name in ("v", "nv")
                           else batch[name])
          for name in ("r", "v", "nv", "term", "trunc")],
        torch.zeros(6), torch.from_numpy(keys), gamma=0.8,
        trace_lambda=0.95, trace_clip_param=1.0)
    torch.testing.assert_close(corrected, torch.from_numpy(batch["adv"][:, 0]))


def test_preopc_uses_the_future_not_current_ratio():
    # delta is one at every row. Ratios are [irrelevant, 0.5, 2.0].
    corrected = preopc_advantages(
        rewards=torch.ones(3), values=torch.zeros(3), next_values=torch.zeros(3),
        terminated=torch.zeros(3), truncated=torch.zeros(3),
        preceding_log_ratio=torch.tensor([100.0, np.log(0.5), np.log(2.0)]),
        keys=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        gamma=0.5, trace_lambda=0.8, trace_clip_param=1.0)
    torch.testing.assert_close(corrected, torch.tensor([1.28, 1.4, 1.0]))


def test_preopc_distinguishes_termination_from_time_limit_truncation():
    keys = torch.tensor([[0, 0, 0], [0, 1, 0]])
    common = dict(
        rewards=torch.tensor([1.0, 1.0]), values=torch.zeros(2),
        next_values=torch.tensor([10.0, 10.0]),
        preceding_log_ratio=torch.zeros(2), keys=keys,
        gamma=0.5, trace_lambda=0.8, trace_clip_param=1.0)
    terminated = preopc_advantages(
        terminated=torch.tensor([1.0, 0.0]),
        truncated=torch.tensor([0.0, 1.0]), **common)
    truncated = preopc_advantages(
        terminated=torch.tensor([0.0, 0.0]),
        truncated=torch.tensor([1.0, 1.0]), **common)
    # Both stop the trace. Only a truncation bootstraps its final observation.
    assert terminated[0] == pytest.approx(1.0)
    assert truncated[0] == pytest.approx(6.0)


def test_oracle_provenance_is_pinned():
    assert REFERENCE.UPSTREAM_URL == "https://github.com/xihuai18/A2PO-ICLR2023"
    assert len(REFERENCE.UPSTREAM_COMMIT) == 40
