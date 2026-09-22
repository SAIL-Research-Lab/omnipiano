"""Numerically compare our HAPPO surrogate against HARL's implementation.

This is the test that answers "is your HAPPO really HAPPO?" with a number
instead of a paragraph. It runs on CPU in milliseconds and needs no MuJoCo, no
Ray and no GPU -- so it can gate every launch.  The independent reference
transcription pins the exact HARL commit and ``prod`` aggregation convention.
"""

from __future__ import annotations

import pytest
import torch

from omnipiano.multiagent.algos._happo_math import happo_surrogate, ppo_surrogate

from omnipiano.multiagent.external.harl_happo import _adapter as REFERENCE

# Tolerance rationale: both sides are float32 and compute exp/clamp/min in a
# different order, so bit-equality is not achievable. 1e-6 is ~100x the float32
# epsilon on values of order 1 -- tight enough that any algebraic difference
# (a missing detach, a clamp on the wrong quantity, a sign error) fails, loose
# enough that reassociation does not.
TOL = 1e-6


def _batch(n: int = 512, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        # Scale chosen so the PPO ratio spans both sides of the clip boundary:
        # a test where clipping never activates would not test clipping.
        "logp_new": torch.randn(n, generator=g) * 0.3,
        "logp_old": torch.randn(n, generator=g) * 0.3,
        "advantages": torch.randn(n, generator=g),
        "log_factor": torch.randn(n, generator=g) * 0.5,
    }


@pytest.mark.parametrize("clip", [0.05, 0.2, 0.5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_our_happo_surrogate_matches_the_reference(clip: float, seed: int):
    b = _batch(seed=seed)
    ours = happo_surrogate(b["logp_new"], b["logp_old"], b["advantages"],
                           b["log_factor"], clip)
    theirs = REFERENCE.happo_surrogate_reference(
        logp_new=b["logp_new"], logp_old=b["logp_old"],
        advantages=b["advantages"], compound_log_factor=b["log_factor"],
        clip_param=clip)
    torch.testing.assert_close(ours, theirs, rtol=TOL, atol=TOL)


def test_clipping_is_actually_exercised_by_the_fixture():
    """A tolerance test on inputs that never trip the clip proves nothing."""
    b = _batch()
    ratio = torch.exp(b["logp_new"] - b["logp_old"])
    assert (ratio > 1.2).any() and (ratio < 0.8).any(), (
        "the fixture never leaves the clip region, so the comparison above does "
        "not test the clipped branch -- widen the logp scale")


@pytest.mark.parametrize("seed", [0, 1])
def test_reference_also_reduces_to_ppo_at_m_equals_one(seed: int):
    """Cross-check that OUR reading of the reference's semantics is right.

    If the reference does not itself collapse to PPO at M == 1, then either the
    adapter is wired wrong or the reference parameterises M differently (e.g. it
    takes the factor rather than its log). Either way, agreement above would be
    agreement on the wrong quantity.
    """
    b = _batch(seed=seed)
    zero = torch.zeros_like(b["advantages"])
    theirs = REFERENCE.happo_surrogate_reference(
        logp_new=b["logp_new"], logp_old=b["logp_old"],
        advantages=b["advantages"], compound_log_factor=zero, clip_param=0.2)
    ppo = ppo_surrogate(b["logp_new"], b["logp_old"], b["advantages"], 0.2)
    torch.testing.assert_close(theirs, ppo, rtol=TOL, atol=TOL)


def test_provenance_is_recorded_for_the_oracle():
    """An oracle without a pinned commit is an anecdote."""
    assert REFERENCE.UPSTREAM_COMMIT and len(REFERENCE.UPSTREAM_COMMIT) >= 12
    assert REFERENCE.UPSTREAM_URL.startswith("https://github.com/")
