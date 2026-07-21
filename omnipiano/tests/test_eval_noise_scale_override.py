"""Gate — public make() kwarg `eval_noise_scale` (§12 decision 7; promoted
from the private `_eval_noise_scale_override` on 2026-07-21: eval scale is a
measurement parameter, not part of the registered task identity, so it lives
at the call site — users test any scale without registering extra ids).

Verifies:
  - the kwarg scales every effective magnitude field (action/obs/reward share
    the single effective_robust_config passed to RobustWrapper);
  - 0.0 → clean eval (a valid value, distinct from omitted);
  - omitted → 1.0 matched eval (protocol default, decisions 10/11);
  - passing it at mode="train" raises (misuse);
  - negative / non-finite values raise.
"""
import numpy as np
import pytest

from omnipiano.envs import registration
from omnipiano.wrappers.robust_wrapper import RobustWrapper

# Gaussian action env at σ=0.10; Uniform obs env; reward Gaussian env — pick
# one per channel so we can read the scaled field off RobustWrapper.config
# (effective_robust_config is one object carrying ALL scaled fields).
_A = "OmniPiano-ClairDeLune-A-Gauss-P10-v0"   # action_noise_std = 0.10
_R = "OmniPiano-ClairDeLune-R-Gauss-P30-v0"   # reward_noise_std = 0.30
_O = "OmniPiano-ClairDeLune-O-Uniform-P10-v0"  # obs uniform ±0.10


def _robust(env) -> RobustWrapper:
    p = env
    while not isinstance(p, RobustWrapper):
        p = p.env
    return p


def _cfg(env_id, **make_kw):
    env = registration.make(env_id, **make_kw)
    try:
        return _robust(env).config
    finally:
        env.close()


# --------------------------------------------------------------------------
# Override scales the effective magnitude
# --------------------------------------------------------------------------
def test_override_scales_action_std():
    cfg = _cfg(_A, mode="eval", eval_noise_scale=2.0)
    assert cfg.action_noise_std == pytest.approx(0.10 * 2.0)


def test_override_scales_reward_std():
    cfg = _cfg(_R, mode="eval", eval_noise_scale=2.0)
    assert cfg.reward_noise_std == pytest.approx(0.30 * 2.0)


def test_override_scales_uniform_bounds():
    cfg = _cfg(_O, mode="eval", eval_noise_scale=0.5)
    assert cfg.obs_noise_uniform_low == pytest.approx(-0.10 * 0.5)
    assert cfg.obs_noise_uniform_high == pytest.approx(+0.10 * 0.5)


def test_override_zero_is_clean():
    # 0.0 is a valid override (distinct from None) → clean eval.
    cfg = _cfg(_A, mode="eval", eval_noise_scale=0.0)
    assert cfg.action_noise_std == 0.0
    assert not cfg.is_channel_active("action")


# --------------------------------------------------------------------------
# Omitted kwarg → matched default (1.0)
# --------------------------------------------------------------------------
def test_no_override_uses_matched_default():
    # Omitted eval_noise_scale → 1.0 → matched → std unchanged.
    cfg = _cfg(_A, mode="eval")
    assert cfg.action_noise_std == pytest.approx(0.10)


def test_train_mode_ignores_scale_field():
    cfg = _cfg(_A, mode="train")
    assert cfg.action_noise_std == pytest.approx(0.10)  # train scale is always 1.0


# --------------------------------------------------------------------------
# Misuse guards
# --------------------------------------------------------------------------
def test_override_at_train_raises():
    with pytest.raises(ValueError, match="only valid with mode='eval'"):
        registration.make(_A, mode="train", eval_noise_scale=2.0)


def test_negative_override_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        registration.make(_A, mode="eval", eval_noise_scale=-1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_scale_raises(bad):
    # Validation moved here from RobustConfig.__post_init__ when the field
    # became a make() kwarg (2026-07-21).
    with pytest.raises(ValueError, match="must be finite"):
        registration.make(_A, mode="eval", eval_noise_scale=bad)
