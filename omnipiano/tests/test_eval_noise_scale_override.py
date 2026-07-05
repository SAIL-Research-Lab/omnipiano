"""Phase 1 · gate — private make() kwarg `_eval_noise_scale_override` (§12 decision 7).

The robustness sweep harness scans a grid of eval noise magnitudes on ONE
registered training env without registering ~135 extra ids. Verifies:
  - override scales every effective magnitude field (action/obs/reward share the
    single effective_robust_config passed to RobustWrapper);
  - override=0.0 → clean eval (a valid override, distinct from None);
  - override=None → falls back to the registered eval_noise_scale (=1.0 matched);
  - override at mode="train" raises (misuse);
  - negative override raises.
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
    cfg = _cfg(_A, mode="eval", _eval_noise_scale_override=2.0)
    assert cfg.action_noise_std == pytest.approx(0.10 * 2.0)


def test_override_scales_reward_std():
    cfg = _cfg(_R, mode="eval", _eval_noise_scale_override=2.0)
    assert cfg.reward_noise_std == pytest.approx(0.30 * 2.0)


def test_override_scales_uniform_bounds():
    cfg = _cfg(_O, mode="eval", _eval_noise_scale_override=0.5)
    assert cfg.obs_noise_uniform_low == pytest.approx(-0.10 * 0.5)
    assert cfg.obs_noise_uniform_high == pytest.approx(+0.10 * 0.5)


def test_override_zero_is_clean():
    # 0.0 is a valid override (distinct from None) → clean eval.
    cfg = _cfg(_A, mode="eval", _eval_noise_scale_override=0.0)
    assert cfg.action_noise_std == 0.0
    assert not cfg.is_channel_active("action")


# --------------------------------------------------------------------------
# Fallback to registered eval_noise_scale when not passed
# --------------------------------------------------------------------------
def test_no_override_uses_matched_default():
    # eval_noise_scale defaults to 1.0 → matched → std unchanged.
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
        registration.make(_A, mode="train", _eval_noise_scale_override=2.0)


def test_negative_override_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        registration.make(_A, mode="eval", _eval_noise_scale_override=-1.0)
