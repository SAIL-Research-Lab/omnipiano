"""Phase 0 · S3 gate — mode rename + eval_noise_scale plumbing (matched eval).

Verifies registration.make()'s effective-config logic (§0.2 / §0.6.3,
decisions 10/11; updated 2026-07-21: ``eval_noise_scale`` is a PUBLIC
``make()`` kwarg, no longer a RobustConfig field — RobustConfig defines
training-time noise only):
  - `log_split` param is renamed to `mode` (old kwarg now rejected);
  - at mode="eval", all magnitude fields are multiplied by the
    eval_noise_scale kwarg; omitted => 1.0 = matched eval (same noise as
    training);
  - reward is treated like action/obs: scaled by eval_noise_scale, NOT
    force-zeroed, and no warning is emitted (decision 11);
  - uniform bounds scale correctly.

Effective config is read off the constructed RobustWrapper's `.config`.
Uses the shortest piece (ForElise) for fast env builds.
"""
import warnings

import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration
from omnipiano.wrappers.robust_wrapper import RobustWrapper

_BASE = "RoboPianist-repertoire-150-ForElise-v0"


def _register_once(env_id: str, **kwargs) -> None:
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, **kwargs)


# Training-noise-only configs (scale is a make() kwarg, not a field).
_register_once(
    "OmniPianoTest-S3-Action-v0",
    robust_config=RobustConfig(action_noise_std=0.10),
)
_register_once(
    "OmniPianoTest-S3-Reward-v0",
    robust_config=RobustConfig(reward_noise_std=0.30),
)
_register_once(
    "OmniPianoTest-S3-Uniform-v0",
    robust_config=RobustConfig(
        noise_dist="uniform",
        action_noise_uniform_low=-0.10,
        action_noise_uniform_high=0.10,
    ),
)


def _robust_wrapper(env) -> RobustWrapper:
    ptr = env
    while not isinstance(ptr, RobustWrapper):
        ptr = ptr.env
    return ptr


def _config_of(env_id: str, mode: str, **make_kw) -> RobustConfig:
    env = registration.make(env_id, mode=mode, **make_kw)
    try:
        return _robust_wrapper(env).config
    finally:
        env.close()


# --------------------------------------------------------------------------
# mode rename
# --------------------------------------------------------------------------
def test_old_log_split_kwarg_rejected():
    # After the rename, `log_split` is no longer a named param → it falls into
    # **kwargs and is rejected by the runtime-bypass whitelist (ValueError).
    with pytest.raises(ValueError, match="log_split"):
        registration.make("OmniPianoTest-S3-Action-v0", log_split="eval")


def test_mode_train_and_eval_both_construct():
    for m in ("train", "eval"):
        env = registration.make("OmniPianoTest-S3-Action-v0", mode=m)
        env.close()


# --------------------------------------------------------------------------
# eval_noise_scale kwarg — action (gaussian)
# --------------------------------------------------------------------------
def test_action_train_is_unscaled():
    cfg = _config_of("OmniPianoTest-S3-Action-v0", "train")
    assert cfg.action_noise_std == pytest.approx(0.10)  # scale 1.0


def test_action_eval_scale_halves():
    cfg = _config_of("OmniPianoTest-S3-Action-v0", "eval", eval_noise_scale=0.5)
    assert cfg.action_noise_std == pytest.approx(0.05)  # 0.10 * 0.5


def test_action_default_eval_is_matched():
    # Omitted eval_noise_scale => 1.0 => eval noise == training noise
    # (matched, RG-comparable). This is the headline default-eval behavior.
    cfg = _config_of("OmniPianoTest-S3-Action-v0", "eval")
    assert cfg.action_noise_std == pytest.approx(0.10)   # 0.10 * 1.0


def test_action_scale_zero_is_clean_eval():
    # eval_noise_scale=0.0 explicitly => clean/nominal eval (opt-in).
    env = registration.make(
        "OmniPianoTest-S3-Action-v0", mode="eval", eval_noise_scale=0.0
    )
    try:
        rw = _robust_wrapper(env)
        assert rw.config.action_noise_std == pytest.approx(0.0)   # 0.10 * 0.0
        assert rw._channel_active("action") is False
    finally:
        env.close()


# --------------------------------------------------------------------------
# reward scales like action/obs at eval — matched by default (decision 11):
# NOT force-zeroed, no warn (obs["reward"] must stay in-distribution vs train)
# --------------------------------------------------------------------------
def test_reward_noise_present_at_train():
    cfg = _config_of("OmniPianoTest-S3-Reward-v0", "train")
    assert cfg.reward_noise_std == pytest.approx(0.30)


def test_reward_matched_at_eval_no_force_zero_no_warn():
    # Omitted scale => 1.0 => reward noise kept at training level (matched),
    # NOT force-zeroed (decision 11), and no warning is emitted.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = _config_of("OmniPianoTest-S3-Reward-v0", "eval")
    assert cfg.reward_noise_std == pytest.approx(0.30)   # matched, not zeroed
    assert [w for w in rec if "Reward noise" in str(w.message)] == []


def test_reward_scales_to_clean_with_explicit_zero():
    # eval_noise_scale=0.0 => reward scaled to clean at eval (same scaling
    # mechanism as action/obs).
    cfg = _config_of("OmniPianoTest-S3-Reward-v0", "eval", eval_noise_scale=0.0)
    assert cfg.reward_noise_std == pytest.approx(0.0)     # 0.30 * 0.0


# --------------------------------------------------------------------------
# uniform bounds scaling
# --------------------------------------------------------------------------
def test_uniform_train_unscaled():
    cfg = _config_of("OmniPianoTest-S3-Uniform-v0", "train")
    assert cfg.action_noise_uniform_low == pytest.approx(-0.10)
    assert cfg.action_noise_uniform_high == pytest.approx(0.10)


def test_uniform_eval_scale_halves():
    cfg = _config_of("OmniPianoTest-S3-Uniform-v0", "eval", eval_noise_scale=0.5)
    assert cfg.action_noise_uniform_low == pytest.approx(-0.05)
    assert cfg.action_noise_uniform_high == pytest.approx(0.05)


# --------------------------------------------------------------------------
# RobustConfig no longer carries the field (2026-07-21 decision)
# --------------------------------------------------------------------------
def test_robust_config_has_no_eval_noise_scale_field():
    with pytest.raises(TypeError):
        RobustConfig(eval_noise_scale=1.0)  # unexpected keyword
