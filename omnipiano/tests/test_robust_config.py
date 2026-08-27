"""Phase 0 · S1 gate — RobustConfig field schema + validation.

Verifies:
  - all fields, including nested physical-environment noise, have defaults;
  - backward compatibility (the pre-Phase-0 2-field usage still constructs
    identically), so the two existing robust envs are unaffected;
  - __post_init__ raises on invalid uniform bounds (low>high, non-finite);
  - __post_init__ raises on noise_dist / field mismatch (mis-registration);
  - valid gaussian / uniform (symmetric + asymmetric) / shift configs pass.

Pure-dataclass test — no mujoco stepping required (only the import chain
pulls mujoco in via omnipiano.safety.constraints).
"""
import numpy as np
import pytest

from omnipiano.configs import RobustConfig


# --------------------------------------------------------------------------
# Field schema + defaults
# --------------------------------------------------------------------------
def test_default_config_all_zero_gaussian():
    cfg = RobustConfig()
    # Gaussian σ
    assert cfg.action_noise_std == 0.0
    assert cfg.obs_noise_std == 0.0
    assert cfg.reward_noise_std == 0.0
    # Uniform bounds
    for ch in ("action", "obs", "reward"):
        assert getattr(cfg, f"{ch}_noise_uniform_low") == 0.0
        assert getattr(cfg, f"{ch}_noise_uniform_high") == 0.0
    # Shift
    assert cfg.action_noise_shift == 0.0
    assert cfg.obs_noise_shift == 0.0
    assert cfg.reward_noise_shift == 0.0
    # Meta — dist selector only; eval scale is a make() kwarg (2026-07-21),
    # not a config field.
    assert cfg.noise_dist == "gaussian"


def test_exactly_17_fields():
    # 2026-07-21: eval_noise_scale left the config (make() kwarg now);
    # the 3 optional per-channel dist overrides arrived.
    from dataclasses import fields
    names = {f.name for f in fields(RobustConfig)}
    expected = {
        "action_noise_std", "obs_noise_std", "reward_noise_std",
        "action_noise_uniform_low", "action_noise_uniform_high",
        "obs_noise_uniform_low", "obs_noise_uniform_high",
        "reward_noise_uniform_low", "reward_noise_uniform_high",
        "action_noise_shift", "obs_noise_shift", "reward_noise_shift",
        "noise_dist",
        "action_noise_dist", "obs_noise_dist", "reward_noise_dist",
        "environment_noise",
    }
    assert names == expected, f"field set drift: {names ^ expected}"
    assert len(names) == 17


# --------------------------------------------------------------------------
# Backward compatibility — the 2 existing robust envs construct unchanged
# --------------------------------------------------------------------------
def test_backward_compat_action_robust():
    # FantaisieImpromptu-ActionRobust-v0 uses action_noise_std only.
    cfg = RobustConfig(action_noise_std=0.01)
    assert cfg.action_noise_std == 0.01
    assert cfg.obs_noise_std == 0.0
    assert cfg.noise_dist == "gaussian"   # default → gaussian path unchanged


def test_backward_compat_obs_robust():
    # ClairDeLune-ObservationRobust-v0 uses obs_noise_std only.
    cfg = RobustConfig(obs_noise_std=0.01)
    assert cfg.obs_noise_std == 0.01
    assert cfg.action_noise_std == 0.0
    assert cfg.noise_dist == "gaussian"


# --------------------------------------------------------------------------
# __post_init__ — uniform bound validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ch", ["action", "obs", "reward"])
def test_uniform_low_gt_high_raises(ch):
    with pytest.raises(ValueError, match="must be <= high"):
        RobustConfig(noise_dist="uniform",
                     **{f"{ch}_noise_uniform_low": 0.1,
                        f"{ch}_noise_uniform_high": -0.1})


@pytest.mark.parametrize("ch", ["action", "obs", "reward"])
def test_uniform_nonfinite_raises(ch):
    with pytest.raises(ValueError, match="must be finite"):
        RobustConfig(noise_dist="uniform",
                     **{f"{ch}_noise_uniform_low": -np.inf,
                        f"{ch}_noise_uniform_high": 0.1})


# --------------------------------------------------------------------------
# __post_init__ — noise_dist / field consistency
# --------------------------------------------------------------------------
def test_gaussian_with_shift_raises():
    with pytest.raises(ValueError, match="noise_dist='gaussian'"):
        RobustConfig(noise_dist="gaussian", action_noise_shift=0.1)


def test_gaussian_with_uniform_raises():
    with pytest.raises(ValueError, match="noise_dist='gaussian'"):
        RobustConfig(noise_dist="gaussian",
                     obs_noise_uniform_low=-0.1, obs_noise_uniform_high=0.1)


def test_uniform_with_std_raises():
    with pytest.raises(ValueError, match="noise_dist='uniform'"):
        RobustConfig(noise_dist="uniform", action_noise_std=0.1)


def test_uniform_with_shift_raises():
    with pytest.raises(ValueError, match="noise_dist='uniform'"):
        RobustConfig(noise_dist="uniform", reward_noise_shift=0.1)


def test_shift_with_std_raises():
    with pytest.raises(ValueError, match="noise_dist='shift'"):
        RobustConfig(noise_dist="shift", obs_noise_std=0.1)


def test_shift_with_uniform_raises():
    with pytest.raises(ValueError, match="noise_dist='shift'"):
        RobustConfig(noise_dist="shift",
                     action_noise_uniform_low=-0.1, action_noise_uniform_high=0.1)


# --------------------------------------------------------------------------
# Valid configs across the 3 distributions
# --------------------------------------------------------------------------
def test_valid_gaussian():
    cfg = RobustConfig(noise_dist="gaussian", action_noise_std=0.05)
    assert cfg.action_noise_std == 0.05


def test_valid_uniform_symmetric():
    # Option 4a: natural params, symmetric [-level, +level], NO √3 conversion.
    cfg = RobustConfig(noise_dist="uniform",
                       action_noise_uniform_low=-0.10,
                       action_noise_uniform_high=0.10)
    assert cfg.action_noise_uniform_low == -0.10
    assert cfg.action_noise_uniform_high == 0.10


def test_valid_uniform_asymmetric():
    # Advanced: biased sensor drift, low <= high still enforced.
    cfg = RobustConfig(noise_dist="uniform",
                       obs_noise_uniform_low=-0.02,
                       obs_noise_uniform_high=0.10)
    assert cfg.obs_noise_uniform_low == -0.02
    assert cfg.obs_noise_uniform_high == 0.10


def test_valid_shift():
    cfg = RobustConfig(noise_dist="shift", reward_noise_shift=0.30)
    assert cfg.reward_noise_shift == 0.30


# --------------------------------------------------------------------------
# __post_init__ — fail-fast validation of noise_dist + non-negative magnitudes
# --------------------------------------------------------------------------
def test_invalid_noise_dist_raises():
    # A typo must fail at construction, not silently produce a clean env.
    with pytest.raises(ValueError, match="noise_dist must be one of"):
        RobustConfig(noise_dist="gausian")


@pytest.mark.parametrize("ch", ["action", "obs", "reward"])
def test_negative_std_raises(ch):
    with pytest.raises(ValueError, match="must be >= 0"):
        RobustConfig(**{f"{ch}_noise_std": -0.05})


# (eval_noise_scale is no longer a RobustConfig field — 2026-07-21 decision;
# its validation now lives in make() and is tested in
# test_eval_noise_scale_override.py.)


# --------------------------------------------------------------------------
# __post_init__ — finiteness (NaN passes any `< 0` check, so it needs its
# own gate; a NaN magnitude would otherwise poison training mid-episode)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ch", ["action", "obs", "reward"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_std_raises(ch, bad):
    with pytest.raises(ValueError, match="noise_std must be finite"):
        RobustConfig(**{f"{ch}_noise_std": bad})


@pytest.mark.parametrize("ch", ["action", "obs", "reward"])
@pytest.mark.parametrize("bad", [float("nan"), float("-inf")])
def test_nonfinite_shift_raises(ch, bad):
    with pytest.raises(ValueError, match="noise_shift must be finite"):
        RobustConfig(noise_dist="shift", **{f"{ch}_noise_shift": bad})
