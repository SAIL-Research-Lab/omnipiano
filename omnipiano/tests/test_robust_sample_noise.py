"""Phase 0 · S2 gate — RobustWrapper._channel_active / _sample_noise dispatch.

Tests the per-distribution (Option C.3) noise sampling in isolation:
  - _channel_active picks the right field set per noise_dist;
  - gaussian path is BIT-IDENTICAL to a direct rng.normal(0, std) call
    (this is the core equivalence guarantee, §0.7 — the pre-Phase-0 code
    sampled exactly `rng.normal(0, std, size=shape)`);
  - uniform path has correct support / mean / std (symmetric + asymmetric);
  - shift path is a deterministic constant and consumes NO rng draw.

Uses ``RobustWrapper.__new__`` to get an instance with only ``.config`` set,
bypassing the env-walking ``__init__`` — the two helpers read only
``self.config`` and the passed rng, so no environment is needed (fast).
"""
import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.wrappers.robust_wrapper import RobustWrapper


def _wrapper(cfg: RobustConfig) -> RobustWrapper:
    w = RobustWrapper.__new__(RobustWrapper)  # bypass __init__ (no env walk)
    w.config = cfg
    return w


# --------------------------------------------------------------------------
# _channel_active — field selection per noise_dist
# --------------------------------------------------------------------------
def test_channel_active_gaussian():
    w = _wrapper(RobustConfig(noise_dist="gaussian", action_noise_std=0.05))
    assert w._channel_active("action") is True
    assert w._channel_active("obs") is False
    assert w._channel_active("reward") is False


def test_channel_active_uniform():
    w = _wrapper(RobustConfig(noise_dist="uniform",
                              obs_noise_uniform_low=-0.1, obs_noise_uniform_high=0.1))
    assert w._channel_active("obs") is True
    assert w._channel_active("action") is False


def test_channel_active_shift():
    w = _wrapper(RobustConfig(noise_dist="shift", reward_noise_shift=0.3))
    assert w._channel_active("reward") is True
    assert w._channel_active("action") is False


def test_channel_active_unknown_dist_raises():
    w = _wrapper(RobustConfig())
    w.config.noise_dist = "bogus"  # bypass Literal/validation
    with pytest.raises(ValueError, match="Unknown noise_dist"):
        w._channel_active("action")


# --------------------------------------------------------------------------
# gaussian — bit-identical to direct rng.normal (equivalence gate)
# --------------------------------------------------------------------------
def test_sample_noise_gaussian_bit_identical():
    w = _wrapper(RobustConfig(noise_dist="gaussian", action_noise_std=0.05))
    shape = (23,)
    got = w._sample_noise(np.random.default_rng(12345), "action", shape)
    ref = np.random.default_rng(12345).normal(0.0, 0.05, size=shape)
    np.testing.assert_array_equal(got, ref)


def test_sample_noise_gaussian_zero_std_all_zero():
    w = _wrapper(RobustConfig(noise_dist="gaussian", action_noise_std=0.0))
    got = w._sample_noise(np.random.default_rng(0), "action", (5,))
    assert np.allclose(got, 0.0)


# --------------------------------------------------------------------------
# uniform — support / mean / std correctness
# --------------------------------------------------------------------------
def test_sample_noise_uniform_symmetric():
    w = _wrapper(RobustConfig(noise_dist="uniform",
                              action_noise_uniform_low=-0.10,
                              action_noise_uniform_high=0.10))
    s = w._sample_noise(np.random.default_rng(0), "action", (200_000,))
    assert s.min() >= -0.10 and s.max() <= 0.10
    assert abs(float(s.mean())) < 5e-3
    # U[-a, a] std = a / sqrt(3)
    assert abs(float(s.std()) - 0.10 / np.sqrt(3.0)) < 5e-3


def test_sample_noise_uniform_asymmetric():
    w = _wrapper(RobustConfig(noise_dist="uniform",
                              obs_noise_uniform_low=-0.02,
                              obs_noise_uniform_high=0.10))
    s = w._sample_noise(np.random.default_rng(0), "obs", (200_000,))
    assert s.min() >= -0.02 and s.max() <= 0.10
    assert abs(float(s.mean()) - 0.04) < 5e-3  # (lo+hi)/2


# --------------------------------------------------------------------------
# shift — deterministic constant, NO rng consumption
# --------------------------------------------------------------------------
def test_sample_noise_shift_constant():
    w = _wrapper(RobustConfig(noise_dist="shift", action_noise_shift=0.3))
    got = w._sample_noise(np.random.default_rng(0), "action", (7,))
    np.testing.assert_array_equal(got, np.full((7,), 0.3))


def test_sample_noise_shift_does_not_consume_rng():
    w = _wrapper(RobustConfig(noise_dist="shift", action_noise_shift=0.3))
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    for _ in range(10):
        w._sample_noise(rng2, "action", (7,))  # shift branch draws nothing
    # rng2 untouched by the shift sampler → same next draw as the pristine rng1
    assert float(rng1.random()) == float(rng2.random())


def test_sample_noise_unknown_dist_raises():
    w = _wrapper(RobustConfig())
    w.config.noise_dist = "bogus"
    with pytest.raises(ValueError, match="Unknown noise_dist"):
        w._sample_noise(np.random.default_rng(0), "action", (3,))
