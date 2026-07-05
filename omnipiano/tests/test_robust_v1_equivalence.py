"""Phase 0 · §0.7 equivalence gate — the two registered v1 robust envs.

Phase 0 reworked the whole robust surface (RobustConfig 2->14 fields, noise
dispatch moved onto RobustConfig, eval_noise_scale plumbing, reward noise,
obs-dist refactor, OBS_NOISE_SEED_OFFSET 31415->20000). The invariant that
MUST survive: the two existing robust envs still produce a DETERMINISTIC,
reproducible noise sequence at a given master seed (train mode), on the
correct channel, and the gaussian draw is bit-identical to a plain
rng.normal(0, std) (so the field expansion changed no RNG arithmetic).

Note: the obs-noise *values* intentionally changed (seed offset 31415->20000,
the documented §0.5 exception); this test asserts reproducibility across builds
at the same seed, not equality to the pre-Phase-0 sequence.
"""
import numpy as np
import pytest

import omnipiano.envs  # noqa: F401  (triggers v1 env registration)
from omnipiano.configs import RobustConfig
from omnipiano.envs import registration
from omnipiano.utils.info_keys import InfoKeys

_ACTION_ENV = "OmniPiano-FantaisieImpromptu-ActionRobust-v0"   # action_noise_std=0.01
_OBS_ENV = "OmniPiano-ClairDeLune-ObservationRobust-v0"        # obs_noise_std=0.01
_N = 20  # << episode length, so no reset happens mid-run


def _noise_seqs(env_id, seed):
    env = registration.make(env_id, seed=seed)  # train mode → scale 1.0
    try:
        env.reset(seed=seed)
        act = np.full(env.action_space.shape, 0.3, dtype=np.float32)
        a_l2, o_l2 = [], []
        for _ in range(_N):
            _, _, term, trunc, info = env.step(act)
            a_l2.append(float(info[InfoKeys.ROBUST_NOISE_ACTION_L2]))
            o_l2.append(float(info[InfoKeys.ROBUST_NOISE_OBS_L2]))
            assert not (term or trunc), "episode ended within _N steps; lower _N"
        return np.array(a_l2), np.array(o_l2)
    finally:
        env.close()


def test_action_env_reproducible_and_channel_correct():
    a1, o1 = _noise_seqs(_ACTION_ENV, seed=123)
    a2, o2 = _noise_seqs(_ACTION_ENV, seed=123)
    # bit-exact reproducibility across independent builds at the same seed
    np.testing.assert_array_equal(a1, a2)
    # correct channel: action noise on, obs noise off
    assert np.all(a1 > 0.0)
    assert np.all(o1 == 0.0)


def test_obs_env_reproducible_and_channel_correct():
    a1, o1 = _noise_seqs(_OBS_ENV, seed=123)
    a2, o2 = _noise_seqs(_OBS_ENV, seed=123)
    np.testing.assert_array_equal(o1, o2)
    # correct channel: obs noise on, action noise off
    assert np.all(o1 > 0.0)
    assert np.all(a1 == 0.0)


def test_noise_differs_across_seeds():
    a_s1, _ = _noise_seqs(_ACTION_ENV, seed=1)
    a_s2, _ = _noise_seqs(_ACTION_ENV, seed=2)
    assert not np.array_equal(a_s1, a_s2)


def test_gaussian_draw_bit_identical_to_rng_normal():
    # The RobustConfig(action_noise_std=0.01) surface used by the registered
    # env must sample exactly rng.normal(0, 0.01) — the field expansion +
    # dispatch refactor changed no RNG arithmetic.
    cfg = RobustConfig(action_noise_std=0.01)
    shape = (23,)
    got = cfg.sample_noise(np.random.default_rng(2024), "action", shape)
    ref = np.random.default_rng(2024).normal(0.0, 0.01, size=shape)
    np.testing.assert_array_equal(got, ref)
