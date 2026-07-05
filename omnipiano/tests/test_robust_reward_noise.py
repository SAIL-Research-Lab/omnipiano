"""Phase 0 · S4 gate — reward noise injection + Method-6 reward-slot override.

Verifies (decision 11: reward matched, NOT force-zeroed):
  - a reward-noise task noises the returned scalar reward, and the returned
    reward == clean(decomposition sum) + info["robust/noise_reward"];
  - Method 6 overrides obs["reward"] with the noised observed reward r_obs
    (so the policy's reward memory matches what it received);
  - clean / action-only tasks get NO reward noise (info key == 0, reward
    equals the clean decomposition sum).

The clean per-step reward is reconstructed as the sum of the ``task/*_reward``
decomposition terms (MetricsWrapper reads pre-noise physics), which is exactly
the ep_return_true basis (§0.6 / decision 11).
"""
import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration
from omnipiano.utils.info_keys import InfoKeys
from omnipiano.wrappers.robust_wrapper import RobustWrapper

_BASE = "RoboPianist-repertoire-150-ForElise-v0"


def _reg(env_id: str, **kw) -> None:
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, **kw)


_reg("OmniPianoTest-S4-RewardGauss-v0",
     robust_config=RobustConfig(noise_dist="gaussian", reward_noise_std=0.5))
_reg("OmniPianoTest-S4-Clean-v0")
_reg("OmniPianoTest-S4-ActionGauss-v0",
     robust_config=RobustConfig(noise_dist="gaussian", action_noise_std=0.05))


def _robust(env) -> RobustWrapper:
    p = env
    while not isinstance(p, RobustWrapper):
        p = p.env
    return p


def _clean_reward(info) -> float:
    """Clean per-step reward = sum of the task/*_reward decomposition terms
    (pre-noise, from MetricsWrapper) — the ep_return_true basis."""
    return float(sum(
        v for k, v in info.items()
        if isinstance(k, str) and k.startswith("task/") and k.endswith("_reward")
    ))


def _first_step(env_id, seed=0):
    env = registration.make(env_id, seed=seed)  # train mode
    env.reset(seed=seed)
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    return env, obs, float(reward), info


def test_reward_noise_applied_and_matches_clean_plus_noise():
    env, obs, reward, info = _first_step("OmniPianoTest-S4-RewardGauss-v0")
    try:
        rn = float(info[InfoKeys.ROBUST_NOISE_REWARD])
        assert rn != 0.0                                   # noise injected
        assert reward == pytest.approx(_clean_reward(info) + rn, abs=1e-4)
    finally:
        env.close()


def test_reward_slot_overridden_to_noised_reward():
    env, obs, reward, info = _first_step("OmniPianoTest-S4-RewardGauss-v0")
    try:
        rw = _robust(env)
        assert rw._reward_slice is not None
        slot = np.asarray(obs[rw._reward_slice]).ravel()
        # Method 6: obs["reward"] == the noised observed reward the agent got.
        assert slot[0] == pytest.approx(reward, abs=1e-4)
    finally:
        env.close()


def test_clean_env_no_reward_noise():
    env, obs, reward, info = _first_step("OmniPianoTest-S4-Clean-v0")
    try:
        assert info[InfoKeys.ROBUST_NOISE_REWARD] == 0.0
        assert reward == pytest.approx(_clean_reward(info), abs=1e-4)  # not noised
    finally:
        env.close()


def test_action_only_task_no_reward_noise():
    env, obs, reward, info = _first_step("OmniPianoTest-S4-ActionGauss-v0")
    try:
        assert info[InfoKeys.ROBUST_NOISE_REWARD] == 0.0   # reward channel inactive
        assert reward == pytest.approx(_clean_reward(info), abs=1e-4)  # reward clean
    finally:
        env.close()
