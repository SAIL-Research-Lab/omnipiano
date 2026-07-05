"""Phase 0 · S6 gate — eval CSV robust columns (SafeRecordEpisodeStatistics).

Verifies the eval CSV gains (§0.6, decision 11):
  - eval_noise_scale (robustness-curve locator),
  - ep_return_true (clean/denoised return = sum of reward-decomposition terms),
  - ep_noise_{action_l2,obs_l2,reward} (per-episode summed injected noise),
and the key cross-check for reward-noise tasks:
  ep_return (received) - ep_return_true == ep_noise_reward.

`ep_return` keeps its meaning (received/accumulated return; noised for reward
tasks); ep_return_true is the new clean headline. F1 remains noise-immune.
Runs one full ForElise episode per env (module-scoped, fixed action).
"""
import csv
import os

import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration

_BASE = "RoboPianist-repertoire-150-ForElise-v0"

_NEW_COLS = [
    "eval_noise_scale", "ep_return_true",
    "ep_noise_action_l2", "ep_noise_obs_l2", "ep_noise_reward",
]


def _reg(env_id, **kw):
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, **kw)


_reg("OmniPianoTest-S6-Reward-v0",
     robust_config=RobustConfig(noise_dist="gaussian", reward_noise_std=0.5))
_reg("OmniPianoTest-S6-Action-v0",
     robust_config=RobustConfig(noise_dist="gaussian", action_noise_std=0.05))
_reg("OmniPianoTest-S6-Clean-v0")


def _run_and_read(env_id, log_dir, seed=0):
    env = registration.make(env_id, log_dir=str(log_dir), mode="eval", seed=seed)
    try:
        env.reset(seed=seed)
        act = np.zeros(env.action_space.shape, dtype=np.float32)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(act)
            done = bool(term or trunc)
    finally:
        env.close()
    csvs = [f for f in os.listdir(log_dir) if f.startswith("eval_episode_metrics_")]
    assert len(csvs) == 1, csvs
    with open(os.path.join(log_dir, csvs[0])) as f:
        rows = list(csv.DictReader(f))
    assert rows
    return rows[-1]


@pytest.fixture(scope="module")
def reward_row(tmp_path_factory):
    return _run_and_read("OmniPianoTest-S6-Reward-v0",
                         tmp_path_factory.mktemp("s6_reward"))


@pytest.fixture(scope="module")
def action_row(tmp_path_factory):
    return _run_and_read("OmniPianoTest-S6-Action-v0",
                         tmp_path_factory.mktemp("s6_action"))


@pytest.fixture(scope="module")
def clean_row(tmp_path_factory):
    return _run_and_read("OmniPianoTest-S6-Clean-v0",
                         tmp_path_factory.mktemp("s6_clean"))


def test_new_columns_present(reward_row):
    for col in _NEW_COLS:
        assert col in reward_row, f"missing column {col}"


def test_eval_noise_scale_matched_default(reward_row):
    assert float(reward_row["eval_noise_scale"]) == pytest.approx(1.0)


def test_reward_true_noised_crosscheck(reward_row):
    ep_return = float(reward_row["ep_return"])          # received (noised)
    ep_return_true = float(reward_row["ep_return_true"])  # clean
    ep_noise_reward = float(reward_row["ep_noise_reward"])
    assert ep_noise_reward != 0.0                        # reward noise injected
    # received - true == summed reward noise
    assert (ep_return - ep_return_true) == pytest.approx(ep_noise_reward, abs=1e-2)


def test_action_env_reward_clean_and_action_noise_logged(action_row):
    assert float(action_row["ep_noise_action_l2"]) > 0.0
    assert float(action_row["ep_noise_reward"]) == 0.0
    # no reward noise -> received == clean
    assert float(action_row["ep_return"]) == pytest.approx(
        float(action_row["ep_return_true"]), abs=1e-2)


def test_clean_env_all_noise_zero(clean_row):
    assert float(clean_row["ep_noise_action_l2"]) == 0.0
    assert float(clean_row["ep_noise_obs_l2"]) == 0.0
    assert float(clean_row["ep_noise_reward"]) == 0.0
    assert float(clean_row["ep_return"]) == pytest.approx(
        float(clean_row["ep_return_true"]), abs=1e-2)
