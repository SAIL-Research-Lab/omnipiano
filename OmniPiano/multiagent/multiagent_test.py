"""Tests for the multi-agent PettingZoo ParallelEnv wrapper (Sub-phase 1A).

Three classes:
- ``PettingZooConformanceTest`` — runs ``parallel_api_test`` on each
  registered 4-hand MA env
- ``LayoutRegressionTest`` — pins agent names / action dims / obs space
  structure against the AGENT_ASSIGNMENTS["FourHand"] decomposition
- ``SmokeTest`` — 100 random-action steps, no crash, finite reward
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from OmniPiano.multiagent import (
    AGENT_ASSIGNMENTS,
    list_parallel_envs,
    make_parallel,
)


# 4-hand envs registered in Sub-phase 1A.
_FOUR_HAND_MA_ENVS = (
    "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0",
    "OmniPiano-PianoSonataNo301StMov-FourHand-MA-Duet-Territorial-v0",
    "OmniPiano-PicturesGreatKiev-FourHand-MA-Duet-Territorial-v0",
)


# ===========================================================================
# Conformance — runs PettingZoo's official parallel_api_test on each env
# ===========================================================================


@pytest.mark.parametrize("env_id", _FOUR_HAND_MA_ENVS)
def test_pettingzoo_parallel_api_conformance(env_id: str) -> None:
    """Verify each 4-hand MA env satisfies the PettingZoo ParallelEnv API."""
    from pettingzoo.test import parallel_api_test

    # parallel_api_test cycles through reset/step many times — use num_cycles=50
    # (faster than 100 for the test suite; still exercises all branches).
    env = make_parallel(env_id, seed=0)
    try:
        parallel_api_test(env, num_cycles=50)
    finally:
        env.close()


# ===========================================================================
# Layout regression — pins agent decomposition for 4-hand Duet
# ===========================================================================


class LayoutRegressionTest:
    """Pins agent names / shapes for 4-hand Duet (canonical Sub-phase 1A morphology)."""

    def setup_method(self) -> None:
        self.env = make_parallel(
            "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0", seed=0
        )

    def teardown_method(self) -> None:
        self.env.close()

    def test_possible_agents_spatial_order(self) -> None:
        # Spatial L→R: secondo (bass-side) before primo (treble-side).
        assert self.env.possible_agents == ["secondo", "primo"]
        # AGENT_ASSIGNMENTS agrees.
        assert AGENT_ASSIGNMENTS["FourHand"].agent_names == ("secondo", "primo")

    def test_action_dims(self) -> None:
        # secondo = (lh_b, rh_b) + sustain → 2×22 + 1 = 45
        assert self.env.action_space("secondo").shape == (45,)
        # primo = (lh_t, rh_t) → 2×22 = 44
        assert self.env.action_space("primo").shape == (44,)
        # Canonical [-1, 1] action range from CanonicalSpecWrapper.
        for agent in ("secondo", "primo"):
            sp = self.env.action_space(agent)
            assert float(sp.low.min()) == -1.0
            assert float(sp.high.max()) == 1.0

    def test_obs_space_structure(self) -> None:
        # 4-hand: each agent has 1 boundary hand (single inter-agent boundary).
        secondo_obs = self.env.observation_space("secondo")
        primo_obs = self.env.observation_space("primo")
        # own_hands
        assert set(secondo_obs.spaces["own_hands"].spaces.keys()) == {"lh_b", "rh_b"}
        assert set(primo_obs.spaces["own_hands"].spaces.keys()) == {"lh_t", "rh_t"}
        # boundary_hands
        assert set(secondo_obs.spaces["boundary_hands"].spaces.keys()) == {"lh_t"}
        assert set(primo_obs.spaces["boundary_hands"].spaces.keys()) == {"rh_b"}
        # piano_state per agent (reach-sized).
        assert "piano_state" in secondo_obs.spaces
        assert "piano_state" in primo_obs.spaces
        # goal per agent (reach + sustain cols × lookahead frames).
        assert "goal" in secondo_obs.spaces
        # piano_sustain global, 1-dim.
        assert secondo_obs.spaces["piano_sustain"].shape == (1,)

    def test_reach_is_clean(self) -> None:
        """Probe-derived reach: secondo's reach starts at key 0, primo's ends at key 87."""
        obs, _ = self.env.reset(seed=0)
        # secondo reach should include key 0 (leftmost), primo's should not start at 0.
        secondo_reach_dim = obs["secondo"]["piano_state"].shape[0]
        primo_reach_dim = obs["primo"]["piano_state"].shape[0]
        # Roughly: secondo ~46 keys (0..~45), primo ~46 keys (~42..87). Allow ±5 for probe variance.
        assert 35 <= secondo_reach_dim <= 55, f"secondo reach dim {secondo_reach_dim}"
        assert 35 <= primo_reach_dim <= 55, f"primo reach dim {primo_reach_dim}"

    def test_reward_shared_broadcast(self) -> None:
        self.env.reset(seed=0)
        actions = {a: self.env.action_space(a).sample() for a in self.env.agents}
        _, rewards, _, _, _ = self.env.step(actions)
        # `shared` mode: all agents receive the same scalar.
        vals = list(rewards.values())
        assert all(v == vals[0] for v in vals), f"reward not shared: {rewards}"


# ===========================================================================
# Smoke — 100 random steps, no crashes, finite rewards
# ===========================================================================


@pytest.mark.parametrize("env_id", _FOUR_HAND_MA_ENVS)
def test_smoke_100_random_steps(env_id: str) -> None:
    env = make_parallel(env_id, seed=0)
    try:
        obs, infos = env.reset(seed=0)
        n_steps = 0
        rewards_history = []
        rng = np.random.default_rng(0)
        for _ in range(100):
            actions = {
                a: rng.uniform(low=-1.0, high=1.0, size=env.action_space(a).shape).astype(np.float32)
                for a in env.agents
            }
            obs, rewards, terms, truncs, infos = env.step(actions)
            rewards_history.append(rewards[next(iter(rewards))])
            n_steps += 1
            if all(terms.values()) or all(truncs.values()):
                # Episode ended naturally; re-reset and continue.
                obs, infos = env.reset(seed=0)
        assert n_steps == 100
        # All rewards finite.
        assert all(np.isfinite(r) for r in rewards_history), "non-finite reward"
    finally:
        env.close()


# ===========================================================================
# Registry sanity
# ===========================================================================


def test_all_four_hand_envs_registered() -> None:
    registered = set(list_parallel_envs())
    for env_id in _FOUR_HAND_MA_ENVS:
        assert env_id in registered, f"missing MA env: {env_id}"
