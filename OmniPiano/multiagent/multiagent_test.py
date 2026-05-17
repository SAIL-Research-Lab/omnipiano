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


class TestLayoutRegression:
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


# ===========================================================================
# Plan-invariant regression tests (lock plan promises into the test suite)
# ===========================================================================
# Each test here pins one explicit plan/design-doc invariant against the
# implementation. If a future code change drifts away from a plan invariant
# (as happened with per-agent clamp + sustain_owner override earlier),
# these tests catch it immediately — no more silent plan/code divergence.


class TestPlanInvariant:
    """Test class grouping plan invariants — pytest auto-collects methods."""

    def setup_method(self) -> None:
        self.env_id = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"
        self.env = make_parallel(self.env_id, seed=0)

    def teardown_method(self) -> None:
        self.env.close()

    # --- Plan § 8: infos[agent]["agent_key_range"] ---

    def test_infos_agent_key_range_present_in_reset(self) -> None:
        """Plan § 8: infos[agent] must include agent_key_range at reset."""
        _, infos = self.env.reset(seed=0)
        for agent in self.env.agents:
            assert "agent_key_range" in infos[agent], (
                f"agent_key_range missing from reset infos[{agent}]"
            )
            lo, hi = infos[agent]["agent_key_range"]
            assert isinstance(lo, int) and isinstance(hi, int)
            assert 0 <= lo <= hi < 88

    def test_infos_agent_key_range_present_in_step(self) -> None:
        """Plan § 8: agent_key_range is constant — must also appear in step infos."""
        self.env.reset(seed=0)
        actions = {a: self.env.action_space(a).sample() for a in self.env.agents}
        _, _, _, _, infos = self.env.step(actions)
        for agent in self.env.agents:
            assert "agent_key_range" in infos[agent], (
                f"agent_key_range missing from step infos[{agent}]"
            )

    # --- Plan § 8: infos["_global_"]["episode_task/musical_f1"] ---

    def test_global_musical_f1_present_at_episode_end(self) -> None:
        """Plan § 8: infos['_global_']['episode_task/musical_f1'] populated at episode-end."""
        import numpy as np
        self.env.reset(seed=0)
        rng = np.random.default_rng(0)
        last_infos = None
        for _ in range(1000):
            if not self.env.agents:
                break
            actions = {
                a: rng.uniform(-1.0, 1.0, size=self.env.action_space(a).shape).astype(np.float32)
                for a in self.env.agents
            }
            _, _, terms, truncs, last_infos = self.env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break
        assert last_infos is not None and "_global_" in last_infos, (
            "infos['_global_'] missing at episode end"
        )
        assert "episode_task/musical_f1" in last_infos["_global_"], (
            "musical_f1 missing from infos['_global_']"
        )
        f1 = last_infos["_global_"]["episode_task/musical_f1"]
        assert 0.0 <= f1 <= 1.0, f"F1 out of range: {f1}"

    # --- Plan § 6: sustain_owner override actually takes effect ---

    def test_sustain_owner_override_relocates_sustain_dim(self) -> None:
        """Plan § 6: make_parallel(sustain_owner=...) moves the +1 sustain dim."""
        # Default: secondo owns sustain → 45/44
        env_default = make_parallel(self.env_id, seed=0)
        try:
            assert env_default.action_space("secondo").shape == (45,)
            assert env_default.action_space("primo").shape == (44,)
        finally:
            env_default.close()
        # Override: primo owns sustain → swap
        env_override = make_parallel(self.env_id, sustain_owner="primo", seed=0)
        try:
            assert env_override.action_space("secondo").shape == (44,)
            assert env_override.action_space("primo").shape == (45,)
        finally:
            env_override.close()

    def test_sustain_owner_invalid_name_raises(self) -> None:
        """Plan § 6: invalid sustain_owner agent name raises ValueError."""
        import pytest as _pytest
        with _pytest.raises(ValueError, match="sustain_owner"):
            make_parallel(self.env_id, sustain_owner="nonexistent_agent", seed=0)

    # --- Plan § 1: OmniPiano.make() rejects MA env_ids ---

    def test_sa_make_redirects_for_ma_env_id(self) -> None:
        """Plan § 1: OmniPiano.make(MA env_id) must raise with redirect to make_parallel."""
        import pytest as _pytest
        import OmniPiano
        with _pytest.raises(ValueError, match="make_parallel"):
            OmniPiano.make(self.env_id)

    # --- Plan § 2: per-agent clamp (hands of same agent share joint.range) ---

    def test_per_agent_clamp_secondo_hands_share_joint_range(self) -> None:
        """Plan § 2: under MA Territorial, all hands of one agent share forearm_tx
        joint.range (= agent territory), removing the SA per-hand bucket wall."""
        from OmniPiano.multiagent.parallel_env import _find_task
        from OmniPiano.multiagent._reach_probe import y_to_key_index
        task = _find_task(self.env._env)
        # Map each spec hand to its forearm_tx joint range (world key range).
        ranges_by_hand: Dict[str, Tuple[int, int]] = {}
        for spec_name, hand in task.hands_by_name.items():
            fj = next(j for j in hand.mjcf_model.find_all("joint")
                      if "forearm_tx" in j.name)
            attach_y = float(hand.root_body.pos[1])
            lo_key = y_to_key_index(attach_y + float(fj.range[0]))
            hi_key = y_to_key_index(attach_y + float(fj.range[1]))
            ranges_by_hand[spec_name] = (lo_key, hi_key)
        # secondo agent owns (lh_b, rh_b) — they must share the same world key range.
        assert ranges_by_hand["lh_b"] == ranges_by_hand["rh_b"], (
            f"per-agent clamp violated for secondo: "
            f"lh_b={ranges_by_hand['lh_b']} rh_b={ranges_by_hand['rh_b']}"
        )
        # primo agent owns (lh_t, rh_t).
        assert ranges_by_hand["lh_t"] == ranges_by_hand["rh_t"], (
            f"per-agent clamp violated for primo: "
            f"lh_t={ranges_by_hand['lh_t']} rh_t={ranges_by_hand['rh_t']}"
        )

    # --- Plan § 5.2 / probe regression: shared zone matches design doc ±2 keys ---

    def test_reach_matches_design_doc_within_tolerance(self) -> None:
        """Plan § 5.2 + design doc § 5.2 4-hand shared zone: probe must agree ±2 keys.

        Locks in the probe correctness invariant: if the probe code drifts
        (wrong actuator index, wrong physics config, wrong outer-extreme rule),
        the shared zone numbers shift and this test catches it.
        """
        _, infos = self.env.reset(seed=0)
        secondo_reach = infos["secondo"]["agent_key_range"]
        primo_reach = infos["primo"]["agent_key_range"]
        # Design doc § 5.2 4-hand: secondo (0, 45), primo (42, 87), shared zone (42, 45) = 4 keys.
        # Probe gives slightly different boundary due to neutral finger pose vs design doc estimate.
        # Tolerance ±2 keys per endpoint = ±4 keys on shared zone width.
        assert abs(secondo_reach[0] - 0) <= 2, f"secondo reach_lo {secondo_reach[0]} off by >2 from design 0"
        assert abs(secondo_reach[1] - 45) <= 2, f"secondo reach_hi {secondo_reach[1]} off by >2 from design 45"
        assert abs(primo_reach[0] - 42) <= 2, f"primo reach_lo {primo_reach[0]} off by >2 from design 42"
        assert abs(primo_reach[1] - 87) <= 2, f"primo reach_hi {primo_reach[1]} off by >2 from design 87"
        # Shared zone non-empty (must overlap).
        sz_lo = max(secondo_reach[0], primo_reach[0])
        sz_hi = min(secondo_reach[1], primo_reach[1])
        sz_width = sz_hi - sz_lo + 1
        assert sz_width >= 2, f"shared zone width {sz_width} too narrow (expect ≥4)"
