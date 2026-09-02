"""Tests for the multi-agent PettingZoo ParallelEnv wrapper.

Covered sub-phases:
  1A — 4-hand Duet (3 envs):     symmetric 2-agent, 1 inter-agent boundary
  1B — 3-hand MainSolo (4 envs): introduces 1-hand agent edge case
  1C — 5-hand Trio (1 env):      3-agent + multi-boundary (center_soloist
                                 has 2 neighbors)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from omnipiano.multiagent import (
    AGENT_ASSIGNMENTS,
    list_parallel_envs,
    make_parallel,
)
from omnipiano.multiagent.coordination_metrics import (
    BASE_TEAM_RETURN,
    COMMON_AREA_DUPLICATE_PRESS_RATE,
    COMMON_AREA_SUCCESS_RATE,
    COMMON_AREA_TARGET_COUNT,
    INTER_AGENT_COLLISION_PENALTY_COEF,
    INTER_AGENT_COLLISION_PENALTY_RETURN,
    INTER_AGENT_COLLISION_STEP_RATE,
    OBSERVED_STEP_COUNT,
    SHAPED_TEAM_RETURN,
)


# Sub-phase 1A: 4-hand Duet.
_FOUR_HAND_MA_ENVS = (
    "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0",
    "OmniPiano-PianoSonataNo301StMov-FourHand-MA-Duet-Territorial-v0",
    "OmniPiano-PicturesGreatKiev-FourHand-MA-Duet-Territorial-v0",
)

# Sub-phase 1B: 3-hand MainSolo.
_THREE_HAND_MA_ENVS = (
    "OmniPiano-WinterWind-ThreeHand-MA-MainSolo-Territorial-v0",
    "OmniPiano-PicturesGreatKiev-ThreeHand-MA-MainSolo-Territorial-v0",
    "OmniPiano-PolonaiseOp40No1-ThreeHand-MA-MainSolo-Territorial-v0",
    "OmniPiano-PianoSonataNo281StMov-ThreeHand-MA-MainSolo-Territorial-v0",
)

# Sub-phase 1C: 5-hand Trio. Only WinterWind — PicturesGreatKiev 5-hand
# StaticPartition is deferred at the SA layer (see Option C decision +
# envs/__init__.py:1055 comment).
_FIVE_HAND_MA_ENVS = (
    "OmniPiano-WinterWind-FiveHand-MA-Trio-Territorial-v0",
)

# Aggregate for tests that should cover every registered MA env.
_ALL_MA_ENVS = _FOUR_HAND_MA_ENVS + _THREE_HAND_MA_ENVS + _FIVE_HAND_MA_ENVS


# ===========================================================================
# Conformance — runs PettingZoo's official parallel_api_test on each env
# ===========================================================================


@pytest.mark.parametrize("env_id", _ALL_MA_ENVS)
def test_pettingzoo_parallel_api_conformance(env_id: str) -> None:
    """Verify each registered MA env satisfies the PettingZoo ParallelEnv API."""
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
# 3-hand MainSolo layout regression — pins 1-hand agent edge case
# ===========================================================================


class TestThreeHandLayoutRegression:
    """Pins agent decomposition for 3-hand MainSolo (Sub-phase 1B).

    Validates the 1-hand agent code path: treble_soloist controls only `rh`
    (action dim 22, NO sustain slice since secondo is sustain owner),
    boundary_hands has exactly 1 neighbor (secondo's rh_c).
    """

    def setup_method(self) -> None:
        self.env = make_parallel(
            "OmniPiano-WinterWind-ThreeHand-MA-MainSolo-Territorial-v0", seed=0
        )

    def teardown_method(self) -> None:
        self.env.close()

    def test_possible_agents_spatial_order(self) -> None:
        # Spatial L→R: secondo (bass + middle, 2 hands) before treble_soloist (1 hand).
        assert self.env.possible_agents == ["secondo", "treble_soloist"]
        assert AGENT_ASSIGNMENTS["ThreeHand"].agent_names == ("secondo", "treble_soloist")

    def test_action_dims_one_hand_agent_has_no_sustain(self) -> None:
        # secondo = (lh, rh_c) + sustain → 2×22 + 1 = 45
        assert self.env.action_space("secondo").shape == (45,)
        # treble_soloist = (rh) only → 22 (sustain belongs to secondo, no +1)
        assert self.env.action_space("treble_soloist").shape == (22,)
        for agent in ("secondo", "treble_soloist"):
            sp = self.env.action_space(agent)
            assert float(sp.low.min()) == -1.0
            assert float(sp.high.max()) == 1.0

    def test_obs_space_structure_three_hand(self) -> None:
        secondo_obs = self.env.observation_space("secondo")
        soloist_obs = self.env.observation_space("treble_soloist")
        # own_hands
        assert set(secondo_obs.spaces["own_hands"].spaces.keys()) == {"lh", "rh_c"}
        assert set(soloist_obs.spaces["own_hands"].spaces.keys()) == {"rh"}
        # boundary_hands: 1 neighbor each
        assert set(secondo_obs.spaces["boundary_hands"].spaces.keys()) == {"rh"}
        assert set(soloist_obs.spaces["boundary_hands"].spaces.keys()) == {"rh_c"}

    def test_reach_three_hand_matches_design_doc(self) -> None:
        """3-hand reach numbers from probe should match design doc § 5.2 within ±2 keys.

        Design doc: secondo (0, 61), treble_soloist (51, 87); shared zone
        51-62 ~= 12 keys. The runtime-aligned probe gives slightly different
        endpoints due to neutral-finger-pose conservativeness; tolerance ±2.
        """
        _, infos = self.env.reset(seed=0)
        sec_lo, sec_hi = infos["secondo"]["agent_key_range"]
        ts_lo, ts_hi = infos["treble_soloist"]["agent_key_range"]
        assert abs(sec_lo - 0) <= 2, f"secondo reach_lo {sec_lo} drift > 2 from design 0"
        assert abs(sec_hi - 61) <= 2, f"secondo reach_hi {sec_hi} drift > 2 from design 61"
        assert abs(ts_lo - 51) <= 2, f"treble_soloist reach_lo {ts_lo} drift > 2 from design 51"
        assert abs(ts_hi - 87) <= 2, f"treble_soloist reach_hi {ts_hi} drift > 2 from design 87"
        # Shared zone non-empty (must overlap ≥ 8 keys per design doc ~12).
        sz_lo = max(sec_lo, ts_lo)
        sz_hi = min(sec_hi, ts_hi)
        sz_width = sz_hi - sz_lo + 1
        assert sz_width >= 8, f"3-hand shared zone width {sz_width} too narrow (expect ≥8)"

    def test_per_agent_clamp_secondo_hands_share_joint_range_three_hand(self) -> None:
        """Plan § 2.2 under 3-hand: secondo's lh and rh_c must share forearm_tx joint.range."""
        from omnipiano.multiagent.parallel_env import _find_task
        from omnipiano.multiagent._reach_probe import y_to_key_index
        task = _find_task(self.env._env)
        ranges_by_hand: Dict[str, tuple] = {}
        for spec_name, hand in task.hands_by_name.items():
            fj = next(j for j in hand.mjcf_model.find_all("joint")
                      if "forearm_tx" in j.name)
            attach_y = float(hand.root_body.pos[1])
            lo_key = y_to_key_index(attach_y + float(fj.range[0]))
            hi_key = y_to_key_index(attach_y + float(fj.range[1]))
            ranges_by_hand[spec_name] = (lo_key, hi_key)
        # secondo agent owns (lh, rh_c) — must share key range = territory [0, 58].
        assert ranges_by_hand["lh"] == ranges_by_hand["rh_c"], (
            f"per-agent clamp violated for 3-hand secondo: "
            f"lh={ranges_by_hand['lh']} rh_c={ranges_by_hand['rh_c']}"
        )
        # treble_soloist agent owns only rh — its own clamp = territory [59, 87].
        assert ranges_by_hand["rh"][0] >= 58, (
            f"treble_soloist (rh) clamp lo {ranges_by_hand['rh'][0]} should be >= 58"
        )


# ===========================================================================
# 5-hand Trio layout regression — pins 3-agent + multi-boundary edge case
# ===========================================================================


class TestFiveHandLayoutRegression:
    """Pins agent decomposition for 5-hand Trio (Sub-phase 1C).

    Validates the 3-agent + multi-boundary code path: center_soloist has
    2 neighbors (left_secondo and right_primo) so its `boundary_hands`
    Dict has 2 keys, not 1 — exercising the wrapper's data-driven
    boundary iteration. Also validates two simultaneous per-agent clamps:
    left_secondo and right_primo each have their own (lh, rh) duet
    sharing a per-agent territory clamp.
    """

    def setup_method(self) -> None:
        self.env = make_parallel(
            "OmniPiano-WinterWind-FiveHand-MA-Trio-Territorial-v0", seed=0
        )

    def teardown_method(self) -> None:
        self.env.close()

    def test_possible_agents_spatial_order(self) -> None:
        # Spatial L→R: left_secondo, center_soloist, right_primo
        assert self.env.possible_agents == [
            "left_secondo", "center_soloist", "right_primo"
        ]
        assert AGENT_ASSIGNMENTS["FiveHand"].agent_names == (
            "left_secondo", "center_soloist", "right_primo"
        )

    def test_action_dims_three_agent_five_hand(self) -> None:
        # left_secondo (sustain owner) = (lh_b, rh_b) + sustain → 45
        assert self.env.action_space("left_secondo").shape == (45,)
        # center_soloist = (rh_c) only → 22 (1-hand, no sustain)
        assert self.env.action_space("center_soloist").shape == (22,)
        # right_primo = (lh_t, rh_t) → 44 (no sustain — owned by left_secondo)
        assert self.env.action_space("right_primo").shape == (44,)
        for agent in ("left_secondo", "center_soloist", "right_primo"):
            sp = self.env.action_space(agent)
            assert float(sp.low.min()) == -1.0
            assert float(sp.high.max()) == 1.0

    def test_center_soloist_has_two_boundary_hands(self) -> None:
        """The defining 1C edge case: center_soloist's boundary_hands
        Dict has 2 keys (left + right neighbors), not 1."""
        obs_space = self.env.observation_space("center_soloist")
        boundary_keys = set(obs_space.spaces["boundary_hands"].spaces.keys())
        # Left neighbor (left_secondo) contributes its rightmost hand `rh_b`.
        # Right neighbor (right_primo) contributes its leftmost hand `lh_t`.
        assert boundary_keys == {"rh_b", "lh_t"}, (
            f"center_soloist boundary_hands should be {{rh_b, lh_t}}, got {boundary_keys}"
        )
        # Outer agents see only 1 neighbor (the center) → exactly 1 boundary hand.
        left_boundary = set(
            self.env.observation_space("left_secondo").spaces["boundary_hands"].spaces.keys()
        )
        right_boundary = set(
            self.env.observation_space("right_primo").spaces["boundary_hands"].spaces.keys()
        )
        assert left_boundary == {"rh_c"}, f"left_secondo boundary = {left_boundary}"
        assert right_boundary == {"rh_c"}, f"right_primo boundary = {right_boundary}"

    def test_obs_space_structure_five_hand(self) -> None:
        left_obs = self.env.observation_space("left_secondo")
        center_obs = self.env.observation_space("center_soloist")
        right_obs = self.env.observation_space("right_primo")
        # own_hands per agent
        assert set(left_obs.spaces["own_hands"].spaces.keys()) == {"lh_b", "rh_b"}
        assert set(center_obs.spaces["own_hands"].spaces.keys()) == {"rh_c"}
        assert set(right_obs.spaces["own_hands"].spaces.keys()) == {"lh_t", "rh_t"}

    def test_reach_five_hand_matches_design_doc(self) -> None:
        """5-hand reach + shared zones match design doc § 5.2 within ±2 keys.

        Design doc:
          left_secondo   (0, 38)  — lh_b LEFT to rh_b RIGHT
          center_soloist (28, 55) — rh_c LEFT to rh_c RIGHT
          right_primo    (50, 87) — lh_t LEFT to rh_t RIGHT
        Shared zones (intersection of adjacent reaches):
          left ↔ center: keys 28-38 (~11 keys, both pinkies at boundary)
          center ↔ right: keys 50-55 (~6 keys, narrow due to pinky-only)
        """
        _, infos = self.env.reset(seed=0)
        L_lo, L_hi = infos["left_secondo"]["agent_key_range"]
        C_lo, C_hi = infos["center_soloist"]["agent_key_range"]
        R_lo, R_hi = infos["right_primo"]["agent_key_range"]

        assert abs(L_lo - 0) <= 2,  f"left_secondo reach_lo {L_lo} drift > 2"
        assert abs(L_hi - 38) <= 2, f"left_secondo reach_hi {L_hi} drift > 2"
        assert abs(C_lo - 28) <= 2, f"center_soloist reach_lo {C_lo} drift > 2"
        assert abs(C_hi - 55) <= 2, f"center_soloist reach_hi {C_hi} drift > 2"
        assert abs(R_lo - 50) <= 2, f"right_primo reach_lo {R_lo} drift > 2"
        assert abs(R_hi - 87) <= 2, f"right_primo reach_hi {R_hi} drift > 2"

        # Two inter-agent shared zones — both non-empty.
        sz1_lo = max(L_lo, C_lo); sz1_hi = min(L_hi, C_hi)
        sz2_lo = max(C_lo, R_lo); sz2_hi = min(C_hi, R_hi)
        sz1_w = sz1_hi - sz1_lo + 1
        sz2_w = sz2_hi - sz2_lo + 1
        # Design: ~11 keys left↔center, ~6 keys center↔right; tolerance ±2.
        assert 9 <= sz1_w <= 13, f"5-hand left↔center shared zone width {sz1_w} (expect ~11)"
        assert 4 <= sz2_w <= 8,  f"5-hand center↔right shared zone width {sz2_w} (expect ~6)"

    def test_per_agent_clamp_five_hand(self) -> None:
        """Plan § 2.2: per-agent clamp under 5-hand — left_secondo and
        right_primo each have 2 hands sharing one joint.range; center_soloist
        has its own (single-hand) clamp."""
        from omnipiano.multiagent.parallel_env import _find_task
        from omnipiano.multiagent._reach_probe import y_to_key_index
        task = _find_task(self.env._env)
        ranges_by_hand: Dict[str, tuple] = {}
        for spec_name, hand in task.hands_by_name.items():
            fj = next(j for j in hand.mjcf_model.find_all("joint")
                      if "forearm_tx" in j.name)
            attach_y = float(hand.root_body.pos[1])
            lo_key = y_to_key_index(attach_y + float(fj.range[0]))
            hi_key = y_to_key_index(attach_y + float(fj.range[1]))
            ranges_by_hand[spec_name] = (lo_key, hi_key)
        # left_secondo (lh_b, rh_b) share clamp = territory [0, 35]
        assert ranges_by_hand["lh_b"] == ranges_by_hand["rh_b"], (
            f"per-agent clamp violated for 5-hand left_secondo: "
            f"lh_b={ranges_by_hand['lh_b']} rh_b={ranges_by_hand['rh_b']}"
        )
        # right_primo (lh_t, rh_t) share clamp = territory [53, 87]
        assert ranges_by_hand["lh_t"] == ranges_by_hand["rh_t"], (
            f"per-agent clamp violated for 5-hand right_primo: "
            f"lh_t={ranges_by_hand['lh_t']} rh_t={ranges_by_hand['rh_t']}"
        )
        # rh_c is its own agent (center_soloist), with territory [36, 52]
        # — different from both neighbors.
        assert ranges_by_hand["rh_c"] != ranges_by_hand["lh_b"]
        assert ranges_by_hand["rh_c"] != ranges_by_hand["lh_t"]


# ===========================================================================
# Smoke — 100 random steps, no crashes, finite rewards
# ===========================================================================


@pytest.mark.parametrize("env_id", _ALL_MA_ENVS)
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


def test_all_ma_envs_registered() -> None:
    registered = set(list_parallel_envs())
    for env_id in _ALL_MA_ENVS:
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

    def test_collision_penalty_is_shared_once_and_updates_actor_oar(self) -> None:
        env = make_parallel(
            self.env_id,
            seed=0,
            inter_agent_collision_penalty_coef=0.25,
        )
        try:
            env.reset(seed=0)
            # Isolate reward plumbing from the physics contact state.  The
            # pure classifier has separate coverage for inter vs intra-agent
            # contacts; this forces one positive indicator on this step.
            env._coordination_tracker.observe_step = lambda physics, task: True
            actions = {
                agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
            observations, rewards, _, _, infos = env.step(actions)

            expected = env._episode_base_team_return - 0.25
            assert env._episode_collision_penalty_return == pytest.approx(-0.25)
            assert all(
                reward == pytest.approx(expected) for reward in rewards.values()
            )
            for agent in rewards:
                assert infos[agent][
                    "step_coordination/inter_agent_collision"
                ] is True
                assert infos[agent][
                    "step_reward/inter_agent_collision_penalty"
                ] == pytest.approx(-0.25)
                assert observations[agent]["prev_reward"][0] == pytest.approx(
                    expected
                )
        finally:
            env.close()

    def test_collision_penalty_updates_mappo_global_oar(self) -> None:
        env = make_parallel(
            self.env_id,
            seed=0,
            flatten_obs=True,
            include_global_state=True,
            inter_agent_collision_penalty_coef=0.25,
        )
        try:
            env.reset(seed=0)
            env._coordination_tracker.observe_step = lambda physics, task: True
            actions = {
                agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                for agent in env.agents
            }
            observations, rewards, _, _, _ = env.step(actions)
            expected = next(iter(rewards.values()))

            component_offset = 0
            previous_reward_offset = None
            for name, width in env._global_state_components():
                if name == "prev_reward":
                    previous_reward_offset = component_offset
                    break
                component_offset += width
            assert previous_reward_offset is not None

            for agent, observation in observations.items():
                global_start, _ = env.obs_layout(agent)["global_state"]
                assert observation[
                    global_start + previous_reward_offset
                ] == pytest.approx(expected)
        finally:
            env.close()

    # --- Plan § 8: infos["_global_"]["episode_task/musical_f1"] ---

    def test_global_evaluation_metrics_present_at_episode_end(self) -> None:
        """Musical and MARL coordination metrics are populated at episode end."""
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

        global_metrics = last_infos["_global_"]
        assert global_metrics[COMMON_AREA_TARGET_COUNT] > 0, (
            "WinterWind four-hand benchmark must contain common-area targets"
        )
        for key in (
            COMMON_AREA_SUCCESS_RATE,
            COMMON_AREA_DUPLICATE_PRESS_RATE,
            INTER_AGENT_COLLISION_STEP_RATE,
        ):
            assert key in global_metrics, f"coordination metric missing: {key}"
            assert 0.0 <= global_metrics[key] <= 1.0, (
                f"coordination metric out of range: {key}={global_metrics[key]}"
            )
        assert global_metrics[OBSERVED_STEP_COUNT] > 0
        assert global_metrics[INTER_AGENT_COLLISION_PENALTY_COEF] == 0.0
        assert global_metrics[INTER_AGENT_COLLISION_PENALTY_RETURN] == 0.0
        assert global_metrics[SHAPED_TEAM_RETURN] == pytest.approx(
            global_metrics[BASE_TEAM_RETURN]
        )

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

    # --- Plan § 1: omnipiano.make() rejects MA env_ids ---

    def test_sa_make_redirects_for_ma_env_id(self) -> None:
        """Plan § 1: omnipiano.make(MA env_id) must raise with redirect to make_parallel."""
        import pytest as _pytest
        import omnipiano
        with _pytest.raises(ValueError, match="make_parallel"):
            omnipiano.make(self.env_id)

    # --- Plan § 2: per-agent clamp (hands of same agent share joint.range) ---

    def test_per_agent_clamp_secondo_hands_share_joint_range(self) -> None:
        """Plan § 2: under MA Territorial, all hands of one agent share forearm_tx
        joint.range (= agent territory), removing the SA per-hand bucket wall."""
        from omnipiano.multiagent.parallel_env import _find_task
        from omnipiano.multiagent._reach_probe import y_to_key_index
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
        assert sz_width >= 2, f"shared zone width {sz_width} too narrow (expect ≥2)"


# ---------------------------------------------------------------------------
# make_parallel guard — SA envs carrying safety constraints or action/reward
# robust channels are rejected (the MA chain has no gym-layer Safety/Robust
# equivalent; silently dropping them would corrupt benchmark data). Obs-noise
# SA envs remain allowed (DmEnvObsNoiseWrapper is dm_env-layer).
# ---------------------------------------------------------------------------
def test_make_parallel_rejects_safety_sa_env():
    from omnipiano.multiagent import registration as ma_reg

    ma_id = "OmniPianoTest-MA-GuardSafety-v0"
    if ma_id not in ma_reg._ma_registry:
        ma_reg.register_parallel(
            id=ma_id,
            sa_env_id="OmniPiano-ClairDeLune-CollisionSafe-v0",
            morphology="ThreeHand",
        )
    try:
        with pytest.raises(ValueError, match="active safety constraints"):
            make_parallel(ma_id)
    finally:
        ma_reg._ma_registry.pop(ma_id, None)


def test_make_parallel_rejects_action_robust_sa_env():
    from omnipiano.multiagent import registration as ma_reg

    ma_id = "OmniPianoTest-MA-GuardRobustA-v0"
    if ma_id not in ma_reg._ma_registry:
        ma_reg.register_parallel(
            id=ma_id,
            sa_env_id="OmniPiano-ClairDeLune-A-Gauss-P05-v0",
            morphology="ThreeHand",
        )
    try:
        with pytest.raises(ValueError, match="robust channel"):
            make_parallel(ma_id)
    finally:
        ma_reg._ma_registry.pop(ma_id, None)
