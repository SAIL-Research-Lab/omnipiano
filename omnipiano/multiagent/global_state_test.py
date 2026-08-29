"""The global state must be strictly ADDITIVE to the IPPO observation.

If the `own` block is not bit-identical to the IPPO observation, then IPPO and
MAPPO policies do not see the same input and the comparison is confounded.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnipiano.multiagent import make_parallel

_ENV_ID = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"


def _paired_rollout(n_steps: int = 20):
    plain = make_parallel(_ENV_ID, seed=0, flatten_obs=True)
    ctde = make_parallel(_ENV_ID, seed=0, flatten_obs=True, include_global_state=True)
    try:
        obs_plain, _ = plain.reset(seed=0)
        obs_ctde, _ = ctde.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(n_steps):
            actions = {
                a: rng.uniform(-1.0, 1.0, size=plain.action_space(a).shape).astype(np.float32)
                for a in plain.agents
            }
            yield plain, ctde, obs_plain, obs_ctde
            obs_plain, r_plain, _, _, _ = plain.step(actions)
            obs_ctde, r_ctde, _, _, _ = ctde.step(actions)
            for agent in r_plain:
                assert r_plain[agent] == pytest.approx(r_ctde[agent], abs=0.0), (
                    "adding the global state changed the reward stream"
                )
    finally:
        plain.close()
        ctde.close()


def test_own_block_matches_ippo_observation_bitwise() -> None:
    for plain, ctde, obs_plain, obs_ctde in _paired_rollout():
        for agent in plain.possible_agents:
            layout = ctde.obs_layout(agent)
            own_start, own_stop = layout["own"]
            gs_start, gs_stop = layout["global_state"]
            np.testing.assert_array_equal(
                obs_ctde[agent][own_start:own_stop],
                obs_plain[agent],
                err_msg=f"agent {agent}: own block differs from the IPPO obs",
            )
            assert gs_stop - gs_start == ctde.global_state_dim
            assert obs_ctde[agent].shape[0] == (gs_stop - gs_start) + obs_plain[agent].shape[0]


def test_global_state_carries_full_piano_state() -> None:
    """The centralized critic must see all 88 keys, not just its own reach."""
    ctde = make_parallel(_ENV_ID, seed=0, include_global_state=True, flatten_obs=True)
    plain = make_parallel(_ENV_ID, seed=0, flatten_obs=False)
    try:
        obs_ctde, _ = ctde.reset(seed=0)
        obs_dict, _ = plain.reset(seed=0)
        agent = ctde.possible_agents[0]
        gs_start, _ = ctde.obs_layout(agent)["global_state"]
        components = ctde._global_state_components()
        offset = gs_start
        found = False
        for name, dim in components:
            if name == "piano_state":
                piano = obs_ctde[agent][offset:offset + dim]
                assert dim == 88, "global state must expose all 88 keys"
                own_reach = obs_dict[agent]["piano_state"]
                assert piano.shape[0] > own_reach.shape[0], (
                    "global piano state is not wider than the agent's own reach"
                )
                found = True
                break
            offset += dim
        assert found
    finally:
        ctde.close()
        plain.close()


def test_agent_onehot_distinguishes_agents() -> None:
    ctde = make_parallel(_ENV_ID, seed=0, include_global_state=True, flatten_obs=True)
    try:
        obs, _ = ctde.reset(seed=0)
        n_agents = len(ctde.possible_agents)
        seen = []
        for agent in ctde.possible_agents:
            gs_start, gs_stop = ctde.obs_layout(agent)["global_state"]
            onehot = obs[agent][gs_stop - n_agents:gs_stop]
            assert onehot.sum() == pytest.approx(1.0)
            seen.append(int(np.argmax(onehot)))
        assert sorted(seen) == list(range(n_agents)), "agent one-hots collide"
    finally:
        ctde.close()