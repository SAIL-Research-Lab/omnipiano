"""PettingZoo ``ParallelEnv`` wrapper over the SA OmniPiano dm_env chain.

The MA shim sits on top of the dm_env-layer chain (which mirrors SA 1:1
minus ``ConcatObservationWrapper`` — see ``ma_territorial_impl_plan.md``
§ 10(a)). It:

  * splits the global Dict observation into per-agent Dicts
    (own_hands / boundary_hands / partitioned piano_state+goal / global sustain
    + optional prev_action/prev_reward slices from OAR)
  * reassembles per-agent canonical [-1, 1] action dicts into the flat
    SA action vector expected by ``CanonicalSpecWrapper`` → ``before_step``
  * broadcasts a single global scalar reward to all agents (``shared`` mode)
  * synchronizes terminations / truncations across agents

Phase 1 supports ``obs_visibility="own_plus_boundary"`` + ``reward_mode="shared"``
only — other modes raise NotImplementedError until Phase 2.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import dm_env
import gymnasium as gym
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from omnipiano.multiagent.assignment import (
    MorphologyAssignment,
    compute_agent_territory,
    compute_boundary_hands,
    compute_inter_agent_boundaries,
)


# Phase 1 supported modes.
_SUPPORTED_OBS_MODES = frozenset({"own_plus_boundary"})
_SUPPORTED_REWARD_MODES = frozenset({"shared"})


class OmniPianoParallelEnv(ParallelEnv):
    """4-hand / 3-hand / 5-hand Territorial multi-agent OmniPiano env.

    The dm_env chain passed in via ``env_builder`` is expected to already
    include (in this order):
      EpisodeStatistics → MidiEvaluation → [DmEnvObsNoise (opt)] →
      [ObservationActionReward (opt)] → [FrameStacking (opt)] →
      CanonicalSpec(clip=True) → SinglePrecision

    i.e., everything from SA chain except ``ConcatObservationWrapper``.

    Args:
        env_builder: ``Callable[[Optional[int]], dm_env.Environment]`` —
            takes a seed (or None) and returns a freshly-built dm_env chain.
            Same closure pattern as SA's ``DmEnvToGymnasium._build_dm_env_chain``.
        assignment: morphology agent decomposition (from AGENT_ASSIGNMENTS).
        hand_key_ranges: per-hand SA bucket key ranges (the source-of-truth
            for territory and boundary computation).
        agent_reaches: per-agent precise reach (from compute_agent_reach).
        seed: initial seed.
        obs_visibility: Phase 1 supports only "own_plus_boundary".
        reward_mode: Phase 1 supports only "shared".
        flatten_obs: if True, each agent's observation_space and obs values
            are flattened to a single Box via gymnasium.spaces.utils.flatten.
            Default False (Dict preserved).
    """

    metadata = {"render_modes": [], "name": "OmniPianoParallel-v0"}

    def __init__(
        self,
        env_builder,
        assignment,
        hand_key_ranges,
        agent_reaches,
        *,
        seed: Optional[int] = None,
        obs_visibility: str = "own_plus_boundary",
        reward_mode: str = "shared",
        flatten_obs: bool = False,
        sustain_owner: Optional[str] = None,
        include_global_state: bool = False,
    ) -> None:
        if obs_visibility not in _SUPPORTED_OBS_MODES:
            raise NotImplementedError(
                f"Phase 1 supports obs_visibility in {sorted(_SUPPORTED_OBS_MODES)}, "
                f"got {obs_visibility!r}. Other modes are Phase 2."
            )
        if reward_mode not in _SUPPORTED_REWARD_MODES:
            raise NotImplementedError(
                f"Phase 1 supports reward_mode in {sorted(_SUPPORTED_REWARD_MODES)}, "
                f"got {reward_mode!r}. Other modes are Phase 2."
            )

        self._env_builder = env_builder
        self._assignment = assignment
        self._hand_key_ranges = dict(hand_key_ranges)
        self._agent_reaches = dict(agent_reaches)
        self._obs_visibility = obs_visibility
        self._reward_mode = reward_mode
        self._flatten_obs = flatten_obs
        # CTDE: when True, every agent additionally receives the *centralized*
        # state s used by MAPPO's critic.  The actor must never read it -- see
        # ``_ctde_module.CtdePPOTorchRLModule`` which enforces that by slicing.
        self._include_global_state = include_global_state
        if include_global_state and not flatten_obs:
            raise NotImplementedError(
                "include_global_state=True currently requires flatten_obs=True: "
                "the CTDE RLModule addresses the actor / critic inputs by index "
                "into one flat observation vector."
            )

        # Resolve sustain owner: caller override (e.g. for Bizet/Dvořák
        # exception pieces or paper ablations) takes precedence over the
        # morphology default (`is_sustain_owner=True` flag in
        # AGENT_ASSIGNMENTS; see plan § 4 / § 6 + design doc § 4.3 for
        # rationale on the default = bass-side agent).
        if sustain_owner is None:
            self._sustain_owner_name = assignment.sustain_owner
        else:
            if sustain_owner not in {a.name for a in assignment.agents}:
                raise ValueError(
                    f"sustain_owner={sustain_owner!r} is not a valid agent "
                    f"for morphology {assignment.morphology!r}. "
                    f"Valid: {sorted(a.name for a in assignment.agents)}"
                )
            self._sustain_owner_name = sustain_owner

        # Derived structures.
        self._territories = compute_agent_territory(
            assignment, self._hand_key_ranges
        )
        self._boundaries = compute_inter_agent_boundaries(
            assignment, self._territories
        )
        # {agent_name: {neighbor_agent_name: that_neighbor's_boundary_hand_name}}
        self._boundary_hands_map = compute_boundary_hands(
            assignment, self._hand_key_ranges
        )

        self.possible_agents: List[str] = list(assignment.agent_names)
        self.agents: List[str] = []

        # Build initial dm_env chain to extract specs.
        self._current_seed = seed
        self._env: dm_env.Environment = env_builder(seed)
        self._dm_obs_spec = self._env.observation_spec()
        self._dm_action_spec = self._env.action_spec()

        # Discover OAR presence (extra "action"/"reward" keys in obs spec).
        self._has_oar = (
            "action" in self._dm_obs_spec and "reward" in self._dm_obs_spec
        )

        # Locate the MidiEvaluationWrapper in the dm_env chain — used to
        # extract episode-end musical F1/precision/recall metrics into
        # infos["_global_"] (see plan § 8). Cached once at init; not
        # re-discovered after env_builder rebuilds (the wrapper class is
        # the same; the instance changes but we walk fresh each time we
        # need it).
        self._midi_eval_wrapper_cls = _import_midi_evaluation_wrapper_class()

        # Build action-layout map: hand_name → slice(start, end) in flat action.
        self._hand_action_slices = self._build_action_layout()
        
        # Fixed spatial L->R hand order for a permutation-consistent global
        # state.  Must never depend on dict iteration order.
        self._all_hands_spatial: Tuple[str, ...] = tuple(
            hand
            for agent in self._assignment.agents
            for hand in agent.hand_names
        )
        self._agent_index = {
            agent.name: i for i, agent in enumerate(self._assignment.agents)
        }

        # Build per-agent action spaces first (needed for obs space construction
        # when OAR adds a prev_action slice).
        self._action_spaces = self._build_action_spaces()
        # Build per-agent obs spaces (Dict).
        self._dict_observation_spaces = self._build_observation_spaces()

        if flatten_obs:
            self._flat_observation_spaces = {
                a: gym.spaces.utils.flatten_space(s)
                for a, s in self._dict_observation_spaces.items()
            }
        else:
            self._flat_observation_spaces = None  # type: ignore

    # ---------------- PettingZoo API ----------------

    def observation_space(self, agent: str) -> gym.spaces.Space:
        if self._flatten_obs:
            return self._flat_observation_spaces[agent]
        return self._dict_observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, dict]]:
        # Gymnasium/PettingZoo contract: two consecutive reset(seed=S) calls must
        # produce bit-identical episodes.  ``_env.reset()`` only starts a new
        # episode; it does NOT rewind the dm_env chain's RNGs
        # (``composer.Environment._random_state`` and
        # ``DmEnvObsNoiseWrapper._rng`` are seeded at construction and only ever
        # advance).  Rebuilding the chain is the only way to rewind them, so
        # rebuild on *every* explicit seed -- matching the SA adapter at
        # ``omnipiano/envs/dm_env_adapter.py``.  Cost is ~0.9 s and is only paid
        # once per worker during training (RLlib passes an explicit seed only on
        # the first reset) and once per evaluation episode (which already uses a
        # fresh seed per episode).
        if seed is not None:
            self._env = self._env_builder(seed)
            self._current_seed = seed

        ts = self._env.reset()
        self.agents = list(self.possible_agents)
        obs = self._split_observation(ts.observation)
        infos = self._build_per_agent_infos()
        return obs, infos

    def step(
        self, actions: Mapping[str, np.ndarray]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        flat_action = self._reassemble_action(actions)
        ts = self._env.step(flat_action)

        obs = self._split_observation(ts.observation)

        # shared reward: broadcast scalar to all agents.
        reward_scalar = float(ts.reward) if ts.reward is not None else 0.0
        rewards = {a: reward_scalar for a in self.agents}

        # dm_env -> PettingZoo termination semantics:
        #   ts.last() and discount == 0  → terminated (true failure / end)
        #   ts.last() and discount != 0  → truncated (time-limit)
        # For piano: episode ends naturally when MIDI finishes (no special
        # discount=0 unless wrong_press_termination triggers). We treat
        # "discount == 0" as terminated and "discount > 0 with last" as truncated.
        last = ts.last()
        if last:
            terminated = ts.discount is not None and float(ts.discount) == 0.0
            truncated = not terminated
        else:
            terminated = False
            truncated = False
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}

        infos = self._build_per_agent_infos()
        # On episode-end, attach env-level musical metrics under "_global_"
        # (PettingZoo permits underscore-prefixed keys for env-level info;
        # see plan § 8). MidiEvaluationWrapper computes these at last() and
        # caches them in its deque; get_musical_metrics() raises ValueError
        # if no episode has finished yet (shouldn't happen here because we
        # gated on `last`, but guard defensively).
        if last:
            try:
                midi_eval = _find_wrapper(self._env, self._midi_eval_wrapper_cls)
                metrics = midi_eval.get_musical_metrics()
                infos["_global_"] = {
                    "episode_task/musical_f1": float(metrics["f1"]),
                    "episode_task/musical_precision": float(metrics["precision"]),
                    "episode_task/musical_recall": float(metrics["recall"]),
                    "episode_task/sustain_f1": float(metrics["sustain_f1"]),
                }
            except (ValueError, RuntimeError):
                # Metrics not yet available — leave _global_ unpopulated.
                pass

        # Once all agents are done, clear self.agents (PettingZoo convention).
        if last:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _build_per_agent_infos(self) -> Dict[str, dict]:
        """Construct per-agent info dict. Currently includes only `agent_key_range`
        (constant, used by user code to decode agent-local key indices back to
        global piano key indices)."""
        return {
            a: {"agent_key_range": self._agent_reaches[a]}
            for a in self.agents
        }

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def render(self) -> None:
        return None

    # ---------------- Internal: action layout ----------------

    def _build_action_layout(self) -> Dict[str, slice]:
        """Map each hand name to its slice in the flat dm_env action vector.

        dm_env action is `[hand_0_actuators | hand_1_actuators | ... | sustain]`
        in `task.hands` spec order. With the N-hand action-layout axiom (N≥3
        spatial L→R), this order equals the spec tuple order.

        The action_spec from CanonicalSpec is nested per the `tree` library
        (likely a single BoundedArray for piano_with_shadow_hands). We use
        size information by introspecting the underlying task.hands.
        """
        # Walk into the dm_env chain to find the task object (for per-hand sizes).
        task = _find_task(self._env)

        # ShadowHand instances have name='lh_b_shadow_hand' (mjcf model name),
        # but the canonical hand identifier used in AGENT_ASSIGNMENTS /
        # hand_specs is just 'lh_b'. The canonical mapping lives in
        # `task.hands_by_name`, whose key order matches `task.hands` order.
        slices: Dict[str, slice] = {}
        offset = 0
        for spec_name, hand in task.hands_by_name.items():
            size = hand.action_spec(self._env.physics).shape[0]
            slices[spec_name] = slice(offset, offset + size)
            offset += size
        # sustain occupies the last element.
        self._sustain_idx = offset
        self._flat_action_dim = offset + 1
        return slices

    def _reassemble_action(
        self, actions: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Per-agent action Dict → flat dm_env action vector.

        Each agent's action vector (canonical [-1, 1]) is its hands' 22-DoF
        slices concatenated in spatial L→R order, with sustain_owner having a
        +1 sustain dim at the end.
        """
        flat = np.zeros(self._flat_action_dim, dtype=np.float32)
        for agent in self._assignment.agents:
            agent_act = np.asarray(actions[agent.name], dtype=np.float32)
            cursor = 0
            for hand_name in agent.hand_names:
                sl = self._hand_action_slices[hand_name]
                size = sl.stop - sl.start
                flat[sl] = agent_act[cursor : cursor + size]
                cursor += size
            if agent.name == self._sustain_owner_name:
                # Sustain is the LAST element of the sustain_owner's action.
                flat[self._sustain_idx] = agent_act[cursor]
                cursor += 1
            if cursor != agent_act.shape[0]:
                raise ValueError(
                    f"action for agent {agent.name!r} has size {agent_act.shape[0]} "
                    f"but expected to consume {cursor} elements"
                )
        return flat

    def _build_action_spaces(self) -> Dict[str, gym.spaces.Box]:
        """Per-agent action_space — Box(-1, 1) since CanonicalSpec is in chain."""
        spaces: Dict[str, gym.spaces.Box] = {}
        for agent in self._assignment.agents:
            dim = sum(
                self._hand_action_slices[h].stop - self._hand_action_slices[h].start
                for h in agent.hand_names
            )
            if agent.name == self._sustain_owner_name:
                dim += 1
            spaces[agent.name] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(dim,), dtype=np.float32
            )
        return spaces

    # ---------------- Internal: obs splitting ----------------

    def _build_observation_spaces(self) -> Dict[str, gym.spaces.Dict]:
        spaces: Dict[str, gym.spaces.Dict] = {}
        # Determine goal lookahead from dm_env obs spec.
        goal_spec = self._dm_obs_spec["goal"]
        goal_flat = int(goal_spec.shape[0])
        # goal_flat = (lookahead + 1) * (num_keys + 1) = (lookahead + 1) * 89
        if goal_flat % 89 != 0:
            raise RuntimeError(
                f"unexpected goal shape {goal_spec.shape}; not divisible by 89"
            )
        self._goal_lookahead_plus_1 = goal_flat // 89

        # Joint-pos dim from any one hand spec.
        any_hand_key = next(
            k for k in self._dm_obs_spec if k.endswith("/joints_pos")
        )
        joints_dim = int(self._dm_obs_spec[any_hand_key].shape[0])
        self._joints_dim = joints_dim

        # piano_state dim (88 keys).
        piano_state_dim = int(self._dm_obs_spec["piano/state"].shape[0])
        if piano_state_dim != 88:
            raise RuntimeError(
                f"unexpected piano/state dim {piano_state_dim}, expected 88"
            )

        # Pre-cache per-agent slicing info.
        self._agent_reach_slice: Dict[str, slice] = {}
        self._agent_goal_dim: Dict[str, int] = {}
        for agent_name, (lo, hi) in self._agent_reaches.items():
            reach_dim = hi - lo + 1
            self._agent_reach_slice[agent_name] = slice(lo, hi + 1)
            self._agent_goal_dim[agent_name] = (
                self._goal_lookahead_plus_1 * (reach_dim + 1)  # +1 for sustain col
            )

        for agent in self._assignment.agents:
            reach_lo, reach_hi = self._agent_reaches[agent.name]
            reach_dim = reach_hi - reach_lo + 1
            goal_dim = self._agent_goal_dim[agent.name]

            sub_spaces: Dict[str, gym.spaces.Space] = {}

            # own_hands: each owned hand's joints_pos (Box).
            own_hands_dict: Dict[str, gym.spaces.Space] = {}
            for hand_name in agent.hand_names:
                own_hands_dict[hand_name] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(joints_dim,), dtype=np.float32
                )
            sub_spaces["own_hands"] = gym.spaces.Dict(own_hands_dict)

            # boundary_hands: neighbor agents' boundary-hand joints_pos.
            boundary_dict: Dict[str, gym.spaces.Space] = {}
            for neighbor_name, neighbor_hand in self._boundary_hands_map[
                agent.name
            ].items():
                boundary_dict[neighbor_hand] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(joints_dim,), dtype=np.float32
                )
            sub_spaces["boundary_hands"] = gym.spaces.Dict(boundary_dict)

            # piano_state slice (reach keys).
            sub_spaces["piano_state"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(reach_dim,), dtype=np.float32
            )
            # piano_sustain: global, 1-dim.
            sub_spaces["piano_sustain"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
            # goal slice (reach keys + sustain col, all lookahead frames).
            sub_spaces["goal"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(goal_dim,), dtype=np.float32
            )

            # If OAR is in the dm_env chain, expose per-agent prev_action +
            # shared prev_reward slices.
            if self._has_oar:
                sub_spaces["prev_action"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self._action_spaces[agent.name].shape,
                    dtype=np.float32,
                )
                sub_spaces["prev_reward"] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                )

            spaces[agent.name] = gym.spaces.Dict(sub_spaces)
            
        if self._include_global_state:
            gs_dim = self.global_state_dim
            spaces = {
                agent_name: gym.spaces.Dict(
                    {
                        # gymnasium's Dict sorts keys, so "global_state" precedes
                        # "own"; obs_layout() computes the real offsets anyway.
                        "global_state": gym.spaces.Box(
                            low=-np.inf, high=np.inf,
                            shape=(gs_dim,), dtype=np.float32,
                        ),
                        "own": own_space,
                    }
                )
                for agent_name, own_space in spaces.items()
            }

        return spaces

    def _split_observation(
        self, dm_obs: Mapping[str, np.ndarray]
    ) -> Dict[str, Any]:
        """dm_env Dict obs → per-agent obs (Dict or flat)."""
        # Reshape goal: (lookahead+1, 89). Last column = sustain.
        goal_flat = np.asarray(dm_obs["goal"], dtype=np.float32)
        goal = goal_flat.reshape(self._goal_lookahead_plus_1, 89)
        piano_state = np.asarray(dm_obs["piano/state"], dtype=np.float32)
        piano_sustain = np.asarray(
            dm_obs["piano/sustain_state"], dtype=np.float32
        ).reshape(1)

        per_agent: Dict[str, Any] = {}
        for agent in self._assignment.agents:
            reach_sl = self._agent_reach_slice[agent.name]

            own_hands = {
                hand_name: np.asarray(
                    dm_obs[f"{hand_name}_shadow_hand/joints_pos"], dtype=np.float32
                )
                for hand_name in agent.hand_names
            }
            boundary_hands = {
                neighbor_hand: np.asarray(
                    dm_obs[f"{neighbor_hand}_shadow_hand/joints_pos"],
                    dtype=np.float32,
                )
                for neighbor_hand in self._boundary_hands_map[agent.name].values()
            }
            # Slice goal: keep all frames, slice key columns + last sustain col.
            agent_goal = np.concatenate(
                [goal[:, reach_sl], goal[:, -1:]], axis=1
            ).reshape(-1)
            agent_obs: Dict[str, Any] = {
                "own_hands": own_hands,
                "boundary_hands": boundary_hands,
                "piano_state": piano_state[reach_sl].copy(),
                "piano_sustain": piano_sustain.copy(),
                "goal": agent_goal.astype(np.float32),
            }

            if self._has_oar:
                # OAR's prev_action is the full SA flat action (shape
                # (flat_action_dim,)). Slice to this agent's per-hand
                # actuators + sustain (if owner).
                prev_action_flat = np.asarray(dm_obs["action"], dtype=np.float32)
                slices = []
                for hand_name in agent.hand_names:
                    sl = self._hand_action_slices[hand_name]
                    slices.append(prev_action_flat[sl])
                if agent.name == self._sustain_owner_name:
                    slices.append(prev_action_flat[self._sustain_idx : self._sustain_idx + 1])
                agent_obs["prev_action"] = np.concatenate(slices).astype(np.float32)
                # prev_reward is a global scalar.
                prev_reward = np.asarray(dm_obs["reward"], dtype=np.float32).reshape(1)
                agent_obs["prev_reward"] = prev_reward

            if self._flatten_obs:
                agent_obs = gym.spaces.utils.flatten(
                    self._dict_observation_spaces[agent.name], agent_obs
                )
                
            # CTDE: wrap BEFORE flattening, exactly once.
            #
            # gymnasium sorts Dict keys, so "global_state" precedes "own" and the
            # flat layout is [global_state | own]. Because the `own` sub-dict is
            # byte-identical to the non-CTDE observation, its flattened block is
            # bit-identical to the IPPO observation -- which is what makes an
            # IPPO/MAPPO comparison a single-factor ablation. Locked by
            # global_state_test.test_own_block_matches_ippo_observation_bitwise.
            if self._include_global_state:
                agent_obs = {
                    "global_state": self._build_global_state(dm_obs, agent.name),
                    "own": agent_obs,
                }

            if self._flatten_obs:
                agent_obs = gym.spaces.utils.flatten(
                    self._dict_observation_spaces[agent.name], agent_obs
                )

            per_agent[agent.name] = agent_obs

        return per_agent
    
    # ---------------- Internal: CTDE global state ----------------
    def _global_state_components(self) -> Tuple[Tuple[str, int], ...]:
        """(name, dim) of every global-state block, in concatenation order.

        Content, following MAPPO's *Agent-Specific* (AS) global-state variant
        (Yu et al., 2022, NeurIPS D&B, section 5.2):

          * every hand's joint positions, in fixed spatial L->R order
            (permutation-consistent; no agent-id embedding needed)
          * the FULL 88-key piano state (each agent's own obs only sees its
            own reach slice)
          * the global sustain state
          * the FULL goal tensor over all lookahead frames
          * [if the OAR wrapper is present] the full joint previous action and
            the previous shared reward
          * a one-hot agent id, which is what makes this AS rather than EP

        Note that under a *shared* team reward every agent has the same return,
        so an EP-style critic (env features only) is already unbiased.  The
        one-hot only lets a per-agent critic specialize; ablate it by dropping
        the last ``num_agents`` entries.
        """
        components = [
            (f"joints/{hand}", int(self._joints_dim))
            for hand in self._all_hands_spatial
        ]
        components.append(("piano_state", 88))
        components.append(("piano_sustain", 1))
        components.append(("goal", int(self._goal_lookahead_plus_1 * 89)))
        if self._has_oar:
            components.append(("prev_joint_action", int(self._flat_action_dim)))
            components.append(("prev_reward", 1))
        components.append(("agent_onehot", len(self._assignment.agents)))
        return tuple(components)

    @property
    def global_state_dim(self) -> int:
        return sum(dim for _, dim in self._global_state_components())

    def _build_global_state(
        self, dm_obs: Mapping[str, np.ndarray], agent_name: str
    ) -> np.ndarray:
        parts = [
            np.asarray(dm_obs[f"{hand}_shadow_hand/joints_pos"], dtype=np.float32)
            for hand in self._all_hands_spatial
        ]
        parts.append(np.asarray(dm_obs["piano/state"], dtype=np.float32))
        parts.append(
            np.asarray(dm_obs["piano/sustain_state"], dtype=np.float32).reshape(1)
        )
        parts.append(np.asarray(dm_obs["goal"], dtype=np.float32).ravel())
        if self._has_oar:
            parts.append(np.asarray(dm_obs["action"], dtype=np.float32).ravel())
            parts.append(np.asarray(dm_obs["reward"], dtype=np.float32).reshape(1))
        onehot = np.zeros(len(self._assignment.agents), dtype=np.float32)
        onehot[self._agent_index[agent_name]] = 1.0
        parts.append(onehot)

        state = np.concatenate(parts).astype(np.float32)
        expected = self.global_state_dim
        if state.shape != (expected,):
            raise RuntimeError(
                f"global state for agent {agent_name!r} has shape "
                f"{state.shape}, expected ({expected},)"
            )
        return state

    def obs_layout(self, agent: str) -> Dict[str, Tuple[int, int]]:
        """Flatten offsets ``{component: (start, stop)}`` for one agent.

        Computed by walking the *actual* iteration order of the Dict space so
        the layout never depends on an assumption about gymnasium's key sorting.
        The CTDE RLModule uses this to slice actor and critic inputs.
        """
        space = self._dict_observation_spaces[agent]
        layout: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for key, sub_space in space.spaces.items():
            width = int(gym.spaces.utils.flatdim(sub_space))
            layout[key] = (offset, offset + width)
            offset += width
        return layout


# ===========================================================================
# Utility: drill into dm_env chain to find the underlying composer.Task.
# ===========================================================================


def _find_task(env: dm_env.Environment):
    """Walk down dm_env wrappers to find the inner composer.Task."""
    cur = env
    # dm_env_wrappers expose `_environment`; composer.Environment exposes `task`.
    while True:
        if hasattr(cur, "task"):
            return cur.task
        if hasattr(cur, "_environment"):
            cur = cur._environment
        else:
            raise RuntimeError(
                f"could not find composer.Task on env chain {type(env).__name__}"
            )


def _find_wrapper(env: dm_env.Environment, wrapper_cls):
    """Walk down dm_env wrappers to find an instance of `wrapper_cls`."""
    cur = env
    while True:
        if isinstance(cur, wrapper_cls):
            return cur
        if hasattr(cur, "_environment"):
            cur = cur._environment
        else:
            raise RuntimeError(
                f"could not find {wrapper_cls.__name__} in dm_env chain "
                f"starting from {type(env).__name__}"
            )


def _import_midi_evaluation_wrapper_class():
    """Lazy import so that module-load order doesn't trip over robopianist."""
    from omnipiano.envs.robopianist.wrappers import MidiEvaluationWrapper
    return MidiEvaluationWrapper
