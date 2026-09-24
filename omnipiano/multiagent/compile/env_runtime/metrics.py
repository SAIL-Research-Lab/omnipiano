"""Agent-aware coordination metrics for multi-hand piano environments.

The metric tracker in this module is deliberately observational: it reads the
ground-truth task trajectory, piano activation state, and MuJoCo contacts, but
never modifies observations, actions, or episode termination.  A separate
pure helper converts the exact tracked collision indicator into optional MARL
reward shaping; the environment wrapper decides whether to apply it.

The headline metrics are:

* common-area target-note success rate;
* common-area target notes pressed by more than one agent;
* control-step rate with a physical collision between different agents.

"Common area" means a piano key contained in the precise fingertip reach of at
least two agents.  Target notes are counted once at onset, rather than once per
active control frame, so sustained notes do not dominate the rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


NUM_PIANO_KEYS = 88
DEFAULT_ONSET_WINDOW_STEPS = 1
COLLISION_MARGIN = 1e-8

COMMON_AREA_SUCCESS_RATE = (
    "episode_coordination/common_area_success_rate"
)
COMMON_AREA_DUPLICATE_PRESS_RATE = (
    "episode_coordination/common_area_duplicate_press_rate"
)
INTER_AGENT_COLLISION_STEP_RATE = (
    "episode_coordination/inter_agent_collision_step_rate"
)

COORDINATION_RATE_METRICS = (
    COMMON_AREA_SUCCESS_RATE,
    COMMON_AREA_DUPLICATE_PRESS_RATE,
    INTER_AGENT_COLLISION_STEP_RATE,
)

COMMON_AREA_TARGET_COUNT = "episode_coordination/common_area_target_count"
COMMON_AREA_SUCCESS_COUNT = "episode_coordination/common_area_success_count"
COMMON_AREA_DUPLICATE_PRESS_COUNT = (
    "episode_coordination/common_area_duplicate_press_count"
)
INTER_AGENT_COLLISION_STEP_COUNT = (
    "episode_coordination/inter_agent_collision_step_count"
)
OBSERVED_STEP_COUNT = "episode_coordination/observed_step_count"

# Episode reward-accounting fields emitted by the MARL wrapper when the
# optional cross-agent collision penalty is active.  They are kept separate
# from the observational coordination rates so the shaped team return can be
# reconstructed exactly from an evaluation artifact.
BASE_TEAM_RETURN = "episode_reward/base_team_return"
INTER_AGENT_COLLISION_PENALTY_RETURN = (
    "episode_reward/inter_agent_collision_penalty_return"
)
SHAPED_TEAM_RETURN = "episode_reward/shaped_team_return"
INTER_AGENT_COLLISION_PENALTY_COEF = (
    "episode_reward/inter_agent_collision_penalty_coef"
)

# Observation-robustness telemetry.  The dm_env wrapper reports one L2 norm
# over the global raw observation before it is split across agents, so these
# values are team/environment metrics rather than per-agent sums.
OBS_NOISE_L2_SUM = "episode_robust/obs_noise_l2_sum"
OBS_NOISE_L2_MEAN = "episode_robust/obs_noise_l2_mean"


def apply_inter_agent_collision_penalty(
    base_reward: float,
    *,
    inter_agent_collision: bool,
    coefficient: float,
) -> Tuple[float, float]:
    """Apply ``-coefficient * 1[cross-agent contact]`` exactly once.

    Returns ``(shaped_reward, penalty_term)`` where ``penalty_term`` is zero
    or negative.  Keeping this operation pure makes the reward/metric
    alignment independently testable without MuJoCo.
    """
    base = float(base_reward)
    coef = float(coefficient)
    if not np.isfinite(base):
        raise ValueError(f"base_reward must be finite, got {base_reward!r}")
    if not np.isfinite(coef) or coef < 0.0:
        raise ValueError(
            "inter-agent collision penalty coefficient must be finite and "
            f"non-negative, got {coefficient!r}"
        )
    penalty = -coef if bool(inter_agent_collision) else 0.0
    return base + penalty, penalty


def compute_common_keys(
    agent_reaches: Mapping[str, Tuple[int, int]],
    *,
    num_keys: int = NUM_PIANO_KEYS,
) -> Tuple[int, ...]:
    """Return keys in the fingertip reach of at least two distinct agents."""
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")
    if len(agent_reaches) < 2:
        return ()

    reach_sets: Dict[int, Set[str]] = {key: set() for key in range(num_keys)}
    for agent, reach in agent_reaches.items():
        if len(reach) != 2:
            raise ValueError(f"invalid reach for agent {agent!r}: {reach!r}")
        lo, hi = int(reach[0]), int(reach[1])
        if not 0 <= lo <= hi < num_keys:
            raise ValueError(
                f"reach for agent {agent!r} must lie in [0, {num_keys - 1}], "
                f"got ({lo}, {hi})"
            )
        for key in range(lo, hi + 1):
            reach_sets[key].add(agent)
    return tuple(key for key, agents in reach_sets.items() if len(agents) >= 2)


def _target_onsets(
    note_trajectory: Sequence[Sequence[Any]],
    common_keys: Set[int],
) -> Dict[int, Tuple[int, ...]]:
    """Extract 0->1 target transitions on common keys from a note trajectory."""
    onsets: Dict[int, Tuple[int, ...]] = {}
    previous: Set[int] = set()
    for step_index, notes in enumerate(note_trajectory):
        current = {int(note.key) for note in notes}
        common_onsets = tuple(sorted((current - previous) & common_keys))
        if common_onsets:
            onsets[step_index] = common_onsets
        previous = current
    return onsets


@dataclass
class _PendingOnset:
    key: int
    final_step: int
    success: bool = False
    duplicate_press: bool = False


class CoordinationEpisodeAccumulator:
    """Pure episode accumulator, separated from MuJoCo contact extraction."""

    def __init__(
        self,
        common_keys: Iterable[int],
        *,
        onset_window_steps: int = DEFAULT_ONSET_WINDOW_STEPS,
    ) -> None:
        if onset_window_steps <= 0:
            raise ValueError("onset_window_steps must be positive")
        self.common_keys = frozenset(int(key) for key in common_keys)
        if any(key < 0 or key >= NUM_PIANO_KEYS for key in self.common_keys):
            raise ValueError("common keys must lie in [0, 87]")
        self.onset_window_steps = int(onset_window_steps)
        self._initialized = False
        self._finalized = False

    def reset(
        self,
        note_trajectory: Sequence[Sequence[Any]],
        *,
        collision_metric_available: bool = True,
    ) -> None:
        self._onsets_by_step = _target_onsets(
            note_trajectory, set(self.common_keys)
        )
        self._target_count = sum(len(keys) for keys in self._onsets_by_step.values())
        self._step_index = 0
        self._observed_step_count = 0
        self._success_count = 0
        self._duplicate_press_count = 0
        self._collision_step_count = 0
        self._collision_metric_available = bool(collision_metric_available)
        self._pending: list[_PendingOnset] = []
        self._initialized = True
        self._finalized = False

    def observe_step(
        self,
        piano_activation: Sequence[bool],
        key_contact_agents: Mapping[int, Set[str]],
        *,
        inter_agent_collision: bool,
    ) -> None:
        if not self._initialized:
            raise RuntimeError("coordination accumulator must be reset before use")
        if self._finalized:
            raise RuntimeError("cannot observe a finalized coordination episode")

        activation = np.asarray(piano_activation, dtype=bool).reshape(-1)
        if activation.size != NUM_PIANO_KEYS:
            raise ValueError(
                f"piano activation must have {NUM_PIANO_KEYS} values, "
                f"got {activation.size}"
            )

        for key in self._onsets_by_step.get(self._step_index, ()):
            self._pending.append(
                _PendingOnset(
                    key=key,
                    final_step=self._step_index + self.onset_window_steps - 1,
                )
            )

        for onset in self._pending:
            if activation[onset.key]:
                onset.success = True
                if len(key_contact_agents.get(onset.key, set())) >= 2:
                    onset.duplicate_press = True

        remaining: list[_PendingOnset] = []
        for onset in self._pending:
            if self._step_index >= onset.final_step:
                self._success_count += int(onset.success)
                self._duplicate_press_count += int(onset.duplicate_press)
            else:
                remaining.append(onset)
        self._pending = remaining

        if self._collision_metric_available and inter_agent_collision:
            self._collision_step_count += 1
        self._observed_step_count += 1
        self._step_index += 1

    def finalize(self) -> Dict[str, Optional[float]]:
        if not self._initialized:
            raise RuntimeError("coordination accumulator must be reset before use")
        if not self._finalized:
            # An episode may terminate before the final onset window expires.  Count
            # evidence already observed, while unvisited future target onsets remain
            # failures through the fixed full-trajectory denominator.
            for onset in self._pending:
                self._success_count += int(onset.success)
                self._duplicate_press_count += int(onset.duplicate_press)
            self._pending = []
            self._finalized = True

        target_denominator = float(self._target_count)
        step_denominator = float(self._observed_step_count)
        return {
            COMMON_AREA_SUCCESS_RATE: (
                self._success_count / target_denominator
                if self._target_count > 0
                else None
            ),
            COMMON_AREA_DUPLICATE_PRESS_RATE: (
                self._duplicate_press_count / target_denominator
                if self._target_count > 0
                else None
            ),
            INTER_AGENT_COLLISION_STEP_RATE: (
                self._collision_step_count / step_denominator
                if self._collision_metric_available and self._observed_step_count > 0
                else None
            ),
            COMMON_AREA_TARGET_COUNT: float(self._target_count),
            COMMON_AREA_SUCCESS_COUNT: float(self._success_count),
            COMMON_AREA_DUPLICATE_PRESS_COUNT: float(
                self._duplicate_press_count
            ),
            INTER_AGENT_COLLISION_STEP_COUNT: (
                float(self._collision_step_count)
                if self._collision_metric_available
                else None
            ),
            OBSERVED_STEP_COUNT: float(self._observed_step_count),
        }


def classify_contacts(
    contacts: Iterable[Any],
    *,
    hand_geom_to_agent: Mapping[int, str],
    fingertip_geom_to_agent: Mapping[int, str],
    key_geom_to_key: Mapping[int, int],
    common_keys: Set[int],
    collision_margin: float = COLLISION_MARGIN,
) -> Tuple[Dict[int, Set[str]], bool]:
    """Classify one control frame's active MuJoCo contacts.

    Returns ``({key_index: contacting_agent_names}, inter_agent_collision)``.
    Multiple fingertip geoms from the same agent collapse to one agent name.
    """
    key_contact_agents: Dict[int, Set[str]] = {}
    inter_agent_collision = False

    for contact in contacts:
        if float(contact.dist) > collision_margin:
            continue
        geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])

        agent1 = hand_geom_to_agent.get(geom1)
        agent2 = hand_geom_to_agent.get(geom2)
        if agent1 is not None and agent2 is not None and agent1 != agent2:
            inter_agent_collision = True

        fingertip_agent = fingertip_geom_to_agent.get(geom1)
        key = key_geom_to_key.get(geom2)
        if fingertip_agent is None or key is None:
            fingertip_agent = fingertip_geom_to_agent.get(geom2)
            key = key_geom_to_key.get(geom1)
        if fingertip_agent is not None and key in common_keys:
            key_contact_agents.setdefault(int(key), set()).add(fingertip_agent)

    return key_contact_agents, inter_agent_collision


class CoordinationMetricsTracker:
    """Bridge between an OmniPiano task/physics pair and the pure accumulator."""

    def __init__(
        self,
        assignment: Any,
        agent_reaches: Mapping[str, Tuple[int, int]],
        *,
        onset_window_steps: int = DEFAULT_ONSET_WINDOW_STEPS,
    ) -> None:
        self._assignment = assignment
        self.common_keys = compute_common_keys(agent_reaches)
        self._common_key_set = set(self.common_keys)
        self._accumulator = CoordinationEpisodeAccumulator(
            self.common_keys, onset_window_steps=onset_window_steps
        )
        self._initialized = False

    @staticmethod
    def _geom_id(physics: Any, geom: Any) -> int:
        element_ids = np.asarray(physics.bind(geom).element_id).reshape(-1)
        if element_ids.size != 1:
            raise RuntimeError(
                f"expected one compiled geom id for {geom!r}, got {element_ids}"
            )
        return int(element_ids[0])

    @staticmethod
    def _is_collision_geom(physics: Any, geom_id: int) -> bool:
        return not (
            int(physics.model.geom_contype[geom_id]) == 0
            and int(physics.model.geom_conaffinity[geom_id]) == 0
        )

    @staticmethod
    def _insert_unique(mapping: Dict[int, Any], key: int, value: Any) -> None:
        previous = mapping.get(key)
        if previous is not None and previous != value:
            raise RuntimeError(
                f"compiled geom id {key} maps to both {previous!r} and {value!r}"
            )
        mapping[key] = value

    def _build_geom_caches(self, physics: Any, task: Any) -> None:
        hand_geom_to_agent: Dict[int, str] = {}
        fingertip_geom_to_agent: Dict[int, str] = {}

        for hand_name, hand in task.hands_by_name.items():
            agent_name = self._assignment.agent_of(hand_name)
            for geom in hand.mjcf_model.find_all("geom"):
                geom_id = self._geom_id(physics, geom)
                if self._is_collision_geom(physics, geom_id):
                    self._insert_unique(
                        hand_geom_to_agent, geom_id, agent_name
                    )
            for body in hand.fingertip_bodies:
                for geom in body.find_all("geom"):
                    geom_id = self._geom_id(physics, geom)
                    if self._is_collision_geom(physics, geom_id):
                        self._insert_unique(
                            fingertip_geom_to_agent, geom_id, agent_name
                        )

        key_geom_to_key: Dict[int, int] = {}
        for key_index, key_body in enumerate(task.piano.keys):
            for geom in key_body.geom:
                geom_id = self._geom_id(physics, geom)
                if self._is_collision_geom(physics, geom_id):
                    self._insert_unique(key_geom_to_key, geom_id, key_index)

        if not hand_geom_to_agent:
            raise RuntimeError("found no active hand collision geoms")
        if not fingertip_geom_to_agent:
            raise RuntimeError("found no active fingertip collision geoms")
        mapped_keys = set(key_geom_to_key.values())
        if mapped_keys != set(range(NUM_PIANO_KEYS)):
            raise RuntimeError(
                "expected one or more active geoms for each of 88 piano keys, "
                f"mapped {len(mapped_keys)} keys"
            )

        self._hand_geom_to_agent = hand_geom_to_agent
        self._fingertip_geom_to_agent = fingertip_geom_to_agent
        self._key_geom_to_key = key_geom_to_key

    def reset(self, physics: Any, task: Any) -> None:
        # A seeded ParallelEnv.reset() rebuilds the dm_env and MuJoCo model, so
        # compiled geom ids must be rebuilt on every episode reset.
        self._build_geom_caches(physics, task)
        self._accumulator.reset(
            task._notes,
            collision_metric_available=not bool(
                getattr(task, "_disable_hand_collisions", False)
            ),
        )
        self._initialized = True

    def observe_step(self, physics: Any, task: Any) -> bool:
        """Observe one frame and return the exact collision indicator logged."""
        if not self._initialized:
            raise RuntimeError("coordination tracker must be reset before use")
        key_contact_agents, inter_agent_collision = classify_contacts(
            physics.data.contact,
            hand_geom_to_agent=self._hand_geom_to_agent,
            fingertip_geom_to_agent=self._fingertip_geom_to_agent,
            key_geom_to_key=self._key_geom_to_key,
            common_keys=self._common_key_set,
        )
        self._accumulator.observe_step(
            np.asarray(task.piano.activation, dtype=bool).copy(),
            key_contact_agents,
            inter_agent_collision=inter_agent_collision,
        )
        return inter_agent_collision

    def finalize(self) -> Dict[str, Optional[float]]:
        if not self._initialized:
            raise RuntimeError("coordination tracker must be reset before use")
        return self._accumulator.finalize()
