"""Agent-decomposition definitions for the multi-agent N-hand morphologies.

Each `MorphologyAssignment` describes how an N-hand morphology's hands group
into agents (one human = one agent), plus enough metadata for the PettingZoo
wrapper to derive per-agent observation/action shapes and boundary-hand
relationships purely from this table.

Conventions (see `static_partition_design.md` § 4.5 N-hand action-layout axiom):
- `hand_names` within an agent are spatial left→right (= the SA `task.hands`
  subsequence for that agent, since SA itself is spatial L→R for N≥3)
- `agents` order is spatial left→right across the keyboard
- exactly one agent in each morphology has `is_sustain_owner=True` — by
  convention the bass-side agent (`secondo` / `left_secondo`); see
  `multi_agent_design.md` § 4.3 for the Henle-grounded rationale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class AgentDef:
    """One agent within a multi-agent morphology."""

    name: str
    hand_names: Tuple[str, ...]  # spatial L→R order within this agent
    is_sustain_owner: bool = False


@dataclass(frozen=True)
class MorphologyAssignment:
    """The full agent decomposition for one N-hand morphology."""

    morphology: str  # "ThreeHand" / "FourHand" / "FiveHand"
    agent_setup: str  # "MainSolo" / "Duet" / "Trio" (token used in env_id)
    agents: Tuple[AgentDef, ...]  # spatial L→R order across keyboard

    def __post_init__(self) -> None:
        # Exactly one sustain owner per morphology.
        owners = [a.name for a in self.agents if a.is_sustain_owner]
        if len(owners) != 1:
            raise ValueError(
                f"MorphologyAssignment {self.morphology!r}: must have exactly "
                f"one sustain_owner agent, got {owners}"
            )
        # Unique agent names.
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(
                f"MorphologyAssignment {self.morphology!r}: agent names "
                f"must be unique, got {names}"
            )

    @property
    def agent_names(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.agents)

    @property
    def sustain_owner(self) -> str:
        return next(a.name for a in self.agents if a.is_sustain_owner)

    def agent_of(self, hand_name: str) -> str:
        for a in self.agents:
            if hand_name in a.hand_names:
                return a.name
        raise KeyError(
            f"hand {hand_name!r} not owned by any agent in "
            f"{self.morphology!r} (agents: {self.agents})"
        )


# ===========================================================================
# Phase 1 morphology assignments — Territorial series only.
#
# 4-hand and 5-hand: spatial = action layout (SA spec order matches).
# 3-hand: post-2026-05-12 migration to spatial (lh, rh_c, rh); spatial L→R
#         agent grouping: secondo=(lh, rh_c), treble_soloist=(rh,).
#
# Single-hand agents follow the `<spatial>_soloist` rule (3-hand → treble,
# 5-hand → center); see `multi_agent_design.md` § 3.5.
# ===========================================================================

AGENT_ASSIGNMENTS: Dict[str, MorphologyAssignment] = {
    "ThreeHand": MorphologyAssignment(
        morphology="ThreeHand",
        agent_setup="MainSolo",
        agents=(
            AgentDef(
                name="secondo",
                hand_names=("lh", "rh_c"),
                is_sustain_owner=True,
            ),
            AgentDef(
                name="treble_soloist",
                hand_names=("rh",),
            ),
        ),
    ),
    "FourHand": MorphologyAssignment(
        morphology="FourHand",
        agent_setup="Duet",
        agents=(
            AgentDef(
                name="secondo",
                hand_names=("lh_b", "rh_b"),
                is_sustain_owner=True,
            ),
            AgentDef(
                name="primo",
                hand_names=("lh_t", "rh_t"),
            ),
        ),
    ),
    "FiveHand": MorphologyAssignment(
        morphology="FiveHand",
        agent_setup="Trio",
        agents=(
            AgentDef(
                name="left_secondo",
                hand_names=("lh_b", "rh_b"),
                is_sustain_owner=True,
            ),
            AgentDef(
                name="center_soloist",
                hand_names=("rh_c",),
            ),
            AgentDef(
                name="right_primo",
                hand_names=("lh_t", "rh_t"),
            ),
        ),
    ),
}


# ===========================================================================
# Helpers — territory & boundary derivation
# ===========================================================================


def compute_agent_territory(
    assignment: MorphologyAssignment,
    hand_key_ranges: Mapping[str, Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """Per-agent territory = union of the agent's hands' SA bucket key_ranges.

    Since agent.hand_names are spatial L→R contiguous and SA buckets are
    contiguous by construction, the union is always a contiguous range
    (lo of leftmost hand, hi of rightmost hand).
    """
    territories: Dict[str, Tuple[int, int]] = {}
    for agent in assignment.agents:
        ranges = [hand_key_ranges[h] for h in agent.hand_names]
        lo = min(r[0] for r in ranges)
        hi = max(r[1] for r in ranges)
        territories[agent.name] = (lo, hi)
    return territories


def compute_inter_agent_boundaries(
    assignment: MorphologyAssignment,
    territories: Mapping[str, Tuple[int, int]],
) -> Tuple[Tuple[str, str, int, int], ...]:
    """Return inter-agent boundaries as (left_agent, right_agent, lo_key, hi_key) tuples.

    For Territorial series, adjacent agent territories are disjoint and abut
    at one key (lo_key = left.hi, hi_key = right.lo, with hi_key == lo_key + 1).
    Returned in spatial L→R order.
    """
    ordered = assignment.agents  # already spatial L→R
    boundaries: list = []
    for i in range(len(ordered) - 1):
        left, right = ordered[i], ordered[i + 1]
        lo = territories[left.name][1]
        hi = territories[right.name][0]
        if hi != lo + 1:
            raise ValueError(
                f"Territorial: adjacent territories must abut at one key. "
                f"{left.name} ends at {lo}, {right.name} starts at {hi}."
            )
        boundaries.append((left.name, right.name, lo, hi))
    return tuple(boundaries)


def compute_boundary_hands(
    assignment: MorphologyAssignment,
    hand_key_ranges: Mapping[str, Tuple[int, int]],
) -> Dict[str, Dict[str, str]]:
    """For each agent, return {neighbor_agent_name: that_agent's_boundary_hand_name}.

    The "boundary hand" of a neighboring agent (with respect to a given
    inter-agent boundary) is that neighbor's hand whose SA bucket is closest
    to the boundary key. For Territorial series, this is the neighbor's
    innermost hand (the one whose bucket abuts the inter-agent boundary).
    """
    territories = compute_agent_territory(assignment, hand_key_ranges)
    boundaries = compute_inter_agent_boundaries(assignment, territories)

    # Initialize empty dict for each agent.
    result: Dict[str, Dict[str, str]] = {a.name: {} for a in assignment.agents}

    agents_by_name = {a.name: a for a in assignment.agents}

    for left_name, right_name, lo_key, hi_key in boundaries:
        left_agent = agents_by_name[left_name]
        right_agent = agents_by_name[right_name]

        # Left agent's boundary hand toward right agent = its rightmost hand
        # (largest bucket hi). For Territorial that's its last hand in
        # spatial L→R order within agent.
        left_boundary_hand = max(
            left_agent.hand_names,
            key=lambda h: hand_key_ranges[h][1],
        )
        # Right agent's boundary hand toward left agent = its leftmost hand
        # (smallest bucket lo). For Territorial that's its first hand in
        # spatial L→R order within agent.
        right_boundary_hand = min(
            right_agent.hand_names,
            key=lambda h: hand_key_ranges[h][0],
        )

        # Agent on left side of boundary sees right agent's leftmost hand.
        result[left_name][right_name] = right_boundary_hand
        # Agent on right side of boundary sees left agent's rightmost hand.
        result[right_name][left_name] = left_boundary_hand

    return result


# ===========================================================================
# Per-agent reach probe (precise, cached)
# ===========================================================================
#
# Imported lazily inside the function to avoid circular import with
# OmniPiano.envs.* at module load time (the probe needs to build a dm_env).


def compute_agent_reach(
    assignment: MorphologyAssignment,
    hand_specs: Sequence,  # Sequence[HandSpec]
    *,
    n_settle_steps: int = 10,
    control_timestep: float = 0.05,
    gravity_compensation: bool = True,
    disable_hand_collisions: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Return {agent_name: (reach_lo_key, reach_hi_key)} via simulator probe.

    For each agent, pin the relevant hand's `forearm_tx` joint qpos to its
    clamp extreme (bypassing PD dynamics), let fingers settle around the
    pinned wrist for `n_settle_steps` env.step calls, then read the 5
    fingertip world-y coordinates and convert to key indices via
    `key_index_to_y`'s inverse.

    Args:
        n_settle_steps: number of env.step() calls between qpos pinning and
            fingertip readout. Default 10 — empirically sufficient for finger
            PD to converge to neutral pose. See `_reach_probe.py`.
        control_timestep / gravity_compensation / disable_hand_collisions:
            physics-affecting BenchmarkEnvConfig fields. Caller should pass
            the same values as the runtime env so probe physics align with
            training physics. Defaults match `BenchmarkEnvConfig` runtime
            defaults (0.05 / True / False).

    The agent's reach is the contiguous (min, max) range of keys reachable
    by any fingertip when the agent's outer hand is at its outer clamp
    extreme (see `_reach_probe.probe_agent_reach` for the full algorithm).

    Result is cached per (morphology, hand_specs identity, n_settle_steps,
    control_timestep, gravity_compensation, disable_hand_collisions).
    """
    # Import here to avoid module-load circularity.
    from OmniPiano.multiagent._reach_probe import cached_probe_agent_reach

    return cached_probe_agent_reach(
        assignment,
        hand_specs,
        n_settle_steps=n_settle_steps,
        control_timestep=control_timestep,
        gravity_compensation=gravity_compensation,
        disable_hand_collisions=disable_hand_collisions,
    )
