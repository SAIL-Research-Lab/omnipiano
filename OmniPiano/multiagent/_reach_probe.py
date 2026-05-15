"""Per-agent reach probe (precise simulator-based, cached).

Drive each hand's ``forearm_tx`` to its clamp extremes (raw ctrl ±5), let
physics settle, read fingertip world-y positions, convert to key indices.
Per-agent reach = union of keys reachable by any fingertip of any of the
agent's hands across both forearm extremes.

Method mirrors the ad-hoc probe used during shared-zone measurement
(see ``multi_agent_design.md`` § 5.2 / ``static_partition_design.md`` § 3.1):
the same procedure validated the 4-hand / 3-hand / 5-hand empirical numbers
in those design docs. This file codifies the throw-away script into a
reusable cached tool.
"""

from __future__ import annotations

import functools
from typing import Dict, Sequence, Tuple

import numpy as np
from dm_control import composer

from OmniPiano.envs.robopianist.suite.tasks import piano_with_shadow_hands
from OmniPiano.envs.robopianist.music import midi_file
from OmniPiano.tasks.hand_spec import HandSpec, key_index_to_y
from robopianist.models.piano import piano_constants as piano_consts

# Imported here only — avoid pulling heavy proto at module load if probe never runs.
try:
    from note_seq import music_pb2
except ImportError:  # pragma: no cover — note_seq is a hard dep of robopianist
    music_pb2 = None  # type: ignore


# Force ctrl values used to push forearm_tx to its clamp extremes. dm_control
# clips ctrl to the actuator's ctrlrange — passing ±5 deliberately overshoots
# the typical [−0.5, 0.5] forearm range so the joint pegs against its limit.
_FOREARM_PUSH_CTRL = 5.0

# Number of physics substeps to let the forearm settle at the clamp limit
# (matches the 500-step measurement used in design-doc § 5.2).
_DEFAULT_SUBSTEPS = 500


# ===========================================================================
# Key-index ↔ Y lookup
# ===========================================================================


@functools.lru_cache(maxsize=1)
def _key_y_table() -> np.ndarray:
    """Pre-computed Y coordinates of every key center. Shape (88,)."""
    return np.array(
        [key_index_to_y(k) for k in range(piano_consts.NUM_KEYS)], dtype=np.float64
    )


def y_to_key_index(y: float) -> int:
    """Nearest key index to the given arena-frame Y coordinate.

    Saturates to [0, 87] for out-of-range y values (fingertip past piano edge).
    """
    table = _key_y_table()
    return int(np.argmin(np.abs(table - y)))


# ===========================================================================
# Probe
# ===========================================================================


def _build_probe_env(
    hand_specs: Sequence[HandSpec],
    *,
    control_timestep: float = 0.01,
) -> composer.Environment:
    """Construct a minimal dm_env for reach probing — no MIDI scoring needed."""
    if music_pb2 is None:
        raise ImportError("note_seq.music_pb2 is required for probe.")
    seq = music_pb2.NoteSequence()
    # Long-enough MIDI so the probe (~10 env.step calls) doesn't auto-terminate
    # the episode mid-probe — terminate-on-MIDI-end would reset physics and
    # invalidate the cached hand reference.
    _probe_duration = max(60 * control_timestep, 1.0)  # ≥1s safety
    seq.notes.add(
        start_time=0.0,
        end_time=_probe_duration,
        velocity=60,
        pitch=midi_file.note_name_to_midi_number("C4"),
        part=1,
    )
    seq.total_time = _probe_duration
    seq.tempos.add(qpm=60)
    midi = midi_file.MidiFile(seq=seq)
    task = piano_with_shadow_hands.PianoWithShadowHands(
        midi=midi,
        n_steps_lookahead=0,
        control_timestep=control_timestep,
        change_color_on_activation=False,
        disable_fingering_reward=True,
        hand_specs=list(hand_specs),
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


def _settle_with_forearm_ctrl(
    env: composer.Environment,
    target_hand_name: str,
    forearm_ctrl: float,
    substeps: int,
) -> np.ndarray:
    """Drive `target_hand_name`'s `forearm_tx` to `forearm_ctrl`, settle, read fingertips.

    Returns: fingertip world-y, shape (5,).
    """
    task = env.task
    target_hand = task.hands_by_name[target_hand_name]
    physics = env.physics

    # Build a single flat action vector: native scale, all neutral except
    # target's forearm_tx pushed to extreme.
    action_dim = task.action_spec(physics).shape[0]
    action = np.zeros(action_dim, dtype=np.float64)

    # Find the offset of target_hand's actions within the flat vector
    # (mirroring before_step's offset loop).
    offset = 0
    target_offset = None
    target_size = None
    for hand in task.hands:
        size = hand.action_spec(physics).shape[0]
        if hand is target_hand:
            target_offset = offset
            target_size = size
            break
        offset += size
    if target_offset is None:
        raise KeyError(f"hand {target_hand_name} not in task.hands")

    # forearm_tx is the FIRST DoF in a Shadow Hand's action vector (see
    # shadow_hand._DEFAULT_FOREARM_DOFS) — index 0 within the hand's slice.
    action[target_offset] = forearm_ctrl

    # Step until forearm settles. Each env.step does `control_timestep /
    # physics_timestep` substeps internally — typical ratio 50:1. To get
    # ~500 substeps we call env.step ~10 times.
    physics_per_control = max(1, substeps // 50)
    for _ in range(physics_per_control):
        ts = env.step(action)
        if ts.last():
            # episode ended unexpectedly mid-probe (shouldn't with long-enough
            # probe MIDI, but guard anyway) — break before refs go stale.
            break

    fingertip_xpos = physics.bind(target_hand.fingertip_sites).xpos.copy()
    return fingertip_xpos[:, 1]  # Y-axis only


def _probe_hand_reach_keys(
    env: composer.Environment,
    hand_name: str,
    *,
    substeps: int = _DEFAULT_SUBSTEPS,
) -> set:
    """Return the set of key indices any of `hand_name`'s 5 fingertips can reach.

    Probes both forearm_tx extremes; takes the union of reached keys.
    """
    reached: set = set()
    for ctrl in (-_FOREARM_PUSH_CTRL, +_FOREARM_PUSH_CTRL):
        env.reset()
        ys = _settle_with_forearm_ctrl(env, hand_name, ctrl, substeps)
        for y in ys:
            reached.add(y_to_key_index(float(y)))
    return reached


def probe_agent_reach(
    assignment,
    hand_specs: Sequence[HandSpec],
    *,
    substeps: int = _DEFAULT_SUBSTEPS,
) -> Dict[str, Tuple[int, int]]:
    """Return {agent_name: (reach_lo, reach_hi)} via simulator probe.

    For each agent, probes each of its hands (both forearm_tx extremes,
    finger pose neutral) and takes the union of fingertip-reachable keys.
    The reach is reported as the inclusive (min, max) key index of that union.
    """
    env = _build_probe_env(hand_specs)
    try:
        reaches: Dict[str, Tuple[int, int]] = {}
        for agent in assignment.agents:
            agent_reached: set = set()
            for hand_name in agent.hand_names:
                agent_reached |= _probe_hand_reach_keys(
                    env, hand_name, substeps=substeps
                )
            if not agent_reached:
                raise RuntimeError(
                    f"agent {agent.name!r} reached zero keys — probe failed"
                )
            reaches[agent.name] = (min(agent_reached), max(agent_reached))
        return reaches
    finally:
        env.close()


# ===========================================================================
# Caching
# ===========================================================================
# We cache by morphology name + a hash of the hand_specs. Hand specs are
# frozen dataclasses, so they're hashable through their tuple of fields.

_REACH_CACHE: Dict[Tuple[str, Tuple], Dict[str, Tuple[int, int]]] = {}


def cached_probe_agent_reach(
    assignment,
    hand_specs: Sequence[HandSpec],
    *,
    substeps: int = _DEFAULT_SUBSTEPS,
) -> Dict[str, Tuple[int, int]]:
    """Cached front-end to :func:`probe_agent_reach`.

    Cache key = (morphology, tuple(hand_specs)). HandSpec is a frozen
    dataclass — directly hashable.
    """
    key = (assignment.morphology, tuple(hand_specs))
    if key not in _REACH_CACHE:
        _REACH_CACHE[key] = probe_agent_reach(assignment, hand_specs, substeps=substeps)
    return _REACH_CACHE[key]
