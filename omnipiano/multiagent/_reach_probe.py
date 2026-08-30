"""Per-agent reach probe (precise simulator-based, cached).

Algorithm:
    1. Build a minimal dm_env using the SAME task kwargs as runtime
       (control_timestep, gravity_compensation, ...) so probe physics
       match what the agent will actually train on.
    2. For each agent, probe only the **physically realizable** extreme
       configurations (skipping infeasible ones under the non-crossing
       constraint — forearms at the same height can't swap order):

         * multi-hand agent: pin leftmost-hand's ``forearm_tx`` qpos to
           the LEFT clamp extreme → fingertip MIN gives agent's leftmost
           reach. Pin rightmost-hand's qpos to the RIGHT clamp extreme
           → fingertip MAX gives agent's rightmost reach.
         * single-hand agent: pin that hand at both extremes; MIN of
           LEFT-extreme fingertips + MAX of RIGHT-extreme fingertips.

       Inner-hand inner-extreme configurations (e.g. lh_b pushed to
       the agent's RIGHT under MA Territorial) are physically infeasible
       and intentionally NOT probed. See ``ma_territorial_impl_plan.md``
       § 10(c) for the LH/RH thumb-asymmetry argument showing these
       inner extremes never extend reach beyond outer extremes anyway.
    3. Use qpos pinning (bypassing actuator PD dynamics) so the joint
       reaches the clamp extreme deterministically — no convergence
       time to tune. Fingers receive ctrl=0 and settle to neutral pose
       via env.step()'s internal physics substeps.
    4. Read fingertip world-y from MJCF ``fingertip_sites``; convert to
       nearest key index via the keyboard's MJCF placement formula.
    5. Report ``(reach_lo, reach_hi)`` per agent — a contiguous closed
       interval (continuous wrist sliding fills the interior).

Mirrors the ad-hoc probe used during shared-zone measurement
(``multi_agent_design.md`` § 5.2, ``static_partition_design.md`` § 3.1).
This file codifies the throw-away script into a reusable cached tool.

Caveat — conservative envelope: fingers are at neutral (ctrl=0) pose.
Agent trained with active finger extension can press ~2-3 keys beyond
this envelope per side per hand (mostly thumb extension). Reach reported
here is therefore a lower bound on what the trained agent can achieve.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from dm_control import composer

from omnipiano.envs.robopianist.suite.tasks import piano_with_shadow_hands
from omnipiano.envs.robopianist.music import midi_file
from omnipiano.tasks.hand_spec import HandSpec, key_index_to_y
from robopianist.models.piano import piano_constants as piano_consts

# Module-level import; light-weight because this module itself is only
# imported lazily (by assignment.compute_agent_reach) when the probe runs.
try:
    from note_seq import music_pb2
except ImportError:  # pragma: no cover — note_seq is a hard dep of robopianist
    music_pb2 = None  # type: ignore


# Direction sentinel for pinning forearm_tx at its clamp extremes. Only the
# SIGN is ever used (the joint is pinned by direct qpos writes, not ctrl);
# the magnitude is irrelevant.
_FOREARM_PUSH_CTRL = 5.0

# Number of env.step() calls between qpos pinning and reading fingertips —
# lets the fingers settle around the pinned forearm. Under the default
# runtime-aligned probe env (control_timestep=0.05, physics_timestep=0.005,
# i.e. 10:1 ratio), 10 env.steps ≈ 100 physics substeps — empirically
# sufficient for finger PD to converge to the ctrl=0 neutral pose. If a
# caller overrides probe `control_timestep` (e.g. to match a non-default
# `BenchmarkEnvConfig`), the per-env.step physics substep count scales
# accordingly; settling time in sim seconds is what matters, not raw step
# count, so this default usually remains adequate.
_DEFAULT_N_SETTLE_STEPS = 10


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
    control_timestep: float = 0.05,
    gravity_compensation: bool = True,
    disable_hand_collisions: bool = False,
) -> composer.Environment:
    """Construct a minimal dm_env for reach probing.

    Probe physics must match the runtime env's physics so reach numbers
    reflect what the agent will actually train on. The three kwargs here
    are the BenchmarkEnvConfig fields that affect physics-step behavior:

      * ``control_timestep``: sets env.step's substep count + finger PD
        convergence window per step. Default 0.05 matches
        ``BenchmarkEnvConfig.control_timestep``.
      * ``gravity_compensation``: if True, gravity on hand bodies is
        offset by per-joint torques. Affects finger settle pose under
        ctrl=0 (without compensation, fingers droop). Default True
        matches ``BenchmarkEnvConfig.gravity_compensation``.
      * ``disable_hand_collisions``: if True, inter-hand body collisions
        are disabled (probe rarely hits this, but match runtime).
        Default False matches runtime.

    Defaults are chosen to align with `BenchmarkEnvConfig` runtime defaults
    so probe-via-default-args is automatically runtime-aligned. Callers
    that override `BenchmarkEnvConfig` should pass the same overrides
    here (see `make_parallel` in `registration.py`).

    Probe does not need MIDI scoring / lookahead / reward / colorization,
    so those task kwargs are hardcoded to the cheapest values.
    """
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
        n_steps_lookahead=0,                            # probe doesn't read goal
        control_timestep=control_timestep,              # runtime-aligned
        gravity_compensation=gravity_compensation,      # runtime-aligned
        disable_hand_collisions=disable_hand_collisions,  # runtime-aligned
        change_color_on_activation=False,               # probe doesn't render
        disable_fingering_reward=True,                  # probe doesn't compute reward
        hand_specs=list(hand_specs),
    )
    return composer.Environment(task, strip_singleton_obs_buffer_dim=True)


def _settle_with_forearm_ctrl(
    env: composer.Environment,
    target_hand_name: str,
    forearm_ctrl: float,
    n_settle_steps: int,
) -> np.ndarray:
    """Pin `target_hand_name`'s `forearm_tx` joint at clamp limit, settle, read fingertips.

    Bypasses the actuator's PD dynamics — directly writes `qpos` to the clamp
    extreme on each step so the joint stays pinned regardless of PD settling
    time. Other joints receive ctrl=0 so fingers settle to their neutral pose
    via env.step()'s internal physics substeps.

    Args:
        n_settle_steps: number of env.step() calls between qpos pinning and
            reading fingertips. Under the default runtime-aligned probe env
            (control_timestep=0.05, physics_timestep=0.005, i.e. 10:1
            ratio), 10 env.steps ≈ 100 physics substeps — empirically
            enough for finger PD to converge to neutral pose. Increase only
            if probe env timesteps are overridden to small values or
            observed fingers don't converge.

    Returns: fingertip world-y, shape (5,).
    """
    task = env.task
    target_hand = task.hands_by_name[target_hand_name]
    physics = env.physics

    # Find the forearm_tx joint and its clamp range.
    forearm_joint = None
    for j in target_hand.mjcf_model.find_all("joint"):
        if "forearm_tx" in j.name:
            forearm_joint = j
            break
    if forearm_joint is None:
        raise RuntimeError(
            f"hand {target_hand_name!r} has no forearm_tx joint"
        )

    joint_range = forearm_joint.range
    if joint_range is None:
        raise RuntimeError(
            f"forearm_tx for {target_hand_name!r} has no joint range; "
            f"cannot determine clamp extreme"
        )
    target_qpos = float(joint_range[1] if forearm_ctrl > 0 else joint_range[0])

    # Neutral action for all actuators (lets fingers settle to ctrl=0 pose).
    action_dim = task.action_spec(physics).shape[0]
    action = np.zeros(action_dim, dtype=np.float64)

    # Step several control timesteps to let the fingers / wrist joints
    # settle around the pinned forearm position. We re-pin qpos after each
    # env.step (which advances physics and would otherwise move the joint
    # off-target if the actuator's ctrl pulls it back).
    for _ in range(n_settle_steps):
        # Pin forearm_tx to clamp limit by direct qpos write.
        physics.bind(forearm_joint).qpos = target_qpos
        # Also zero its velocity so it doesn't drift back.
        physics.bind(forearm_joint).qvel = 0.0
        ts = env.step(action)
        if ts.last():
            break

    # Re-pin one more time before reading (in case the last step moved it).
    physics.bind(forearm_joint).qpos = target_qpos
    physics.bind(forearm_joint).qvel = 0.0

    fingertip_xpos = physics.bind(target_hand.fingertip_sites).xpos.copy()
    return fingertip_xpos[:, 1]  # Y-axis only


def probe_agent_reach(
    assignment,
    hand_specs: Sequence[HandSpec],
    *,
    n_settle_steps: int = _DEFAULT_N_SETTLE_STEPS,
    control_timestep: float = 0.05,
    gravity_compensation: bool = True,
    disable_hand_collisions: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Return {agent_name: (reach_lo, reach_hi)} via simulator probe.

    Only probes the **physically realizable** extreme configurations:
      * single-hand agent: both forearm extremes of that hand
      * multi-hand agent: leftmost-hand at its LEFT extreme (gives agent's
        leftmost fingertip reach) + rightmost-hand at its RIGHT extreme
        (gives agent's rightmost fingertip reach)

    We do NOT probe "inner extremes" like lh_b at the agent's RIGHT extreme
    or rh_b at the agent's LEFT extreme — those configurations are physically
    infeasible under the non-crossing constraint (forearms at the same height
    can't swap order). Per `ma_territorial_impl_plan.md` § 10(c) discussion,
    the agent's reach BOUNDS are always determined by outer-hand-outer-extreme
    because LH/RH thumb-direction asymmetry makes inner-extreme fingertip
    overshoot strictly dominated by outer-extreme overshoot.

    Caller is expected to pass `hand_specs` whose `key_range` reflects the
    actual physical clamp at runtime — typically the per-agent territory
    (= union of bucket key_ranges) for the MA Territorial path. See
    `omnipiano.multiagent.registration.make_parallel` for the override site.
    """
    env = _build_probe_env(
        hand_specs,
        control_timestep=control_timestep,
        gravity_compensation=gravity_compensation,
        disable_hand_collisions=disable_hand_collisions,
    )
    try:
        reaches: Dict[str, Tuple[int, int]] = {}
        for agent in assignment.agents:
            if len(agent.hand_names) == 1:
                # Single-hand agent (e.g. 3-hand treble_soloist, 5-hand
                # center_soloist). Both extremes of that hand are needed —
                # there's no "inner" / "outer" distinction with one hand.
                h = agent.hand_names[0]
                ys_left = _drive_and_read_fingertips(
                    env, h, forearm_extreme="LEFT", n_settle_steps=n_settle_steps
                )
                ys_right = _drive_and_read_fingertips(
                    env, h, forearm_extreme="RIGHT", n_settle_steps=n_settle_steps
                )
                reach_lo = min(y_to_key_index(float(y)) for y in ys_left)
                reach_hi = max(y_to_key_index(float(y)) for y in ys_right)
            else:
                # Multi-hand agent: leftmost-hand LEFT + rightmost-hand RIGHT.
                # `agent.hand_names` is in spatial L→R order by convention
                # (see AGENT_ASSIGNMENTS).
                leftmost = agent.hand_names[0]
                rightmost = agent.hand_names[-1]
                ys_left = _drive_and_read_fingertips(
                    env, leftmost, forearm_extreme="LEFT", n_settle_steps=n_settle_steps
                )
                ys_right = _drive_and_read_fingertips(
                    env, rightmost, forearm_extreme="RIGHT", n_settle_steps=n_settle_steps
                )
                reach_lo = min(y_to_key_index(float(y)) for y in ys_left)
                reach_hi = max(y_to_key_index(float(y)) for y in ys_right)
            if reach_hi < reach_lo:
                raise RuntimeError(
                    f"agent {agent.name!r}: invalid reach "
                    f"({reach_lo}, {reach_hi}) — probe failed"
                )
            reaches[agent.name] = (reach_lo, reach_hi)
        return reaches
    finally:
        env.close()


def _drive_and_read_fingertips(
    env: composer.Environment,
    hand_name: str,
    *,
    forearm_extreme: str,  # "LEFT" or "RIGHT"
    n_settle_steps: int,
) -> np.ndarray:
    """Reset env, pin hand's forearm_tx to its joint LEFT/RIGHT extreme, return fingertip Ys."""
    env.reset()
    sign = -_FOREARM_PUSH_CTRL if forearm_extreme == "LEFT" else +_FOREARM_PUSH_CTRL
    return _settle_with_forearm_ctrl(env, hand_name, sign, n_settle_steps)



# ===========================================================================
# Caching
# ===========================================================================
# Cache key tuple structure (all parameters that affect probe results):
#   (morphology: str,
#    hand_specs: Tuple[HandSpec, ...],
#    n_settle_steps: int,
#    control_timestep: float,
#    gravity_compensation: bool,
#    disable_hand_collisions: bool)
# HandSpec is a frozen dataclass — directly hashable. Including all
# probe-affecting fields prevents stale-result reuse when callers vary
# settle time or physics config for ablations.

_REACH_CACHE: Dict[Tuple[Any, ...], Dict[str, Tuple[int, int]]] = {}


def cached_probe_agent_reach(
    assignment,
    hand_specs: Sequence[HandSpec],
    *,
    n_settle_steps: int = _DEFAULT_N_SETTLE_STEPS,
    control_timestep: float = 0.05,
    gravity_compensation: bool = True,
    disable_hand_collisions: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Cached front-end to :func:`probe_agent_reach`.

    Cache key includes all parameters that affect probe results (morphology
    + hand_specs + n_settle_steps + physics-config kwargs). Changes to any
    of these invalidate the cache entry so ablations don't get stale data.
    """
    key = (
        assignment.morphology,
        tuple(hand_specs),
        n_settle_steps,
        control_timestep,
        gravity_compensation,
        disable_hand_collisions,
    )
    if key not in _REACH_CACHE:
        _REACH_CACHE[key] = probe_agent_reach(
            assignment,
            hand_specs,
            n_settle_steps=n_settle_steps,
            control_timestep=control_timestep,
            gravity_compensation=gravity_compensation,
            disable_hand_collisions=disable_hand_collisions,
        )
    return _REACH_CACHE[key]