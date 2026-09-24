"""Multi-agent env registrations (PettingZoo ParallelEnv) — Territorial series.

Mirrors `omnipiano/envs/__init__.py` structure but for MA envs. Each MA env
wraps an existing SA StaticPartition env via the PettingZoo ParallelEnv
shim in `omnipiano.multiagent`.

Phase 1 scope (per ma_territorial_impl_plan.md § 11):
  Sub-phase 1A: 4-hand Duet (3 envs)
  Sub-phase 1B: 3-hand MainSolo (4 envs)
  Sub-phase 1C: 5-hand Trio (1 env — WinterWind only;
                              PicturesGreatKiev 5-hand SA deferred per
                              `envs/__init__.py:1055` comment)
"""

from omnipiano.multiagent import register_parallel


# ===========================================================================
# Sub-phase 1A: 4-hand Territorial Duet
# ===========================================================================
# AgentSetup = "Duet" → agents = (secondo, primo); see
# `omnipiano.multiagent.AGENT_ASSIGNMENTS["FourHand"]`.
# Underlying SA envs: OmniPiano-{Piece}-FourHand-StaticPartition-v0.

_FOUR_HAND_PIECES = (
    "WinterWind",          # smoke piece, morphology-ladder anchor
    "PianoSonataNo301StMov",
    "PicturesGreatKiev",
)

for _piece in _FOUR_HAND_PIECES:
    register_parallel(
        id=f"OmniPiano-{_piece}-FourHand-MA-Duet-Territorial-v0",
        sa_env_id=f"OmniPiano-{_piece}-FourHand-StaticPartition-v0",
        morphology="FourHand",
    )


# ===========================================================================
# Sub-phase 1B: 3-hand Territorial MainSolo
# ===========================================================================
# AgentSetup = "MainSolo" → agents = (secondo, treble_soloist); see
# `omnipiano.multiagent.AGENT_ASSIGNMENTS["ThreeHand"]`.
#   secondo        (sustain owner): lh + rh_c → action dim 45 (2×22 + 1)
#   treble_soloist (1-hand agent):  rh        → action dim 22
# Underlying SA envs: OmniPiano-{Piece}-ThreeHand-StaticPartition-v0.
# This sub-phase validates the 1-hand-agent code path (single-hand
# probe branch, no-sustain action layout, single-neighbor boundary_hands).

_THREE_HAND_PIECES = (
    "WinterWind",            # smoke piece, morphology-ladder anchor
    "PicturesGreatKiev",
    "PolonaiseOp40No1",
    "PianoSonataNo281StMov",
)

for _piece in _THREE_HAND_PIECES:
    register_parallel(
        id=f"OmniPiano-{_piece}-ThreeHand-MA-MainSolo-Territorial-v0",
        sa_env_id=f"OmniPiano-{_piece}-ThreeHand-StaticPartition-v0",
        morphology="ThreeHand",
    )

# Phase 1B registers 4 MA envs.


# ===========================================================================
# Sub-phase 1C: 5-hand Territorial Trio
# ===========================================================================
# AgentSetup = "Trio" → agents = (left_secondo, center_soloist, right_primo);
# see `omnipiano.multiagent.AGENT_ASSIGNMENTS["FiveHand"]`.
#   left_secondo  (sustain owner): lh_b + rh_b → action dim 45 (2×22 + 1)
#   center_soloist (1-hand agent): rh_c        → action dim 22
#   right_primo                  : lh_t + rh_t → action dim 44
# Underlying SA envs: OmniPiano-{Piece}-FiveHand-StaticPartition-v0.
# This sub-phase validates: (a) N≥3 agent code path, (b) per-agent
# multi-boundary obs assembly — center_soloist has 2 neighbors
# (left_secondo and right_primo), so its boundary_hands Dict has 2 keys
# ({"rh_b": ..., "lh_t": ...}) rather than the single neighbor seen in
# 3-hand / 4-hand morphologies.
#
# Scope note (Option C, 2026-05-15): only WinterWind is registered, not
# the originally planned 2 envs. The 5-hand StaticPartition variant for
# PicturesGreatKiev was deliberately deferred at the SA layer (see
# `envs/__init__.py:1055` comment — paper-quality 5-hand L-1 runs are
# ~2.3× wall-clock of WinterWind, deferred until needed). Adding it
# later is a one-line addition to both SA `envs/__init__.py` and this
# register-loop. Sub-phase 1C verification still meaningful with 1 env:
# the value gate is "does the wrapper handle 3-agent + multi-boundary
# correctly", not "do N pieces all train".

_FIVE_HAND_PIECES = (
    "WinterWind",            # smoke piece, morphology-ladder anchor
    # "PicturesGreatKiev",   # deferred per SA-side decision (envs/__init__.py:1055)
)

for _piece in _FIVE_HAND_PIECES:
    register_parallel(
        id=f"OmniPiano-{_piece}-FiveHand-MA-Trio-Territorial-v0",
        sa_env_id=f"OmniPiano-{_piece}-FiveHand-StaticPartition-v0",
        morphology="FiveHand",
    )
