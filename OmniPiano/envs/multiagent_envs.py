"""Multi-agent env registrations (PettingZoo ParallelEnv) — Territorial series.

Mirrors `OmniPiano/envs/__init__.py` structure but for MA envs. Each MA env
wraps an existing SA StaticPartition env via the PettingZoo ParallelEnv
shim in `OmniPiano.multiagent`.

Phase 1 scope: 4-hand only (Sub-phase 1A). 3-hand (1B) and 5-hand (1C)
register-loops will be added in subsequent sub-phases.
"""

from OmniPiano.multiagent import register_parallel


# ===========================================================================
# Sub-phase 1A: 4-hand Territorial Duet
# ===========================================================================
# AgentSetup = "Duet" → agents = (secondo, primo); see
# `OmniPiano.multiagent.assignment.AGENT_ASSIGNMENTS["FourHand"]`.
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

# Phase 1A registers 3 MA envs. Sub-phase 1B (3-hand) and 1C (5-hand) will
# extend this file with their respective register-loops.
