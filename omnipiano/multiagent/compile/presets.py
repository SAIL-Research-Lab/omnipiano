"""Lightweight music and default agent-assignment presets.

User piano-key intervals are inclusive 1..88; internal intervals are 0..87.
These buckets describe the fixed embodiment, not per-agent action limits.
"""

from omnipiano.tasks.multi_hand_layouts import MULTI_HAND_LAYOUTS


# Compiler-friendly tuple view over the shared SA/MA physical layout source.
LAYOUTS = {
    num_hands: (
        layout.morphology,
        tuple(hand.name for hand in layout.hands),
        tuple(hand.key_range for hand in layout.hands),
    )
    for num_hands, layout in MULTI_HAND_LAYOUTS.items()
}

DEFAULT_ASSIGNMENTS = {
    3: (("secondo", (0, 1)), ("treble_soloist", (2,))),
    4: (("secondo", (0, 1)), ("primo", (2, 3))),
    5: (("left_secondo", (0, 1)), ("center_soloist", (2,)),
        ("right_primo", (3, 4))),
}

# Historical env-id tokens.  Ownership itself lives only in
# ``DEFAULT_ASSIGNMENTS``; ``env_runtime/topology.py`` materializes the typed
# runtime objects from this table instead of maintaining a second hand-written
# copy.
AGENT_SETUPS = {3: "MainSolo", 4: "Duet", 5: "Trio"}

SONGS = {
    "TwinkleTwinkleLittleStar": "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    "WinterWind": "RoboPianist-repertoire-150-EtudeOp25No11-v0",
    "PicturesGreatKiev": (
        "RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0"
    ),
    "PianoSonataNo301StMov": "RoboPianist-repertoire-150-PianoSonataNo301StMov-v0",
    "PianoSonataNo281StMov": "RoboPianist-repertoire-150-PianoSonataNo281StMov-v0",
    "PolonaiseOp40No1": "RoboPianist-repertoire-150-PolonaiseOp40No1-v0",
    "ForElise": "RoboPianist-repertoire-150-ForElise-v0",
    "ClairDeLune": "RoboPianist-repertoire-150-ClairDeLune-v0",
    "NocturneOp9No2": "RoboPianist-repertoire-150-NocturneOp9No2-v0",
}
