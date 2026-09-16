"""Dependency-free source of truth for the supported N-hand embodiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HandLayout:
    name: str
    side: str
    position: Tuple[float, float, float]
    group: str
    key_range: Tuple[int, int]


@dataclass(frozen=True)
class MultiHandLayout:
    morphology: str
    hands: Tuple[HandLayout, ...]


MULTI_HAND_LAYOUTS = {
    3: MultiHandLayout("ThreeHand", (
        HandLayout("lh", "LEFT", (0.4, -0.4051, 0.13), "bass", (0, 28)),
        HandLayout("rh_c", "RIGHT", (0.4, 0.0, 0.13), "middle", (29, 58)),
        HandLayout("rh", "RIGHT", (0.4, 0.4056, 0.13), "treble", (59, 87)),
    )),
    4: MultiHandLayout("FourHand", (
        HandLayout("lh_b", "LEFT", (0.4, -0.4521, 0.13), "bass", (0, 21)),
        HandLayout("rh_b", "RIGHT", (0.4, -0.1527, 0.13), "mid_bass", (22, 43)),
        HandLayout("lh_t", "LEFT", (0.4, 0.1528, 0.13), "mid_treble", (44, 65)),
        HandLayout("rh_t", "RIGHT", (0.4, 0.4526, 0.13), "treble", (66, 87)),
    )),
    5: MultiHandLayout("FiveHand", (
        HandLayout("lh_b", "LEFT", (0.4, -0.4817, 0.13), "bass", (0, 17)),
        HandLayout("rh_b", "RIGHT", (0.4, -0.2345, 0.13), "low_mid", (18, 35)),
        HandLayout("rh_c", "RIGHT", (0.4, 0.0061, 0.13), "middle", (36, 52)),
        HandLayout("lh_t", "LEFT", (0.4, 0.2468, 0.13), "high_mid", (53, 70)),
        HandLayout("rh_t", "RIGHT", (0.4, 0.4879, 0.13), "treble", (71, 87)),
    )),
}


__all__ = ["HandLayout", "MULTI_HAND_LAYOUTS", "MultiHandLayout"]
