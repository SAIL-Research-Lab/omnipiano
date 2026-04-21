"""Per-hand specification for N-hand piano tasks.

A `HandSpec` fully describes where a single hand is placed, how it is oriented,
which forearm DoFs it has, and which logical group it belongs to (used for
multi-agent partitioning). The base `PianoTask` consumes a list of `HandSpec`s
and builds one `ShadowHand` per spec — this replaces the old hard-coded
right/left pair while preserving backward compatibility via the default specs.
"""

from dataclasses import dataclass, field
from typing import Tuple

from robopianist.models.hands import HandSide, shadow_hand


# Default poses — reproduce the existing 2-hand configuration exactly.
_LEFT_HAND_POSITION: Tuple[float, float, float] = (0.4, -0.15, 0.13)
_RIGHT_HAND_POSITION: Tuple[float, float, float] = (0.4, 0.15, 0.13)
_DEFAULT_QUATERNION: Tuple[float, float, float, float] = (-1, -1, 1, 1)


@dataclass(frozen=True)
class HandSpec:
    """Immutable description of a single hand to be attached to the task.

    Attributes:
        name: Unique identifier for this hand within the task. Used verbatim as
            the MJCF model stem: ``f"{name}_shadow_hand"`` — so this directly
            controls observation keys like ``{name}_shadow_hand/joints_pos``.
            Canonical convention: "rh" / "lh" for the default pair (preserves
            the original RoboPianist obs-key contract), "rh_c" for a third
            center hand, "rh_t1" / "lh_b1" for future 4-hand bimanual splits.
            Must be unique across all specs in one task.
        side: Which body-side XML model to load (RIGHT or LEFT). Determines the
            MJCF prefix ("rh_" / "lh_") and the hand geometry.
        position: (x, y, z) attach position in arena coordinates.
        quaternion: (w, x, y, z) attach orientation.
        attachment_yaw: Additional Z-axis rotation in degrees, composed on top
            of `quaternion`. Sign is flipped for LEFT hands to match the
            original symmetric posture convention.
        forearm_dofs: Names of forearm joints/actuators to expose.
        reduced_action_space: If True, drop the excluded DoFs from the action
            space for this hand.
        group: Logical group label. Used for multi-agent partitioning
            (e.g., "low_register" vs "high_register") and for grouping reward
            terms. "default" means no special grouping.
    """

    name: str
    side: HandSide
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float] = _DEFAULT_QUATERNION
    attachment_yaw: float = 0.0
    forearm_dofs: Tuple[str, ...] = shadow_hand._DEFAULT_FOREARM_DOFS
    reduced_action_space: bool = False
    group: str = "default"


def default_two_hand_specs() -> Tuple[HandSpec, HandSpec]:
    """The canonical (rh, lh) pair used by existing 2-hand tasks.

    Names are "rh" / "lh" so that the resulting MJCF model names
    (``f"{spec.name}_shadow_hand"`` → ``rh_shadow_hand`` / ``lh_shadow_hand``)
    match the original RoboPianist convention exactly — this keeps observation
    keys like ``rh_shadow_hand/joints_pos`` unchanged for downstream
    consumers (training scripts, log parsers, tests).
    """
    return (
        HandSpec(
            name="rh",
            side=HandSide.RIGHT,
            position=_RIGHT_HAND_POSITION,
        ),
        HandSpec(
            name="lh",
            side=HandSide.LEFT,
            position=_LEFT_HAND_POSITION,
        ),
    )
