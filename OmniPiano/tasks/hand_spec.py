"""Per-hand specification for N-hand piano tasks.

A `HandSpec` fully describes where a single hand is placed, how it is oriented,
which forearm DoFs it has, and which logical group it belongs to (used for
multi-agent partitioning). The base `PianoTask` consumes a list of `HandSpec`s
and builds one `ShadowHand` per spec — this replaces the old hard-coded
right/left pair while preserving backward compatibility via the default specs.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from robopianist.models.hands import HandSide, shadow_hand
from robopianist.models.piano import piano_constants as piano_consts


# Default poses — reproduce the existing 2-hand configuration exactly.
_LEFT_HAND_POSITION: Tuple[float, float, float] = (0.4, -0.15, 0.13)
_RIGHT_HAND_POSITION: Tuple[float, float, float] = (0.4, 0.15, 0.13)
_DEFAULT_QUATERNION: Tuple[float, float, float, float] = (-1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Piano key index → arena Y coordinate (for static partitioning, Level 1+).
# ---------------------------------------------------------------------------

# Mod-12 indices of black keys, anchored at A0. The pattern is:
#   0:A  1:A# 2:B  3:C  4:C# 5:D  6:D# 7:E  8:F  9:F# 10:G 11:G#
# Matches `piano.Piano.is_key_black`'s lookup table.
_BLACK_KEYS_MOD12 = frozenset({1, 4, 6, 9, 11})


def key_index_to_y(key_idx: int) -> float:
    """Center Y coordinate (arena/world frame) of the i-th piano key.

    ``key_idx=0`` is A0 (MIDI 21, leftmost / lowest pitch);
    ``key_idx=87`` is C8 (MIDI 108, rightmost / highest pitch).

    Mirrors the *exact* placement formulas in ``piano_mjcf.build``:
      - white keys: ``y = -L/2 + W/2 + white_idx * (W + S)``
      - lone black (A#0, key_idx=1): gap center between white_0 and white_1
        ``y = -L/2 + W + S/2``
      - twin/triplet blacks: placed flush with the left edge of the
        right-adjacent white key (a 0.5 mm shift from gap center; this
        matches piano_mjcf's `(j+1+block_index)*(W+S)` formulation):
        ``y = -L/2 + white_before * (W + S)``

    The lone-vs-twin/triplet asymmetry is inherited from piano_mjcf — the
    lone black uses gap-center placement, while twin/triplet blacks are
    offset by +SPACING/2. We match physics exactly here so partition Y
    bounds align with the actual key body xpos.
    """
    if not 0 <= key_idx < piano_consts.NUM_KEYS:
        raise ValueError(
            f"key_idx must be in [0, {piano_consts.NUM_KEYS}), got {key_idx}"
        )
    pitch = piano_consts.WHITE_KEY_WIDTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS
    half_piano = piano_consts.PIANO_LENGTH * 0.5
    # Number of white keys at indices strictly less than key_idx.
    white_before = sum(
        1 for j in range(key_idx) if (j % 12) not in _BLACK_KEYS_MOD12
    )

    if (key_idx % 12) not in _BLACK_KEYS_MOD12:
        # White key: index equals ``white_before``.
        return (
            -half_piano
            + 0.5 * piano_consts.WHITE_KEY_WIDTH
            + white_before * pitch
        )

    # Black key.
    if key_idx == 1:
        # Lone black (A#0) — piano_mjcf uses gap-center placement here only.
        return (
            -half_piano
            + piano_consts.WHITE_KEY_WIDTH
            + 0.5 * piano_consts.SPACING_BETWEEN_WHITE_KEYS
        )
    # Twin / triplet black — placed at left edge of right-adjacent white
    # (where right-adjacent white has index ``white_before``).
    return -half_piano + white_before * pitch


def key_range_to_y_range(
    lo_key: int, hi_key: int, edge_buffer: float = 0.0
) -> Tuple[float, float]:
    """Convert an inclusive piano key index range to an arena Y range.

    Args:
        lo_key: Leftmost (lowest pitch) key index, 0..87.
        hi_key: Rightmost (highest pitch) key index, lo_key..87.
        edge_buffer: Optional slack (meters) added to each end. The wrist's
            Y need not be exactly under the key center for the fingertip to
            press it, so a small buffer (~half a white-key width) lets the
            hand reach edge-of-region keys with a comfortable posture.
            Default 0 = strict (the wrist can land anywhere from the center
            of the leftmost assigned key to the center of the rightmost).

    Returns ``(y_lo, y_hi)`` in arena coordinates, ready to drop into a
    ``HandSpec.y_range`` field.
    """
    if not (0 <= lo_key <= hi_key < piano_consts.NUM_KEYS):
        raise ValueError(
            f"key_range must satisfy 0 <= lo <= hi < {piano_consts.NUM_KEYS}, "
            f"got ({lo_key}, {hi_key})"
        )
    if edge_buffer < 0:
        raise ValueError(f"edge_buffer must be non-negative, got {edge_buffer}")
    return (
        key_index_to_y(lo_key) - edge_buffer,
        key_index_to_y(hi_key) + edge_buffer,
    )


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
        y_range: (Optional) Static partition: arena Y bounds (meters) the
            hand's ``forearm_tx`` slider is allowed to reach. Mutually
            exclusive with ``key_range``. If neither is set, the hand can
            reach the full keyboard (default Level-3 behavior preserved).
        key_range: (Optional) Static partition: inclusive piano key index
            range, 0..87 (0 = A0, 87 = C8). Resolved to a y_range via
            ``key_range_to_y_range`` at task-build time. Prefer this over
            ``y_range`` for music-semantic registrations. Mutually exclusive
            with ``y_range``.
    """

    name: str
    side: HandSide
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float] = _DEFAULT_QUATERNION
    attachment_yaw: float = 0.0
    forearm_dofs: Tuple[str, ...] = shadow_hand._DEFAULT_FOREARM_DOFS
    reduced_action_space: bool = False
    group: str = "default"
    # Level-1 / Level-2 partition fields (mutually exclusive; both optional).
    y_range: Optional[Tuple[float, float]] = None
    key_range: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        # Frozen dataclass: __post_init__ can validate but cannot mutate.
        if self.y_range is not None and self.key_range is not None:
            raise ValueError(
                f"HandSpec {self.name!r}: y_range and key_range are mutually "
                f"exclusive — got both y_range={self.y_range} and "
                f"key_range={self.key_range}. Use exactly one."
            )
        if self.key_range is not None:
            lo, hi = self.key_range
            if not (0 <= lo <= hi < piano_consts.NUM_KEYS):
                raise ValueError(
                    f"HandSpec {self.name!r}: invalid key_range "
                    f"{self.key_range}; require 0 <= lo <= hi < "
                    f"{piano_consts.NUM_KEYS}."
                )
        if self.y_range is not None:
            lo, hi = self.y_range
            if not lo < hi:
                raise ValueError(
                    f"HandSpec {self.name!r}: y_range {self.y_range} must "
                    f"satisfy lo < hi."
                )

    @property
    def resolved_y_range(self) -> Optional[Tuple[float, float]]:
        """The Y bounds to enforce on this hand's forearm_tx slider.

        Returns the y_range if set; otherwise resolves key_range via
        ``key_range_to_y_range``; otherwise None (no partition — fall back
        to full-keyboard reach in the caller). This is the SINGLE source of
        truth used by ``base.py:_add_hand`` — that code must not inspect
        ``key_range`` directly.
        """
        if self.y_range is not None:
            return self.y_range
        if self.key_range is not None:
            return key_range_to_y_range(*self.key_range)
        return None


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


def default_four_hand_specs() -> Tuple[HandSpec, HandSpec, HandSpec, HandSpec]:
    """4-hand morphology — two (LH, RH) duet pairs, alternating L-R-L-R.

    Layout (y in arena coords; +y = treble side):
        lh_b at -0.45  (bass,        Secondo's LH)
        rh_b at -0.15  (mid_bass,    Secondo's RH)
        lh_t at +0.15  (mid_treble,  Primo's   LH)
        rh_t at +0.45  (treble,      Primo's   RH)

    Mirrors real piano-four-hands practice: two players sit at one piano,
    each contributing one (LH, RH) pair to a contiguous register slice.
    Secondo plays the bass half, Primo plays the treble half. Within each
    player's pair, the LH sits to the LEFT of the RH (bass-side hand on
    lower notes), giving the spatial pattern L-R-L-R.

    Why this matters vs. an L-L-R-R "stacked-by-side" layout:
      - Each spatial pair == one human's two hands (~0.30 m apart);
        the L-L-R-R alternative would make the "outer pair" 0.90 m apart,
        not anatomically a pair at all.
      - Inner-pair thumbs (rh_b at -0.15 + lh_t at +0.15) face *outward*,
        leaving a thumb-gap at the keyboard center — natural for
        boundary-region sharing between Secondo and Primo.
      - Each duet pair can independently consume the standard 2-hand PIG
        fingering convention (RH=fingers 1-5, LH=6-10) on its own slice,
        making future per-pair-fingering rewards trivial to wire in.

    Spacing 0.30 m matches the 2-hand and 3-hand precedent. Inner hands at
    ±0.15 coincide with the default 2-hand positions. Returned in spatial
    left→right order (bass → treble) so action vector layout matches
    "leftmost hand first".

    All hands use the default Shadow Hand DoFs / forearm DoFs from
    ``HandSpec``'s field defaults — N-hand variants don't change a
    single hand's action space, only the count. Per-hand reduced-action
    or custom forearm overrides should be done at register() time by
    constructing the tuple manually instead of using this helper.
    """
    return (
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.45, 0.13), group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.15, 0.13), group="mid_bass"),
        HandSpec(name="lh_t", side=HandSide.LEFT,
                 position=(0.4, +0.15, 0.13), group="mid_treble"),
        HandSpec(name="rh_t", side=HandSide.RIGHT,
                 position=(0.4, +0.45, 0.13), group="treble"),
    )


def default_five_hand_specs() -> Tuple[
    HandSpec, HandSpec, HandSpec, HandSpec, HandSpec
]:
    """5-hand morphology — L-R-L-R-R alternating with attach positions
    aligned to the natural key_range bucket centers.

    Layout (spatial left → right; y in arena coords):
        lh_b   at -0.4817  (LEFT,  bass        bucket = keys 0-17)
        rh_b   at -0.2345  (RIGHT, low_mid     bucket = keys 18-35)
        lh_c   at +0.0061  (LEFT,  middle      bucket = keys 36-52)
        rh_t2  at +0.2468  (RIGHT, high_mid    bucket = keys 53-70)
        rh_t1  at +0.4879  (RIGHT, treble      bucket = keys 71-87)

    Why L-R-L-R-R (and not the previous L-L-R-R-R or the "ideal" L-R-L-R-?)

      - **L-R alternation in the bass half mirrors the 4-hand duet design.**
        Adjacent (LH, RH) pair at (-0.48, -0.23) reads as Secondo's hands.
      - **Center hand at y=0 is LEFT** (`lh_c`), not RIGHT (was `rh_c`).
        This deliberately breaks the 3-hand `rh_c` convention because with
        5 hands and a fixed 2-LH / 3-RH ratio, putting LH in the center
        gives every L-R adjacent pair a natural "duet" spatial structure
        on the bass half (see above). The treble half (rh_t2 + rh_t1)
        becomes a "two RH" stack — analogous to the L-L-R-R original
        4-hand layout but only on one side, accepting that 5 hands cannot
        decompose into perfect duet pairs.
      - **Asymmetric (3 RH + 2 LH)** is preserved from the prior version;
        flipping to 3 LH + 2 RH would only require mirroring all sides.

    Why the non-uniform 0.24-spacing positions (instead of 0.20 uniform)

      Each spec.position[1] is set to the CENTER of its assigned key_range
      bucket (computed via `key_index_to_y`), so that the forearm_tx joint
      range derived from `(key_range_to_y_range(bucket) - position[1])` is
      symmetric around 0. With the previous 0.20-uniform spacing, outer
      hands had asymmetric forearm ranges biased toward the keyboard
      center (e.g., bass-most hand could slide -0.20 m left but only
      +0.04 m right). This made the "neutral" home posture sit at the
      outer edge of the partition, not the middle — physically awkward
      and biased exploration.

      Spacing between adjacent hands is ~0.241 m (vs 0.20 previously).
      Shadow Hand mesh width ~0.10 m → ~0.14 m gap, still comfortable.
      Outermost hands at ±0.488 m (vs ±0.40 previously) approach the
      keyboard edge ±0.61 m, but stay well within reachable territory.

    All hands use the default Shadow Hand DoFs / forearm DoFs from
    ``HandSpec``'s field defaults — N-hand variants don't change a
    single hand's action space, only the count.

    Returned in spatial left→right order (bass → treble) so action vector
    layout matches "leftmost hand first" — same convention as
    ``default_four_hand_specs``.
    """
    # Positions = each bucket's geometric center, computed as
    # ``0.5 * (key_index_to_y(lo) + key_index_to_y(hi))`` — i.e., the
    # MIDPOINT OF THE TWO ENDPOINT KEY Y COORDINATES, *not*
    # ``key_index_to_y((lo+hi)/2)`` (the latter would differ by up to
    # 12 mm because white/black key Y spacing is non-uniform). With this
    # choice, the forearm_tx joint range is symmetric around the attach
    # point (asymmetry < 0.0002 m, verified by FiveHandPartitionPosition-
    # AlignmentTest::test_forearm_range_symmetric).
    # Bucket boundaries: [(0,17), (18,35), (36,52), (53,70), (71,87)].
    # Positions are CACHED as float constants below for deterministic
    # registry behavior across Python runs.
    return (
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.4817, 0.13), group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.2345, 0.13), group="low_mid"),
        HandSpec(name="lh_c", side=HandSide.LEFT,
                 position=(0.4, +0.0061, 0.13), group="middle"),
        HandSpec(name="rh_t2", side=HandSide.RIGHT,
                 position=(0.4, +0.2468, 0.13), group="high_mid"),
        HandSpec(name="rh_t1", side=HandSide.RIGHT,
                 position=(0.4, +0.4879, 0.13), group="treble"),
    )
