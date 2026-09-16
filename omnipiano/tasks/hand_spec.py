"""Per-hand specification for N-hand piano tasks.

A `HandSpec` fully describes where a single hand is placed, how it is oriented,
which forearm DoFs it has, and which logical group it belongs to (used for
multi-agent partitioning). The base `PianoTask` consumes a list of `HandSpec`s
and builds one `ShadowHand` per spec — this replaces the old hard-coded
right/left pair while preserving backward compatibility via the default specs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from robopianist.models.hands import HandSide, shadow_hand
from robopianist.models.piano import piano_constants as piano_consts

from omnipiano.tasks.multi_hand_layouts import MULTI_HAND_LAYOUTS


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
            center hand, "lh_b"/"rh_b"/"lh_t"/"rh_t" for the 4-hand
            bimanual split (see default_four_hand_specs).
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


def _multi_hand_specs(num_hands: int, *, partitioned: bool) -> Tuple[HandSpec, ...]:
    """Materialize HandSpecs from the shared dependency-free layout table."""
    try:
        layout = MULTI_HAND_LAYOUTS[num_hands]
    except KeyError as exc:
        raise ValueError(
            f"num_hands must be one of {sorted(MULTI_HAND_LAYOUTS)}, got {num_hands}"
        ) from exc
    return tuple(
        HandSpec(
            name=hand.name,
            side=HandSide[hand.side],
            position=hand.position,
            group=hand.group,
            key_range=hand.key_range if partitioned else None,
        )
        for hand in layout.hands
    )


def default_partitioned_hand_specs(num_hands: int) -> Tuple[HandSpec, ...]:
    """Return the canonical static-partition specs for a 3/4/5-hand layout."""
    return _multi_hand_specs(num_hands, partitioned=True)


def default_three_hand_specs() -> Tuple[HandSpec, HandSpec, HandSpec]:
    """3-hand morphology — canonical (LH, RH) pair + one center hand.

    Layout (y in arena coords; +y = treble side; positions = bucket centers
    of the canonical 29/30/29-key partition; see N-hand morphology axiom):
        rh    at +0.4056  (treble  bucket = keys 59-87, 29 keys)
        lh    at -0.4051  (bass    bucket = keys 0-28,  29 keys)
        rh_c  at  0.0000  (middle  bucket = keys 29-58, 30 keys)

    N-hand morphology axiom: each hand's attach position[1] equals the
    geometric center of its canonical key-range bucket — i.e.,
    ``0.5 * (key_index_to_y(lo) + key_index_to_y(hi))``. Under any Level-1
    StaticPartition variant of the 3-hand morphology this makes the
    forearm_tx joint range symmetric around the attach point. Per-hand
    forearm range asymmetry is ~5e-5 m on the outer hands and exactly 0
    on the center hand (the residual is purely 4-decimal rounding noise
    on the position values; the test threshold in
    ``ThreeHandPartitionPositionAlignmentTest::test_forearm_range_symmetric``
    is 0.05 m, so the actual margin is ~1000× over the bound). Under
    partition-free Prototype variants the position is identical, so
    L-3 Prototype vs L-1 StaticPartition comparisons isolate the
    partition constraint cleanly. See `static_partition_design.md`
    § N-hand axiom.

    Bucket scheme 29/30/29 chosen over the alternatives 29/29/30,
    30/29/29, 30/28/30 because: (a) middle bucket strictly centered at
    y=0 (cleanest layout for the center hand); (b) middle bucket widest
    at 30 keys aligns with the bass/middle/treble register distribution
    of typical piano repertoire (most note density in middle register);
    (c) outer-pair POSITION asymmetry around y=0 is ~5e-4 m (driven by
    the piano's intrinsic non-symmetry — A#0 lone black key shifts the
    geometry by ~0.5 mm; this is geometric truth, not a design defect,
    and is orthogonal to the per-hand forearm range asymmetry above).

    Returned in spatial left→right order (bass → treble) so action
    vector layout matches "leftmost hand first" — same convention as
    ``default_four_hand_specs`` / ``default_five_hand_specs``. This is
    the N-hand morphology action-layout axiom for N≥3
    (``default_two_hand_specs`` is exempt because it freezes the
    original RoboPianist (rh, lh) convention as a direct baseline; see
    ``static_partition_design.md`` § N-hand axiom).
    """
    return _multi_hand_specs(3, partitioned=False)  # type: ignore[return-value]


def default_four_hand_specs() -> Tuple[HandSpec, HandSpec, HandSpec, HandSpec]:
    """4-hand morphology — two (LH, RH) duet pairs, alternating L-R-L-R.

    Layout (y in arena coords; +y = treble side; positions = bucket centers
    of the canonical 4 × 22-key partition, see N-hand morphology axiom):
        lh_b at -0.4521  (bass        bucket = keys 0-21)
        rh_b at -0.1527  (mid_bass    bucket = keys 22-43)
        lh_t at +0.1528  (mid_treble  bucket = keys 44-65)
        rh_t at +0.4526  (treble      bucket = keys 66-87)

    N-hand morphology axiom: each hand's attach position[1] equals the
    geometric center of its canonical key-range bucket — i.e.,
    ``0.5 * (key_index_to_y(lo) + key_index_to_y(hi))``. This makes the
    forearm_tx joint range symmetric around the attach point under any
    Level-1 StaticPartition variant. Per-hand forearm range asymmetry
    is ~5e-5 to ~1e-4 m (purely 4-decimal rounding noise on position
    values; the test threshold in
    ``FourHandPartitionPositionAlignmentTest::test_forearm_range_symmetric``
    is 0.05 m, so the actual margin is ~500× over the bound). Under
    partition-free Prototype variants the position is identical, so
    L-3 Prototype vs L-1 StaticPartition comparisons isolate the
    partition constraint cleanly. See `static_partition_design.md`
    § N-hand axiom for the full rationale.

    Mirrors real piano-four-hands practice: two players sit at one piano,
    each contributing one (LH, RH) pair to a contiguous register slice.
    Secondo plays the bass half, Primo plays the treble half. Within each
    player's pair, the LH sits to the LEFT of the RH (bass-side hand on
    lower notes), giving the spatial pattern L-R-L-R.

    Why this matters vs. an L-L-R-R "stacked-by-side" layout:
      - Each spatial pair == one human's two hands (~0.30 m apart);
        the L-L-R-R alternative would make the "outer pair" ~0.90 m apart,
        not anatomically a pair at all.
      - Inner-pair thumbs (rh_b + lh_t) face *outward*, leaving a
        thumb-gap at the keyboard center — natural for boundary-region
        sharing between Secondo and Primo.
      - Each duet pair can independently consume the standard 2-hand PIG
        fingering convention (RH=fingers 1-5, LH=6-10) on its own slice,
        making future per-pair-fingering rewards trivial to wire in.

    Adjacent-hand spacing is ~0.30 m (0.2994 / 0.3055 / 0.2998); inner
    hands at ±0.1527 m are very close to the default 2-hand ±0.15 m
    positions (offset ~0.0027 m, visually imperceptible) but are not
    identical — the axiom prioritizes partition-symmetry over exact
    2-hand backward compat. Returned in spatial left→right order
    (bass → treble) so action vector layout matches "leftmost hand first".

    All hands use the default Shadow Hand DoFs / forearm DoFs from
    ``HandSpec``'s field defaults — N-hand variants don't change a
    single hand's action space, only the count. Per-hand reduced-action
    or custom forearm overrides should be done at register() time by
    constructing the tuple manually instead of using this helper.
    """
    return _multi_hand_specs(4, partitioned=False)  # type: ignore[return-value]


def default_five_hand_specs() -> Tuple[
    HandSpec, HandSpec, HandSpec, HandSpec, HandSpec
]:
    """5-hand morphology — L-R-R-L-R alternating with attach positions
    aligned to the natural key_range bucket centers.

    Layout (spatial left → right; y in arena coords):
        lh_b   at -0.4817  (LEFT,  bass        bucket = keys 0-17)
        rh_b   at -0.2345  (RIGHT, low_mid     bucket = keys 18-35)
        rh_c   at +0.0061  (RIGHT, middle      bucket = keys 36-52)
        lh_t   at +0.2468  (LEFT,  high_mid    bucket = keys 53-70)
        rh_t   at +0.4879  (RIGHT, treble      bucket = keys 71-87)

    Why L-R-R-L-R (and not the earlier L-R-L-R-R or L-L-R-R-R)

      The L-R-R-L-R layout decomposes cleanly into THREE anatomically
      valid agents under the multi-agent wrapper:

          left player  : (lh_b LEFT, rh_b RIGHT)   spacing 0.247 m  — (LH, RH) duet
          center solo  : (rh_c RIGHT)              single hand
          right player : (lh_t LEFT, rh_t RIGHT)   spacing 0.241 m  — (LH, RH) duet

      Both outer agents get a STANDARD (LH, RH) pair at ~0.24 m spacing
      (matches a single human's two-handed reach, mirroring the 4-hand
      Secondo/Primo duet design). The center hand is a soloist — this is
      the only "asymmetric" agent, but it has only one hand so anatomy
      is trivially fine.

      The earlier L-R-L-R-R layout (with `lh_c` LEFT at center +
      `rh_t2 RIGHT, rh_t1 RIGHT` stacked on the treble half) put TWO
      right hands on a single right-side agent — anatomically impossible
      for a single human. L-R-R-L-R fixes this by moving one LEFT hand
      from center to the treble side, freeing the center to be a single
      RIGHT hand.

      Side ratio (3 RIGHT + 2 LEFT) is preserved from the prior version;
      naming convention now matches the 3-hand center hand (`rh_c`) and
      uses the 4-hand-style `_b` / `_t` suffixes for outer-pair zones.

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
    # point. Per-hand forearm range asymmetry is ~5e-5 to ~1e-4 m (purely
    # 4-decimal rounding noise on the position values; the test threshold
    # in ``FiveHandPartitionPositionAlignmentTest::test_forearm_range_symmetric``
    # is 0.05 m, so the actual margin is ~500× over the bound).
    # Bucket boundaries: [(0,17), (18,35), (36,52), (53,70), (71,87)].
    # Positions are CACHED as float constants below for deterministic
    # registry behavior across Python runs.
    return _multi_hand_specs(5, partitioned=False)  # type: ignore[return-value]
