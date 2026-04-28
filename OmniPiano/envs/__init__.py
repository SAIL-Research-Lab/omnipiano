"""OmniPiano environment registration.

All benchmark tasks are registered here via register() calls.
Importing this module triggers registration (side-effect on import).

Naming convention:
    OmniPiano-{PieceName}-{TaskType}-v0

All formal benchmark tasks use PIG dataset pieces.
Debug pieces (TwinkleTwinkleLittleStar, etc.) are NOT registered here;
they exist only for installation verification (see README Quick Start).
"""

from OmniPiano.envs.registration import register
from OmniPiano.configs import SafetyConfig, RobustConfig, TaskVariantConfig, BenchmarkEnvConfig
from OmniPiano.safety.constraints import (
    JointMagnitudeConstraint,
    MultiJointSharedMagnitudeConstraint,
    MultiJointSummedMagnitudeConstraint,
    HandCollisionConstraint,
    HandCollisionForceConstraint,
    TotalActuatorPowerConstraint,
    InjuredJointPowerConstraint,
)
from OmniPiano.tasks.hand_spec import HandSpec
from robopianist.models.hands import HandSide

# ===========================================================================
# Ablation: annotation-based vs OT fingering reward
#
# Paired tasks for empirically validating whether the OT fingering reward
# (Hungarian assignment, no per-finger supervision) provides comparable F1
# to the annotation-based fingering reward used in the original RoboPianist
# paper. Both variants use the SAME piece (For Elise, annotated in PIG),
# SAME base env, NO safety constraints, and differ only in
# ``BenchmarkEnvConfig.disable_fingering_reward``. Final F1 difference is
# attributable solely to the fingering-reward signal.
# ---------------------------------------------------------------------------
# ForElise: RH A3-E7 (35 unique keys, 399 steps). Annotated; shortest PIG
# piece — good smoke-test target for reward-signal ablations.
# ===========================================================================
register(
    id="OmniPiano-ForElise-FingeringAnn-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    # Defaults: empty SafetyConfig, BenchmarkEnvConfig(disable_fingering_reward=False)
    # → annotation-based fingering reward, no safety overhead.
)
register(
    id="OmniPiano-ForElise-FingeringOT-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    # → OT (Hungarian) fingering reward, no safety overhead. Paired with
    # FingeringAnn-v0 above for the ablation; everything else identical.
)

# ===========================================================================
# Task Type 1: Right-Hand Wrist Limit
#
# Structural conflict: wide RH range forces large wrist movements, but the
# constraint caps wrist magnitude — the agent must find alternative fingering.
# ---------------------------------------------------------------------------
# ForElise: RH A3-E7 (35 unique keys, 399 steps).  The wide right-hand range
# spanning over 4 octaves makes wrist constraint highly restrictive.
# ===========================================================================
register(
    id="OmniPiano-ForElise-WristLimit-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    safety_config=SafetyConfig(
        constraints=[
            JointMagnitudeConstraint(index=1, max_magnitude=0.8, penalty_coef=5.0),
        ]
    ),
)

# ===========================================================================
# Task Type 2: Right-Hand Only (task variant, no safety cost)
#
# Tests whether a single hand can perform a piece originally written for two.
# ---------------------------------------------------------------------------
# NocturneOp9No2: RH E4-D6, LH G#1-A#4 (769 steps).  Classic melody + bass
# accompaniment structure; freezing LH removes harmonic support and tests
# whether RH alone can maintain the melodic line.
# ===========================================================================
register(
    id="OmniPiano-NocturneOp9No2-RightHandOnly-v0",
    base_env_name="RoboPianist-repertoire-150-NocturneOp9No2-v0",
    task_config=TaskVariantConfig(left_hand_immobile=True),
)

# ===========================================================================
# Task Type 3: Collision-Safe (binary collision detection)
#
# Cost = 1 per step if any hand-hand geom contact exists, else 0.
# ---------------------------------------------------------------------------
# ClairDeLune: RH G#3-G#5, LH G#2-A4, overlap=13 semitones (597 steps).
# Debussy's gentle dynamics with hands in overlapping registers make collision
# avoidance a subtle but persistent challenge.
#
# MapleLeafRag: RH G#2-G#6, LH G#1-C#5, overlap=29 semitones (401 steps).
# Ragtime's stride pattern with extreme overlap creates a harder variant
# where collision is nearly unavoidable without careful coordination.
# ===========================================================================
register(
    id="OmniPiano-ClairDeLune-CollisionSafe-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    safety_config=SafetyConfig(
        constraints=[
            HandCollisionConstraint(penalty_coef=1.0),
        ]
    ),
)

register(
    id="OmniPiano-MapleLeafRag-CollisionSafe-v0",
    base_env_name="RoboPianist-repertoire-150-MapleLeafRag-v0",
    safety_config=SafetyConfig(
        constraints=[
            HandCollisionConstraint(penalty_coef=1.0),
        ]
    ),
)

# ===========================================================================
# Task Type 4: Power-Constrained (dense actuator power cost)
#
# Cost = total actuator power per step.  Structural conflict: expressive
# playing requires high force and velocity, both of which increase power.
# ---------------------------------------------------------------------------
# PolonaiseOp53 ("Heroic"): RH D#3-F5, LH D#1-G#4, overlap=17 (655 steps).
# Chopin's powerful polonaise demands fortissimo octave passages — the agent
# must find energy-efficient voicing without sacrificing musical intensity.
#
# FantaisieImpromptu: RH C#4-B6, LH E2-A4, overlap=8 (800 steps).
# Fast continuous arpeggios at presto tempo — high velocity across all
# actuators creates sustained power demand throughout the entire piece.
# ===========================================================================
register(
    id="OmniPiano-PolonaiseOp53-PowerConstrained-v0",
    base_env_name="RoboPianist-repertoire-150-PolonaiseOp53-v0",
    safety_config=SafetyConfig(
        constraints=[
            TotalActuatorPowerConstraint(penalty_coef=1.0),
        ]
    ),
)

register(
    id="OmniPiano-FantaisieImpromptu-PowerConstrained-v0",
    base_env_name="RoboPianist-repertoire-150-FantaisieImpromptu-v0",
    safety_config=SafetyConfig(
        constraints=[
            TotalActuatorPowerConstraint(penalty_coef=1.0),
        ]
    ),
)

# ===========================================================================
# Task Type 5: Action Robustness (Gaussian noise on actions)
#
# No safety cost.  Tests policy robustness under execution noise (std=0.01).
# ---------------------------------------------------------------------------
# FantaisieImpromptu: fast alternating fingering at presto tempo (800 steps).
# Even 1% action noise can cause wrong key presses at this speed — the most
# sensitive piece to action perturbation among candidates.
# ===========================================================================
register(
    id="OmniPiano-FantaisieImpromptu-ActionRobust-v0",
    base_env_name="RoboPianist-repertoire-150-FantaisieImpromptu-v0",
    robust_config=RobustConfig(
        action_noise_std=0.01,
        obs_noise_std=0.0,
    ),
)

# ===========================================================================
# Task Type 6: Collision Force (continuous normal contact force cost)
#
# Cost = sum of normal contact forces between hands.  Dense signal that
# differentiates light brush from hard impact.
# ---------------------------------------------------------------------------
# MapleLeafRag: overlap=29 semitones (401 steps).  High overlap in a short
# piece guarantees frequent hand proximity.  The agent cannot trivially keep
# hands apart because the music demands overlapping key ranges.
#
# EtudeOp10No12 ("Revolutionary"): overlap=33 semitones (3089 steps).
# Maximum hand overlap in the entire PIG dataset.  Left-hand fast chromatic
# runs cross deeply into right-hand territory — the most extreme collision
# force challenge.  Long episode also tests sustained force control.
# ===========================================================================
register(
    id="OmniPiano-MapleLeafRag-CollisionForce-v0",
    base_env_name="RoboPianist-repertoire-150-MapleLeafRag-v0",
    safety_config=SafetyConfig(
        constraints=[
            HandCollisionForceConstraint(penalty_coef=1.0),
        ]
    ),
)

register(
    id="OmniPiano-EtudeOp10No12-CollisionForce-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp10No12-v0",
    safety_config=SafetyConfig(
        constraints=[
            HandCollisionForceConstraint(penalty_coef=1.0),
        ]
    ),
)

# ===========================================================================
# Task Type 7: Observation Robustness (Gaussian noise on observations)
#
# No safety cost.  Tests policy robustness under sensor noise (std=0.01).
# ---------------------------------------------------------------------------
# ClairDeLune: soft dynamics require precise position sensing (597 steps).
# Observation noise degrades the agent's ability to judge finger-to-key
# distance, which is critical for the gentle, controlled touch this piece
# demands.
# ===========================================================================
register(
    id="OmniPiano-ClairDeLune-ObservationRobust-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    robust_config=RobustConfig(
        action_noise_std=0.0,
        obs_noise_std=0.01,
    ),
)

# ===========================================================================
# Task Type 8: Joint Injury (per-joint actuator power cost)
#
# Definition:
#     Penalize the physical usage of a specific joint or joint chain via
#     actuator power |force| * |velocity|.  
#
# Simulates a pianist's joint injury: the injured joints can still move, but
# their actuator power (|force| * |velocity|) is penalized as cost.
#
# Three injury scopes target different levels of the kinematic chain,
# testing fundamentally different compensation strategies:
#
# 8a. Wrist Injury — orientation hub (mid-chain)
#     Injured: WRJ1 + WRJ2 (2 DoF).  Realistic: carpal tunnel / tendinitis.
#     OT fingering enabled (disable_fingering_reward=True): all three injury
#     tasks share the same fingering convention so the injury cost is the
#     only constraint shaping finger choice; the agent compensates with
#     forearm translation instead of wrist rotation.
#     Tests: kinematic redundancy resolution (local rotation → global shift).
# ---------------------------------------------------------------------------
# ForElise: RH A3-E7 (4+ octaves, 399 steps).  The extreme right-hand range
# demands constant wrist adjustment.  Same piece as WristLimit (Task 1) to
# enable direct comparison: WristLimit caps action magnitude, WristInjury
# caps physical power — two orthogonal constraints on the same joint.
# ===========================================================================
register(
    id="OmniPiano-ForElise-WristInjury-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            InjuredJointPowerConstraint(
                hand="right", joint_names=("WRJ1", "WRJ2"), penalty_coef=1.0,
            ),
        ]
    ),
)

# ===========================================================================
# 8b. Thumb Injury — end effector (distal)
#     Injured: THJ1-THJ5 (5 DoF).  Realistic: De Quervain's tendinitis.
#     OT fingering enabled (disable_fingering_reward=True): the agent is free
#     to reassign keystrokes from the injured thumb to other fingers.
#     Tests: discrete task reallocation in high-dimensional action space.
# ---------------------------------------------------------------------------
# NocturneOp9No2: RH E4-D6, LH G#1-A#4 (769 steps).  The melodic line
# relies on the thumb as a stable low-note anchor; injury forces a complete
# rethinking of fingering strategy.
# ===========================================================================
register(
    id="OmniPiano-NocturneOp9No2-ThumbInjury-v0",
    base_env_name="RoboPianist-repertoire-150-NocturneOp9No2-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            InjuredJointPowerConstraint(
                hand="right",
                joint_names=("THJ1", "THJ2", "THJ3", "THJ4", "THJ5"),
                penalty_coef=1.0,
            ),
        ]
    ),
)

# ===========================================================================
# 8c. Forearm Injury — global positioning (proximal)
#     Injured: forearm_tx + forearm_ty (2 DoF).  Realistic: elbow/forearm
#     muscle strain limiting arm translation across the keyboard.
#     OT fingering enabled (disable_fingering_reward=True): consistent with
#     8a and 8b — the agent reaches distant keys by compensating with wrist
#     rotation and finger stretching, and is free to reassign keys when the
#     MIDI-specified fingering becomes unreachable under mobility loss.
#     Tests: extreme posture control under mobility loss.
# ---------------------------------------------------------------------------
# FantaisieImpromptu: RH C#4-B6 (3+ octaves, 800 steps).  Presto arpeggios
# across a wide range require constant forearm repositioning — the injury
# constraint directly conflicts with the piece's physical demands.
# ===========================================================================
register(
    id="OmniPiano-FantaisieImpromptu-ForearmInjury-v0",
    base_env_name="RoboPianist-repertoire-150-FantaisieImpromptu-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            InjuredJointPowerConstraint(
                hand="right",
                joint_names=("forearm_tx", "forearm_ty"),
                penalty_coef=1.0,
            ),
        ]
    ),
)

# ===========================================================================
# Task Type 9: Multi-Joint OT Magnitude Limits
#
# All tasks in this family enable OT fingering (disable_fingering_reward=True)
# so the agent is free to reassign keys across fingers when the constrained
# joint group becomes too expensive to use.  The safety cost is the summed
# excess magnitude over a shared threshold across a named joint group.
#
# Shared threshold for family comparability:
#     max_magnitude = 0.4, penalty_coef = 1.0
# ===========================================================================

# ===========================================================================
# 9a. Bilateral Middle Finger Limit OT
#
# Constrained group: both hands' middle-finger chains
#     RH: indices 10,11,12 = MFJ4, MFJ3, MFJ0
#     LH: indices 32,33,34 = MFJ4, MFJ3, MFJ0
#
# OT fingering lets the agent move work away from the middle fingers toward
# index/ring/thumb instead of merely shrinking the same original finger plan.
# ---------------------------------------------------------------------------
# ClairDeLune: slower, expressive two-hand texture makes it easier to observe
# fine-grained finger reassignment rather than outright failure from speed.
# ===========================================================================
register(
    id="OmniPiano-ClairDeLune-BimanualMiddleFingerLimitOT-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            MultiJointSharedMagnitudeConstraint(
                indices=(10, 11, 12, 32, 33, 34),
                max_magnitude=0.4,
                penalty_coef=1.0,
                group_name="bimanual_middle_finger",
            ),
        ]
    ),
)

# ===========================================================================
# 9b. Bilateral Wrist + Middle Finger Limit OT
#
# Constrained group:
#     RH wrist 0,1 + RH middle 10,11,12
#     LH wrist 22,23 + LH middle 32,33,34
#
# This is a stronger variant: the agent cannot rely on the natural
# wrist+middle-finger combination and must re-plan both finger assignment and
# local posture under OT fingering.
# ---------------------------------------------------------------------------
# MapleLeafRag: dense, interleaved two-hand passages magnify the cost of
# staying with the original comfortable wrist/middle-finger strategy.
# ===========================================================================
register(
    id="OmniPiano-MapleLeafRag-BimanualWristMiddleLimitOT-v0",
    base_env_name="RoboPianist-repertoire-150-MapleLeafRag-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            MultiJointSharedMagnitudeConstraint(
                indices=(0, 1, 10, 11, 12, 22, 23, 32, 33, 34),
                max_magnitude=0.4,
                penalty_coef=1.0,
                group_name="bimanual_wrist_middle",
            ),
        ]
    ),
)

# ===========================================================================
# 9c. Left Wrist + Middle Finger Limit OT
#
# Constrained group: LH wrist 22,23 + LH middle 32,33,34
#
# This asymmetric version tests whether the policy shifts more complex work
# to the healthier right hand when only the left-hand wrist+middle chain is
# restricted.  OT fingering exposes that cross-hand redistribution option.
# ---------------------------------------------------------------------------
# NocturneOp9No2: naturally asymmetric melody/accompaniment structure makes
# left-hand impairment and right-hand compensation easy to interpret.
# ===========================================================================
register(
    id="OmniPiano-NocturneOp9No2-LeftWristMiddleLimitOT-v0",
    base_env_name="RoboPianist-repertoire-150-NocturneOp9No2-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            MultiJointSharedMagnitudeConstraint(
                indices=(22, 23, 32, 33, 34),
                max_magnitude=0.4,
                penalty_coef=1.0,
                group_name="left_wrist_middle",
            ),
        ]
    ),
)

# ===========================================================================
# Task Type 10: Summed-Magnitude OT Budgets
#
# Definition:
#     Penalize a coordinated joint chain only when the summed normalized
#     command magnitude across the whole group exceeds a shared total budget.
#     This is a chain-level budget benchmark rather than a per-joint ceiling.
#
# This family limits the TOTAL normalized action magnitude of a coordinated
# joint chain, rather than capping each joint independently.  OT fingering is
# enabled for all tasks so the policy can redistribute notes away from the
# constrained chain instead of only shrinking the same original finger plan.
#
# Budget design:
#     max_summed_magnitude = x * len(indices)
#     penalty_coef = 1.0
#
# This keeps the average allowed magnitude per joint at x while still
# allowing within-group trade-offs.
# ===========================================================================

# ===========================================================================
# 10a. Bimanual Thumb Budget OT
#
# Constrained group:
#     RH thumb 2,3,4,5,6
#     LH thumb 24,25,26,27,28
#
# Thumb-under and thumb-led anchor motions are central to piano technique, so
# limiting only the thumbs is the cleanest summed-budget test of finger
# reassignment under OT fingering.
# ---------------------------------------------------------------------------
# NocturneOp9No2: melody + accompaniment structure gives the policy room to
# redistribute work away from the thumbs while preserving musical continuity.
# ===========================================================================
register(
    id="OmniPiano-NocturneOp9No2-BimanualThumbBudgetOT-v0",
    base_env_name="RoboPianist-repertoire-150-NocturneOp9No2-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            MultiJointSummedMagnitudeConstraint(
                indices=(2, 3, 4, 5, 6, 24, 25, 26, 27, 28),
                max_summed_magnitude=4.0,
                penalty_coef=1.0,
                group_name="bimanual_thumb_budget",
            ),
        ]
    ),
)

# ===========================================================================
# 10b. Bimanual Wrist + Thumb Budget OT
#
# Constrained group:
#     RH wrist 0,1 + RH thumb 2,3,4,5,6
#     LH wrist 22,23 + LH thumb 24,25,26,27,28
#
# Wrist and thumb form a realistic piano motion chain: wrist adjustment and
# thumb-under technique frequently work together in scales, arpeggios, and
# passagework.  A summed budget models limited total authority over that chain.
# ---------------------------------------------------------------------------
# FantaisieImpromptu: rapid arpeggios and thumb-led repositioning make this a
# high-pressure test of OT reassignment plus local posture replanning.
# ===========================================================================
register(
    id="OmniPiano-FantaisieImpromptu-BimanualWristThumbBudgetOT-v0",
    base_env_name="RoboPianist-repertoire-150-FantaisieImpromptu-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            MultiJointSummedMagnitudeConstraint(
                indices=(0, 1, 2, 3, 4, 5, 6, 22, 23, 24, 25, 26, 27, 28),
                max_summed_magnitude=5.6,
                penalty_coef=1.0,
                group_name="bimanual_wrist_thumb_budget",
            ),
        ]
    ),
)

# ===========================================================================
# 10c. Bimanual Thumb + Little Finger Budget OT
#
# Constrained group:
#     RH thumb 2,3,4,5,6 + RH little 16,17,18,19
#     LH thumb 24,25,26,27,28 + LH little 38,39,40,41
#
# Thumb and little finger define span-heavy octave/chord technique.  Limiting
# their total budget tests whether the policy can avoid over-relying on the
# classic thumb-fifth-finger span strategy and instead reassign notes inward.
# ---------------------------------------------------------------------------
# PolonaiseOp53: octave/chord-heavy writing makes thumb-little coordination a
# core mechanical demand, so this budget directly conflicts with the piece.
# ===========================================================================
register(
    id="OmniPiano-PolonaiseOp53-BimanualThumbLittleBudgetOT-v0",
    base_env_name="RoboPianist-repertoire-150-PolonaiseOp53-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    safety_config=SafetyConfig(
        constraints=[
            MultiJointSummedMagnitudeConstraint(
                indices=(2, 3, 4, 5, 6, 16, 17, 18, 19, 24, 25, 26, 27, 28, 38, 39, 40, 41),
                max_summed_magnitude=7.2,
                penalty_coef=1.0,
                group_name="bimanual_thumb_little_budget",
            ),
        ]
    ),
)

# ===========================================================================
# Task Type 10: Three-Hand Pianist (morphological prototype)
#
# Definition:
#     A third hand is attached near the keyboard center alongside the canonical
#     right/left pair, giving the agent 3x ShadowHands (66 hand actuators total
#     = 3 hands x 22 per Shadow Hand). The piano's sustain pedal is a separate
#     actuator owned by the piano model (not any hand), so the full env
#     action_spec is 66 + 1 = 67; it stays 1 regardless of hand count.
#     `disable_fingering_reward=True` forces the OT (optimal-transport) reward
#     path, which matches all 5 * N_hands = 15 fingertips to the currently
#     required keys via Hungarian assignment — so the third hand is directly
#     reward-constrained (fingertips are pulled toward keys; forearm-collision
#     and energy penalties also enumerate all N*(N-1)/2 hand pairs / N hands).
#
# The three hands are labeled by logical `group` (treble / middle / bass) as
# pure metadata for a future multi-agent wrapper; `group` is NOT read by the
# reward or observation pipeline today.
#
# Same piece as Task 8c (FantaisieImpromptu): a presto-arpeggio virtuoso work
# with wide range and fast voice-changes — rich enough that a third hand can
# plausibly help the OT assignment find lower-cost matchings.
# ===========================================================================
register(
    id="OmniPiano-FantaisieImpromptu-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-FantaisieImpromptu-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=(
        HandSpec(
            name="rh",
            side=HandSide.RIGHT,
            position=(0.4, 0.30, 0.13),
            group="treble",
        ),
        HandSpec(
            name="lh",
            side=HandSide.LEFT,
            position=(0.4, -0.30, 0.13),
            group="bass",
        ),
        HandSpec(
            name="rh_c",
            side=HandSide.RIGHT,
            position=(0.4, 0.0, 0.13),
            group="middle",
        ),
    ),
)

# ForElise 3-hand companion to ThreeHandPrototype above. Same 3-hand
# morphology (treble / middle / bass) + OT fingering (annotation PIG data
# covers only 2 hands, so the third hand must be OT-supervised). Shorter
# and easier piece than FantaisieImpromptu — kept as a smoke test for
# the 3-hand pipeline and as an empirical baseline showing that on a
# genuinely 2-hand piece the 3rd hand stays vestigial under plain OT
# (companion to WinterWind below where the wide range can plausibly
# engage all three).
register(
    id="OmniPiano-ForElise-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=(
        HandSpec(
            name="rh",
            side=HandSide.RIGHT,
            position=(0.4, 0.30, 0.13),
            group="treble",
        ),
        HandSpec(
            name="lh",
            side=HandSide.LEFT,
            position=(0.4, -0.30, 0.13),
            group="bass",
        ),
        HandSpec(
            name="rh_c",
            side=HandSide.RIGHT,
            position=(0.4, 0.0, 0.13),
            group="middle",
        ),
    ),
)

# Chopin Étude Op.25 No.11 ("Winter Wind") — short piece with extremely
# wide pitch range, intended as the "3-hand can plausibly help" target:
#   - Duration ~15.8s (shortest among PIG entries with range >= 65 semitones,
#     even shorter than ForElise's 19.9s; episode length 314 steps).
#   - Pitch range 69 semitones (MIDI 32-101, ≈ 5.75 octaves) — naturally
#     splits across bass / middle / treble.
#   - 261 notes / 15.8s ≈ 16.5 notes/sec — dense enough that |K_t| often
#     reaches 4-6 simultaneous fingertip targets, so OT can usefully
#     spread assignment across 3 hands rather than always picking the
#     closest 2.
# Same 3-hand morphology + OT fingering as the ForElise variant above;
# the only difference is the piece. Pair the two runs to ablate
# "wide-range repertoire helps 3rd-hand engagement under plain OT".
register(
    id="OmniPiano-WinterWind-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=(
        HandSpec(
            name="rh",
            side=HandSide.RIGHT,
            position=(0.4, 0.30, 0.13),
            group="treble",
        ),
        HandSpec(
            name="lh",
            side=HandSide.LEFT,
            position=(0.4, -0.30, 0.13),
            group="bass",
        ),
        HandSpec(
            name="rh_c",
            side=HandSide.RIGHT,
            position=(0.4, 0.0, 0.13),
            group="middle",
        ),
    ),
)
