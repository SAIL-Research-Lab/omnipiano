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
from OmniPiano.tasks.hand_spec import (
    HandSpec,
    default_four_hand_specs,
    default_five_hand_specs,
)
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
# Repertoire choice — Chopin Étude Op.25 No.11 ("Winter Wind"):
#   - Duration ~15.8s (short enough for fast iteration; episode length 314 steps).
#   - Pitch range 69 semitones (MIDI 32-101, ~5.75 octaves) — naturally splits
#     across bass / middle / treble registers.
#   - 261 notes / 15.8s ~16.5 notes/sec — dense enough that |K_t| often reaches
#     4-6 simultaneous fingertip targets, so OT can usefully spread assignment
#     across 3 hands rather than always picking the closest 2.
#
# Earlier prototypes also registered FantaisieImpromptu (too long for fast
# iteration on this morphology) and ForElise (a genuinely 2-hand piece — 3rd
# hand empirically stayed vestigial under plain OT). Both removed; WinterWind
# kept as the single canonical 3-hand task.
# ===========================================================================
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

# ForElise variant of the 3-hand prototype. Originally removed when
# pruning 3-hand registrations (ForElise's 2-hand-native repertoire makes
# the 3rd hand stay vestigial under plain OT). RESTORED here because the
# trained checkpoint at examples/logs/sac_3hand_forelise_1/best_model.zip
# (SAC 5M, F1 ≈ 0.53 — substantially higher than 3-hand WinterWind's
# 0.31) is the cleanest 3-hand demo for the README morphology ladder.
# Same hand_specs layout as WinterWind 3-hand for cross-piece
# comparability.
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

# ===========================================================================
# Task 11 (preview): 4-hand prototypes — two (LH, RH) duet pairs.
#
# Spatial layout L-R-L-R alternating along the keyboard (y = -0.45 / -0.15
# / +0.15 / +0.45), mirroring real piano-four-hands practice: Secondo
# (bass-side player) contributes the left pair (LH bass + RH mid-bass);
# Primo (treble-side player) contributes the right pair (LH mid-treble +
# RH treble). Inner two hands coincide with the default 2-hand positions,
# so this layout reads as "the 2-hand baseline split into two duet pairs
# each ~0.30 m wide". Logical groups: bass / mid_bass / mid_treble /
# treble — pure metadata for future multi-agent wrappers; not consumed
# by reward or observation pipelines today.
#
# Same OT-fingering setup as the 3-hand variant (PIG annotation only covers
# 2 hands, so any N>2 morphology MUST use OT). Two pieces selected to test
# distinct hypotheses about when extra hands help:
#
#   * ``WinterWind`` (Chopin Étude Op.25 No.11) — extends the 3-hand ladder
#     to 4-hand on the same piece, isolating "one more hand" from "different
#     repertoire". Wide pitch range (69 semitones) gives the extra hand a
#     plausibly useful role.
#   * ``LaCampanella`` (Liszt) — chosen specifically because human virtuosos
#     widely recognize this étude as approaching "two hands aren't enough":
#     extreme pitch jumps, fast repeated-note ornamentation, sustained chord
#     + melody texture in the same passage. Strong candidate for "4 hands
#     genuinely beat 3" — if the policy can't show clear gains here, the
#     extra hand is unlikely to help anywhere in PIG-150.
#
# Earlier prototypes also registered FantaisieImpromptu (too long for fast
# iteration; the 4/5-hand ladder needs more wall-clock per run) and ForElise
# (a genuinely 2-hand piece — extra hands stayed vestigial under plain OT
# even at 3-hand, so 4/5-hand re-running this control would be redundant).
# Both removed.
#
# See ``OmniPiano.tasks.hand_spec.default_four_hand_specs`` for the exact
# layout (positions, groups, naming).
# ===========================================================================
register(
    id="OmniPiano-WinterWind-FourHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_four_hand_specs(),
)
register(
    id="OmniPiano-LaCampanella-FourHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-LaCampanella-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_four_hand_specs(),
)

# ===========================================================================
# Task 12 (preview): 5-hand prototypes — extend FourHandPrototype with one
# center hand (rh_c) inserted at y=0.
#
# Spacing tightens from 0.30 (used by 2/3/4-hand) to 0.20: at 0.30 the
# outermost hand would push to ±0.60, past the playable keyboard edge.
# Shadow Hand mesh width (~0.10 m) still fits comfortably at 0.20 spacing
# without inter-hand mesh collisions. Center hand uses HandSide.RIGHT for
# consistency with the existing 3-hand ``rh_c``.
#
# Repertoire selection: WinterWind + Pictures-At-An-Exhibition-GreatKiev.
# Picked from a full PIG-150 scan that bucketed every piece into the 5
# spatial regions used by ``default_five_hand_specs`` and ranked by
# (a) all-5-bucket coverage, (b) frequency of ≥3-bucket simultaneous
# activity, (c) polyphony density, (d) length:
#   * WinterWind (Chopin Étude Op.25 No.11) — 314 steps, all 5 buckets
#     non-empty (min 5.1%), 42% of steps with ≥3 buckets active. Shared
#     with 3/4-hand morphology ladder for clean "5 vs 4 hands" delta.
#   * PicturesAtAnExhibitionGreatKiev (Mussorgsky) — 720 steps but the
#     ONLY PIG-150 piece that combines (i) all 5 buckets ≥7%, (ii) 51%
#     ≥3-bucket activity, (iii) 11% ≥4-bucket activity, (iv) polyphony
#     mean 3.78 / max 8. Genuinely "5 hands' worth" of material.
#
# The original LaCampanella choice was dropped: its lowest pitch is key
# 30, leaving the bass bucket (keys 0-17) completely empty — the leftmost
# 5-hand slot would be permanently idle, masking the policy quality.
#
# See ``OmniPiano.tasks.hand_spec.default_five_hand_specs`` for the exact
# layout (positions, groups, naming).
# ===========================================================================
register(
    id="OmniPiano-WinterWind-FiveHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_five_hand_specs(),
)
register(
    id="OmniPiano-PicturesGreatKiev-FiveHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_five_hand_specs(),
)


# ===========================================================================
# Task 13: Level-1 (static partition) variants of the 4-hand morphology.
#
# Identical to ``OmniPiano-WinterWind-FourHandPrototype-v0`` (same OT reward,
# same hand attach positions, same MIDI) EXCEPT each hand carries a
# ``key_range`` that hard-clamps its forearm_tx slider to a non-overlapping
# slice of the 88-key piano. This is the "Level 1" rung of the multi-hand
# benchmark: the OT-only Level-3 baseline learns to park the outer hands at
# the keyboard edges (no per-hand gradient signal under pure OT — see the
# diagnosis in three_hand_design notes), so Level-1 ablates that failure by
# physically forbidding outer hands from reaching the central registers.
#
# Boundaries: bass / mid-bass / mid-treble / treble, each ~22 keys (~2
# octaves), aligned with classical piano-four-hands Primo/Secondo practice.
# Inner pair (rh_t2 / lh_b2) coincides with the default 2-hand attach
# positions, so Level-1 reads as "2-hand baseline + 2 outer hands each
# constrained to its own register".
# ===========================================================================
register(
    id="OmniPiano-WinterWind-FourHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=(
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.45, 0.13), key_range=(0, 21),
                 group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.15, 0.13), key_range=(22, 43),
                 group="mid_bass"),
        HandSpec(name="lh_t", side=HandSide.LEFT,
                 position=(0.4, +0.15, 0.13), key_range=(44, 65),
                 group="mid_treble"),
        HandSpec(name="rh_t", side=HandSide.RIGHT,
                 position=(0.4, +0.45, 0.13), key_range=(66, 87),
                 group="treble"),
    ),
)
register(
    id="OmniPiano-LaCampanella-FourHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-LaCampanella-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=(
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.45, 0.13), key_range=(0, 21),
                 group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.15, 0.13), key_range=(22, 43),
                 group="mid_bass"),
        HandSpec(name="lh_t", side=HandSide.LEFT,
                 position=(0.4, +0.15, 0.13), key_range=(44, 65),
                 group="mid_treble"),
        HandSpec(name="rh_t", side=HandSide.RIGHT,
                 position=(0.4, +0.45, 0.13), key_range=(66, 87),
                 group="treble"),
    ),
)


# ===========================================================================
# Task 14: Level-1 (static partition) variant of the 5-hand morphology.
#
# Same algorithm/reward setup as the 5-hand prototypes (Task 12) EXCEPT each
# hand carries a ``key_range`` that hard-clamps its forearm_tx slider to a
# non-overlapping slice of the 88-key piano. Five evenly-keyed buckets:
#   bass     keys  0-17  (18 keys, A0-D2)
#   low_mid  keys 18-35  (18 keys, D#2-G3)
#   middle   keys 36-52  (17 keys, G#3-C5)  ← contains middle C (key 39)
#   high_mid keys 53-70  (18 keys, C#5-F#6)
#   treble   keys 71-87  (17 keys, G6-C8)
#
# Hand attach positions are set to each bucket's geometric center —
# specifically ``0.5 * (key_index_to_y(lo) + key_index_to_y(hi))``, the
# midpoint of the two endpoint key Y coordinates (NOT key_index_to_y at
# the midpoint key index, which differs by up to 12 mm because key Y
# spacing is not uniform across white/black keys). This makes each hand's
# forearm_tx joint range symmetric around its rest position. See
# ``default_five_hand_specs`` for rationale on why we use bucket-center
# positions instead of uniform 0.20 m spacing.
#
# Sides follow the L-R-L-R-R duet-extension pattern from the prototype:
#   bass duet (Secondo)  : lh_b (LEFT) + rh_b (RIGHT)
#   center solo          : lh_c (LEFT)
#   treble pair          : rh_t2 + rh_t1 (both RIGHT, stacked)
#
# WinterWind chosen as the first 5-hand partition env: shortest viable
# 5-hand piece (314 steps), shared with 3/4-hand morphology ladder, and
# all 5 buckets have ≥5% of note events under the env-loader path
# (no idle hand). PicturesGreatKiev variant is paper-quality but 720
# steps; can be added later for the demo run.
# ===========================================================================
register(
    id="OmniPiano-WinterWind-FiveHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=(
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.4817, 0.13), key_range=(0, 17),
                 group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.2345, 0.13), key_range=(18, 35),
                 group="low_mid"),
        HandSpec(name="lh_c", side=HandSide.LEFT,
                 position=(0.4, +0.0061, 0.13), key_range=(36, 52),
                 group="middle"),
        HandSpec(name="rh_t2", side=HandSide.RIGHT,
                 position=(0.4, +0.2468, 0.13), key_range=(53, 70),
                 group="high_mid"),
        HandSpec(name="rh_t1", side=HandSide.RIGHT,
                 position=(0.4, +0.4879, 0.13), key_range=(71, 87),
                 group="treble"),
    ),
)
