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
    default_three_hand_specs,
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
# Repertoire roster (5 pieces; see inline per-registration empirical
# numbers from the 2026-05 PIG-150 rerun under the canonical 29/30/29
# partition):
#   - WinterWind (Chopin Étude Op.25 No.11) — ANCHOR. 314 steps,
#     eq3=0%. Shared with 4/5-hand on the same MIDI for clean
#     "one more hand" delta. Short enough for fast iteration.
#   - ForElise (Beethoven) — PEDAGOGICAL NEGATIVE EXAMPLE. Prototype only,
#     no StaticPartition. 399 steps, eq3=0%, 79% middle bucket — a
#     genuinely 2-hand piece where the 3rd hand stays vestigial under
#     plain OT. Demonstrates the OT winner-takes-all pathology that
#     motivates Level-1 partition.
#   - PicturesGreatKiev (Mussorgsky) — CROSS-LADDER STRESS. 720 steps,
#     eq3=34.9% ⭐ (highest in PIG-150). Shared with 4/5-hand. The
#     "genuinely needs all 3 hands" extreme.
#   - PolonaiseOp40No1 (Chopin) — HIGH-eq3 MEDIUM LENGTH. 563 steps,
#     eq3=26.8%. Trade-off between PicturesGreatKiev (longer) and
#     WinterWind (no simultaneity).
#   - PianoSonataNo281StMov (Mozart K.545) — SHORT + HIGH eq3.
#     301 steps (shortest top-tier), eq3=22.6%. Use for many-seed
#     ablations / fast iteration when WinterWind's eq3=0% is too easy.
#
# Earlier prototypes also registered FantaisieImpromptu (too long for fast
# iteration) and PianoSonataNo301StMov (3-hand registration dropped in
# 2026-05 — dominated by PolonaiseOp40No1 on every 3-hand metric at
# similar length; PianoSonataNo301StMov is still registered for 4-hand
# where it serves as a bucket-imbalanced control).
#
# Positions = bucket centers of the canonical 29/30/29-key partition
# (N-hand morphology axiom — see ``default_three_hand_specs``):
#   rh    at +0.4056  (treble  bucket = keys 59-87, 29 keys)
#   lh    at -0.4051  (bass    bucket = keys 0-28,  29 keys)
#   rh_c  at  0.0000  (middle  bucket = keys 29-58, 30 keys, centered)
# 3-hand Prototype and StaticPartition share these positions exactly,
# so the L-3 Prototype vs L-1 StaticPartition ablation isolates the
# partition constraint cleanly (only forearm_tx joint range differs:
# full keyboard vs. clamped to its own bucket).
#
# MIGRATION NOTE (2026-05): positions moved from ±0.30 / 0.0 →
# bucket-center values above for cross-task consistency with 4/5-hand.
# Trained checkpoints from before this migration (e.g.,
# examples/logs/sac_3hand_forelise_1/best_model.zip) are not directly
# transferable; the README demo will need a fresh training run.
# ===========================================================================
# ---------------------------------------------------------------------------
# Repertoire empirical numbers below are from a full PIG-150 scan under the
# canonical 29/30/29 3-bucket partition [(0,28),(29,58),(59,87)], computed
# via ``examples/repertoire/analyze_3hand.py`` with env-loader path +
# trim_silence=True (matches BenchmarkEnvConfig defaults). All metrics
# represent what a trained agent actually observes in the registered env.
# Notation: B0=bass / B1=middle / B2=treble; eq3 = % of steps with all 3
# buckets simultaneously active; geq2 = % of steps with ≥ 2 buckets active.
# ---------------------------------------------------------------------------

# WinterWind (Chopin Étude Op.25 No.11): canonical morphology-ladder
# ANCHOR — shared with 4-hand and 5-hand on the same MIDI for clean
# "one more hand" delta measurements.
#   B0=17.5% / B1=65.2% / B2=17.3%   (min_bkt=17.34%)
#   geq2=82.80%, eq3=0%   poly_mean=3.73, length=314 steps
# Healthy distribution across all 3 buckets but eq3=0% (3 hands never
# simultaneously active) — measures "do 3 hands help when they don't
# have to coordinate"; short length makes it the fast-iteration anchor.
register(
    id="OmniPiano-WinterWind-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_three_hand_specs(),
)

# ForElise: 3-hand pedagogical NEGATIVE EXAMPLE — registered as Prototype
# (no StaticPartition variant — see Task 10b).
#   B0=10.5% / B1=79.1% / B2=10.4%   (min_bkt=10.40%)
#   geq2=24.06%, eq3=0%   poly_mean=1.88, length=399 steps
# Genuinely 2-hand repertoire (79% of notes in the middle bucket); under
# plain OT the 3rd hand stays vestigial. Useful for: (a) demonstrating
# the OT "winner-takes-all" pathology that motivates Level-1, (b) probing
# how Level-3 policies fail to engage extra hands on 2-hand-native pieces.
register(
    id="OmniPiano-ForElise-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_three_hand_specs(),
)

# Note: PianoSonataNo301StMov was previously registered for 3-hand
# (Prototype + StaticPartition) under the claim it was "UNIQUE in PIG-150"
# for 3-bucket simultaneity (9.4% eq3 under the old 29/29/30 partition).
# The 2026-05 PIG-150 rerun under the 29/30/29 partition found
# PolonaiseOp40No1 (eq3=26.8%, score 197) dominates Sonata No.30
# (eq3=10.3%, score 118) on every 3-hand metric at the same length
# range. 3-hand registrations dropped; PianoSonataNo301StMov is still
# registered for 4-hand (where its bucket-imbalanced distribution serves
# as a useful control / repertoire-diversity piece — see Task 11).

# PicturesGreatKiev (Mussorgsky, Pictures at an Exhibition: The Great
# Gate of Kiev): 3-hand CROSS-LADDER STRESS-TEST — shared with 4-hand
# and 5-hand on the same MIDI as a second cross-cutting anchor
# alongside WinterWind.
#   B0=13.6% / B1=55.1% / B2=31.3%   (min_bkt=13.60%)
#   geq2=75.83%, eq3=34.86% ⭐  poly_mean=3.78, length=720 steps
# Rank #1 / 39 filter-passers — highest 3-hand score (208.5) and
# extreme eq3=34.9% (next-best is PolonaiseOp40No1 at 26.8%). Genuinely
# requires all 3 hands at once in over a third of all steps. The 720-
# step length is the trade-off; pair with WinterWind for short runs.
register(
    id="OmniPiano-PicturesGreatKiev-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_three_hand_specs(),
)

# PolonaiseOp40No1 (Chopin, "Military" Polonaise A-major): 3-hand
# HIGH-eq3 ALTERNATIVE — second-best 3-hand piece by score in PIG-150.
#   B0=19.3% / B1=69.1% / B2=11.6%   (min_bkt=11.55%)
#   geq2=80.11%, eq3=26.82%   poly_mean=4.81, length=563 steps
# Rank #2 / 39 (score 197.0). High polyphony (4.81 — highest among
# 3-hand top-tier) and very high eq3. Use as a "less than PicturesGreatKiev
# but cheaper at 563 vs 720 steps" intermediate stress-test.
register(
    id="OmniPiano-PolonaiseOp40No1-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-PolonaiseOp40No1-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_three_hand_specs(),
)

# PianoSonataNo281StMov (Mozart, K.545 "Facile", 1st mvt): 3-hand
# SHORT-FAST-ITERATION ALTERNATIVE — shortest top-tier 3-hand piece in
# PIG-150 at 301 steps (4% shorter than WinterWind's 314).
#   B0=28.0% / B1=65.3% / B2=6.7%   (min_bkt=6.70%)
#   geq2=87.71%, eq3=22.59%   poly_mean=4.27, length=301 steps
# Rank #4 / 39 (score 172.8). High eq3 (22.6%) at shorter-than-WinterWind
# length. Use for: (a) ablations needing many seeds per piece, (b)
# probing whether 3-hand learning generalizes across short pieces, (c)
# a Mozart-K.545 (foundational repertoire) reference point.
register(
    id="OmniPiano-PianoSonataNo281StMov-ThreeHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-PianoSonataNo281StMov-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_three_hand_specs(),
)

# ===========================================================================
# Task 10b: Level-1 (static partition) variants of the 3-hand morphology.
#
# Identical to the 3-hand Prototype registrations above (same OT reward,
# same hand attach positions, same MIDI) EXCEPT each hand carries a
# ``key_range`` that hard-clamps its forearm_tx slider to a non-overlapping
# slice of the 88-key piano. Same Level-1 motivation as 4/5-hand: the
# OT-only Level-3 baseline can't propagate per-hand gradient signal in
# multi-hand settings, so static partition physically forbids outer hands
# from idling at the keyboard edges.
#
# Canonical 3-bucket partition (29/30/29 keys):
#   bass     keys  0-28  (29 keys)   ← lh
#   middle   keys 29-58  (30 keys, contains middle C = key 39, centered at y=0)
#   treble   keys 59-87  (29 keys)   ← rh
# Hand attach positions exactly match the bucket centers (see
# ``default_three_hand_specs``). Bucket scheme 29/30/29 chosen over
# alternatives 29/29/30, 30/29/29, 30/28/30 because middle bucket
# centers exactly at y=0 AND middle bucket is the widest at 30 keys,
# aligning with the bass/middle/treble register density of typical piano
# repertoire.
#
# StaticPartition pieces (4 total): WinterWind + PicturesGreatKiev +
# PolonaiseOp40No1 + PianoSonataNo281StMov.
# All share the 29/30/29 partition; the only differences across registrations
# are the MIDI and (downstream) the per-bucket empirical activity.
# Empirical numbers (B0/B1/B2 distribution, eq3, length) are documented
# inline at the corresponding Prototype registrations in Task 10 above —
# the StaticPartition variants share those numbers exactly (same MIDI,
# same buckets, only the forearm_tx joint range differs).
#
# ForElise StaticPartition is intentionally NOT registered: as a genuinely
# 2-hand piece (79% of notes in the middle bucket), partitioning the
# outer hands would only enforce the already-empirical vestigiality
# rather than test the partition mechanism.
# ===========================================================================


def _three_hand_partition_specs(*, group_override=None):
    """Inline 3-hand partition specs (29/30/29 keys, positions at
    bucket centers per the N-hand morphology axiom)."""
    return (
        HandSpec(name="rh", side=HandSide.RIGHT,
                 position=(0.4, +0.4056, 0.13), key_range=(59, 87),
                 group="treble"),
        HandSpec(name="lh", side=HandSide.LEFT,
                 position=(0.4, -0.4051, 0.13), key_range=(0, 28),
                 group="bass"),
        HandSpec(name="rh_c", side=HandSide.RIGHT,
                 position=(0.4, 0.0, 0.13), key_range=(29, 58),
                 group="middle"),
    )


# WinterWind 3-hand StaticPartition — morphology-ladder anchor (314 steps,
# eq3=0%); see Task 10 WinterWind comment.
register(
    id="OmniPiano-WinterWind-ThreeHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_three_hand_partition_specs(),
)
# PicturesGreatKiev 3-hand StaticPartition — cross-ladder stress-test
# (720 steps, eq3=34.9% ⭐); see Task 10 PicturesGreatKiev comment.
register(
    id="OmniPiano-PicturesGreatKiev-ThreeHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_three_hand_partition_specs(),
)
# PolonaiseOp40No1 3-hand StaticPartition — high-eq3 alternative
# (563 steps, eq3=26.8%, poly=4.81); see Task 10 PolonaiseOp40No1 comment.
register(
    id="OmniPiano-PolonaiseOp40No1-ThreeHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-PolonaiseOp40No1-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_three_hand_partition_specs(),
)
# PianoSonataNo281StMov 3-hand StaticPartition — short fast-iteration
# (301 steps, eq3=22.6%); see Task 10 PianoSonataNo281StMov comment.
register(
    id="OmniPiano-PianoSonataNo281StMov-ThreeHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-PianoSonataNo281StMov-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_three_hand_partition_specs(),
)


# ===========================================================================
# Task 11 (preview): 4-hand prototypes — two (LH, RH) duet pairs.
#
# Spatial layout L-R-L-R alternating along the keyboard at bucket-center
# positions (y ≈ -0.4521 / -0.1527 / +0.1528 / +0.4526), mirroring real
# piano-four-hands practice: Secondo (bass-side player) contributes the
# left pair (LH bass + RH mid-bass); Primo (treble-side player) contributes
# the right pair (LH mid-treble + RH treble). Positions are derived from
# the canonical 4 × 22-key partition's bucket centers (N-hand morphology
# axiom), so Prototype and StaticPartition share the same attach geometry
# — only the forearm_tx joint range differs (full keyboard vs. clamped
# to its own bucket). Adjacent-hand spacing ~0.30 m. Logical groups:
# bass / mid_bass / mid_treble / treble — pure metadata for future
# multi-agent wrappers; not consumed by reward or observation pipelines
# today.
#
# Same OT-fingering setup as the 3-hand variant (PIG annotation only covers
# 2 hands, so any N>2 morphology MUST use OT). Repertoire empirical
# numbers below are from a full PIG-150 scan under the canonical 4-bucket
# partition [(0,21),(22,43),(44,65),(66,87)] via
# ``examples/repertoire/analyze_4hand.py`` (env-loader path +
# trim_silence=True). Notation: B0=bass / B1=mid_bass / B2=mid_treble /
# B3=treble; geq3 = % of steps with ≥ 3 buckets simultaneously active;
# eq4 = % of steps with all 4 buckets simultaneously active.
#
# Strict 4-hand quality filter (min_bkt ≥ 5%, geq3 ≥ 10%, length ≤ 800):
# **only 2 pieces** in PIG-150 pass — WinterWind and PicturesGreatKiev.
# PianoSonataNo301StMov is the best near-miss (min_bkt=4.2%, just under).
# Future replacements should similarly satisfy (a) min_bkt ≥ 5%,
# (b) geq3 ≥ 15%, with empirical numbers documented inline.
#
# Earlier prototypes also registered FantaisieImpromptu (too long for fast
# iteration; the 4/5-hand ladder needs more wall-clock per run), ForElise
# (genuinely 2-hand — extra hands stayed vestigial under plain OT even at
# 3-hand), and ``LaCampanella`` (Liszt; lowest pitch is key 30, leaving
# B0 = 0% under the canonical partition; ``lh_b`` would be permanently
# idle, masking policy quality). All dropped.
#
# See ``OmniPiano.tasks.hand_spec.default_four_hand_specs`` for the exact
# layout (positions, groups, naming).
# ===========================================================================

# WinterWind (Chopin Étude Op.25 No.11): canonical morphology-ladder
# ANCHOR — shared with 3-hand and 5-hand on the same MIDI.
#   B0=9.9% / B1=52.9% / B2=28.6% / B3=8.5%   (min_bkt=8.5%)
#   geq3=19.1%, eq4=0%   poly_mean=3.73, length=314 steps
# Passes strict 4-hand filter; rank #2 / 2 in PIG-150. Healthy across
# all 4 buckets despite eq4=0%; short length makes it the fast-iteration
# anchor.
register(
    id="OmniPiano-WinterWind-FourHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_four_hand_specs(),
)
# PianoSonataNo301StMov (Mozart K.330, 1st mvt): 4-hand
# REPERTOIRE-DIVERSITY candidate — kept despite NOT passing the strict
# filter (min_bkt=4.2% < 5% guideline). Best non-passing piece by score.
#   B0=4.2% / B1=26.9% / B2=56.8% / B3=12.1%   (min_bkt=4.20%)
#   geq3=23.8%, eq4=2.45%   poly_mean=3.59, length=571 steps
# Higher geq3 and eq4 than WinterWind, at the cost of a thin bass
# bucket. Use for: stress-testing 4-hand synchrony on a piece with
# under-utilized outer hand; documents the L-1 partition's effect on a
# bucket-imbalanced piece.
register(
    id="OmniPiano-PianoSonataNo301StMov-FourHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-PianoSonataNo301StMov-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_four_hand_specs(),
)
# PicturesGreatKiev (Mussorgsky, Pictures at an Exhibition: The Great
# Gate of Kiev): 4-hand CROSS-LADDER STRESS-TEST — shared with 3-hand
# and 5-hand on the same MIDI.
#   B0=12.1% / B1=36.2% / B2=33.2% / B3=18.6%   (min_bkt=12.06%)
#   geq3=40.0%, eq4=10.14% ⭐  poly_mean=3.78, length=720 steps
# Passes strict 4-hand filter; rank #1 / 2 in PIG-150 (score 185.3).
# The ONLY piece in PIG-150 with eq4 > 10% — genuinely needs all 4
# hands at once in over 10% of all steps. Length 720 is the trade-off;
# pair with WinterWind for short runs.
register(
    id="OmniPiano-PicturesGreatKiev-FourHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_four_hand_specs(),
)

# ===========================================================================
# Task 12 (preview): 5-hand prototypes — L-R-L-R-R alternating extension
# of the 4-hand layout, with a LEFT center hand `lh_c`.
#
# Attach positions at the geometric centers of the canonical 5-bucket
# partition (18/18/17/18/17 keys; bucket boundaries [(0,17), (18,35),
# (36,52), (53,70), (71,87)]) per the N-hand morphology axiom — see
# ``default_five_hand_specs``:
#   lh_b   at -0.4817  (bass     bucket = keys 0-17)
#   rh_b   at -0.2345  (low_mid  bucket = keys 18-35)
#   lh_c   at +0.0061  (middle   bucket = keys 36-52, centered ~y=0)
#   rh_t2  at +0.2468  (high_mid bucket = keys 53-70)
#   rh_t1  at +0.4879  (treble   bucket = keys 71-87)
#
# Adjacent-hand spacing ~0.241 m (non-uniform, reflecting non-uniform
# piano key spacing); Shadow Hand mesh ~0.10 m → ~0.14 m gap, well
# clear of inter-hand collision. Center hand is LEFT (`lh_c`), NOT
# right — with 5 hands and a 2-LH / 3-RH count, putting LH at center
# gives the bass half a natural Secondo (LH, RH) duet pair; the treble
# half is an unavoidable (rh_t2, rh_t1) stack-of-two-RHs (only one
# half can decompose cleanly under odd N).
#
# Repertoire empirical numbers below are from a full PIG-150 scan under
# the canonical 5-bucket partition [(0,17),(18,35),(36,52),(53,70),(71,87)]
# via ``examples/repertoire/analyze_5hand.py`` (env-loader path +
# trim_silence=True). Notation: B0=bass / B1=low_mid / B2=middle /
# B3=high_mid / B4=treble; geq4 = % of steps with ≥ 4 buckets active;
# eq5 = % of steps with all 5 buckets simultaneously active.
#
# Strict 5-hand quality filter (min_bkt ≥ 5%, geq4 ≥ 10%): **only 1 piece**
# in PIG-150 passes — PicturesGreatKiev. WinterWind is the only OTHER
# piece with all 5 buckets non-empty (min_bkt=5.1%) but eq4=2.2% < 10%.
# These two are the entire 5-hand repertoire; no replacement is
# available in PIG-150.
#
# LaCampanella was previously considered but DROPPED: lowest pitch is
# key 30, leaving B0 (keys 0-17) completely empty → the leftmost 5-hand
# slot would be permanently idle, masking policy quality.
#
# See ``OmniPiano.tasks.hand_spec.default_five_hand_specs`` for the exact
# layout (positions, groups, naming).
# ===========================================================================

# WinterWind (Chopin Étude Op.25 No.11): canonical morphology-ladder
# ANCHOR — shared with 3-hand and 4-hand on the same MIDI.
#   B0=7.4% / B1=21.9% / B2=46.9% / B3=18.7% / B4=5.1%   (min_bkt=5.12%)
#   geq3=42.4%, geq4=2.23%, eq5=0%   poly_mean=3.73, length=314 steps
# All 5 buckets non-empty (just barely on the outer ones). FAILS strict
# 5-hand filter on the geq4 criterion (2.2% < 10%), but remains valuable
# as the short fast-iteration anchor and as the only ≤600-step 5-hand
# piece in PIG-150.
register(
    id="OmniPiano-WinterWind-FiveHandPrototype-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_five_hand_specs(),
)
# PicturesGreatKiev (Mussorgsky, Pictures at an Exhibition: The Great
# Gate of Kiev): 5-hand FLAGSHIP — the only PIG-150 piece passing the
# strict 5-hand filter, AND the only piece scoring rank #1 on all three
# of 3/4/5-hand morphologies (cross-ladder anchor).
#   B0=12.1% / B1=10.5% / B2=34.6% / B3=35.9% / B4=7.0%   (min_bkt=6.99%)
#   geq4=11.25% ⭐, eq5=0.69%   poly_mean=3.78, poly_max=8, length=720 steps
# Passes strict 5-hand filter. The 720-step length is the trade-off;
# pair with WinterWind for short runs.
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
# Boundaries: bass / mid-bass / mid-treble / treble, each 22 keys (~2
# octaves), aligned with classical piano-four-hands Primo/Secondo practice.
# Attach positions equal the geometric center of each hand's key_range
# bucket (N-hand morphology axiom — see ``default_four_hand_specs``):
# forearm_tx joint range is symmetric around the rest pose under each
# partition, so the policy's "home posture" sits in the middle of its
# allowed slide range rather than biased to one edge. Prototype (Task 11)
# shares these positions exactly, so L-3 Prototype vs L-1 StaticPartition
# differs ONLY in the partition constraint.
# ===========================================================================
def _four_hand_partition_specs():
    """Inline 4-hand partition specs (22 keys each, positions at bucket
    centers per the N-hand morphology axiom)."""
    return (
        HandSpec(name="lh_b", side=HandSide.LEFT,
                 position=(0.4, -0.4521, 0.13), key_range=(0, 21),
                 group="bass"),
        HandSpec(name="rh_b", side=HandSide.RIGHT,
                 position=(0.4, -0.1527, 0.13), key_range=(22, 43),
                 group="mid_bass"),
        HandSpec(name="lh_t", side=HandSide.LEFT,
                 position=(0.4, +0.1528, 0.13), key_range=(44, 65),
                 group="mid_treble"),
        HandSpec(name="rh_t", side=HandSide.RIGHT,
                 position=(0.4, +0.4526, 0.13), key_range=(66, 87),
                 group="treble"),
    )


# WinterWind 4-hand StaticPartition — morphology-ladder anchor (314 steps,
# min_bkt=8.5%, geq3=19.1%); see Task 11 WinterWind comment.
register(
    id="OmniPiano-WinterWind-FourHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-EtudeOp25No11-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_four_hand_partition_specs(),
)
# PianoSonataNo301StMov 4-hand StaticPartition — diversity (571 steps,
# min_bkt=4.2%, eq4=2.45%); see Task 11 PianoSonataNo301StMov comment.
# LaCampanella was previously the diversity choice but DROPPED (4-bucket
# bass=0% → lh_b permanently idle).
register(
    id="OmniPiano-PianoSonataNo301StMov-FourHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-PianoSonataNo301StMov-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_four_hand_partition_specs(),
)
# PicturesGreatKiev 4-hand StaticPartition — cross-ladder stress-test
# (720 steps, min_bkt=12.1%, eq4=10.1% ⭐ — only piece in PIG-150 with
# eq4 > 10%); see Task 11 PicturesGreatKiev comment.
register(
    id="OmniPiano-PicturesGreatKiev-FourHand-StaticPartition-v0",
    base_env_name="RoboPianist-repertoire-150-PicturesAtAnExhibitionGreatKiev-v0",
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=_four_hand_partition_specs(),
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
