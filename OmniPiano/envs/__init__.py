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
from OmniPiano.configs import SafetyConfig, RobustConfig, TaskVariantConfig
from OmniPiano.safety.constraints import (
    JointMagnitudeConstraint,
    HandCollisionConstraint,
    HandCollisionForceConstraint,
    TotalActuatorPowerConstraint,
    InjuredJointPowerConstraint,
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
# Simulates a pianist's joint injury: the injured joints can still move, but
# their actuator power (|force| * |velocity|) is penalized as cost.
#
# Three injury scopes target different levels of the kinematic chain,
# testing fundamentally different compensation strategies:
#
# 8a. Wrist Injury — orientation hub (mid-chain)
#     Injured: WRJ1 + WRJ2 (2 DoF).  Realistic: carpal tunnel / tendinitis.
#     MIDI fingering retained: the agent must use the CORRECT fingers but
#     compensate with forearm translation instead of wrist rotation.
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
    task_config=TaskVariantConfig(disable_fingering_reward=True),
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
#     MIDI fingering retained: the agent must reach distant keys using the
#     correct fingers, compensating with extreme wrist rotation and finger
#     stretching instead of arm translation.
#     Tests: extreme posture control under mobility loss.
# ---------------------------------------------------------------------------
# FantaisieImpromptu: RH C#4-B6 (3+ octaves, 800 steps).  Presto arpeggios
# across a wide range require constant forearm repositioning — the injury
# constraint directly conflicts with the piece's physical demands.
# ===========================================================================
register(
    id="OmniPiano-FantaisieImpromptu-ForearmInjury-v0",
    base_env_name="RoboPianist-repertoire-150-FantaisieImpromptu-v0",
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
