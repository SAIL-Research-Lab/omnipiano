"""OmniPiano environment registration.

All benchmark tasks are registered here via register() calls.
Importing this module triggers registration (side-effect on import).
"""

from OmniPiano.envs.registration import register
from OmniPiano.configs import SafetyConfig, RobustConfig, TaskVariantConfig
from OmniPiano.safety.constraints import (
    JointMagnitudeConstraint,
    HandCollisionConstraint,
    TotalActuatorPowerConstraint,
)

# ---------------------------------------------------------------------------
# Debug / Example Tasks
# ---------------------------------------------------------------------------
register(
    id="OmniPiano-debug-Twinkle-WristLimit-v0",
    base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    safety_config=SafetyConfig(
        constraints=[
            JointMagnitudeConstraint(index=1, max_magnitude=0.5, penalty_coef=5.0),
            JointMagnitudeConstraint(index=23, max_magnitude=0.5, penalty_coef=5.0),
        ]
    ),
    robust_config=RobustConfig(action_noise_std=0.05, obs_noise_std=0.01),
    task_config=TaskVariantConfig(left_hand_immobile=False, right_hand_immobile=False),
)

# ---------------------------------------------------------------------------
# Right-Hand-WristPitch-Limit Tasks
# ---------------------------------------------------------------------------
register(
    id="OmniPiano-Twinkle-RightHandWristLimit-v0",
    base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    safety_config=SafetyConfig(
        constraints=[
            JointMagnitudeConstraint(index=1, max_magnitude=0.8, penalty_coef=5.0),
        ]
    ),
)

# ---------------------------------------------------------------------------
# Right-Hand-Only Tasks
# ---------------------------------------------------------------------------
register(
    id="OmniPiano-Twinkle-RightHandOnly-v0",
    base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    task_config=TaskVariantConfig(left_hand_immobile=True),
)

# ---------------------------------------------------------------------------
# Collision-Safe Tasks
# ---------------------------------------------------------------------------
register(
    id="OmniPiano-Twinkle-CollisionSafe-v0",
    base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    safety_config=SafetyConfig(
        constraints=[
            HandCollisionConstraint(penalty_coef=1.0),
        ]
    ),
)

# ---------------------------------------------------------------------------
# Power-Constrained Tasks
# ---------------------------------------------------------------------------
register(
    id="OmniPiano-Twinkle-PowerConstrained-v0",
    base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    safety_config=SafetyConfig(
        constraints=[
            TotalActuatorPowerConstraint(penalty_coef=1.0),
        ]
    ),
)

