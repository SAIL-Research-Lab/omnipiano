from dataclasses import dataclass
from typing import Optional
from safe_robopianist.configs import SafetyConfig, RobustConfig, TaskVariantConfig
from safe_robopianist.safety.constraints import JointMagnitudeConstraint

@dataclass
class TaskSpec:
    """Specification for a registered benchmark task."""
    base_env_name: str
    safety_config: Optional[SafetyConfig] = None
    robust_config: Optional[RobustConfig] = None
    task_config: Optional[TaskVariantConfig] = None

# The official registry of SafeRoboPianist tasks
REGISTERED_TASKS = {
    # -------------------------------------------------------------------------
    # Example / Debug Tasks
    # -------------------------------------------------------------------------
    "SafeRoboPianist-debug-Twinkle-WristLimit-v0": TaskSpec(
        base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        safety_config=SafetyConfig(
            constraints=[
                # Limit Right Hand Wrist Pitch (WRJ1, index 1) magnitude to 0.5
                JointMagnitudeConstraint(index=1, max_magnitude=0.5, penalty_coef=5.0),
                # Limit Left Hand Wrist Pitch (WRJ1, index 23) magnitude to 0.5
                JointMagnitudeConstraint(index=23, max_magnitude=0.5, penalty_coef=5.0)
            ]
        ),
        robust_config=RobustConfig(
            action_noise_std=0.05,
            obs_noise_std=0.01
        ),
        task_config=TaskVariantConfig(
            left_hand_immobile=False,
            right_hand_immobile=False
        )
    ),
    "SafeRoboPianist-Twinkle-RightHandWristLimit-v0": TaskSpec(
        base_env_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        safety_config=SafetyConfig(
            constraints=[
                JointMagnitudeConstraint(index=1, max_magnitude=0.8, penalty_coef=5.0)
            ]
        )
    ),
    
    # Add more tasks here as we implement them!
}
