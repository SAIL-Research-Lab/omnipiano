"""Metrics definitions and info dictionary keys for SafeRoboPianist.

This module defines the standard keys used in the `info` dictionary
to ensure consistency across wrappers and logging callbacks.

Naming convention:
- step_safety/* : Safety related metrics for a single step (costs, violations)
- robust/*      : Robustness related metrics (noise magnitudes)
- task/*        : Task specific metrics (musical performance, reward terms)
"""


class InfoKeys:
    # Safety
    STEP_SAFETY_COST_TOTAL = "step_safety/cost_total"
    STEP_SAFETY_VIOLATION_JOINT_LIMIT = "step_safety/violation_joint_limit"
    
    # Robustness
    ROBUST_NOISE_ACTION_L2 = "robust/noise_action_l2"
    ROBUST_NOISE_OBS_L2 = "robust/noise_obs_l2"
    
    # Task (Musical Metrics)
    TASK_F1 = "task/f1"
    TASK_KEY_PRECISION = "task/key_precision"
    TASK_KEY_RECALL = "task/key_recall"
    TASK_SUSTAIN_F1 = "task/sustain_f1"
    TASK_SUSTAIN_PRECISION = "task/sustain_precision"
    TASK_SUSTAIN_RECALL = "task/sustain_recall"
    
    # Task (Reward Terms)
    TASK_REWARD_ENERGY = "task/reward_energy"
    TASK_REWARD_FINGERING = "task/reward_fingering"
    TASK_REWARD_FOREARM = "task/reward_forearm"
    TASK_REWARD_KEY_PRESS = "task/reward_key_press"
    TASK_REWARD_SUSTAIN = "task/reward_sustain"

class EpisodeInfoKeys:
    """Keys for episode-level aggregated metrics (injected at done)."""
    # Safety
    EPISODE_SAFETY_COST_TOTAL = "episode_safety/cost_total"
    EPISODE_SAFETY_VIOLATIONS = "episode_safety/violations"
    
    # Task (Musical Metrics)
    EPISODE_TASK_F1 = "episode_task/f1"
    EPISODE_TASK_KEY_PRECISION = "episode_task/key_precision"
    EPISODE_TASK_KEY_RECALL = "episode_task/key_recall"
    EPISODE_TASK_SUSTAIN_F1 = "episode_task/sustain_f1"
    EPISODE_TASK_SUSTAIN_PRECISION = "episode_task/sustain_precision"
    EPISODE_TASK_SUSTAIN_RECALL = "episode_task/sustain_recall"
    
    # Task (Reward Terms)
    EPISODE_TASK_REWARD_ENERGY = "episode_task/reward_energy"
    EPISODE_TASK_REWARD_FINGERING = "episode_task/reward_fingering"
    EPISODE_TASK_REWARD_FOREARM = "episode_task/reward_forearm"
    EPISODE_TASK_REWARD_KEY_PRESS = "episode_task/reward_key_press"
    EPISODE_TASK_REWARD_SUSTAIN = "episode_task/reward_sustain"
