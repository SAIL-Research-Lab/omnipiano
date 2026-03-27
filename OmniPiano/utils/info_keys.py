"""Metrics definitions and info dictionary keys for OmniPiano.

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
    STEP_SAFETY_VIOLATION_ANY = "step_safety/violation_any"
    
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
    TASK_ENERGY_REWARD = "task/energy_reward"
    TASK_FINGERING_REWARD = "task/fingering_reward"
    TASK_FOREARM_REWARD = "task/forearm_reward"
    TASK_KEY_PRESS_REWARD = "task/key_press_reward"
    TASK_SUSTAIN_REWARD = "task/sustain_reward"

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
    EPISODE_TASK_ENERGY_REWARD = "episode_task/energy_reward"
    EPISODE_TASK_FINGERING_REWARD = "episode_task/fingering_reward"
    EPISODE_TASK_FOREARM_REWARD = "episode_task/forearm_reward"
    EPISODE_TASK_KEY_PRESS_REWARD = "episode_task/key_press_reward"
    EPISODE_TASK_SUSTAIN_REWARD = "episode_task/sustain_reward"
