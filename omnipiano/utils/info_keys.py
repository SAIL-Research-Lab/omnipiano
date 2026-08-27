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
    # Per-step reward noise (signed scalar; can be negative). Matched at eval
    # (decision 11) — NOT force-zeroed. Aggregated per-episode into the eval
    # CSV `ep_noise_reward` column; ep_return_noised - ep_return_true == sum.
    ROBUST_NOISE_REWARD = "robust/noise_reward"
    # Episode-stationary physical perturbations. They are repeated in every
    # step's info so generic Gym loggers can observe them without a special
    # reset-info path; episode CSV writers keep the latest value (do not sum).
    ROBUST_NOISE_GRAVITY = "robust/noise_gravity"
    ROBUST_NOISE_CONTACT_FRICTION = "robust/noise_contact_friction"
    ROBUST_ENV_GRAVITY_Z = "robust/env_gravity_z"
    ROBUST_ENV_CONTACT_FRICTION_SLIDING = (
        "robust/env_contact_friction_sliding"
    )
    ROBUST_ENV_HAND_POSITION_L2 = "robust/env_hand_position_l2"
    ROBUST_ENV_HAND_POSITION_MAX_L2 = "robust/env_hand_position_max_l2"
    ROBUST_ENV_HAND_POSITION_OFFSETS = "robust/env_hand_position_offsets"
    
    # Task (Musical Metrics) — per-step
    # TODO: not yet emitted by any wrapper. Reserved placeholders for
    # future per-step F1/P/R logging. The values are well-defined
    # mathematically (this-step sklearn binary precision_recall_fscore_support
    # over the ground-truth vs played key set, identical to what
    # ``MidiEvaluationWrapper._compute_key_press_metrics`` computes per-step
    # before averaging into the episode metric), but the upstream wrapper
    # keeps them as local variables and only exposes the mean via
    # ``get_musical_metrics()``. To enable, MetricsWrapper would need to
    # recompute them per step from ``task.piano.activation`` + ``task._notes``
    # — useful for per-step F1 curves within an episode (where in the song
    # is the policy weakest?). Do NOT read these keys from info today —
    # they are never present.
    TASK_F1 = "task/f1"
    TASK_KEY_PRECISION = "task/key_precision"
    TASK_KEY_RECALL = "task/key_recall"
    TASK_SUSTAIN_F1 = "task/sustain_f1"
    TASK_SUSTAIN_PRECISION = "task/sustain_precision"
    TASK_SUSTAIN_RECALL = "task/sustain_recall"
    
    # Task (Reward Terms)
    TASK_ENERGY_REWARD = "task/energy_reward"
    TASK_FINGERING_REWARD = "task/fingering_reward"
    TASK_OT_FINGERING_REWARD = "task/ot_fingering_reward"
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
    EPISODE_TASK_OT_FINGERING_REWARD = "episode_task/ot_fingering_reward"
    EPISODE_TASK_FOREARM_REWARD = "episode_task/forearm_reward"
    EPISODE_TASK_KEY_PRESS_REWARD = "episode_task/key_press_reward"
    EPISODE_TASK_SUSTAIN_REWARD = "episode_task/sustain_reward"
