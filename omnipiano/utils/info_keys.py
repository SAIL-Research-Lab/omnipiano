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

    # Reward-independent benchmark v2 musical metrics.  ``EPISODE_TASK_F1``
    # above is the historical frame-macro score and is retained only for
    # compatibility; NOTE_EVENT_F1 and KEY_TIME_MICRO_F1 are paper metrics.
    EPISODE_TASK_NOTE_EVENT_PRECISION = "episode_task/note_event_precision"
    EPISODE_TASK_NOTE_EVENT_RECALL = "episode_task/note_event_recall"
    EPISODE_TASK_NOTE_EVENT_F1 = "episode_task/note_event_f1"
    EPISODE_TASK_NOTE_EVENT_F1_100MS = "episode_task/note_event_f1_100ms"
    EPISODE_TASK_NOTE_ONSET_MAE_MS = "episode_task/note_onset_mae_ms"
    EPISODE_TASK_NOTE_OFFSET_MAE_MS = "episode_task/note_offset_mae_ms"
    EPISODE_TASK_NOTE_DURATION_MAE_MS = "episode_task/note_duration_mae_ms"
    EPISODE_TASK_KEY_TIME_MICRO_PRECISION = "episode_task/key_time_micro_precision"
    EPISODE_TASK_KEY_TIME_MICRO_RECALL = "episode_task/key_time_micro_recall"
    EPISODE_TASK_KEY_TIME_MICRO_F1 = "episode_task/key_time_micro_f1"
    EPISODE_TASK_ACTIVE_KEY_TIME_PRECISION = "episode_task/active_key_time_precision"
    EPISODE_TASK_ACTIVE_KEY_TIME_RECALL = "episode_task/active_key_time_recall"
    EPISODE_TASK_ACTIVE_KEY_TIME_F1 = "episode_task/active_key_time_f1"
    EPISODE_TASK_REST_FALSE_POSITIVE_RATE = "episode_task/rest_false_positive_rate"
    EPISODE_TASK_REST_FALSE_POSITIVE_KEYS_PER_SECOND = (
        "episode_task/rest_false_positive_keys_per_second"
    )

    # Attribution, cooperation, and physical-efficiency metrics.
    EPISODE_COORDINATION_ACTIVE_AGENT_COUNT = "episode_coordination/active_agent_count"
    EPISODE_COORDINATION_EFFECTIVE_ACTIVE_AGENTS = (
        "episode_coordination/effective_active_agents"
    )
    EPISODE_COORDINATION_AGENT_COVERAGE = "episode_coordination/agent_coverage"
    EPISODE_COORDINATION_TASK_IDLE_AGENT_RATE = (
        "episode_coordination/task_idle_agent_rate"
    )
    EPISODE_COORDINATION_MOTOR_IDLE_AGENT_RATE = (
        "episode_coordination/motor_idle_agent_rate"
    )
    EPISODE_COORDINATION_WORKLOAD_L1_MISMATCH = (
        "episode_coordination/workload_l1_mismatch"
    )
    EPISODE_COORDINATION_WORKLOAD_JS_DIVERGENCE = (
        "episode_coordination/workload_js_divergence"
    )
    EPISODE_COORDINATION_UNATTRIBUTED_NOTE_RATE = (
        "episode_coordination/unattributed_note_rate"
    )
    EPISODE_COORDINATION_DUPLICATE_HAND_NOTE_RATE = (
        "episode_coordination/duplicate_hand_note_rate"
    )
    EPISODE_COORDINATION_DUPLICATE_AGENT_NOTE_RATE = (
        "episode_coordination/duplicate_agent_note_rate"
    )
    EPISODE_COORDINATION_INTER_HAND_COLLISION_STEP_RATE = (
        "episode_coordination/inter_hand_collision_step_rate"
    )
    EPISODE_COORDINATION_INTER_HAND_COLLISION_EVENT_COUNT = (
        "episode_coordination/inter_hand_collision_event_count"
    )
    EPISODE_COORDINATION_INTER_AGENT_COLLISION_STEP_RATE = (
        "episode_coordination/inter_agent_collision_step_rate"
    )
    EPISODE_COORDINATION_INTER_AGENT_COLLISION_EVENT_COUNT = (
        "episode_coordination/inter_agent_collision_event_count"
    )
    EPISODE_PHYSICAL_ACTUATOR_WORK_JOULE = "episode_physical/actuator_work_joule"
    EPISODE_PHYSICAL_WORK_PER_CORRECT_EVENT = (
        "episode_physical/work_per_correct_event"
    )
    
    # Task (Reward Terms)
    EPISODE_TASK_ENERGY_REWARD = "episode_task/energy_reward"
    EPISODE_TASK_FINGERING_REWARD = "episode_task/fingering_reward"
    EPISODE_TASK_OT_FINGERING_REWARD = "episode_task/ot_fingering_reward"
    EPISODE_TASK_FOREARM_REWARD = "episode_task/forearm_reward"
    EPISODE_TASK_KEY_PRESS_REWARD = "episode_task/key_press_reward"
    EPISODE_TASK_SUSTAIN_REWARD = "episode_task/sustain_reward"


_TASK_METRICS = frozenset({
    "episode_steps",
    "episode_duration_seconds",
    "precision",
    "recall",
    "f1",
    "sustain_precision",
    "sustain_recall",
    "sustain_f1",
    "note_event_precision",
    "note_event_recall",
    "note_event_f1",
    "note_event_f1_100ms",
    "note_onset_tolerance_seconds",
    "note_event_tp",
    "note_event_fp",
    "note_event_fn",
    "note_onset_mae_ms",
    "note_offset_mae_ms",
    "note_duration_mae_ms",
    "key_time_micro_precision",
    "key_time_micro_recall",
    "key_time_micro_f1",
    "active_key_time_precision",
    "active_key_time_recall",
    "active_key_time_f1",
    "target_active_step_ratio",
    "rest_frame_count",
    "rest_false_positive_rate",
    "rest_false_positive_keys",
    "rest_false_positive_keys_per_second",
})

_COORDINATION_METRICS = frozenset({
    "contact_attribution_available",
    "contact_attribution_valid",
    "contact_attribution_nonfinite_event_count",
    "contact_attribution_nonfinite_sample_count",
    "contact_attribution_negative_event_count",
    "contact_attribution_negative_sample_count",
    "contact_attribution_window_seconds",
    "contact_force_threshold_n",
    "collision_force_threshold_n",
    "active_agent_count",
    "effective_active_agents",
    "agent_coverage",
    "task_idle_agent_rate",
    "motor_idle_agent_rate",
    "workload_l1_mismatch",
    "workload_js_divergence",
    "unattributed_note_rate",
    "duplicate_hand_note_rate",
    "duplicate_agent_note_rate",
    "inter_hand_collision_step_rate",
    "inter_hand_collision_event_count",
    "inter_hand_collision_force_time_integral_ns",
    "inter_hand_max_collision_force_n",
    "inter_hand_collision_metrics_available",
    "inter_hand_collision_metrics_valid",
    "inter_hand_collision_nonfinite_sample_count",
    "inter_hand_collision_negative_sample_count",
    "inter_agent_collision_step_rate",
    "inter_agent_collision_event_count",
    "inter_agent_collision_force_time_integral_ns",
    "inter_agent_max_collision_force_n",
    "inter_agent_collision_metrics_available",
    "inter_agent_collision_metrics_valid",
    "inter_agent_collision_nonfinite_sample_count",
    "inter_agent_collision_negative_sample_count",
})

_PHYSICAL_METRICS = frozenset({
    "hand_power_available",
    "hand_power_metrics_valid",
    "hand_power_nonfinite_sample_count",
    "hand_power_negative_sample_count",
    "motor_power_threshold_watts",
    "actuator_work_joule",
    "work_per_correct_event",
})

_LEGACY_METRIC_KEYS = {
    "precision": EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION,
    "recall": EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL,
    "f1": EpisodeInfoKeys.EPISODE_TASK_F1,
    "sustain_precision": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_PRECISION,
    "sustain_recall": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_RECALL,
    "sustain_f1": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_F1,
}


def benchmark_metrics_to_episode_info(metrics):
    """Map scorer-native metric names to stable terminal ``info`` keys.

    Hand and agent names are dynamic across morphologies, so their fields use
    ``episode_hand/<name>/...`` and ``episode_agent/<name>/...`` rather than an
    enum-like constant for every possible robot configuration.
    """

    result = {}
    for name, value in metrics.items():
        if name in _LEGACY_METRIC_KEYS:
            key = _LEGACY_METRIC_KEYS[name]
        elif name.startswith("hand/"):
            key = "episode_" + name
        elif name.startswith("agent/"):
            key = "episode_" + name
        elif name in _COORDINATION_METRICS:
            key = "episode_coordination/" + name
        elif name in _PHYSICAL_METRICS:
            key = "episode_physical/" + name
        elif name in _TASK_METRICS:
            key = "episode_task/" + name
        else:
            # Fail loudly when the scorer grows without assigning a namespace;
            # silently dropping a new metric would create incomplete paper logs.
            raise KeyError(f"No episode-info namespace registered for metric {name!r}")
        result[key] = value
    return result
