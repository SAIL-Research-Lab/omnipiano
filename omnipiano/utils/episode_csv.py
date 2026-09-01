"""Shared, schema-locked episode CSV serialization.

Both online SB3 evaluation and post-hoc OmniSafe checkpoint replay must emit
the same columns.  Keeping the schema and row construction here prevents one
writer from silently omitting new benchmark metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omnipiano.benchmark.metrics import METRICS_PROTOCOL_VERSION
from omnipiano.utils.info_keys import EpisodeInfoKeys
from omnipiano.utils.json_utils import strict_json_dumps


# Preserve the pre-Metrics-v2.1 24-column prefix exactly.  Several existing
# plotting scripts read by column name, but retaining the prefix also protects
# older positional consumers while the v2.1 fields are appended.
LEGACY_EPISODE_CSV_HEADER = (
    "env_step_count",
    "episode",
    "time_elapsed",
    "ep_return",
    "ep_length",
    "ep_cost",
    "ep_violations",
    "ep_f1",
    "ep_precision",
    "ep_recall",
    "ep_sustain_f1",
    "ep_sustain_precision",
    "ep_sustain_recall",
    "energy_reward",
    "fingering_reward",
    "ot_fingering_reward",
    "forearm_reward",
    "key_press_reward",
    "sustain_reward",
    "eval_noise_scale",
    "ep_return_true",
    "ep_noise_action_l2",
    "ep_noise_obs_l2",
    "ep_noise_reward",
)

METRICS_V21_EPISODE_CSV_HEADER = (
    "metrics_protocol_version",
    "ep_note_event_precision",
    "ep_note_event_recall",
    "ep_note_event_f1",
    "ep_note_event_f1_100ms",
    "ep_note_onset_mae_ms",
    "ep_note_offset_mae_ms",
    "ep_note_duration_mae_ms",
    "ep_key_time_micro_precision",
    "ep_key_time_micro_recall",
    "ep_key_time_micro_f1",
    "ep_active_key_time_f1",
    "ep_rest_false_positive_rate",
    "ep_rest_false_positive_keys_per_second",
    "ep_effective_active_agents",
    "ep_task_idle_agent_rate",
    "ep_duplicate_agent_note_rate",
    "ep_unattributed_note_rate",
    "ep_inter_agent_collision_step_rate",
    "ep_actuator_work_joule",
    "ep_work_per_correct_event",
    "contact_attribution_available",
    "contact_attribution_valid",
    "contact_attribution_nonfinite_event_count",
    "contact_attribution_nonfinite_sample_count",
    "contact_attribution_negative_event_count",
    "contact_attribution_negative_sample_count",
    "contact_attribution_window_seconds",
    "contact_force_threshold_n",
    "note_onset_tolerance_seconds",
    "collision_force_threshold_n",
    "inter_hand_collision_metrics_available",
    "inter_hand_collision_metrics_valid",
    "inter_hand_collision_nonfinite_sample_count",
    "inter_hand_collision_negative_sample_count",
    "inter_agent_collision_metrics_available",
    "inter_agent_collision_metrics_valid",
    "inter_agent_collision_nonfinite_sample_count",
    "inter_agent_collision_negative_sample_count",
    "hand_power_available",
    "hand_power_metrics_valid",
    "hand_power_nonfinite_sample_count",
    "hand_power_negative_sample_count",
    "motor_power_threshold_watts",
    "hand_metrics_json",
    "agent_metrics_json",
    "benchmark_metrics_json",
)

EPISODE_CSV_HEADER = LEGACY_EPISODE_CSV_HEADER + METRICS_V21_EPISODE_CSV_HEADER


_INFO_KEY_BY_COLUMN = {
    "ep_f1": EpisodeInfoKeys.EPISODE_TASK_F1,
    "ep_precision": EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION,
    "ep_recall": EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL,
    "ep_sustain_f1": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_F1,
    "ep_sustain_precision": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_PRECISION,
    "ep_sustain_recall": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_RECALL,
    "energy_reward": EpisodeInfoKeys.EPISODE_TASK_ENERGY_REWARD,
    "fingering_reward": EpisodeInfoKeys.EPISODE_TASK_FINGERING_REWARD,
    "ot_fingering_reward": EpisodeInfoKeys.EPISODE_TASK_OT_FINGERING_REWARD,
    "forearm_reward": EpisodeInfoKeys.EPISODE_TASK_FOREARM_REWARD,
    "key_press_reward": EpisodeInfoKeys.EPISODE_TASK_KEY_PRESS_REWARD,
    "sustain_reward": EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_REWARD,
    "ep_note_event_precision": "episode_task/note_event_precision",
    "ep_note_event_recall": "episode_task/note_event_recall",
    "ep_note_event_f1": "episode_task/note_event_f1",
    "ep_note_event_f1_100ms": "episode_task/note_event_f1_100ms",
    "ep_note_onset_mae_ms": "episode_task/note_onset_mae_ms",
    "ep_note_offset_mae_ms": "episode_task/note_offset_mae_ms",
    "ep_note_duration_mae_ms": "episode_task/note_duration_mae_ms",
    "ep_key_time_micro_precision": "episode_task/key_time_micro_precision",
    "ep_key_time_micro_recall": "episode_task/key_time_micro_recall",
    "ep_key_time_micro_f1": "episode_task/key_time_micro_f1",
    "ep_active_key_time_f1": "episode_task/active_key_time_f1",
    "ep_rest_false_positive_rate": "episode_task/rest_false_positive_rate",
    "ep_rest_false_positive_keys_per_second": (
        "episode_task/rest_false_positive_keys_per_second"
    ),
    "ep_effective_active_agents": "episode_coordination/effective_active_agents",
    "ep_task_idle_agent_rate": "episode_coordination/task_idle_agent_rate",
    "ep_duplicate_agent_note_rate": (
        "episode_coordination/duplicate_agent_note_rate"
    ),
    "ep_unattributed_note_rate": "episode_coordination/unattributed_note_rate",
    "ep_inter_agent_collision_step_rate": (
        "episode_coordination/inter_agent_collision_step_rate"
    ),
    "ep_actuator_work_joule": "episode_physical/actuator_work_joule",
    "ep_work_per_correct_event": "episode_physical/work_per_correct_event",
    "contact_attribution_available": (
        "episode_coordination/contact_attribution_available"
    ),
    "contact_attribution_valid": "episode_coordination/contact_attribution_valid",
    "contact_attribution_nonfinite_event_count": (
        "episode_coordination/contact_attribution_nonfinite_event_count"
    ),
    "contact_attribution_nonfinite_sample_count": (
        "episode_coordination/contact_attribution_nonfinite_sample_count"
    ),
    "contact_attribution_negative_event_count": (
        "episode_coordination/contact_attribution_negative_event_count"
    ),
    "contact_attribution_negative_sample_count": (
        "episode_coordination/contact_attribution_negative_sample_count"
    ),
    "contact_attribution_window_seconds": (
        "episode_coordination/contact_attribution_window_seconds"
    ),
    "contact_force_threshold_n": "episode_coordination/contact_force_threshold_n",
    "note_onset_tolerance_seconds": "episode_task/note_onset_tolerance_seconds",
    "collision_force_threshold_n": (
        "episode_coordination/collision_force_threshold_n"
    ),
    "inter_hand_collision_metrics_available": (
        "episode_coordination/inter_hand_collision_metrics_available"
    ),
    "inter_hand_collision_metrics_valid": (
        "episode_coordination/inter_hand_collision_metrics_valid"
    ),
    "inter_hand_collision_nonfinite_sample_count": (
        "episode_coordination/inter_hand_collision_nonfinite_sample_count"
    ),
    "inter_hand_collision_negative_sample_count": (
        "episode_coordination/inter_hand_collision_negative_sample_count"
    ),
    "inter_agent_collision_metrics_available": (
        "episode_coordination/inter_agent_collision_metrics_available"
    ),
    "inter_agent_collision_metrics_valid": (
        "episode_coordination/inter_agent_collision_metrics_valid"
    ),
    "inter_agent_collision_nonfinite_sample_count": (
        "episode_coordination/inter_agent_collision_nonfinite_sample_count"
    ),
    "inter_agent_collision_negative_sample_count": (
        "episode_coordination/inter_agent_collision_negative_sample_count"
    ),
    "hand_power_available": "episode_physical/hand_power_available",
    "hand_power_metrics_valid": "episode_physical/hand_power_metrics_valid",
    "hand_power_nonfinite_sample_count": (
        "episode_physical/hand_power_nonfinite_sample_count"
    ),
    "hand_power_negative_sample_count": (
        "episode_physical/hand_power_negative_sample_count"
    ),
    "motor_power_threshold_watts": "episode_physical/motor_power_threshold_watts",
}

_REWARD_COLUMNS = (
    "energy_reward",
    "fingering_reward",
    "ot_fingering_reward",
    "forearm_reward",
    "key_press_reward",
    "sustain_reward",
)

_BENCHMARK_PREFIXES = (
    "episode_task/",
    "episode_coordination/",
    "episode_physical/",
    "episode_hand/",
    "episode_agent/",
)


def build_episode_csv_row(
    *,
    env_step_count: int,
    episode: int,
    time_elapsed: float,
    ep_return: float,
    ep_length: int,
    info: Mapping[str, Any],
    eval_noise_scale: float,
    ep_noise_action_l2: float = 0.0,
    ep_noise_obs_l2: float = 0.0,
    ep_noise_reward: float = 0.0,
    ep_cost: float | None = None,
    ep_violations: int | None = None,
    ep_return_true: float | None = None,
) -> list[Any]:
    """Build one row in :data:`EPISODE_CSV_HEADER` order."""

    values: dict[str, Any] = {
        "env_step_count": env_step_count,
        "episode": episode,
        "time_elapsed": time_elapsed,
        "ep_return": ep_return,
        "ep_length": ep_length,
        "ep_cost": (
            info.get(EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL, 0.0)
            if ep_cost is None
            else ep_cost
        ),
        "ep_violations": (
            info.get(EpisodeInfoKeys.EPISODE_SAFETY_VIOLATIONS, 0)
            if ep_violations is None
            else ep_violations
        ),
        "eval_noise_scale": eval_noise_scale,
        "ep_noise_action_l2": ep_noise_action_l2,
        "ep_noise_obs_l2": ep_noise_obs_l2,
        "ep_noise_reward": ep_noise_reward,
        "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
    }
    values.update(
        (column, info.get(info_key, ""))
        for column, info_key in _INFO_KEY_BY_COLUMN.items()
    )

    if ep_return_true is None:
        ep_return_true = sum(
            float(values[column])
            for column in _REWARD_COLUMNS
            if values[column] not in ("", None)
        )
    values["ep_return_true"] = ep_return_true

    benchmark_metrics = {
        key: value
        for key, value in info.items()
        if key.startswith(_BENCHMARK_PREFIXES)
    }
    values["hand_metrics_json"] = strict_json_dumps(
        {
            key.removeprefix("episode_hand/"): value
            for key, value in info.items()
            if key.startswith("episode_hand/")
        },
        sort_keys=True,
    )
    values["agent_metrics_json"] = strict_json_dumps(
        {
            key.removeprefix("episode_agent/"): value
            for key, value in info.items()
            if key.startswith("episode_agent/")
        },
        sort_keys=True,
    )
    values["benchmark_metrics_json"] = strict_json_dumps(
        benchmark_metrics,
        sort_keys=True,
    )

    missing = [column for column in EPISODE_CSV_HEADER if column not in values]
    unexpected = sorted(set(values) - set(EPISODE_CSV_HEADER))
    if missing or unexpected:
        raise AssertionError(
            "episode CSV serializer/schema mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    row = [values[column] for column in EPISODE_CSV_HEADER]
    if len(row) != len(EPISODE_CSV_HEADER):
        raise AssertionError("episode CSV header/row length mismatch")
    return row
