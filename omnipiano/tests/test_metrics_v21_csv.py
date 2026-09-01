"""Fast schema gate for Metrics-v2.1 episode CSV output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from omnipiano.benchmark.metrics import METRICS_PROTOCOL_VERSION
from omnipiano.utils.episode_csv import (
    EPISODE_CSV_HEADER,
    LEGACY_EPISODE_CSV_HEADER,
    build_episode_csv_row,
)
from omnipiano.utils.logger_wrapper import SafeRecordEpisodeStatistics


_EXPECTED_LEGACY_HEADER = (
    "env_step_count", "episode", "time_elapsed", "ep_return", "ep_length",
    "ep_cost", "ep_violations", "ep_f1", "ep_precision", "ep_recall",
    "ep_sustain_f1", "ep_sustain_precision", "ep_sustain_recall",
    "energy_reward", "fingering_reward", "ot_fingering_reward",
    "forearm_reward", "key_press_reward", "sustain_reward",
    "eval_noise_scale", "ep_return_true", "ep_noise_action_l2",
    "ep_noise_obs_l2", "ep_noise_reward",
)


class _OneStepEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        info = {
            "episode_task/f1": 0.5,
            "episode_task/note_event_f1": 0.25,
            "episode_task/note_onset_tolerance_seconds": 0.05,
            "episode_coordination/contact_attribution_available": 1.0,
            "episode_coordination/contact_attribution_valid": 0.0,
            "episode_coordination/contact_attribution_nonfinite_event_count": 1.0,
            "episode_coordination/contact_attribution_nonfinite_sample_count": 1.0,
            "episode_coordination/contact_attribution_negative_event_count": 0.0,
            "episode_coordination/contact_attribution_negative_sample_count": 0.0,
            "episode_coordination/contact_attribution_window_seconds": 0.05,
            "episode_coordination/contact_force_threshold_n": 1e-6,
            "episode_coordination/collision_force_threshold_n": 1e-6,
            "episode_physical/hand_power_available": 1.0,
            "episode_physical/hand_power_metrics_valid": 1.0,
            "episode_physical/hand_power_nonfinite_sample_count": 0.0,
            "episode_physical/hand_power_negative_sample_count": 0.0,
            "episode_physical/motor_power_threshold_watts": 1e-6,
            "episode_hand/lh/correct_events": float("nan"),
            "episode_agent/secondo/correct_events": float("nan"),
        }
        return np.zeros(1, dtype=np.float32), 1.0, True, False, info


def test_metrics_v21_csv_has_aligned_quality_and_strict_json_columns(
    tmp_path,
) -> None:
    env = SafeRecordEpisodeStatistics(
        _OneStepEnv(),
        log_dir=tmp_path,
        env_id="v21",
        split="eval",
    )
    env.reset(seed=0)
    env.step(np.zeros(1, dtype=np.float32))
    env.close()

    with (tmp_path / "eval_episode_metrics_v21.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 1
    row = rows[0]
    assert None not in row
    assert row["metrics_protocol_version"] == METRICS_PROTOCOL_VERSION
    assert row["contact_attribution_valid"] == "0.0"
    assert row["contact_attribution_nonfinite_event_count"] == "1.0"
    hand_metrics = json.loads(row["hand_metrics_json"])
    benchmark_metrics = json.loads(row["benchmark_metrics_json"])
    assert hand_metrics["lh/correct_events"] is None
    assert benchmark_metrics["episode_task/note_event_f1"] == 0.25


def test_shared_schema_preserves_legacy_prefix_and_builds_aligned_rows() -> None:
    assert LEGACY_EPISODE_CSV_HEADER == _EXPECTED_LEGACY_HEADER
    assert EPISODE_CSV_HEADER[:24] == _EXPECTED_LEGACY_HEADER
    row = build_episode_csv_row(
        env_step_count=1,
        episode=1,
        time_elapsed=0.1,
        ep_return=1.0,
        ep_length=1,
        info={"episode_hand/lh/correct_events": float("nan")},
        eval_noise_scale=1.0,
    )
    assert len(row) == len(EPISODE_CSV_HEADER)
    record = dict(zip(EPISODE_CSV_HEADER, row))
    assert json.loads(record["hand_metrics_json"])["lh/correct_events"] is None


def test_checkpoint_replay_uses_the_shared_schema_and_row_builder() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "checkpoint_replay_eval.py"
    ).read_text(encoding="utf-8")
    assert "EPISODE_CSV_HEADER" in source
    assert "build_episode_csv_row" in source
    assert "_CSV_HEADER =" not in source
    assert source.index('os.environ.setdefault("MUJOCO_GL", "egl")') < source.index(
        "from omnipiano.configs import BenchmarkProtocolConfig"
    )
