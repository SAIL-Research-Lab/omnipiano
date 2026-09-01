"""Synthetic semantic tests for Benchmark Metrics Protocol v2."""

from __future__ import annotations

import numpy as np
import pytest

from omnipiano.benchmark.metrics import (
    EpisodeTrace,
    NoteEvent,
    attribution_window_weights,
    compute_episode_metrics,
    extract_note_events,
    match_note_events,
)
from omnipiano.benchmark.aggregation import (
    aggregate_seed_scorecards,
    aggregate_seed_values,
    normalized_curve_auc,
    steps_to_threshold,
)
from omnipiano.utils.info_keys import benchmark_metrics_to_episode_info


def _trace(target, actual=None, *, sustain=None, **kwargs):
    target = np.asarray(target, dtype=bool)
    if actual is None:
        actual = target.copy()
    actual = np.asarray(actual, dtype=bool)
    steps = target.shape[0]
    if sustain is None:
        sustain = np.zeros(steps, dtype=bool)
    return EpisodeTrace(
        target_keys=target,
        actual_keys=actual,
        target_sustain=sustain,
        actual_sustain=np.asarray(sustain, dtype=bool),
        control_timestep=0.05,
        **kwargs,
    )


def test_perfect_performance_scores_one() -> None:
    target = np.zeros((12, 88), dtype=bool)
    target[2:5, 40] = True
    target[7:10, [44, 47]] = True
    metrics = compute_episode_metrics(_trace(target))
    assert metrics["note_event_f1"] == 1.0
    assert metrics["key_time_micro_f1"] == 1.0
    assert metrics["active_key_time_f1"] == 1.0
    assert metrics["rest_false_positive_rate"] == 0.0
    assert metrics["note_onset_mae_ms"] == 0.0
    assert metrics["note_offset_mae_ms"] == 0.0


def test_silent_policy_is_not_rescued_by_rest_frames() -> None:
    target = np.zeros((100, 88), dtype=bool)
    target[50, 40] = True
    actual = np.zeros_like(target)
    metrics = compute_episode_metrics(_trace(target, actual))
    # Historical macro F1 looks deceptively excellent because 99 silent frames
    # score one.  Both benchmark metrics correctly score the policy zero.
    assert metrics["f1"] == pytest.approx(0.99)
    assert metrics["note_event_f1"] == 0.0
    assert metrics["active_key_time_f1"] == 0.0
    assert metrics["key_time_micro_f1"] == 0.0


def test_rest_false_positives_are_explicitly_penalized() -> None:
    target = np.zeros((10, 88), dtype=bool)
    target[2:4, 40] = True
    actual = target.copy()
    actual[0, 10] = True
    actual[8:10, 20] = True
    metrics = compute_episode_metrics(_trace(target, actual))
    assert metrics["rest_frame_count"] == 8
    assert metrics["rest_false_positive_rate"] == pytest.approx(3 / 8)
    assert metrics["rest_false_positive_keys"] == 3
    assert metrics["note_event_fp"] == 2
    assert metrics["note_event_f1"] < 1.0


def test_event_tolerance_and_timing_error() -> None:
    target = np.zeros((20, 88), dtype=bool)
    actual = np.zeros_like(target)
    target[10:13, 40] = True
    actual[11:15, 40] = True
    metrics = compute_episode_metrics(_trace(target, actual))
    assert metrics["note_event_f1"] == 1.0  # exactly +50 ms
    assert metrics["note_onset_mae_ms"] == pytest.approx(50.0)
    assert metrics["note_offset_mae_ms"] == pytest.approx(100.0)
    assert metrics["note_duration_mae_ms"] == pytest.approx(50.0)


def test_exact_pitch_is_required_for_event_match() -> None:
    target = np.zeros((8, 88), dtype=bool)
    actual = np.zeros_like(target)
    target[2:4, 40] = True
    actual[2:4, 41] = True
    metrics = compute_episode_metrics(_trace(target, actual))
    assert metrics["note_event_tp"] == 0
    assert metrics["note_event_fp"] == 1
    assert metrics["note_event_fn"] == 1
    assert metrics["note_event_f1"] == 0.0


def test_event_extraction_preserves_rearticulation_after_gap() -> None:
    frames = np.zeros((8, 88), dtype=bool)
    frames[1:3, 40] = True
    frames[4:7, 40] = True
    assert extract_note_events(frames) == (
        NoteEvent(40, 1, 3),
        NoteEvent(40, 4, 7),
    )


def test_matcher_maximizes_cardinality_before_timing_error() -> None:
    target = (NoteEvent(40, 0, 1), NoteEvent(40, 2, 3))
    predicted = (NoteEvent(40, 1, 2), NoteEvent(40, 3, 4))
    matches = match_note_events(
        target, predicted, control_timestep=0.05, onset_tolerance_seconds=0.05
    )
    assert len(matches) == 2


def test_attribution_participation_energy_and_collision_metrics() -> None:
    target = np.zeros((8, 88), dtype=bool)
    target[1:3, 10] = True
    target[4:6, 70] = True
    forces = np.zeros((8, 2, 88), dtype=np.float64)
    forces[1, 0, 10] = 2.0
    forces[4, 1, 70] = 3.0
    power = np.zeros((8, 2), dtype=np.float64)
    power[:, 0] = 2.0
    power[::2, 1] = 4.0
    collision = np.zeros((8, 2, 2), dtype=np.float64)
    collision[2:4, 0, 1] = collision[2:4, 1, 0] = 5.0
    trace = _trace(
        target,
        hand_names=("bass", "treble"),
        hand_to_agent={"bass": "secondo", "treble": "primo"},
        hand_key_ranges={"bass": (0, 43), "treble": (44, 87)},
        key_contact_force=forces,
        hand_power=power,
        hand_collision_force=collision,
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["agent/secondo/correct_events"] == 1
    assert metrics["agent/primo/correct_events"] == 1
    assert metrics["agent/secondo/target_recall"] == 1.0
    assert metrics["agent/primo/target_recall"] == 1.0
    assert metrics["effective_active_agents"] == pytest.approx(2.0)
    assert metrics["task_idle_agent_rate"] == 0.0
    assert metrics["workload_l1_mismatch"] == 0.0
    assert metrics["actuator_work_joule"] == pytest.approx(1.6)
    assert metrics["work_per_correct_event"] == pytest.approx(0.8)
    assert metrics["inter_agent_collision_step_rate"] == pytest.approx(0.25)
    assert metrics["inter_agent_collision_event_count"] == 1
    assert metrics["inter_agent_collision_force_time_integral_ns"] == pytest.approx(0.5)
    assert metrics["inter_agent_max_collision_force_n"] == 5.0


def test_idle_duplicate_and_unattributed_agents_are_detected() -> None:
    target = np.zeros((10, 88), dtype=bool)
    target[1:3, 10] = True
    target[5:7, 70] = True
    forces = np.zeros((10, 2, 88), dtype=np.float64)
    # Both agents contact the first onset; second onset has no attributable
    # fingertip-key contact at all.
    forces[1, :, 10] = (2.0, 1.0)
    trace = _trace(
        target,
        hand_names=("bass", "treble"),
        hand_to_agent={"bass": "secondo", "treble": "primo"},
        hand_key_ranges={"bass": (0, 43), "treble": (44, 87)},
        key_contact_force=forces,
        hand_power=np.zeros((10, 2)),
        hand_collision_force=np.zeros((10, 2, 2)),
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["duplicate_agent_note_rate"] == pytest.approx(0.5)
    assert metrics["unattributed_note_rate"] == pytest.approx(0.5)
    assert metrics["active_agent_count"] == 1
    assert metrics["effective_active_agents"] == 1.0
    assert metrics["task_idle_agent_rate"] == pytest.approx(0.5)
    assert metrics["agent/primo/correct_events"] == 0


def test_contact_attribution_uses_half_open_50ms_window() -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    forces = np.zeros((4, 2, 88), dtype=np.float64)
    forces[1, 0, 40] = 1.0
    # This large force occurs exactly at the excluded 50 ms endpoint.  An
    # inclusive two-frame implementation would incorrectly attribute the note
    # to treble instead of bass.
    forces[2, 1, 40] = 100.0
    trace = _trace(
        target,
        hand_names=("bass", "treble"),
        hand_to_agent={"bass": "secondo", "treble": "primo"},
        hand_key_ranges={"bass": (0, 43), "treble": (44, 87)},
        key_contact_force=forces,
        hand_power=np.zeros((4, 2)),
        hand_collision_force=np.zeros((4, 2, 2)),
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["agent/secondo/correct_events"] == 1
    assert metrics["agent/primo/correct_events"] == 0
    assert metrics["duplicate_agent_note_rate"] == 0.0
    assert attribution_window_weights(0.05, 0.05) == pytest.approx([0.05])
    assert attribution_window_weights(0.02, 0.05) == pytest.approx(
        [0.02, 0.02, 0.01]
    )


def test_every_scorer_metric_has_a_stable_info_namespace() -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    trace = _trace(
        target,
        hand_names=("hand",),
        hand_to_agent={"hand": "agent"},
        hand_key_ranges={"hand": (0, 87)},
        key_contact_force=np.zeros((4, 1, 88)),
        hand_power=np.zeros((4, 1)),
        hand_collision_force=np.zeros((4, 1, 1)),
    )
    info = benchmark_metrics_to_episode_info(compute_episode_metrics(trace))
    assert "episode_task/note_event_f1" in info
    assert "episode_coordination/effective_active_agents" in info
    assert "episode_hand/hand/correct_events" in info
    assert "episode_agent/agent/correct_events" in info


def test_learning_curve_auc_and_threshold() -> None:
    steps = [50_000, 100_000, 150_000]
    scores = [0.0, 0.5, 1.0]
    assert normalized_curve_auc(steps, scores) == pytest.approx(0.5)
    assert steps_to_threshold(steps, scores, 0.75) == pytest.approx(125_000)
    assert steps_to_threshold(steps, scores, 1.1) is None


def test_seed_aggregation_reports_variance_and_failures() -> None:
    stats = aggregate_seed_values([0.2, 0.4, float("nan")])
    assert stats["num_seeds"] == 3
    assert stats["num_successful_seeds"] == 2
    assert stats["failure_rate"] == pytest.approx(1 / 3)
    assert stats["mean"] == pytest.approx(0.3)
    assert stats["std"] == pytest.approx(np.sqrt(0.02))
    cards = aggregate_seed_scorecards({
        0: {"event_f1": 0.2, "reward": 1.0},
        1: {"event_f1": 0.4, "reward": 2.0},
    })
    assert cards["event_f1"]["mean"] == pytest.approx(0.3)
