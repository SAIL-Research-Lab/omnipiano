"""Synthetic semantic tests for Benchmark Metrics Protocol v2."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from omnipiano.benchmark.metrics import (
    DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    METRICS_PROTOCOL_VERSION,
    EpisodeTrace,
    NoteEvent,
    attribute_note_events,
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
from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent.evaluation import export_episode_trace
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


@pytest.mark.parametrize("bad_tolerance", [float("nan"), float("inf")])
def test_matcher_rejects_nonfinite_tolerance(bad_tolerance: float) -> None:
    with pytest.raises(ValueError, match="onset_tolerance_seconds"):
        match_note_events(
            (),
            (),
            control_timestep=0.05,
            onset_tolerance_seconds=bad_tolerance,
        )


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


def test_territory_recall_excludes_correct_events_outside_territory() -> None:
    target = np.zeros((6, 88), dtype=bool)
    target[1, 5] = True
    target[3, 40] = True
    forces = np.zeros((6, 1, 88), dtype=np.float64)
    forces[1, 0, 5] = 1.0
    forces[3, 0, 40] = 1.0
    trace = _trace(
        target,
        hand_names=("bass",),
        hand_to_agent={"bass": "secondo"},
        hand_key_ranges={"bass": (0, 10)},
        key_contact_force=forces,
        hand_power=np.zeros((6, 1)),
        hand_collision_force=np.zeros((6, 1, 1)),
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["hand/bass/correct_events"] == 2
    assert metrics["hand/bass/eligible_correct_events"] == 1
    assert metrics["hand/bass/eligible_target_events"] == 1
    assert metrics["hand/bass/target_recall"] == 1.0
    assert metrics["agent/secondo/correct_events"] == 2
    assert metrics["agent/secondo/eligible_correct_events"] == 1
    assert metrics["agent/secondo/target_recall"] == 1.0


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


def test_contact_threshold_rejects_denormals_but_keeps_real_force() -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    forces = np.zeros((4, 2, 88), dtype=np.float64)
    forces[1, 0, 40] = 1e-310
    forces[1, 1, 40] = 2.0 * DEFAULT_CONTACT_FORCE_THRESHOLD_N
    trace = _trace(
        target,
        hand_names=("denormal", "finite"),
        hand_to_agent={"denormal": "a", "finite": "b"},
        key_contact_force=forces,
        hand_power=np.zeros((4, 2)),
        hand_collision_force=np.zeros((4, 2, 2)),
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["hand/denormal/actual_events"] == 0
    assert metrics["hand/finite/actual_events"] == 1
    assert metrics["contact_attribution_valid"] == 1.0


def test_contact_threshold_is_force_not_impulse_and_primary_is_participant() -> None:
    target = np.zeros((6, 88), dtype=bool)
    target[1:4, 40] = True
    forces = np.zeros((6, 2, 88), dtype=np.float64)
    # Hand zero stays below the 1 N participant threshold for all three
    # samples, but its total impulse exceeds hand one's.  It must not win.
    forces[1:4, 0, 40] = (0.9, 1.0, 0.9)
    forces[1, 1, 40] = 1.1
    trace = EpisodeTrace(
        target_keys=target,
        actual_keys=target,
        target_sustain=np.zeros(6, dtype=bool),
        actual_sustain=np.zeros(6, dtype=bool),
        control_timestep=0.02,
        hand_names=("below", "participant"),
        hand_to_agent={"below": "a", "participant": "b"},
        key_contact_force=forces,
    )
    events = extract_note_events(target)

    attribution = attribute_note_events(
        trace,
        events,
        onset_window_seconds=0.05,
        contact_force_threshold_n=1.0,
    )

    assert attribution.participant_hand_indices == ((1,),)
    assert attribution.primary_hand_indices.tolist() == [1]


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_contact_threshold_fails_loudly(threshold: float) -> None:
    target = np.zeros((2, 88), dtype=bool)
    trace = _trace(target)
    with pytest.raises(ValueError, match="contact_force_threshold_n"):
        compute_episode_metrics(trace, contact_force_threshold_n=threshold)


@pytest.mark.parametrize("window", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_attribution_window_fails_even_without_hands(window: float) -> None:
    target = np.zeros((2, 88), dtype=bool)
    with pytest.raises(ValueError, match="attribution_window_seconds"):
        compute_episode_metrics(
            _trace(target), attribution_window_seconds=window
        )


def test_nonfinite_contact_invalidates_only_attribution_metrics() -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    forces = np.zeros((4, 2, 88), dtype=np.float64)
    forces[1, 0, 40] = np.nan
    forces[1, 1, 40] = 2.0
    trace = _trace(
        target,
        hand_names=("bad", "finite"),
        hand_to_agent={"bad": "a", "finite": "b"},
        key_contact_force=forces,
        hand_power=np.zeros((4, 2)),
        hand_collision_force=np.zeros((4, 2, 2)),
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["note_event_f1"] == 1.0
    assert metrics["contact_attribution_valid"] == 0.0
    assert metrics["contact_attribution_nonfinite_event_count"] == 1.0
    assert metrics["contact_attribution_nonfinite_sample_count"] == 1.0
    assert np.isnan(metrics["active_agent_count"])
    assert np.isnan(metrics["hand/finite/correct_events"])
    assert metrics["work_per_correct_event"] == 0.0


def test_nonfinite_contact_outside_event_window_is_irrelevant() -> None:
    target = np.zeros((5, 88), dtype=bool)
    target[1:3, 40] = True
    forces = np.zeros((5, 1, 88), dtype=np.float64)
    forces[1, 0, 40] = 1.0
    forces[4, 0, 40] = np.nan
    trace = _trace(
        target,
        hand_names=("hand",),
        hand_to_agent={"hand": "agent"},
        key_contact_force=forces,
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["contact_attribution_valid"] == 1.0
    assert metrics["hand/hand/correct_events"] == 1.0


@pytest.mark.parametrize("bad_value", [float("inf"), -1.0])
def test_invalid_relevant_contact_force_fails_attribution_gate(
    bad_value: float,
) -> None:
    target = np.zeros((3, 88), dtype=bool)
    target[1, 40] = True
    forces = np.zeros((3, 1, 88), dtype=np.float64)
    forces[1, 0, 40] = bad_value
    trace = _trace(
        target,
        hand_names=("hand",),
        hand_to_agent={"hand": "agent"},
        key_contact_force=forces,
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["contact_attribution_valid"] == 0.0
    if np.isfinite(bad_value):
        assert metrics["contact_attribution_negative_sample_count"] == 1.0
    else:
        assert metrics["contact_attribution_nonfinite_sample_count"] == 1.0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_collision_invalidates_the_complete_scope(
    bad_value: float,
) -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    collision = np.zeros((4, 2, 2), dtype=np.float64)
    collision[1, 0, 1] = bad_value
    trace = _trace(
        target,
        hand_names=("left", "right"),
        hand_to_agent={"left": "a", "right": "b"},
        key_contact_force=np.zeros((4, 2, 88)),
        hand_power=np.zeros((4, 2)),
        hand_collision_force=collision,
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["note_event_f1"] == 1.0
    for prefix in ("inter_hand_", "inter_agent_"):
        assert metrics[prefix + "collision_metrics_valid"] == 0.0
        assert metrics[prefix + "collision_nonfinite_sample_count"] == 1.0
        for suffix in (
            "collision_step_rate",
            "collision_event_count",
            "collision_force_time_integral_ns",
            "max_collision_force_n",
        ):
            assert np.isnan(metrics[prefix + suffix])


def test_collision_threshold_rejects_denormal_force() -> None:
    target = np.zeros((3, 88), dtype=bool)
    collision = np.zeros((3, 2, 2), dtype=np.float64)
    collision[1, 0, 1] = 1e-310
    trace = _trace(
        target,
        hand_names=("left", "right"),
        hand_to_agent={"left": "a", "right": "b"},
        key_contact_force=np.zeros((3, 2, 88)),
        hand_power=np.zeros((3, 2)),
        hand_collision_force=collision,
    )

    metrics = compute_episode_metrics(trace)

    assert metrics["inter_agent_collision_metrics_valid"] == 1.0
    assert metrics["inter_agent_collision_step_rate"] == 0.0
    assert metrics["inter_agent_collision_event_count"] == 0.0
    assert metrics["inter_agent_collision_force_time_integral_ns"] == 0.0
    assert metrics["inter_agent_max_collision_force_n"] == 0.0


def test_collision_at_exact_threshold_is_excluded() -> None:
    target = np.zeros((3, 88), dtype=bool)
    collision = np.zeros((3, 2, 2), dtype=np.float64)
    collision[1, 0, 1] = 1e-6
    trace = _trace(
        target,
        hand_names=("left", "right"),
        hand_to_agent={"left": "a", "right": "b"},
        key_contact_force=np.zeros((3, 2, 88)),
        hand_collision_force=collision,
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["inter_agent_collision_event_count"] == 0.0


def test_negative_collision_force_fails_the_scope_gate() -> None:
    target = np.zeros((3, 88), dtype=bool)
    collision = np.zeros((3, 2, 2), dtype=np.float64)
    collision[1, 0, 1] = -1.0
    trace = _trace(
        target,
        hand_names=("left", "right"),
        hand_to_agent={"left": "a", "right": "b"},
        key_contact_force=np.zeros((3, 2, 88)),
        hand_collision_force=collision,
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["inter_agent_collision_metrics_valid"] == 0.0
    assert metrics["inter_agent_collision_negative_sample_count"] == 1.0
    assert np.isnan(metrics["inter_agent_collision_event_count"])


def test_intra_agent_bad_collision_only_invalidates_inter_hand_scope() -> None:
    target = np.zeros((3, 88), dtype=bool)
    collision = np.zeros((3, 2, 2), dtype=np.float64)
    collision[1, 0, 1] = np.nan
    trace = _trace(
        target,
        hand_names=("left", "right"),
        hand_to_agent={"left": "same", "right": "same"},
        key_contact_force=np.zeros((3, 2, 88)),
        hand_collision_force=collision,
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["inter_hand_collision_metrics_valid"] == 0.0
    assert metrics["inter_agent_collision_metrics_valid"] == 1.0
    assert metrics["inter_agent_collision_event_count"] == 0.0


@pytest.mark.parametrize("bad_power", [float("nan"), float("inf")])
def test_nonfinite_hand_power_has_an_explicit_gate(bad_power: float) -> None:
    target = np.zeros((2, 88), dtype=bool)
    trace = _trace(
        target,
        hand_names=("hand",),
        hand_to_agent={"hand": "agent"},
        key_contact_force=np.zeros((2, 1, 88)),
        hand_power=np.asarray([[bad_power], [0.0]]),
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["hand_power_available"] == 1.0
    assert metrics["hand_power_metrics_valid"] == 0.0
    assert np.isnan(metrics["actuator_work_joule"])


def test_negative_hand_power_has_an_explicit_gate() -> None:
    target = np.zeros((2, 88), dtype=bool)
    trace = _trace(
        target,
        hand_names=("hand",),
        hand_to_agent={"hand": "agent"},
        key_contact_force=np.zeros((2, 1, 88)),
        hand_power=np.asarray([[-1.0], [0.0]]),
    )
    metrics = compute_episode_metrics(trace)
    assert metrics["hand_power_available"] == 1.0
    assert metrics["hand_power_metrics_valid"] == 0.0
    assert metrics["hand_power_negative_sample_count"] == 1.0
    assert np.isnan(metrics["actuator_work_joule"])


def test_metrics_protocol_version_is_v21() -> None:
    assert METRICS_PROTOCOL_VERSION == "2.1"
    assert BenchmarkProtocolConfig().metrics_protocol_version == (
        METRICS_PROTOCOL_VERSION
    )


def test_exported_audit_uses_same_threshold_semantics(
    tmp_path,
) -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    forces = np.zeros((4, 2, 88), dtype=np.float64)
    forces[1, 0, 40] = 1e-310
    forces[1, 1, 40] = 2.0 * DEFAULT_CONTACT_FORCE_THRESHOLD_N
    trace = _trace(
        target,
        hand_names=("denormal", "finite"),
        hand_to_agent={"denormal": "a", "finite": "b"},
        hand_key_ranges={"denormal": (0, 87), "finite": (0, 87)},
        key_contact_force=forces,
        hand_power=np.zeros((4, 2)),
        hand_collision_force=np.zeros((4, 2, 2)),
    )

    files = export_episode_trace(
        trace,
        tmp_path,
        include_trace_npz=False,
    )

    with open(files["audit"], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads(
        (tmp_path / "episode_trace_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(rows) == 1
    assert rows[0]["contacting_hands"] == "finite"
    assert rows[0]["primary_hand"] == "finite"
    assert rows[0]["attributed"] == "True"
    assert metadata["metrics_protocol_version"] == "2.1"
    assert metadata["contact_force_threshold_n"] == (
        DEFAULT_CONTACT_FORCE_THRESHOLD_N
    )
    assert metadata["contact_attribution_valid"] is True
    assert "trace" not in files


def test_exported_invalid_contact_event_is_not_plain_unattributed(
    tmp_path,
) -> None:
    target = np.zeros((3, 88), dtype=bool)
    target[1, 40] = True
    forces = np.zeros((3, 1, 88), dtype=np.float64)
    forces[1, 0, 40] = np.nan
    trace = _trace(
        target,
        hand_names=("bad",),
        hand_to_agent={"bad": "agent"},
        key_contact_force=forces,
    )

    files = export_episode_trace(trace, tmp_path, include_trace_npz=False)

    with open(files["audit"], newline="", encoding="utf-8") as stream:
        row = list(csv.DictReader(stream))[0]
    assert row["attribution_event_valid"] == "False"
    assert row["attributed"] == ""
    assert row["invalid_reason"] == "nonfinite_contact_force"
    assert row["per_hand_peak_force_n"] == ""
    assert row["per_hand_peak_force_frame"] == ""


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
