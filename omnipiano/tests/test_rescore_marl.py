"""Focused round-trip tests for immutable Metrics-v2.1 rescoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import omnipiano.benchmark.attribution_sensitivity as sensitivity_module
from omnipiano.benchmark.metrics import EpisodeTrace, compute_episode_metrics
from omnipiano.benchmark.attribution_sensitivity import (
    analyze_tree,
    trace_sensitivity_counts,
)
from omnipiano.benchmark.rescore_marl import (
    discover_final_evaluations,
    load_episode_trace,
    rescore_final_evaluation,
    rescore_tree,
)
from omnipiano.multiagent.evaluation import export_episode_trace
from omnipiano.utils.info_keys import benchmark_metrics_to_episode_info
from omnipiano.utils.json_utils import strict_json_dumps


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source_final(tmp_path: Path, *, collision_value: float = 0.0) -> Path:
    final_dir = (
        tmp_path
        / "formal"
        / "mappo"
        / "song"
        / "seed_0"
        / "final_evaluation"
    )
    audit_dir = final_dir / "audit"
    audit_dir.mkdir(parents=True)
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    force = np.zeros((4, 2, 88), dtype=np.float64)
    force[1, 1, 40] = 2.0
    power = np.zeros((4, 2), dtype=np.float64)
    collision = np.zeros((4, 2, 2), dtype=np.float64)
    collision[1, 0, 1] = collision_value
    trace = EpisodeTrace(
        target_keys=target,
        actual_keys=target,
        target_sustain=np.zeros(4, dtype=bool),
        actual_sustain=np.zeros(4, dtype=bool),
        control_timestep=0.05,
        hand_names=("left", "right"),
        hand_to_agent={"left": "a", "right": "b"},
        hand_key_ranges={"left": (0, 43), "right": (44, 87)},
        key_contact_force=force,
        hand_power=power,
        hand_collision_force=collision,
    )
    np.savez_compressed(
        audit_dir / "episode_trace.npz",
        target_keys=trace.target_keys,
        actual_keys=trace.actual_keys,
        target_sustain=trace.target_sustain,
        actual_sustain=trace.actual_sustain,
        key_contact_force=trace.key_contact_force,
        hand_power=trace.hand_power,
        hand_collision_force=trace.hand_collision_force,
        control_timestep=np.asarray(trace.control_timestep),
        hand_names=np.asarray(trace.hand_names),
    )
    (audit_dir / "episode_trace_metadata.json").write_text(
        strict_json_dumps({
            "control_timestep": 0.05,
            "hand_names": ["left", "right"],
            "hand_to_agent": {"left": "a", "right": "b"},
            "hand_key_ranges": {"left": [0, 43], "right": [44, 87]},
            "contact_attribution_window_seconds": 0.05,
            "contact_force_threshold_n": 0.0,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_metrics = benchmark_metrics_to_episode_info(
        compute_episode_metrics(trace)
    )
    old_card = {
        "algorithm": "mappo",
        "env_id": "synthetic",
        "episode_index": 0,
        "episode_steps": 4,
        "metrics_protocol_version": "2.0",
        "protocol_version": "1.1",
        "seed": 10000,
        "team_return": 123.0,
        "metrics": old_metrics,
    }
    (final_dir / "evaluation_episodes.jsonl").write_text(
        strict_json_dumps(old_card, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (final_dir / "evaluation_summary.json").write_text(
        strict_json_dumps({
            "env_id": "synthetic",
            "metrics_protocol_version": "2.0",
            "protocol_version": "1.1",
            "aggregate": {},
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_dir


def test_rescore_roundtrip_preserves_sources_and_versions_outputs(
    tmp_path: Path,
) -> None:
    source = _make_source_final(tmp_path)
    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    before = {path: _sha256(path) for path in source_files}
    destination = tmp_path / "rescored"

    result = rescore_final_evaluation(source, destination)

    after = {path: _sha256(path) for path in source_files}
    assert after == before
    card = json.loads(
        (destination / "evaluation_episodes.jsonl").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (destination / "rescore_manifest.json").read_text(encoding="utf-8")
    )
    checklist = json.loads(
        (destination / "audit" / "contact_attribution_checklist.json").read_text(
            encoding="utf-8"
        )
    )
    audit_metadata = json.loads(
        (destination / "audit" / "episode_trace_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert card["metrics_protocol_version"] == "2.1"
    assert card["team_return"] == 123.0
    assert card["metrics"]["episode_task/note_event_f1"] == 1.0
    assert result["provenance"]["source_artifacts_modified"] is False
    assert manifest["quality_gates"]["contact_attribution_valid"] is True
    assert manifest["outputs"]["manifest"] == str(
        destination.resolve() / "rescore_manifest.json"
    )
    assert checklist["sampling"]["selected_sample_size"] == 1
    assert checklist["schema_version"] == "omnipiano.contact_attribution_checklist.v2"
    assert set(manifest["output_sha256"]) == {
        "evaluation_episodes",
        "evaluation_summary",
        "audit",
        "metadata",
        "checklist",
    }
    assert audit_metadata["metrics_protocol_version"] == "2.1"
    assert not (destination / "audit" / "episode_trace.npz").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rescore_final_evaluation(source, destination)


def test_nonfinite_collision_is_explicit_quality_gate_failure(
    tmp_path: Path,
) -> None:
    source = _make_source_final(tmp_path, collision_value=float("nan"))
    destination = tmp_path / "rescored"

    result = rescore_final_evaluation(source, destination)

    card = json.loads(
        (destination / "evaluation_episodes.jsonl").read_text(encoding="utf-8")
    )
    assert result["quality_gates"]["inter_hand_collision_metrics_valid"] is False
    assert card["metrics"][
        "episode_coordination/inter_hand_collision_event_count"
    ] is None
    assert card["metrics"]["episode_task/note_event_f1"] == 1.0


def test_discovery_finds_only_final_evaluations_with_traces(tmp_path: Path) -> None:
    source = _make_source_final(tmp_path)
    (tmp_path / "unrelated" / "final_evaluation").mkdir(parents=True)
    assert discover_final_evaluations(tmp_path) == (source.resolve(),)


def test_attribution_sensitivity_separates_threshold_and_window() -> None:
    target = np.zeros((4, 88), dtype=bool)
    target[1:3, 40] = True
    force = np.zeros((4, 2, 88), dtype=np.float64)
    force[1, 0, 40] = 1e-310
    force[2, 1, 40] = 2e-6
    trace = EpisodeTrace(
        target_keys=target,
        actual_keys=target,
        target_sustain=np.zeros(4, dtype=bool),
        actual_sustain=np.zeros(4, dtype=bool),
        control_timestep=0.05,
        hand_names=("denormal", "post"),
        hand_to_agent={"denormal": "a", "post": "b"},
        key_contact_force=force,
    )

    counts = trace_sensitivity_counts(
        trace,
        thresholds_n=(0.0, 1e-6),
        windows={"onset_only": (0, 0), "onset_and_post": (0, 1)},
    )

    assert counts["onset_only"]["0"]["attributed_event_count"] == 1
    assert counts["onset_only"]["1e-06"]["attributed_event_count"] == 0
    assert counts["onset_and_post"]["1e-06"]["attributed_event_count"] == 1
    assert counts["onset_and_post"]["1e-06"]["stability_vs_reference"][
        "attribution_gained_event_count"
    ] == 1


def test_sensitivity_report_records_provenance_and_exact_window_match(
    tmp_path: Path,
) -> None:
    _make_source_final(tmp_path)
    report = analyze_tree(
        tmp_path,
        tmp_path / "sensitivity.json",
        thresholds_n=(1e-6,),
        windows={"onset_only": (0, 0)},
    )
    assert report["metrics_protocol_version"] == "2.1"
    assert report["interpretation"]["onset_only_matches_metrics_v2_1"] is True
    assert report["implementation"]["git"]["commit"]
    assert set(report["implementation"]["source_sha256"]) == {
        "attribution_sensitivity",
        "metrics",
        "rescore_marl",
        "json_utils",
    }

    shifted = analyze_tree(
        tmp_path,
        tmp_path / "sensitivity_shifted.json",
        thresholds_n=(1e-6,),
        windows={"onset_only": (0, 1)},
    )
    assert shifted["interpretation"]["onset_only_matches_metrics_v2_1"] is False


def test_sensitivity_atomic_publish_never_clobbers_concurrent_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _make_source_final(tmp_path)
    destination = tmp_path / "sensitivity_race.json"
    original_link = sensitivity_module.os.link

    def _concurrent_publish(source, target):
        Path(target).write_text("concurrent winner\n", encoding="utf-8")
        return original_link(source, target)

    monkeypatch.setattr(sensitivity_module.os, "link", _concurrent_publish)
    with pytest.raises(FileExistsError):
        analyze_tree(
            tmp_path,
            destination,
            thresholds_n=(1e-6,),
            windows={"onset_only": (0, 0)},
        )

    assert destination.read_text(encoding="utf-8") == "concurrent winner\n"
    assert not list(tmp_path.glob(".sensitivity_race.json.*.tmp"))


def test_optional_physical_channels_export_and_load_roundtrip(
    tmp_path: Path,
) -> None:
    target = np.zeros((2, 88), dtype=bool)
    trace = EpisodeTrace(
        target_keys=target,
        actual_keys=target,
        target_sustain=np.zeros(2, dtype=bool),
        actual_sustain=np.zeros(2, dtype=bool),
        control_timestep=0.05,
        hand_names=("hand",),
        hand_to_agent={"hand": "agent"},
        hand_key_ranges={"hand": (0, 87)},
    )
    files = export_episode_trace(trace, tmp_path)

    loaded = load_episode_trace(files["trace"], files["metadata"])

    assert loaded.key_contact_force is None
    assert loaded.hand_power is None
    assert loaded.hand_collision_force is None


def test_tree_rescore_failure_leaves_no_partial_destination(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "source_a"
    first = _make_source_final(first_root)
    second_root = tmp_path / "source_b"
    second = _make_source_final(second_root)
    combined = tmp_path / "combined"
    combined.mkdir()
    # Discovery is path based, so move the two synthetic formal trees under a
    # common input root without altering their files.
    (first_root / "formal").replace(combined / "formal_a")
    (second_root / "formal").replace(combined / "formal_b")
    broken_metadata = next(
        (combined / "formal_b").glob(
            "**/final_evaluation/audit/episode_trace_metadata.json"
        )
    )
    broken_metadata.unlink()
    destination = tmp_path / "transactional_output"

    with pytest.raises(FileNotFoundError):
        rescore_tree(combined, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".transactional_output.tmp-*"))
