"""Fast tests for deterministic contact-attribution audit sampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from omnipiano.benchmark.contact_attribution_checklist import (
    STRATA_ORDER,
    build_contact_attribution_manifest,
    main,
    write_contact_attribution_manifest,
)


FIELDS = (
    "actual_event_index",
    "is_matched_correct_event",
    "attributed",
    "note_name",
    "onset_frame",
    "onset_seconds",
    "video_frame",
    "video_seconds",
    "primary_hand",
    "primary_agent",
    "contacting_hands",
    "per_hand_force_sample_sum_n",
    "attribution_event_valid",
    "contact_nonfinite_sample_count",
    "contact_negative_sample_count",
    "invalid_reason",
)


def _write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, *, matched: bool, attributed: bool) -> dict[str, object]:
    return {
        "actual_event_index": index,
        "is_matched_correct_event": matched,
        "attributed": attributed,
        "note_name": "C4",
        "onset_frame": 10 + index,
        "onset_seconds": (10 + index) * 0.05,
        "video_frame": 11 + index,
        "video_seconds": (11 + index) * 0.05,
        "primary_hand": "lh_b" if attributed else "",
        "primary_agent": "secondo" if attributed else "",
        "contacting_hands": "lh_b" if attributed else "",
        "per_hand_force_sample_sum_n": (
            json.dumps({"lh_b": 2.5}) if attributed else json.dumps({"lh_b": 0.0})
        ),
        "attribution_event_valid": True,
        "contact_nonfinite_sample_count": 0,
        "contact_negative_sample_count": 0,
        "invalid_reason": "",
    }


def test_empty_audit_has_machine_readable_no_events_status(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    _write_audit(audit, [])

    manifest = build_contact_attribution_manifest(audit, seed=7, sample_size=8)

    assert manifest["status"] == "no_eligible_events"
    assert manifest["source"]["actual_event_count"] == 0
    assert manifest["sampling"]["selected_sample_size"] == 0
    assert manifest["samples"] == []
    assert set(manifest["sampling"]["population_by_stratum"].values()) == {0}


def test_sampling_is_deterministic_and_covers_all_nonempty_strata(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    rows = []
    index = 0
    for matched, attributed in ((True, True), (True, False), (False, True), (False, False)):
        for _ in range(3):
            rows.append(_row(index, matched=matched, attributed=attributed))
            index += 1
    _write_audit(audit, rows)

    first = build_contact_attribution_manifest(audit, seed=123, sample_size=4)
    second = build_contact_attribution_manifest(audit, seed=123, sample_size=4)

    assert first == second
    assert first["status"] == "ok"
    assert first["schema_version"] == "omnipiano.contact_attribution_checklist.v2"
    assert (
        first["sampling"]["stratification"]
        == "sensor_valid_then_matched_x_attributed"
    )
    expected = set(STRATA_ORDER) - {"invalid_contact_event"}
    assert set(sample["stratum"] for sample in first["samples"]) == expected
    assert first["sampling"]["selected_by_stratum"]["invalid_contact_event"] == 0
    assert all(
        first["sampling"]["selected_by_stratum"][name] == 1
        for name in expected
    )
    assert len({sample["actual_event_index"] for sample in first["samples"]}) == 4


def test_oversized_request_selects_each_real_event_once(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    _write_audit(audit, [
        _row(3, matched=True, attributed=True),
        _row(8, matched=False, attributed=False),
    ])

    manifest = build_contact_attribution_manifest(audit, seed=0, sample_size=99)

    assert manifest["sampling"]["selected_sample_size"] == 2
    assert manifest["sampling"]["population_exhausted"] is True
    assert {sample["actual_event_index"] for sample in manifest["samples"]} == {3, 8}


def test_cli_writes_json_manifest(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    output = tmp_path / "manifest.json"
    _write_audit(audit, [_row(0, matched=True, attributed=True)])

    assert main([
        str(audit), "--seed", "5", "--sample-size", "1", "--output", str(output)
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["sampling"]["seed"] == 5
    assert payload["samples"][0]["reported_attribution"]["primary_hand"] == "lh_b"


def test_invalid_boolean_fails_loudly(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    row = _row(0, matched=True, attributed=True)
    row["attributed"] = "maybe"
    _write_audit(audit, [row])

    with pytest.raises(ValueError, match="attributed.*boolean"):
        build_contact_attribution_manifest(audit, seed=0, sample_size=1)


def test_invalid_contact_event_has_its_own_stratum(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    row = _row(0, matched=True, attributed=False)
    row.update({
        "attribution_event_valid": False,
        "attributed": "",
        "contact_nonfinite_sample_count": 1,
        "invalid_reason": "nonfinite_contact_force",
    })
    _write_audit(audit, [row])

    manifest = build_contact_attribution_manifest(audit, seed=0, sample_size=1)

    sample = manifest["samples"][0]
    assert sample["stratum"] == "invalid_contact_event"
    assert sample["attributed"] is None
    assert sample["reported_attribution"]["contact_nonfinite_sample_count"] == 1


def test_nonfinite_csv_values_are_rejected(tmp_path: Path) -> None:
    audit = tmp_path / "contact_attribution_audit.csv"
    row = _row(0, matched=True, attributed=True)
    row["onset_seconds"] = "NaN"
    _write_audit(audit, [row])
    with pytest.raises(ValueError, match="onset_seconds.*finite"):
        build_contact_attribution_manifest(audit, seed=0, sample_size=1)

    row = _row(0, matched=True, attributed=True)
    row["per_hand_force_sample_sum_n"] = '{"lh_b": NaN}'
    _write_audit(audit, [row])
    with pytest.raises(ValueError, match="not valid JSON"):
        build_contact_attribution_manifest(audit, seed=0, sample_size=1)


def test_manifest_writer_is_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_contact_attribution_manifest({"undefined": float("nan")}, output)
    assert output.read_text(encoding="utf-8") == '{\n  "undefined": null\n}\n'
    assert json.loads(
        output.read_text(encoding="utf-8"),
        parse_constant=lambda token: pytest.fail(f"non-RFC token {token}"),
    ) == {"undefined": None}
