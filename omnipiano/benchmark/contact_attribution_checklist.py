"""Build a deterministic manual-review manifest for contact attribution.

The input is ``audit/contact_attribution_audit.csv`` produced by OmniPiano's
multi-agent evaluation. Sensor-invalid events receive a dedicated stratum;
the remaining rows are sampled across the Cartesian product of
matched/unmatched and attributed/unattributed events so a small manual review
does not silently inspect only the common success case.

Example::

    python -m omnipiano.benchmark.contact_attribution_checklist \
        evaluation/audit/contact_attribution_audit.csv \
        --seed 0 --sample-size 12 \
        --output evaluation/audit/contact_attribution_checklist.json

The manifest intentionally contains no timestamp: identical input bytes,
seed, and sample size produce identical JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from omnipiano.utils.json_utils import strict_json_dumps


SCHEMA_VERSION = "omnipiano.contact_attribution_checklist.v2"
REQUIRED_COLUMNS = frozenset({
    "actual_event_index",
    "is_matched_correct_event",
    "attributed",
})

# Sensor-invalid events are reviewed first. The diagonal then gives both
# binary axes coverage with two samples; a complete round covers all five
# strata before any stratum receives a second sample.
STRATA_ORDER = (
    "invalid_contact_event",
    "matched_attributed",
    "unmatched_unattributed",
    "matched_unattributed",
    "unmatched_attributed",
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "y"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n"})


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token}")


def _parse_bool(value: Any, *, column: str, line_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"line {line_number}: {column!r} must be a boolean, got {value!r}"
    )


def _parse_int(value: Any, *, column: str, line_number: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: {column!r} must be an integer, got {value!r}"
        ) from exc


def _optional_int(row: Mapping[str, str], name: str, line_number: int) -> int | None:
    value = row.get(name, "").strip()
    return None if not value else _parse_int(value, column=name, line_number=line_number)


def _optional_float(
    row: Mapping[str, str], name: str, line_number: int
) -> float | None:
    value = row.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"line {line_number}: {name!r} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"line {line_number}: {name!r} must be finite, got {value!r}"
        )
    return parsed


def _optional_json_mapping(
    row: Mapping[str, str], name: str, line_number: int
) -> dict[str, Any] | None:
    value = row.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(
            value,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"line {line_number}: {name!r} is not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"line {line_number}: {name!r} must contain a JSON object")
    return parsed


def _stratum(*, matched: bool, attributed: bool | None) -> str:
    if attributed is None:
        return "invalid_contact_event"
    return (
        ("matched" if matched else "unmatched")
        + "_"
        + ("attributed" if attributed else "unattributed")
    )


def _selection_key(seed: int, stratum: str, event_index: int, line_number: int) -> str:
    """Cross-version deterministic pseudo-random ordering within one stratum."""

    token = f"{seed}\0{stratum}\0{event_index}\0{line_number}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def _read_rows(csv_path: Path) -> tuple[list[dict[str, Any]], str]:
    raw_bytes = csv_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing CSV header")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path}: missing required columns: {sorted(missing)}"
            )

        rows: list[dict[str, Any]] = []
        seen_event_indices: set[int] = set()
        for line_number, source_row in enumerate(reader, start=2):
            if None in source_row:
                raise ValueError(f"line {line_number}: row has more values than columns")
            row = {key: (value or "") for key, value in source_row.items()}
            event_index = _parse_int(
                row["actual_event_index"],
                column="actual_event_index",
                line_number=line_number,
            )
            if event_index in seen_event_indices:
                raise ValueError(
                    f"line {line_number}: duplicate actual_event_index {event_index}"
                )
            seen_event_indices.add(event_index)
            matched = _parse_bool(
                row["is_matched_correct_event"],
                column="is_matched_correct_event",
                line_number=line_number,
            )
            event_valid = (
                _parse_bool(
                    row["attribution_event_valid"],
                    column="attribution_event_valid",
                    line_number=line_number,
                )
                if "attribution_event_valid" in row
                else True
            )
            attributed = (
                _parse_bool(
                    row["attributed"],
                    column="attributed",
                    line_number=line_number,
                )
                if event_valid
                else None
            )
            rows.append({
                "source_csv_line": line_number,
                "actual_event_index": event_index,
                "matched": matched,
                "attributed": attributed,
                "attribution_event_valid": event_valid,
                "stratum": _stratum(matched=matched, attributed=attributed),
                "row": row,
            })
    return rows, digest


def _sample_payload(item: Mapping[str, Any], rank: int) -> dict[str, Any]:
    row = item["row"]
    line_number = int(item["source_csv_line"])
    contacting_hands = [
        hand for hand in row.get("contacting_hands", "").split("|") if hand
    ]
    return {
        "sample_id": f"contact-audit-{rank:03d}",
        "stratum": item["stratum"],
        "source_csv_line": line_number,
        "actual_event_index": int(item["actual_event_index"]),
        "matched": bool(item["matched"]),
        "attributed": item["attributed"],
        "attribution_event_valid": bool(item["attribution_event_valid"]),
        "event": {
            "pitch_key_index": _optional_int(row, "pitch_key_index", line_number),
            "midi_pitch": _optional_int(row, "midi_pitch", line_number),
            "note_name": row.get("note_name", "") or None,
            "onset_frame": _optional_int(row, "onset_frame", line_number),
            "onset_seconds": _optional_float(row, "onset_seconds", line_number),
            "offset_frame": _optional_int(row, "offset_frame", line_number),
            "video_frame": _optional_int(row, "video_frame", line_number),
            "video_seconds": _optional_float(row, "video_seconds", line_number),
            "matched_target_event_index": _optional_int(
                row, "matched_target_event_index", line_number
            ),
            "target_onset_frame": _optional_int(
                row, "target_onset_frame", line_number
            ),
            "onset_error_ms": _optional_float(row, "onset_error_ms", line_number),
        },
        "reported_attribution": {
            "primary_hand": row.get("primary_hand", "") or None,
            "primary_agent": row.get("primary_agent", "") or None,
            "contacting_hands": contacting_hands,
            "force_sample_sum_n": _optional_json_mapping(
                row, "per_hand_force_sample_sum_n", line_number
            ),
            "force_impulse_ns": _optional_json_mapping(
                row, "per_hand_force_impulse_ns", line_number
            ),
            "peak_force_n": _optional_json_mapping(
                row, "per_hand_peak_force_n", line_number
            ),
            "peak_force_frame": _optional_json_mapping(
                row, "per_hand_peak_force_frame", line_number
            ),
            "contact_nonfinite_sample_count": _optional_int(
                row, "contact_nonfinite_sample_count", line_number
            ),
            "contact_negative_sample_count": _optional_int(
                row, "contact_negative_sample_count", line_number
            ),
            "invalid_reason": row.get("invalid_reason", "") or None,
        },
        "manual_review": {
            "video_key_onset_visible": None,
            "reported_primary_hand_matches_video": None,
            "contacting_hands_plausible": None,
            "force_trace_plausible": None,
            "sensor_channel_valid": None,
            "reviewer_notes": None,
        },
    }


def build_contact_attribution_manifest(
    audit_csv: str | Path,
    *,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    """Return a deterministic, five-stratum contact-attribution checklist."""

    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    csv_path = Path(audit_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"contact-attribution audit CSV not found: {csv_path}")

    rows, source_sha256 = _read_rows(csv_path)
    grouped = {name: [] for name in STRATA_ORDER}
    for item in rows:
        grouped[item["stratum"]].append(item)

    population_counts = {name: len(grouped[name]) for name in STRATA_ORDER}
    base_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if rows else "no_eligible_events",
        "source": {
            "audit_csv": str(csv_path),
            "sha256": source_sha256,
            "actual_event_count": len(rows),
        },
        "sampling": {
            "seed": int(seed),
            "requested_sample_size": int(sample_size),
            "selected_sample_size": 0,
            "stratification": "sensor_valid_then_matched_x_attributed",
            "strata_order": list(STRATA_ORDER),
            "population_by_stratum": population_counts,
            "selected_by_stratum": {name: 0 for name in STRATA_ORDER},
            "population_exhausted": False,
        },
        "review_instructions": [
            "Seek to event.video_seconds (or event.video_frame) in the evaluation video.",
            "Confirm that the reported key activation is visible at the stated onset.",
            "Compare the visible touching hand with reported_attribution.primary_hand.",
            "Flag zero, implausible, or contradictory force/contact values in reviewer_notes.",
            "Review invalid_contact_event samples as sensor-quality failures, not ordinary unattributed notes.",
        ],
        "samples": [],
    }
    if not rows:
        base_manifest["reason"] = (
            "The audit CSV contains zero actual note-event rows; no manual-review "
            "samples were fabricated."
        )
        return base_manifest

    for name, items in grouped.items():
        items.sort(key=lambda item: _selection_key(
            int(seed),
            name,
            int(item["actual_event_index"]),
            int(item["source_csv_line"]),
        ))

    target_size = min(sample_size, len(rows))
    selected: list[dict[str, Any]] = []
    offsets = {name: 0 for name in STRATA_ORDER}
    while len(selected) < target_size:
        made_progress = False
        for name in STRATA_ORDER:
            offset = offsets[name]
            if offset < len(grouped[name]) and len(selected) < target_size:
                selected.append(grouped[name][offset])
                offsets[name] += 1
                made_progress = True
        if not made_progress:  # Defensive: target_size already bounds this.
            break

    selected_counts = {name: 0 for name in STRATA_ORDER}
    samples = []
    for rank, item in enumerate(selected, start=1):
        selected_counts[item["stratum"]] += 1
        samples.append(_sample_payload(item, rank))

    base_manifest["sampling"].update({
        "selected_sample_size": len(samples),
        "selected_by_stratum": selected_counts,
        "population_exhausted": len(samples) == len(rows),
    })
    base_manifest["samples"] = samples
    return base_manifest


def write_contact_attribution_manifest(
    manifest: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Write a manifest using stable, human-readable JSON formatting."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically sample a contact-attribution audit CSV into a "
            "five-stratum manual-review JSON manifest."
        )
    )
    parser.add_argument("audit_csv", help="evaluation audit/contact_attribution_audit.csv")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    parser.add_argument(
        "--sample-size",
        type=_positive_int,
        default=12,
        help="maximum number of actual events to select (default: 12)",
    )
    parser.add_argument(
        "--output",
        help="write JSON here; if omitted, print the JSON manifest to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_contact_attribution_manifest(
        args.audit_csv,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    rendered = strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        write_contact_attribution_manifest(manifest, args.output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
