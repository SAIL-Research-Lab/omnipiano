"""Offline Metrics-v2.1 rescoring for immutable MARL evaluation traces.

The command never edits a source run. It reconstructs ``EpisodeTrace`` from
``final_evaluation/audit/episode_trace.npz`` plus its JSON metadata, recomputes
the scorecard, and writes a separate result tree with source hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from omnipiano.benchmark.metrics import (
    DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    METRICS_PROTOCOL_VERSION,
    EpisodeTrace,
    compute_episode_metrics,
)
from omnipiano.benchmark.contact_attribution_checklist import (
    build_contact_attribution_manifest,
    write_contact_attribution_manifest,
)
from omnipiano.utils.info_keys import benchmark_metrics_to_episode_info
from omnipiano.utils.json_utils import strict_json_dumps


RESCORE_SCHEMA_VERSION = "omnipiano.marl_rescore.v2.1"
_REQUIRED_TRACE_ARRAYS = frozenset({
    "target_keys",
    "actual_keys",
    "target_sustain",
    "actual_sustain",
    "control_timestep",
    "hand_names",
})
_OPTIONAL_PHYSICAL_ARRAYS = (
    "key_contact_force",
    "hand_power",
    "hand_collision_force",
)
_REQUIRED_MUSICAL_INVARIANT_KEYS = (
    "episode_task/note_event_precision",
    "episode_task/note_event_recall",
    "episode_task/note_event_f1",
    "episode_task/note_event_tp",
    "episode_task/note_event_fp",
    "episode_task/note_event_fn",
    "episode_task/key_time_micro_precision",
    "episode_task/key_time_micro_recall",
    "episode_task/key_time_micro_f1",
)
_MUSICAL_INVARIANT_KEYS = (
    "episode_task/episode_steps",
    "episode_task/episode_duration_seconds",
    "episode_task/key_precision",
    "episode_task/key_recall",
    "episode_task/f1",
    "episode_task/sustain_precision",
    "episode_task/sustain_recall",
    "episode_task/sustain_f1",
    "episode_task/note_event_precision",
    "episode_task/note_event_recall",
    "episode_task/note_event_f1",
    "episode_task/note_event_f1_100ms",
    "episode_task/note_event_tp",
    "episode_task/note_event_fp",
    "episode_task/note_event_fn",
    "episode_task/note_onset_mae_ms",
    "episode_task/note_offset_mae_ms",
    "episode_task/note_duration_mae_ms",
    "episode_task/key_time_micro_precision",
    "episode_task/key_time_micro_recall",
    "episode_task/key_time_micro_f1",
    "episode_task/active_key_time_precision",
    "episode_task/active_key_time_recall",
    "episode_task/active_key_time_f1",
    "episode_task/target_active_step_ratio",
    "episode_task/rest_frame_count",
    "episode_task/rest_false_positive_rate",
    "episode_task/rest_false_positive_keys",
    "episode_task/rest_false_positive_keys_per_second",
    # Phase-1 MA compatibility aliases.
    "episode_task/musical_precision",
    "episode_task/musical_recall",
    "episode_task/musical_f1",
)
_MUSICAL_ALIAS_SOURCES = {
    "episode_task/musical_precision": "episode_task/key_precision",
    "episode_task/musical_recall": "episode_task/key_recall",
    "episode_task/musical_f1": "episode_task/f1",
}
_LEGACY_WRAPPER_METRIC_KEYS = (
    "episode_task/key_precision",
    "episode_task/key_recall",
    "episode_task/f1",
    "episode_task/sustain_precision",
    "episode_task/sustain_recall",
    "episode_task/sustain_f1",
)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token}")


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line, parse_constant=_reject_json_constant)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(path), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    root = run("rev-parse", "--show-toplevel")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "repository_root": root,
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _implementation_provenance() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    sources = {
        "metrics": Path(__file__).with_name("metrics.py"),
        "rescore_marl": Path(__file__).resolve(),
        "contact_attribution_checklist": Path(__file__).with_name(
            "contact_attribution_checklist.py"
        ),
        "info_keys": package_root / "utils" / "info_keys.py",
        "json_utils": package_root / "utils" / "json_utils.py",
        "trace_exporter": package_root / "multiagent" / "evaluation.py",
    }
    return {
        "git": _git_provenance(package_root),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_sha256": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
    }


def load_episode_trace(
    trace_npz: str | Path,
    metadata_json: str | Path,
) -> EpisodeTrace:
    """Reconstruct one trace without guessing agent or territory metadata."""

    trace_path = Path(trace_npz).expanduser().resolve()
    metadata_path = Path(metadata_json).expanduser().resolve()
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    with np.load(trace_path, allow_pickle=False) as arrays:
        missing = _REQUIRED_TRACE_ARRAYS - set(arrays.files)
        if missing:
            raise ValueError(
                f"{trace_path}: missing trace arrays {sorted(missing)}"
            )
        hand_names = tuple(str(value) for value in arrays["hand_names"].tolist())
        metadata_hand_names = tuple(str(value) for value in metadata["hand_names"])
        if hand_names != metadata_hand_names:
            raise ValueError(
                "NPZ/metadata hand order mismatch: "
                f"{hand_names} != {metadata_hand_names}"
            )
        npz_timestep = float(np.asarray(arrays["control_timestep"]).reshape(()))
        metadata_timestep = float(metadata["control_timestep"])
        if not np.isclose(npz_timestep, metadata_timestep, rtol=0.0, atol=0.0):
            raise ValueError(
                "NPZ/metadata control_timestep mismatch: "
                f"{npz_timestep} != {metadata_timestep}"
            )
        hand_to_agent = {
            str(hand): str(agent)
            for hand, agent in metadata["hand_to_agent"].items()
        }
        hand_key_ranges = {
            str(hand): (int(bounds[0]), int(bounds[1]))
            for hand, bounds in metadata["hand_key_ranges"].items()
        }
        declared_channels = metadata.get("physical_channels_available")
        if declared_channels is not None:
            for name in _OPTIONAL_PHYSICAL_ARRAYS:
                declared = bool(declared_channels.get(name, False))
                present = name in arrays.files
                if declared != present:
                    raise ValueError(
                        f"{trace_path}: physical channel {name!r} presence "
                        f"({present}) disagrees with metadata ({declared})"
                    )

        def optional_array(name: str):
            return (
                np.asarray(arrays[name], dtype=np.float64)
                if name in arrays.files
                else None
            )

        return EpisodeTrace(
            target_keys=np.asarray(arrays["target_keys"], dtype=bool),
            actual_keys=np.asarray(arrays["actual_keys"], dtype=bool),
            target_sustain=np.asarray(arrays["target_sustain"], dtype=bool),
            actual_sustain=np.asarray(arrays["actual_sustain"], dtype=bool),
            control_timestep=npz_timestep,
            hand_names=hand_names,
            hand_to_agent=hand_to_agent,
            hand_key_ranges=hand_key_ranges,
            key_contact_force=optional_array("key_contact_force"),
            hand_power=optional_array("hand_power"),
            hand_collision_force=optional_array("hand_collision_force"),
        )


def discover_final_evaluations(input_root: str | Path) -> tuple[Path, ...]:
    """Return deterministic final-evaluation directories containing traces."""

    root = Path(input_root).expanduser().resolve()
    direct_trace = root / "audit" / "episode_trace.npz"
    if direct_trace.is_file():
        return (root,)
    discovered = {
        path.parent.parent
        for path in root.glob("**/final_evaluation/audit/episode_trace.npz")
    }
    return tuple(sorted(discovered))


def _aggregate_values(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    total = int(array.size)
    if not valid.size:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "num_episodes": float(total),
            "num_successful_episodes": 0.0,
            "failure_rate": 1.0 if total else 0.0,
        }
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "num_episodes": float(total),
        "num_successful_episodes": float(valid.size),
        "failure_rate": float(1.0 - valid.size / total),
    }


def _metric_diff(
    old_metrics: Mapping[str, Any],
    new_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    def numeric(value: Any) -> float:
        return float("nan") if value is None else float(value)

    added = sorted(set(new_metrics) - set(old_metrics))
    removed = sorted(set(old_metrics) - set(new_metrics))
    changed = {}
    for name in sorted(set(old_metrics) & set(new_metrics)):
        old_value = numeric(old_metrics[name])
        new_value = numeric(new_metrics[name])
        if not np.isclose(
            old_value,
            new_value,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            changed[name] = {
                "old": old_value,
                "new": new_value,
                "delta": new_value - old_value,
            }
    return {"added": added, "removed": removed, "changed": changed}


def _summarize_cards(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = set.intersection(*(set(card["metrics"]) for card in cards))
    return {
        "num_episodes": len(cards),
        "episode_seeds": [int(card["seed"]) for card in cards],
        "aggregate": {
            "team_return": _aggregate_values(
                [float(card["team_return"]) for card in cards]
            ),
            "episode_steps": _aggregate_values(
                [float(card["episode_steps"]) for card in cards]
            ),
            "metrics": {
                name: _aggregate_values(
                    [float(card["metrics"][name]) for card in cards]
                )
                for name in sorted(metric_names)
            },
        },
    }


def _assert_musical_invariants(
    old_metrics: Mapping[str, Any],
    new_metrics: Mapping[str, Any],
) -> None:
    for name in _REQUIRED_MUSICAL_INVARIANT_KEYS:
        if name not in old_metrics:
            raise ValueError(f"source scorecard is missing invariant metric {name}")
    for name in _MUSICAL_INVARIANT_KEYS:
        if name not in old_metrics:
            continue
        if name not in new_metrics:
            raise ValueError(f"rescored scorecard is missing invariant metric {name}")
        old_value = (
            float("nan") if old_metrics[name] is None else float(old_metrics[name])
        )
        new_value = (
            float("nan") if new_metrics[name] is None else float(new_metrics[name])
        )
        if not np.isclose(
            old_value,
            new_value,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"v2.1 changed reward-independent musical metric {name}: "
                f"{old_metrics[name]} != {new_metrics[name]}"
            )


def _find_snapshot_manifest(final_dir: Path) -> tuple[str | None, str | None]:
    for parent in (final_dir, *final_dir.parents):
        candidate = parent / ".snapshot_manifest_sha256"
        if candidate.is_file():
            return str(candidate), candidate.read_text(encoding="utf-8").strip()
    return None, None


def _rescore_final_evaluation_into(
    source_final_dir: str | Path,
    working_output_dir: str | Path,
    logical_output_dir: str | Path,
    *,
    contact_force_threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    collision_force_threshold_n: float = DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    attribution_window_seconds: float = DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    motor_power_threshold_watts: float = DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
) -> dict[str, Any]:
    """Write one rescore into a transaction-owned working directory."""

    source = Path(source_final_dir).expanduser().resolve()
    destination = Path(working_output_dir).expanduser().resolve()
    logical_destination = Path(logical_output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing working destination: {destination}"
        )
    trace_path = source / "audit" / "episode_trace.npz"
    trace_metadata_path = source / "audit" / "episode_trace_metadata.json"
    episodes_path = source / "evaluation_episodes.jsonl"
    summary_path = source / "evaluation_summary.json"
    for required in (trace_path, trace_metadata_path, episodes_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {required}")

    source_cards = _read_jsonl(episodes_path)
    if len(source_cards) != 1:
        raise ValueError(
            "one exported trace can only rescore one episode, got "
            f"{len(source_cards)} source cards"
        )
    source_summary = _read_json(summary_path)
    trace = load_episode_trace(trace_path, trace_metadata_path)
    raw_metrics = compute_episode_metrics(
        trace,
        attribution_window_seconds=attribution_window_seconds,
        contact_force_threshold_n=contact_force_threshold_n,
        collision_force_threshold_n=collision_force_threshold_n,
        onset_tolerance_seconds=onset_tolerance_seconds,
        motor_power_threshold_watts=motor_power_threshold_watts,
    )
    namespaced_metrics = benchmark_metrics_to_episode_info(raw_metrics)
    # The online wrapper deliberately preserves RoboPianist's historical
    # sklearn frame-macro values for these compatibility keys.  Metrics-v2.1
    # does not redefine them, so an offline rescore carries their immutable
    # source values forward instead of replacing them with a pure-scorer
    # approximation.
    for name in _LEGACY_WRAPPER_METRIC_KEYS:
        if name in source_cards[0]["metrics"]:
            namespaced_metrics[name] = source_cards[0]["metrics"][name]
    for alias, source_name in _MUSICAL_ALIAS_SOURCES.items():
        if alias in source_cards[0]["metrics"]:
            namespaced_metrics[alias] = namespaced_metrics[source_name]
    _assert_musical_invariants(source_cards[0]["metrics"], namespaced_metrics)
    if int(source_cards[0]["episode_steps"]) != int(raw_metrics["episode_steps"]):
        raise RuntimeError(
            "v2.1 changed episode_steps outside the metrics mapping: "
            f"{source_cards[0]['episode_steps']} != {raw_metrics['episode_steps']}"
        )

    new_card = dict(source_cards[0])
    new_card["metrics"] = namespaced_metrics
    new_card["metrics_protocol_version"] = METRICS_PROTOCOL_VERSION
    new_card["rescore_schema_version"] = RESCORE_SCHEMA_VERSION
    source_manifest_path, source_manifest_sha = _find_snapshot_manifest(source)
    source_hashes_before = {
        "trace": _sha256(trace_path),
        "trace_metadata": _sha256(trace_metadata_path),
        "evaluation_episodes": _sha256(episodes_path),
        "evaluation_summary": _sha256(summary_path),
    }
    provenance = {
        "schema_version": RESCORE_SCHEMA_VERSION,
        "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
        "source_final_evaluation": str(source),
        "source_trace": str(trace_path),
        "source_artifact_sha256": source_hashes_before,
        "source_snapshot_manifest_path": source_manifest_path,
        "source_snapshot_manifest_sha256": source_manifest_sha,
        "implementation": _implementation_provenance(),
        "contact_force_threshold_n": float(contact_force_threshold_n),
        "collision_force_threshold_n": float(collision_force_threshold_n),
        "contact_attribution_window_seconds": float(attribution_window_seconds),
        "note_onset_tolerance_seconds": float(onset_tolerance_seconds),
        "motor_power_threshold_watts": float(motor_power_threshold_watts),
        "source_artifacts_modified": False,
    }
    new_card["rescore_provenance"] = provenance
    cards = [new_card]
    new_summary = _summarize_cards(cards)
    if isinstance(source_summary, dict):
        for key, value in source_summary.items():
            if key not in {"aggregate", "num_episodes", "episode_seeds"}:
                new_summary[key] = value
    new_summary["metrics_protocol_version"] = METRICS_PROTOCOL_VERSION
    new_summary["rescore_schema_version"] = RESCORE_SCHEMA_VERSION
    new_summary["rescore_provenance"] = provenance

    destination.mkdir(parents=True, exist_ok=False)
    rescored_episodes_path = destination / "evaluation_episodes.jsonl"
    rescored_episodes_path.write_text(
        strict_json_dumps(new_card, sort_keys=True) + "\n", encoding="utf-8"
    )
    rescored_summary_path = destination / "evaluation_summary.json"
    rescored_summary_path.write_text(
        strict_json_dumps(new_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from omnipiano.multiagent.evaluation import export_episode_trace

    audit_files = export_episode_trace(
        trace,
        destination / "audit",
        contact_force_threshold_n=contact_force_threshold_n,
        collision_force_threshold_n=collision_force_threshold_n,
        attribution_window_seconds=attribution_window_seconds,
        onset_tolerance_seconds=onset_tolerance_seconds,
        motor_power_threshold_watts=motor_power_threshold_watts,
        include_trace_npz=False,
    )
    checklist_path = destination / "audit" / "contact_attribution_checklist.json"
    checklist = build_contact_attribution_manifest(
        audit_files["audit"],
        seed=0,
        sample_size=12,
    )
    checklist["source"]["audit_csv"] = str(
        logical_destination / "audit" / "contact_attribution_audit.csv"
    )
    write_contact_attribution_manifest(checklist, checklist_path)

    def logical_path(path: str | Path) -> str:
        physical = Path(path).resolve()
        return str(logical_destination / physical.relative_to(destination))

    logical_audit_files = {
        name: logical_path(path) for name, path in audit_files.items()
    }
    logical_audit_files["checklist"] = logical_path(checklist_path)
    output_manifest = {
        "provenance": provenance,
        "metric_diff_from_source": _metric_diff(
            source_cards[0]["metrics"], namespaced_metrics
        ),
        "outputs": {
            "evaluation_episodes": logical_path(rescored_episodes_path),
            "evaluation_summary": logical_path(rescored_summary_path),
            **logical_audit_files,
        },
        "quality_gates": {
            "contact_attribution_valid": bool(
                raw_metrics.get("contact_attribution_valid") == 1.0
            ),
            "inter_hand_collision_metrics_valid": bool(
                raw_metrics.get("inter_hand_collision_metrics_valid") == 1.0
            ),
            "inter_agent_collision_metrics_valid": bool(
                raw_metrics.get("inter_agent_collision_metrics_valid") == 1.0
            ),
            "hand_power_metrics_valid": bool(
                raw_metrics.get("hand_power_metrics_valid") == 1.0
            ),
        },
    }
    manifest_path = destination / "rescore_manifest.json"
    output_manifest["outputs"]["manifest"] = str(
        logical_destination / "rescore_manifest.json"
    )
    output_manifest["output_sha256"] = {
        "evaluation_episodes": _sha256(rescored_episodes_path),
        "evaluation_summary": _sha256(rescored_summary_path),
        "audit": _sha256(Path(audit_files["audit"])),
        "metadata": _sha256(Path(audit_files["metadata"])),
        "checklist": _sha256(checklist_path),
    }
    source_hashes_after = {
        "trace": _sha256(trace_path),
        "trace_metadata": _sha256(trace_metadata_path),
        "evaluation_episodes": _sha256(episodes_path),
        "evaluation_summary": _sha256(summary_path),
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(
            f"source artifacts changed during rescore: {source}"
        )
    manifest_path.write_text(
        strict_json_dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def rescore_final_evaluation(
    source_final_dir: str | Path,
    output_dir: str | Path,
    *,
    contact_force_threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    collision_force_threshold_n: float = DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    attribution_window_seconds: float = DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    motor_power_threshold_watts: float = DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
) -> dict[str, Any]:
    """Atomically rescore one immutable final evaluation."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing rescore destination: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction_parent = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    ))
    working = transaction_parent / "payload"
    try:
        result = _rescore_final_evaluation_into(
            source_final_dir,
            working,
            destination,
            contact_force_threshold_n=contact_force_threshold_n,
            collision_force_threshold_n=collision_force_threshold_n,
            attribution_window_seconds=attribution_window_seconds,
            onset_tolerance_seconds=onset_tolerance_seconds,
            motor_power_threshold_watts=motor_power_threshold_watts,
        )
        working.replace(destination)
        transaction_parent.rmdir()
        return result
    except Exception:
        shutil.rmtree(transaction_parent, ignore_errors=True)
        raise


def rescore_tree(
    input_root: str | Path,
    output_root: str | Path,
    *,
    contact_force_threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    collision_force_threshold_n: float = DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    attribution_window_seconds: float = DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    motor_power_threshold_watts: float = DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
) -> dict[str, Any]:
    """Rescore every discovered final evaluation under ``input_root``."""

    source_root = Path(input_root).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    if destination_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output root: {destination_root}"
        )
    final_dirs = discover_final_evaluations(source_root)
    if not final_dirs:
        raise FileNotFoundError(
            f"no final_evaluation episode traces found under {source_root}"
        )
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    transaction_parent = Path(tempfile.mkdtemp(
        prefix=f".{destination_root.name}.tmp-",
        dir=destination_root.parent,
    ))
    working_root = transaction_parent / "payload"
    try:
        results = []
        for final_dir in final_dirs:
            relative = (
                final_dir.name
                if final_dir == source_root
                else str(final_dir.relative_to(source_root))
            )
            result = _rescore_final_evaluation_into(
                final_dir,
                working_root / relative,
                destination_root / relative,
                contact_force_threshold_n=contact_force_threshold_n,
                collision_force_threshold_n=collision_force_threshold_n,
                attribution_window_seconds=attribution_window_seconds,
                onset_tolerance_seconds=onset_tolerance_seconds,
                motor_power_threshold_watts=motor_power_threshold_watts,
            )
            results.append(result)
        quality = [result["quality_gates"] for result in results]
        changed_metric_names = sorted({
            name
            for result in results
            for name in result["metric_diff_from_source"]["changed"]
        })
        added_metric_names = sorted({
            name
            for result in results
            for name in result["metric_diff_from_source"]["added"]
        })
        summary = {
            "schema_version": RESCORE_SCHEMA_VERSION,
            "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
            "input_root": str(source_root),
            "output_root": str(destination_root),
            "num_runs": len(results),
            "contact_force_threshold_n": float(contact_force_threshold_n),
            "collision_force_threshold_n": float(collision_force_threshold_n),
            "contact_attribution_window_seconds": float(
                attribution_window_seconds
            ),
            "note_onset_tolerance_seconds": float(onset_tolerance_seconds),
            "motor_power_threshold_watts": float(motor_power_threshold_watts),
            "quality_gate_failures": {
                key: sum(not bool(item[key]) for item in quality)
                for key in quality[0]
            },
            "metric_change_counts": {
                name: int(sum(
                    name in result["metric_diff_from_source"]["changed"]
                    for result in results
                ))
                for name in changed_metric_names
            },
            "metric_added_counts": {
                name: int(sum(
                    name in result["metric_diff_from_source"]["added"]
                    for result in results
                ))
                for name in added_metric_names
            },
            "runs": results,
        }
        working_root.mkdir(parents=True, exist_ok=True)
        summary_path = working_root / "rescore_tree_summary.json"
        summary_path.write_text(
            strict_json_dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        working_root.replace(destination_root)
        transaction_parent.rmdir()
        return summary
    except Exception:
        shutil.rmtree(transaction_parent, ignore_errors=True)
        raise


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and strictly positive")
    return parsed


def _nonnegative_finite(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescore immutable MARL final traces with Metrics-v2.1"
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--contact-force-threshold-n",
        type=_positive_finite,
        default=DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    )
    parser.add_argument(
        "--collision-force-threshold-n",
        type=_positive_finite,
        default=DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    )
    parser.add_argument(
        "--attribution-window-seconds",
        type=_positive_finite,
        default=DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--onset-tolerance-seconds",
        type=_nonnegative_finite,
        default=DEFAULT_ONSET_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--motor-power-threshold-watts",
        type=_positive_finite,
        default=DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = rescore_tree(
        args.input_root,
        args.output_root,
        contact_force_threshold_n=args.contact_force_threshold_n,
        collision_force_threshold_n=args.collision_force_threshold_n,
        attribution_window_seconds=args.attribution_window_seconds,
        onset_tolerance_seconds=args.onset_tolerance_seconds,
        motor_power_threshold_watts=args.motor_power_threshold_watts,
    )
    print(
        f"[done] rescored {summary['num_runs']} runs -> "
        f"{summary['output_root']}"
    )
    print(
        "[quality_gate_failures] "
        + strict_json_dumps(summary["quality_gate_failures"], sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
