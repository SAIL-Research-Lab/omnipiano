"""Reproducible threshold/window sensitivity for contact attribution traces."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from omnipiano.benchmark.metrics import (
    DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    METRICS_PROTOCOL_VERSION,
    extract_note_events,
    match_note_events,
)
from omnipiano.benchmark.rescore_marl import (
    _git_provenance,
    discover_final_evaluations,
    load_episode_trace,
)
from omnipiano.utils.json_utils import strict_json_dumps


SENSITIVITY_SCHEMA_VERSION = "omnipiano.contact_attribution_sensitivity.v1"
DEFAULT_THRESHOLDS_N = (
    0.0,
    1e-12,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
)
# Inclusive frame offsets around the actual key-activation onset.  These are
# diagnostics only; Metrics-v2.1 retains the frozen [onset, onset+50ms)
# definition until video review establishes a better-aligned physical window.
DEFAULT_WINDOWS = {
    "onset_only": (0, 0),
    "pre_and_onset": (-1, 0),
    "onset_and_post": (0, 1),
    "symmetric_1": (-1, 1),
    "symmetric_2": (-2, 2),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _effective_count(counts: Mapping[str, int]) -> float:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    values = values[values > 0.0]
    if not values.size:
        return 0.0
    shares = values / values.sum()
    return float(np.exp(-np.sum(shares * np.log(shares))))


def _shares(counts: Mapping[str, int]) -> dict[str, float]:
    total = int(sum(counts.values()))
    return {
        name: (value / total if total else 0.0)
        for name, value in counts.items()
    }


def trace_sensitivity_counts(
    trace,
    *,
    thresholds_n: Sequence[float],
    windows: Mapping[str, tuple[int, int]],
) -> dict:
    """Count attribution outcomes for every threshold/window combination."""

    if not thresholds_n:
        raise ValueError("thresholds_n must not be empty")
    if not windows:
        raise ValueError("windows must not be empty")

    actual_events = extract_note_events(trace.actual_keys)
    target_events = extract_note_events(trace.target_keys)
    matches = match_note_events(
        target_events,
        actual_events,
        control_timestep=trace.control_timestep,
        onset_tolerance_seconds=0.05,
    )
    matched_actual = {match.predicted_index for match in matches}
    if trace.key_contact_force is None:
        raise ValueError("contact-attribution sensitivity requires key_contact_force")
    force = np.asarray(trace.key_contact_force, dtype=np.float64)
    result: dict = {}
    outcomes: dict[tuple[str, str], dict] = {}
    for window_name, (start_offset, end_offset) in windows.items():
        if start_offset > end_offset:
            raise ValueError(f"invalid window {window_name}: start > end")
        event_peaks: list[np.ndarray | None] = []
        event_impulses: list[np.ndarray | None] = []
        nonfinite_counts: list[int] = []
        negative_counts: list[int] = []
        for event in actual_events:
            start = max(0, event.onset_frame + int(start_offset))
            stop = min(force.shape[0], event.onset_frame + int(end_offset) + 1)
            window = force[start:stop, :, event.pitch]
            nonfinite = int(np.count_nonzero(~np.isfinite(window)))
            finite = np.isfinite(window)
            negative = int(np.count_nonzero(window[finite] < 0.0))
            nonfinite_counts.append(nonfinite)
            negative_counts.append(negative)
            if nonfinite or negative:
                event_peaks.append(None)
                event_impulses.append(None)
            elif not window.shape[0]:
                event_peaks.append(np.zeros(len(trace.hand_names)))
                event_impulses.append(np.zeros(len(trace.hand_names)))
            else:
                event_peaks.append(window.max(axis=0))
                event_impulses.append(window.sum(axis=0) * trace.control_timestep)
        window_result: dict = {}
        for threshold in thresholds_n:
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError("sensitivity thresholds must be finite and >= 0")
            attributed = 0
            duplicate = 0
            matched_attributed = 0
            primary_indices = np.full(len(actual_events), -1, dtype=np.int32)
            participant_sets: list[tuple[int, ...]] = []
            for event_index, (peaks, impulses) in enumerate(
                zip(event_peaks, event_impulses)
            ):
                if peaks is None:
                    participant_sets.append(())
                    continue
                participant_array = np.flatnonzero(peaks > threshold)
                participants = tuple(int(index) for index in participant_array)
                participant_sets.append(participants)
                participant_count = len(participants)
                attributed += participant_count > 0
                duplicate += participant_count > 1
                matched_attributed += (
                    participant_count > 0 and event_index in matched_actual
                )
                if participant_count:
                    primary_indices[event_index] = int(
                        participant_array[
                            int(np.argmax(impulses[participant_array]))
                        ]
                    )
            key = format(float(threshold), ".12g")
            primary_hand_counts = {
                hand: int(np.count_nonzero(primary_indices == index))
                for index, hand in enumerate(trace.hand_names)
            }
            primary_agent_names = [
                (
                    trace.hand_to_agent[trace.hand_names[index]]
                    if index >= 0
                    else None
                )
                for index in primary_indices
            ]
            agent_names = tuple(dict.fromkeys(trace.hand_to_agent.values()))
            primary_agent_counts = {
                agent: int(sum(name == agent for name in primary_agent_names))
                for agent in agent_names
            }
            window_result[key] = {
                "actual_event_count": len(actual_events),
                "matched_event_count": len(matches),
                "valid_event_count": int(
                    sum(
                        nonfinite == 0 and negative == 0
                        for nonfinite, negative in zip(
                            nonfinite_counts, negative_counts
                        )
                    )
                ),
                "attributed_event_count": int(attributed),
                "matched_attributed_event_count": int(matched_attributed),
                "duplicate_event_count": int(duplicate),
                "nonfinite_event_count": int(
                    sum(count > 0 for count in nonfinite_counts)
                ),
                "nonfinite_sample_count": int(sum(nonfinite_counts)),
                "negative_event_count": int(
                    sum(count > 0 for count in negative_counts)
                ),
                "negative_sample_count": int(sum(negative_counts)),
                "primary_hand_event_count": primary_hand_counts,
                "primary_agent_event_count": primary_agent_counts,
                "active_hand_count": int(sum(
                    value > 0 for value in primary_hand_counts.values()
                )),
                "active_agent_count": int(sum(
                    value > 0 for value in primary_agent_counts.values()
                )),
                "effective_active_agents": _effective_count(primary_agent_counts),
                "agent_contribution_share": _shares(primary_agent_counts),
            }
            outcomes[(window_name, key)] = {
                "primary_hand": primary_indices,
                "primary_agent": primary_agent_names,
                "participants": participant_sets,
            }
        result[window_name] = window_result

    reference_window = "onset_only" if "onset_only" in windows else next(iter(windows))
    threshold_values = tuple(float(value) for value in thresholds_n)
    reference_threshold = (
        DEFAULT_CONTACT_FORCE_THRESHOLD_N
        if DEFAULT_CONTACT_FORCE_THRESHOLD_N in threshold_values
        else threshold_values[0]
    )
    reference_key = format(reference_threshold, ".12g")
    reference = outcomes[(reference_window, reference_key)]
    for window_name, window_result in result.items():
        for key, counts in window_result.items():
            current = outcomes[(window_name, key)]
            reference_hand = reference["primary_hand"]
            current_hand = current["primary_hand"]
            both_attributed = (reference_hand >= 0) & (current_hand >= 0)
            same_hand = both_attributed & (reference_hand == current_hand)
            reference_agent = reference["primary_agent"]
            current_agent = current["primary_agent"]
            same_agent_count = sum(
                left is not None and left == right
                for left, right in zip(reference_agent, current_agent)
            )
            both_agent_count = sum(
                left is not None and right is not None
                for left, right in zip(reference_agent, current_agent)
            )
            counts["stability_vs_reference"] = {
                "both_attributed_event_count": int(np.count_nonzero(both_attributed)),
                "same_primary_hand_event_count": int(np.count_nonzero(same_hand)),
                "primary_hand_flip_event_count": int(
                    np.count_nonzero(both_attributed & ~same_hand)
                ),
                "both_primary_agent_event_count": int(both_agent_count),
                "same_primary_agent_event_count": int(same_agent_count),
                "primary_agent_flip_event_count": int(
                    both_agent_count - same_agent_count
                ),
                "attribution_gained_event_count": int(
                    np.count_nonzero((reference_hand < 0) & (current_hand >= 0))
                ),
                "attribution_lost_event_count": int(
                    np.count_nonzero((reference_hand >= 0) & (current_hand < 0))
                ),
                "participant_set_flip_event_count": int(sum(
                    left != right
                    for left, right in zip(
                        reference["participants"], current["participants"]
                    )
                )),
            }
            current_shares = counts["agent_contribution_share"]
            reference_counts = result[reference_window][reference_key]
            reference_shares = reference_counts["agent_contribution_share"]
            share_names = set(current_shares) | set(reference_shares)
            counts["stability_vs_reference"].update({
                "agent_contribution_share_l1_delta": float(
                    0.5 * sum(
                        abs(current_shares.get(name, 0.0) - reference_shares.get(name, 0.0))
                        for name in share_names
                    )
                ),
                "effective_active_agents_delta": float(
                    counts["effective_active_agents"]
                    - reference_counts["effective_active_agents"]
                ),
            })
    return result


def analyze_tree(
    input_root: str | Path,
    output_json: str | Path,
    *,
    thresholds_n: Sequence[float] = DEFAULT_THRESHOLDS_N,
    windows: Mapping[str, tuple[int, int]] = DEFAULT_WINDOWS,
) -> dict:
    """Analyze every final trace and write one strict, source-hashed report."""

    root = Path(input_root).expanduser().resolve()
    destination = Path(output_json).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite sensitivity report: {destination}")
    final_dirs = discover_final_evaluations(root)
    if not final_dirs:
        raise FileNotFoundError(f"no final evaluation traces under {root}")
    per_run = []
    control_timesteps = set()
    for final_dir in final_dirs:
        trace_path = final_dir / "audit" / "episode_trace.npz"
        metadata_path = final_dir / "audit" / "episode_trace_metadata.json"
        trace = load_episode_trace(trace_path, metadata_path)
        control_timesteps.add(float(trace.control_timestep))
        per_run.append({
            "source_final_evaluation": str(final_dir),
            "source_trace_sha256": _sha256(trace_path),
            "source_trace_metadata_sha256": _sha256(metadata_path),
            "control_timestep_seconds": float(trace.control_timestep),
            "counts": trace_sensitivity_counts(
                trace,
                thresholds_n=thresholds_n,
                windows=windows,
            ),
        })

    threshold_values = tuple(float(value) for value in thresholds_n)
    reference_threshold = (
        DEFAULT_CONTACT_FORCE_THRESHOLD_N
        if DEFAULT_CONTACT_FORCE_THRESHOLD_N in threshold_values
        else threshold_values[0]
    )
    reference_window = (
        "onset_only" if "onset_only" in windows else next(iter(windows))
    )
    reference_key = format(reference_threshold, ".12g")

    aggregate: dict[str, dict[str, dict]] = {}
    for window_name in windows:
        aggregate[window_name] = {}
        for threshold in thresholds_n:
            key = format(float(threshold), ".12g")
            summed = {
                field: int(sum(run["counts"][window_name][key][field] for run in per_run))
                for field in (
                    "actual_event_count",
                    "matched_event_count",
                    "valid_event_count",
                    "attributed_event_count",
                    "matched_attributed_event_count",
                    "duplicate_event_count",
                    "nonfinite_event_count",
                    "nonfinite_sample_count",
                    "negative_event_count",
                    "negative_sample_count",
                )
            }
            summed["attribution_rate"] = (
                summed["attributed_event_count"] / summed["actual_event_count"]
                if summed["actual_event_count"]
                else 0.0
            )
            summed["matched_attribution_rate"] = (
                summed["matched_attributed_event_count"]
                / summed["matched_event_count"]
                if summed["matched_event_count"]
                else 0.0
            )
            summed["conditional_attribution_rate"] = (
                summed["attributed_event_count"] / summed["valid_event_count"]
                if summed["valid_event_count"]
                else 0.0
            )
            for entity_field in (
                "primary_hand_event_count",
                "primary_agent_event_count",
            ):
                entity_names = sorted({
                    name
                    for run in per_run
                    for name in run["counts"][window_name][key][entity_field]
                })
                summed[entity_field] = {
                    name: int(sum(
                        run["counts"][window_name][key][entity_field].get(name, 0)
                        for run in per_run
                    ))
                    for name in entity_names
                }
            summed["active_hand_count"] = int(sum(
                value > 0 for value in summed["primary_hand_event_count"].values()
            ))
            summed["active_agent_count"] = int(sum(
                value > 0 for value in summed["primary_agent_event_count"].values()
            ))
            summed["effective_active_agents"] = _effective_count(
                summed["primary_agent_event_count"]
            )
            summed["agent_contribution_share"] = _shares(
                summed["primary_agent_event_count"]
            )
            stability_fields = (
                "both_attributed_event_count",
                "same_primary_hand_event_count",
                "primary_hand_flip_event_count",
                "both_primary_agent_event_count",
                "same_primary_agent_event_count",
                "primary_agent_flip_event_count",
                "attribution_gained_event_count",
                "attribution_lost_event_count",
                "participant_set_flip_event_count",
            )
            stability = {
                field: int(sum(
                    run["counts"][window_name][key]["stability_vs_reference"][field]
                    for run in per_run
                ))
                for field in stability_fields
            }
            stability["primary_hand_agreement_rate"] = (
                stability["same_primary_hand_event_count"]
                / stability["both_attributed_event_count"]
                if stability["both_attributed_event_count"]
                else 1.0
            )
            stability["primary_agent_agreement_rate"] = (
                stability["same_primary_agent_event_count"]
                / stability["both_primary_agent_event_count"]
                if stability["both_primary_agent_event_count"]
                else 1.0
            )
            summed["stability_vs_reference"] = stability
            aggregate[window_name][key] = summed

    reference_counts = aggregate[reference_window][reference_key]
    reference_shares = reference_counts["agent_contribution_share"]
    for window_result in aggregate.values():
        for counts in window_result.values():
            current_shares = counts["agent_contribution_share"]
            names = set(reference_shares) | set(current_shares)
            counts["stability_vs_reference"].update({
                "agent_contribution_share_l1_delta": float(
                    0.5 * sum(
                        abs(
                            current_shares.get(name, 0.0)
                            - reference_shares.get(name, 0.0)
                        )
                        for name in names
                    )
                ),
                "effective_active_agents_delta": float(
                    counts["effective_active_agents"]
                    - reference_counts["effective_active_agents"]
                ),
            })

    package_root = Path(__file__).resolve().parents[1]
    implementation_sources = {
        "attribution_sensitivity": Path(__file__).resolve(),
        "metrics": Path(__file__).with_name("metrics.py"),
        "rescore_marl": Path(__file__).with_name("rescore_marl.py"),
        "json_utils": package_root / "utils" / "json_utils.py",
    }
    report = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
        "input_root": str(root),
        "implementation": {
            "git": _git_provenance(package_root),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "source_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in implementation_sources.items()
            },
        },
        "num_runs": len(per_run),
        "control_timesteps_seconds": sorted(control_timesteps),
        "thresholds_n": [float(value) for value in thresholds_n],
        "windows_inclusive_frame_offsets": {
            name: [int(bounds[0]), int(bounds[1])]
            for name, bounds in windows.items()
        },
        "interpretation": {
            "reference_threshold_n": reference_threshold,
            "reference_window": reference_window,
            "default_threshold_n": DEFAULT_CONTACT_FORCE_THRESHOLD_N,
            "default_is_numerical_floor_not_physical_calibration": True,
            "metrics_v2_1_window": "[onset, onset + 0.05 seconds)",
            "onset_only_matches_metrics_v2_1": (
                tuple(windows.get("onset_only", ())) == (0, 0)
                and all(
                    np.isclose(value, 0.05, rtol=0.0, atol=1e-12)
                    for value in control_timesteps
                )
            ),
            "paper_claim_requires_manual_video_validation": True,
        },
        "aggregate": aggregate,
        "runs": per_run,
    }
    for run in per_run:
        final_dir = Path(run["source_final_evaluation"])
        if _sha256(final_dir / "audit" / "episode_trace.npz") != run[
            "source_trace_sha256"
        ]:
            raise RuntimeError(f"source trace changed during analysis: {final_dir}")
        if _sha256(
            final_dir / "audit" / "episode_trace_metadata.json"
        ) != run["source_trace_metadata_sha256"]:
            raise RuntimeError(
                f"source trace metadata changed during analysis: {final_dir}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = strict_json_dumps(report, indent=2, sort_keys=True) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish atomically without clobbering a report created by a
        # concurrent scorer after the exists() preflight above.  The temp
        # file lives in destination.parent, so this is a same-filesystem link.
        os.link(temporary_path, destination)
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return report


def _thresholds(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            "thresholds must be a comma-separated list of finite values >= 0"
        )
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze contact-attribution threshold/window sensitivity"
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds-n",
        type=_thresholds,
        default=DEFAULT_THRESHOLDS_N,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = analyze_tree(
        args.input_root,
        args.output,
        thresholds_n=args.thresholds_n,
    )
    print(f"[done] analyzed {report['num_runs']} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
