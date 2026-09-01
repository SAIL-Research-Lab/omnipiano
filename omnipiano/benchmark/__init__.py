"""Benchmark-facing evaluation utilities for OmniPiano."""

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

__all__ = [
    "EpisodeTrace",
    "NoteEvent",
    "attribution_window_weights",
    "compute_episode_metrics",
    "extract_note_events",
    "match_note_events",
    "aggregate_seed_scorecards",
    "aggregate_seed_values",
    "normalized_curve_auc",
    "steps_to_threshold",
]
