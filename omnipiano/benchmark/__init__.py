"""Benchmark-facing evaluation utilities for OmniPiano."""

from omnipiano.benchmark.metrics import (
    DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
    DEFAULT_ONSET_TOLERANCE_SECONDS,
    METRICS_PROTOCOL_VERSION,
    ContactAttribution,
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

__all__ = [
    "DEFAULT_ATTRIBUTION_WINDOW_SECONDS",
    "DEFAULT_COLLISION_FORCE_THRESHOLD_N",
    "DEFAULT_CONTACT_FORCE_THRESHOLD_N",
    "DEFAULT_MOTOR_POWER_THRESHOLD_WATTS",
    "DEFAULT_ONSET_TOLERANCE_SECONDS",
    "METRICS_PROTOCOL_VERSION",
    "ContactAttribution",
    "EpisodeTrace",
    "NoteEvent",
    "attribute_note_events",
    "attribution_window_weights",
    "compute_episode_metrics",
    "extract_note_events",
    "match_note_events",
    "aggregate_seed_scorecards",
    "aggregate_seed_values",
    "normalized_curve_auc",
    "steps_to_threshold",
]
