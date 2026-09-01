"""Pure, reward-independent benchmark metrics for robot piano playing.

The functions in this module intentionally do not depend on Gymnasium,
PettingZoo, dm_control, or MuJoCo.  A simulator wrapper records an
``EpisodeTrace`` and this module turns that trace into scalar episode metrics.
Keeping the metric definitions pure makes them easy to unit test with synthetic
performances and prevents an RL library from changing the scoring semantics.

Conventions
-----------
* Piano key indices are 0..87 (A0..C8).
* A frame corresponds to one environment control step.
* Note events are maximal contiguous runs of an active key.
* Event matching is one-to-one, pitch-exact, and onset-tolerant.  The matcher
  first maximizes the number of matches and then minimizes total onset error.
* The historical frame-macro F1 remains available as ``f1``.  It must not be
  used as the benchmark headline because correctly silent frames receive 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


METRICS_PROTOCOL_VERSION = "2.1"
DEFAULT_ONSET_TOLERANCE_SECONDS = 0.05
DEFAULT_ATTRIBUTION_WINDOW_SECONDS = 0.05
# This is a numerical floor, not a claim about the minimum force required to
# depress a physical piano key.  The frozen MARL traces contain a separated
# cluster of denormal values (roughly 1e-323--1e-311 N); 1e-6 N removes that
# cluster while preserving the stable attribution plateau observed from
# 1e-7--1e-6 N.  Every exported artifact records the exact value so later
# physical calibration can be versioned rather than silently changing scores.
DEFAULT_CONTACT_FORCE_THRESHOLD_N = 1e-6
DEFAULT_COLLISION_FORCE_THRESHOLD_N = 1e-6
DEFAULT_MOTOR_POWER_THRESHOLD_WATTS = 1e-6


@dataclass(frozen=True)
class ContactAttribution:
    """Per-event hand attribution plus channel-validity diagnostics."""

    primary_hand_indices: np.ndarray
    participant_hand_indices: Tuple[Tuple[int, ...], ...]
    event_valid: np.ndarray
    nonfinite_sample_counts_by_event: np.ndarray
    negative_sample_counts_by_event: np.ndarray
    nonfinite_event_count: int
    nonfinite_sample_count: int
    negative_event_count: int
    negative_sample_count: int
    available: bool

    @property
    def valid(self) -> bool:
        return (
            self.available
            and self.nonfinite_event_count == 0
            and self.negative_event_count == 0
        )


def attribution_window_weights(
    control_timestep: float,
    window_seconds: float,
) -> np.ndarray:
    """Return exact interval weights for a half-open attribution window.

    A force sample at frame ``t`` represents the control interval
    ``[t * dt, (t + 1) * dt)``.  The returned weights integrate the half-open
    window ``[onset, onset + window_seconds)`` without counting the endpoint
    as another full frame.  Thus the standard 50 ms window at a 50 ms control
    timestep contains exactly one sample, not two.
    """

    dt = float(control_timestep)
    duration = float(window_seconds)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("control_timestep must be finite and positive")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("window_seconds must be finite and positive")
    ratio = duration / dt
    # Moving an exactly integral ratio one ULP downward avoids an accidental
    # extra sample from floating-point roundoff while leaving non-integral
    # ratios unchanged for ceil().
    sample_count = max(1, int(np.ceil(np.nextafter(ratio, -np.inf))))
    weights = np.full(sample_count, dt, dtype=np.float64)
    weights[-1] = duration - dt * (sample_count - 1)
    if weights[-1] <= 0.0 or weights[-1] > dt * (1.0 + 1e-12):
        raise RuntimeError(
            "invalid attribution-window discretization: "
            f"dt={dt}, duration={duration}, weights={weights}"
        )
    return weights


@dataclass(frozen=True)
class NoteEvent:
    """One key activation event in frame coordinates."""

    pitch: int
    onset_frame: int
    offset_frame: int  # exclusive

    @property
    def duration_frames(self) -> int:
        return self.offset_frame - self.onset_frame


@dataclass(frozen=True)
class EventMatch:
    """Indices of one matched target/predicted event pair."""

    target_index: int
    predicted_index: int


@dataclass
class EpisodeTrace:
    """All simulator-independent data required by the benchmark scorer.

    Optional physical arrays use a fixed hand ordering given by ``hand_names``.
    ``key_contact_force`` has shape ``[T, H, K]`` and stores normal contact
    force in newtons. ``hand_power`` is ``[T, H]`` in watts.
    ``hand_collision_force`` is ``[T, H, H]`` and is symmetric; only its upper
    triangle is consumed.
    """

    target_keys: np.ndarray
    actual_keys: np.ndarray
    target_sustain: np.ndarray
    actual_sustain: np.ndarray
    control_timestep: float
    hand_names: Tuple[str, ...] = ()
    hand_to_agent: Mapping[str, str] = field(default_factory=dict)
    hand_key_ranges: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    key_contact_force: Optional[np.ndarray] = None
    hand_power: Optional[np.ndarray] = None
    hand_collision_force: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.target_keys = np.asarray(self.target_keys, dtype=bool)
        self.actual_keys = np.asarray(self.actual_keys, dtype=bool)
        self.target_sustain = np.asarray(self.target_sustain, dtype=bool).reshape(-1)
        self.actual_sustain = np.asarray(self.actual_sustain, dtype=bool).reshape(-1)
        if self.target_keys.ndim != 2 or self.actual_keys.ndim != 2:
            raise ValueError("target_keys and actual_keys must be [T, K] arrays")
        if self.target_keys.shape != self.actual_keys.shape:
            raise ValueError(
                "target_keys and actual_keys must have identical shapes, got "
                f"{self.target_keys.shape} and {self.actual_keys.shape}"
            )
        steps, n_keys = self.target_keys.shape
        if n_keys != 88:
            raise ValueError(f"OmniPiano traces require 88 keys, got {n_keys}")
        if self.target_sustain.shape != (steps,) or self.actual_sustain.shape != (steps,):
            raise ValueError("sustain arrays must have shape [T] matching key frames")
        if not np.isfinite(self.control_timestep) or self.control_timestep <= 0:
            raise ValueError("control_timestep must be finite and positive")

        n_hands = len(self.hand_names)
        if len(set(self.hand_names)) != n_hands:
            raise ValueError(f"hand_names must be unique, got {self.hand_names}")
        for hand in self.hand_names:
            if hand not in self.hand_to_agent:
                raise ValueError(f"hand_to_agent has no entry for hand {hand!r}")
        self._validate_optional("key_contact_force", self.key_contact_force,
                                (steps, n_hands, n_keys))
        self._validate_optional("hand_power", self.hand_power, (steps, n_hands))
        self._validate_optional(
            "hand_collision_force", self.hand_collision_force,
            (steps, n_hands, n_hands),
        )

    @staticmethod
    def _validate_optional(name: str, value: Optional[np.ndarray], shape: tuple) -> None:
        if value is not None and np.asarray(value).shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {np.asarray(value).shape}")


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision/recall/F1 with explicit empty-set semantics.

    Two empty sets are a perfect match.  If only one side is empty, all three
    scores are zero.  This avoids the misleading per-rest-frame averaging used
    by the legacy metric while retaining mathematically useful edge behavior.
    """

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _binary_counts(target: np.ndarray, actual: np.ndarray) -> Tuple[int, int, int]:
    target = np.asarray(target, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    return (
        int(np.count_nonzero(target & actual)),
        int(np.count_nonzero(~target & actual)),
        int(np.count_nonzero(target & ~actual)),
    )


def extract_note_events(frames: np.ndarray) -> Tuple[NoteEvent, ...]:
    """Convert a binary piano roll ``[T, K]`` to note events."""

    frames = np.asarray(frames, dtype=bool)
    if frames.ndim != 2:
        raise ValueError(f"frames must be a [T, K] array, got {frames.shape}")
    steps, n_keys = frames.shape
    events = []
    for pitch in range(n_keys):
        active = frames[:, pitch]
        padded = np.concatenate(([False], active, [False])).astype(np.int8)
        transitions = np.diff(padded)
        onsets = np.flatnonzero(transitions == 1)
        offsets = np.flatnonzero(transitions == -1)
        events.extend(
            NoteEvent(pitch=pitch, onset_frame=int(on), offset_frame=int(off))
            for on, off in zip(onsets, offsets)
        )
    events.sort(key=lambda event: (event.onset_frame, event.pitch, event.offset_frame))
    return tuple(events)


def _match_one_pitch(
    target: Sequence[Tuple[int, NoteEvent]],
    predicted: Sequence[Tuple[int, NoteEvent]],
    tolerance_frames: float,
) -> Tuple[EventMatch, ...]:
    """Order-preserving DP: max cardinality, then min total onset error."""

    n, m = len(target), len(predicted)
    matches = np.zeros((n + 1, m + 1), dtype=np.int32)
    errors = np.zeros((n + 1, m + 1), dtype=np.float64)
    decisions = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 target, 2 pred, 3 match

    def better(a_matches: int, a_error: float, b_matches: int, b_error: float) -> bool:
        return a_matches > b_matches or (a_matches == b_matches and a_error < b_error)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Skip target is the deterministic tie default.
            best_matches = int(matches[i - 1, j])
            best_error = float(errors[i - 1, j])
            decision = 1
            if better(int(matches[i, j - 1]), float(errors[i, j - 1]),
                      best_matches, best_error):
                best_matches = int(matches[i, j - 1])
                best_error = float(errors[i, j - 1])
                decision = 2
            onset_error = abs(target[i - 1][1].onset_frame - predicted[j - 1][1].onset_frame)
            if onset_error <= tolerance_frames + 1e-12:
                candidate_matches = int(matches[i - 1, j - 1]) + 1
                candidate_error = float(errors[i - 1, j - 1]) + onset_error
                if better(candidate_matches, candidate_error, best_matches, best_error):
                    best_matches = candidate_matches
                    best_error = candidate_error
                    decision = 3
            matches[i, j] = best_matches
            errors[i, j] = best_error
            decisions[i, j] = decision

    result = []
    i, j = n, m
    while i > 0 and j > 0:
        decision = int(decisions[i, j])
        if decision == 3:
            result.append(EventMatch(target[i - 1][0], predicted[j - 1][0]))
            i -= 1
            j -= 1
        elif decision == 2:
            j -= 1
        else:
            i -= 1
    result.reverse()
    return tuple(result)


def match_note_events(
    target_events: Sequence[NoteEvent],
    predicted_events: Sequence[NoteEvent],
    *,
    control_timestep: float,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
) -> Tuple[EventMatch, ...]:
    """One-to-one, exact-pitch note matching under an onset tolerance."""

    if not np.isfinite(control_timestep) or control_timestep <= 0:
        raise ValueError("control_timestep must be finite and positive")
    if (
        not np.isfinite(onset_tolerance_seconds)
        or onset_tolerance_seconds < 0
    ):
        raise ValueError(
            "onset_tolerance_seconds must be finite and non-negative"
        )
    tolerance_frames = onset_tolerance_seconds / control_timestep
    target_by_pitch: Dict[int, list] = {}
    pred_by_pitch: Dict[int, list] = {}
    for index, event in enumerate(target_events):
        target_by_pitch.setdefault(event.pitch, []).append((index, event))
    for index, event in enumerate(predicted_events):
        pred_by_pitch.setdefault(event.pitch, []).append((index, event))

    matches = []
    for pitch in sorted(set(target_by_pitch) | set(pred_by_pitch)):
        matches.extend(
            _match_one_pitch(
                target_by_pitch.get(pitch, ()),
                pred_by_pitch.get(pitch, ()),
                tolerance_frames,
            )
        )
    matches.sort(key=lambda item: item.target_index)
    return tuple(matches)


def _legacy_frame_macro(target: np.ndarray, actual: np.ndarray) -> Tuple[float, float, float]:
    scores = []
    for target_frame, actual_frame in zip(target, actual):
        scores.append(_prf(*_binary_counts(target_frame, actual_frame)))
    if not scores:
        return 1.0, 1.0, 1.0
    return tuple(float(np.mean([score[i] for score in scores])) for i in range(3))


def _event_metrics(
    target_events: Sequence[NoteEvent],
    actual_events: Sequence[NoteEvent],
    matches: Sequence[EventMatch],
    dt: float,
) -> Dict[str, float]:
    tp = len(matches)
    precision, recall, f1 = _prf(tp, len(actual_events) - tp, len(target_events) - tp)
    result: Dict[str, float] = {
        "note_event_precision": precision,
        "note_event_recall": recall,
        "note_event_f1": f1,
        "note_event_tp": float(tp),
        "note_event_fp": float(len(actual_events) - tp),
        "note_event_fn": float(len(target_events) - tp),
    }
    if matches:
        onset_errors = []
        offset_errors = []
        duration_errors = []
        for match in matches:
            target = target_events[match.target_index]
            actual = actual_events[match.predicted_index]
            onset_errors.append(abs(actual.onset_frame - target.onset_frame) * dt * 1000.0)
            offset_errors.append(abs(actual.offset_frame - target.offset_frame) * dt * 1000.0)
            duration_errors.append(abs(actual.duration_frames - target.duration_frames) * dt * 1000.0)
        result.update({
            "note_onset_mae_ms": float(np.mean(onset_errors)),
            "note_offset_mae_ms": float(np.mean(offset_errors)),
            "note_duration_mae_ms": float(np.mean(duration_errors)),
        })
    else:
        # Timing error is undefined without a true positive.  NaN makes failed
        # runs visible instead of pretending their error is zero.
        result.update({
            "note_onset_mae_ms": float("nan"),
            "note_offset_mae_ms": float("nan"),
            "note_duration_mae_ms": float("nan"),
        })
    return result


def attribute_note_events(
    trace: EpisodeTrace,
    actual_events: Sequence[NoteEvent],
    *,
    onset_window_seconds: float = DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    contact_force_threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
) -> ContactAttribution:
    """Attribute note events using force in N and rank participants by impulse.

    A hand participates when its peak force anywhere in the half-open onset
    window is strictly greater than ``contact_force_threshold_n``.  The
    primary hand is the participating hand with the largest integrated
    impulse.  Comparing the threshold to force rather than impulse preserves
    the threshold's physical unit and avoids timestep-dependent semantics.

    Non-finite samples in an event's relevant key/window invalidate that event
    instead of being treated as zero or being selected by ``argmax``.  Samples
    elsewhere in the trace cannot affect attribution and are not counted.
    """

    threshold = float(contact_force_threshold_n)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "contact_force_threshold_n must be finite and strictly positive"
        )

    primary = np.full(len(actual_events), -1, dtype=np.int32)
    event_valid = np.zeros(len(actual_events), dtype=bool)
    nonfinite_counts = np.zeros(len(actual_events), dtype=np.int64)
    negative_counts = np.zeros(len(actual_events), dtype=np.int64)
    participants: list[Tuple[int, ...]] = []
    if trace.key_contact_force is None or not trace.hand_names:
        return ContactAttribution(
            primary_hand_indices=primary,
            participant_hand_indices=tuple(() for _ in actual_events),
            event_valid=event_valid,
            nonfinite_sample_counts_by_event=nonfinite_counts,
            negative_sample_counts_by_event=negative_counts,
            nonfinite_event_count=0,
            nonfinite_sample_count=0,
            negative_event_count=0,
            negative_sample_count=0,
            available=False,
        )
    force = np.asarray(trace.key_contact_force, dtype=np.float64)
    window_weights = attribution_window_weights(
        trace.control_timestep, onset_window_seconds
    )
    nonfinite_event_count = 0
    nonfinite_sample_count = 0
    negative_event_count = 0
    negative_sample_count = 0
    for event_index, event in enumerate(actual_events):
        start = event.onset_frame
        stop = min(force.shape[0], start + len(window_weights))
        force_window = force[start:stop, :, event.pitch]
        finite = np.isfinite(force_window)
        event_nonfinite = int(np.count_nonzero(~finite))
        event_negative = int(np.count_nonzero(force_window[finite] < 0.0))
        nonfinite_counts[event_index] = event_nonfinite
        negative_counts[event_index] = event_negative
        if event_nonfinite:
            nonfinite_event_count += 1
            nonfinite_sample_count += event_nonfinite
        if event_negative:
            negative_event_count += 1
            negative_sample_count += event_negative
        if event_nonfinite or event_negative:
            participants.append(())
            continue
        if not force_window.shape[0]:
            participants.append(())
            continue
        event_valid[event_index] = True
        peak_force = force_window.max(axis=0)
        impulse = (
            force_window * window_weights[: force_window.shape[0], None]
        ).sum(axis=0)
        touched_array = np.flatnonzero(peak_force > threshold)
        touched = tuple(int(i) for i in touched_array)
        participants.append(touched)
        if touched:
            primary[event_index] = int(
                touched_array[int(np.argmax(impulse[touched_array]))]
            )
    return ContactAttribution(
        primary_hand_indices=primary,
        participant_hand_indices=tuple(participants),
        event_valid=event_valid,
        nonfinite_sample_counts_by_event=nonfinite_counts,
        negative_sample_counts_by_event=negative_counts,
        nonfinite_event_count=nonfinite_event_count,
        nonfinite_sample_count=nonfinite_sample_count,
        negative_event_count=negative_event_count,
        negative_sample_count=negative_sample_count,
        available=True,
    )


def _rising_event_count(active: np.ndarray) -> int:
    active = np.asarray(active, dtype=bool).reshape(-1)
    if active.size == 0:
        return 0
    return int(active[0]) + int(np.count_nonzero(~active[:-1] & active[1:]))


def _entropy_effective_count(shares: np.ndarray) -> float:
    positive = np.asarray(shares, dtype=np.float64)
    positive = positive[positive > 0]
    if not positive.size:
        return 0.0
    positive = positive / positive.sum()
    return float(exp(-sum(float(p) * log(float(p)) for p in positive)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def compute_episode_metrics(
    trace: EpisodeTrace,
    *,
    onset_tolerance_seconds: float = DEFAULT_ONSET_TOLERANCE_SECONDS,
    attribution_window_seconds: float = DEFAULT_ATTRIBUTION_WINDOW_SECONDS,
    contact_force_threshold_n: float = DEFAULT_CONTACT_FORCE_THRESHOLD_N,
    collision_force_threshold_n: float = DEFAULT_COLLISION_FORCE_THRESHOLD_N,
    motor_power_threshold_watts: float = DEFAULT_MOTOR_POWER_THRESHOLD_WATTS,
) -> Dict[str, float]:
    """Compute the complete reward-independent episode scorecard."""

    for name, value in (
        ("attribution_window_seconds", attribution_window_seconds),
        ("contact_force_threshold_n", contact_force_threshold_n),
        ("collision_force_threshold_n", collision_force_threshold_n),
        ("motor_power_threshold_watts", motor_power_threshold_watts),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")

    target = trace.target_keys
    actual = trace.actual_keys
    steps = target.shape[0]
    dt = trace.control_timestep
    metrics: Dict[str, float] = {
        "episode_steps": float(steps),
        "episode_duration_seconds": float(steps * dt),
        "note_onset_tolerance_seconds": float(onset_tolerance_seconds),
        "contact_attribution_window_seconds": float(
            attribution_window_seconds
        ),
        "contact_force_threshold_n": float(contact_force_threshold_n),
        "collision_force_threshold_n": float(collision_force_threshold_n),
        "motor_power_threshold_watts": float(motor_power_threshold_watts),
    }

    # Historical compatibility metric (frame macro, empty frames score 1).
    legacy_precision, legacy_recall, legacy_f1 = _legacy_frame_macro(target, actual)
    metrics.update({"precision": legacy_precision, "recall": legacy_recall, "f1": legacy_f1})

    # Key-time micro metrics: every key-time cell has equal weight.
    micro_p, micro_r, micro_f1 = _prf(*_binary_counts(target, actual))
    metrics.update({
        "key_time_micro_precision": micro_p,
        "key_time_micro_recall": micro_r,
        "key_time_micro_f1": micro_f1,
    })
    active_mask = np.any(target, axis=1)
    if np.any(active_mask):
        active_p, active_r, active_f1 = _prf(
            *_binary_counts(target[active_mask], actual[active_mask])
        )
    else:
        active_p, active_r, active_f1 = 1.0, 1.0, 1.0
    metrics.update({
        "active_key_time_precision": active_p,
        "active_key_time_recall": active_r,
        "active_key_time_f1": active_f1,
        "target_active_step_ratio": float(np.mean(active_mask)) if steps else 0.0,
    })
    rest_mask = ~active_mask
    rest_false_frames = np.any(actual[rest_mask], axis=1) if np.any(rest_mask) else np.zeros(0, bool)
    rest_false_keys = int(np.count_nonzero(actual[rest_mask])) if np.any(rest_mask) else 0
    rest_seconds = float(np.count_nonzero(rest_mask) * dt)
    metrics.update({
        "rest_frame_count": float(np.count_nonzero(rest_mask)),
        "rest_false_positive_rate": float(np.mean(rest_false_frames)) if rest_false_frames.size else 0.0,
        "rest_false_positive_keys": float(rest_false_keys),
        "rest_false_positive_keys_per_second": rest_false_keys / rest_seconds if rest_seconds else 0.0,
    })

    sustain_p, sustain_r, sustain_f1 = _prf(
        *_binary_counts(trace.target_sustain, trace.actual_sustain)
    )
    metrics.update({
        "sustain_precision": sustain_p,
        "sustain_recall": sustain_r,
        "sustain_f1": sustain_f1,
    })

    target_events = extract_note_events(target)
    actual_events = extract_note_events(actual)
    matches = match_note_events(
        target_events,
        actual_events,
        control_timestep=dt,
        onset_tolerance_seconds=onset_tolerance_seconds,
    )
    metrics.update(_event_metrics(target_events, actual_events, matches, dt))
    matches_100ms = match_note_events(
        target_events,
        actual_events,
        control_timestep=dt,
        onset_tolerance_seconds=0.1,
    )
    metrics["note_event_f1_100ms"] = _prf(
        len(matches_100ms),
        len(actual_events) - len(matches_100ms),
        len(target_events) - len(matches_100ms),
    )[2]

    hand_names = trace.hand_names
    if not hand_names:
        return metrics

    attribution = attribute_note_events(
        trace,
        actual_events,
        onset_window_seconds=attribution_window_seconds,
        contact_force_threshold_n=contact_force_threshold_n,
    )
    primary_hand = attribution.primary_hand_indices
    participants = attribution.participant_hand_indices
    metrics.update({
        "contact_attribution_available": float(attribution.available),
        "contact_attribution_valid": float(attribution.valid),
        "contact_attribution_nonfinite_event_count": float(
            attribution.nonfinite_event_count
        ),
        "contact_attribution_nonfinite_sample_count": float(
            attribution.nonfinite_sample_count
        ),
        "contact_attribution_negative_event_count": float(
            attribution.negative_event_count
        ),
        "contact_attribution_negative_sample_count": float(
            attribution.negative_sample_count
        ),
    })
    matched_pred_to_target = {m.predicted_index: m.target_index for m in matches}
    unattributed = int(np.count_nonzero(primary_hand < 0))
    duplicate_hands = sum(len(touched) > 1 for touched in participants)
    metrics.update({
        "unattributed_note_rate": unattributed / len(actual_events) if actual_events else 0.0,
        "duplicate_hand_note_rate": duplicate_hands / len(actual_events) if actual_events else 0.0,
    })

    hand_to_agent = dict(trace.hand_to_agent)
    agent_names = tuple(dict.fromkeys(hand_to_agent[hand] for hand in hand_names))
    hand_index = {hand: i for i, hand in enumerate(hand_names)}
    agent_index = {agent: i for i, agent in enumerate(agent_names)}
    hand_actual = np.zeros(len(hand_names), dtype=np.int64)
    hand_correct = np.zeros(len(hand_names), dtype=np.int64)
    hand_eligible_correct = np.zeros(len(hand_names), dtype=np.int64)
    agent_actual = np.zeros(len(agent_names), dtype=np.int64)
    agent_correct = np.zeros(len(agent_names), dtype=np.int64)
    agent_eligible_correct = np.zeros(len(agent_names), dtype=np.int64)
    duplicate_agents = 0
    for event_index, hand_i in enumerate(primary_hand):
        touched_agents = {
            hand_to_agent[hand_names[touched_hand]]
            for touched_hand in participants[event_index]
        }
        duplicate_agents += len(touched_agents) > 1
        if hand_i < 0:
            continue
        hand_actual[hand_i] += 1
        agent_i = agent_index[hand_to_agent[hand_names[hand_i]]]
        agent_actual[agent_i] += 1
        if event_index in matched_pred_to_target:
            hand_correct[hand_i] += 1
            agent_correct[agent_i] += 1
            target_event = target_events[matched_pred_to_target[event_index]]
            primary_hand_name = hand_names[hand_i]
            hand_lo, hand_hi = trace.hand_key_ranges.get(
                primary_hand_name, (0, 87)
            )
            if hand_lo <= target_event.pitch <= hand_hi:
                hand_eligible_correct[hand_i] += 1
            agent_name = hand_to_agent[primary_hand_name]
            if any(
                lo <= target_event.pitch <= hi
                for owned_hand in hand_names
                if hand_to_agent[owned_hand] == agent_name
                for lo, hi in (trace.hand_key_ranges.get(owned_hand, (0, 87)),)
            ):
                agent_eligible_correct[agent_i] += 1
    metrics["duplicate_agent_note_rate"] = (
        duplicate_agents / len(actual_events) if actual_events else 0.0
    )

    # Target opportunities are defined by registered key territories.
    hand_eligible = np.zeros(len(hand_names), dtype=np.int64)
    agent_eligible = np.zeros(len(agent_names), dtype=np.int64)
    for event in target_events:
        eligible_agents = set()
        for hand_i, hand in enumerate(hand_names):
            lo, hi = trace.hand_key_ranges.get(hand, (0, 87))
            if lo <= event.pitch <= hi:
                hand_eligible[hand_i] += 1
                eligible_agents.add(hand_to_agent[hand])
        for agent in eligible_agents:
            agent_eligible[agent_index[agent]] += 1

    power_available = trace.hand_power is not None
    power_nonfinite_count = 0
    power_negative_count = 0
    power_valid = False
    power: Optional[np.ndarray] = None
    if power_available:
        raw_power = np.asarray(trace.hand_power, dtype=np.float64)
        power_nonfinite_count = int(np.count_nonzero(~np.isfinite(raw_power)))
        finite_power = raw_power[np.isfinite(raw_power)]
        power_negative_count = int(np.count_nonzero(finite_power < 0.0))
        power_valid = power_nonfinite_count == 0 and power_negative_count == 0
        if power_valid:
            power = raw_power
    metrics.update({
        "hand_power_available": float(power_available),
        "hand_power_metrics_valid": float(power_valid),
        "hand_power_nonfinite_sample_count": float(power_nonfinite_count),
        "hand_power_negative_sample_count": float(power_negative_count),
    })
    if power_valid and power is not None:
        hand_work = power.sum(axis=0) * dt
        hand_motor_ratio = (
            np.mean(power > motor_power_threshold_watts, axis=0)
            if steps
            else np.zeros(len(hand_names))
        )
    else:
        hand_work = np.full(len(hand_names), np.nan, dtype=np.float64)
        hand_motor_ratio = np.full(len(hand_names), np.nan, dtype=np.float64)

    total_attributed_correct = int(hand_correct.sum())
    for i, hand in enumerate(hand_names):
        prefix = f"hand/{hand}/"
        metrics.update({
            prefix + "actual_events": float(hand_actual[i]),
            prefix + "correct_events": float(hand_correct[i]),
            prefix + "eligible_correct_events": float(hand_eligible_correct[i]),
            prefix + "eligible_target_events": float(hand_eligible[i]),
            prefix + "target_recall": hand_eligible_correct[i] / hand_eligible[i] if hand_eligible[i] else 0.0,
            prefix + "contribution_share": hand_correct[i] / total_attributed_correct if total_attributed_correct else 0.0,
            prefix + "motor_active_ratio": float(hand_motor_ratio[i]),
            prefix + "actuator_work_joule": float(hand_work[i]),
            prefix + "work_per_correct_event": hand_work[i] / hand_correct[i] if hand_correct[i] else float("nan"),
        })

    agent_work = np.zeros(len(agent_names), dtype=np.float64)
    agent_motor = np.zeros(len(agent_names), dtype=np.float64)
    for agent, agent_i in agent_index.items():
        owned = [hand_index[h] for h in hand_names if hand_to_agent[h] == agent]
        agent_work[agent_i] = float(np.sum(hand_work[owned]))
        if owned:
            if not power_valid or power is None:
                agent_motor[agent_i] = float("nan")
            else:
                owned_power = power[:, owned]
                agent_motor[agent_i] = float(
                    np.mean(
                        np.any(
                            owned_power > motor_power_threshold_watts,
                            axis=1,
                        )
                    )
                ) if steps else 0.0

    total_agent_correct = int(agent_correct.sum())
    for i, agent in enumerate(agent_names):
        prefix = f"agent/{agent}/"
        metrics.update({
            prefix + "actual_events": float(agent_actual[i]),
            prefix + "correct_events": float(agent_correct[i]),
            prefix + "eligible_correct_events": float(agent_eligible_correct[i]),
            prefix + "eligible_target_events": float(agent_eligible[i]),
            prefix + "target_recall": agent_eligible_correct[i] / agent_eligible[i] if agent_eligible[i] else 0.0,
            prefix + "contribution_share": agent_correct[i] / total_agent_correct if total_agent_correct else 0.0,
            prefix + "motor_active_ratio": float(agent_motor[i]),
            prefix + "actuator_work_joule": float(agent_work[i]),
            prefix + "work_per_correct_event": agent_work[i] / agent_correct[i] if agent_correct[i] else float("nan"),
        })

    active_agents = int(np.count_nonzero(agent_correct))
    motor_active_agents = (
        int(np.count_nonzero(agent_motor > 0.0)) if power_valid else 0
    )
    metrics.update({
        "active_agent_count": float(active_agents),
        "effective_active_agents": _entropy_effective_count(agent_correct),
        "agent_coverage": active_agents / len(agent_names) if agent_names else 0.0,
        "task_idle_agent_rate": 1.0 - active_agents / len(agent_names) if agent_names else 0.0,
        "motor_idle_agent_rate": (
            1.0 - motor_active_agents / len(agent_names)
            if agent_names and power_valid
            else float("nan")
        ),
        "actuator_work_joule": float(agent_work.sum()),
        "work_per_correct_event": agent_work.sum() / len(matches) if matches else float("nan"),
    })
    if agent_correct.sum() and agent_eligible.sum():
        actual_share = agent_correct / agent_correct.sum()
        target_share = agent_eligible / agent_eligible.sum()
        metrics["workload_l1_mismatch"] = float(0.5 * np.sum(np.abs(actual_share - target_share)))
        metrics["workload_js_divergence"] = _js_divergence(actual_share, target_share)
    else:
        metrics["workload_l1_mismatch"] = float("nan")
        metrics["workload_js_divergence"] = float("nan")

    # A single non-finite force in an event's relevant key/window makes the
    # trace unsuitable for formal attribution claims.  Preserve independent
    # music, eligibility, power, and global efficiency metrics, but make every
    # attribution-derived scalar explicitly missing so consumers cannot
    # accidentally rank policies while ignoring the validity gate.
    if not attribution.valid:
        attribution_top_level = {
            "unattributed_note_rate",
            "duplicate_hand_note_rate",
            "duplicate_agent_note_rate",
            "active_agent_count",
            "effective_active_agents",
            "agent_coverage",
            "task_idle_agent_rate",
            "workload_l1_mismatch",
            "workload_js_divergence",
        }
        attribution_entity_suffixes = (
            "/actual_events",
            "/correct_events",
            "/eligible_correct_events",
            "/target_recall",
            "/contribution_share",
            "/work_per_correct_event",
        )
        for name in tuple(metrics):
            if name in attribution_top_level or (
                (name.startswith("hand/") or name.startswith("agent/"))
                and name.endswith(attribution_entity_suffixes)
            ):
                metrics[name] = float("nan")

    # Physical collision metrics.  Pairwise rising edges define collision
    # events; force is integrated over control steps to produce N*s.
    collision_available = trace.hand_collision_force is not None
    hand_pair_series: Optional[list[np.ndarray]] = None
    agent_pair_series: Optional[list[np.ndarray]] = None
    if collision_available:
        collision = np.asarray(trace.hand_collision_force, dtype=np.float64)
        hand_pair_series = []
        agent_pair_series = []
        for i in range(len(hand_names)):
            for j in range(i + 1, len(hand_names)):
                series = collision[:, i, j]
                hand_pair_series.append(series)
                if hand_to_agent[hand_names[i]] != hand_to_agent[hand_names[j]]:
                    agent_pair_series.append(series)

    def add_collision_metrics(
        prefix: str,
        series_list: Optional[Sequence[np.ndarray]],
    ) -> None:
        value_names = (
            "collision_step_rate",
            "collision_event_count",
            "collision_force_time_integral_ns",
            "max_collision_force_n",
        )
        metrics[prefix + "collision_metrics_available"] = float(
            series_list is not None
        )
        if series_list is None:
            metrics[prefix + "collision_metrics_valid"] = 0.0
            metrics[prefix + "collision_nonfinite_sample_count"] = 0.0
            metrics[prefix + "collision_negative_sample_count"] = 0.0
            for name in value_names:
                metrics[prefix + name] = float("nan")
            return
        if not series_list:
            metrics[prefix + "collision_metrics_valid"] = 1.0
            metrics[prefix + "collision_nonfinite_sample_count"] = 0.0
            metrics[prefix + "collision_negative_sample_count"] = 0.0
            for name in value_names:
                metrics[prefix + name] = 0.0
            return
        stacked = np.stack(series_list, axis=1)
        nonfinite_count = int(np.count_nonzero(~np.isfinite(stacked)))
        finite = np.isfinite(stacked)
        negative_count = int(np.count_nonzero(stacked[finite] < 0.0))
        metrics[prefix + "collision_nonfinite_sample_count"] = float(
            nonfinite_count
        )
        metrics[prefix + "collision_negative_sample_count"] = float(
            negative_count
        )
        metrics[prefix + "collision_metrics_valid"] = float(
            nonfinite_count == 0 and negative_count == 0
        )
        if nonfinite_count or negative_count:
            # Never publish the old mixed state where rate/count looked valid
            # while integral/max serialized as null.  One invalid sample makes
            # all four metrics in that collision scope unavailable.
            for name in value_names:
                metrics[prefix + name] = float("nan")
            return
        filtered = np.where(
            stacked > collision_force_threshold_n,
            np.maximum(stacked, 0.0),
            0.0,
        )
        active = filtered > 0.0
        metrics[prefix + "collision_step_rate"] = (
            float(np.mean(np.any(active, axis=1))) if steps else 0.0
        )
        metrics[prefix + "collision_event_count"] = float(
            sum(
                _rising_event_count(active[:, i])
                for i in range(active.shape[1])
            )
        )
        metrics[prefix + "collision_force_time_integral_ns"] = float(
            filtered.sum() * dt
        )
        metrics[prefix + "max_collision_force_n"] = (
            float(filtered.max()) if filtered.size else 0.0
        )

    add_collision_metrics("inter_hand_", hand_pair_series)
    add_collision_metrics("inter_agent_", agent_pair_series)

    return metrics
