"""Run-level aggregation for learning curves, robustness sweeps, and seeds."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def normalized_curve_auc(
    steps: Sequence[float],
    values: Sequence[float],
    *,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> float:
    """Trapezoidal AUC normalized to [0, 1] over the observed step interval.

    Formal comparisons must use the same first/last evaluation steps.  This
    function deliberately does not invent a step-zero value when the training
    pipeline did not evaluate the untrained policy.
    """

    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("steps and values must be same-length 1D arrays")
    if x.size < 2:
        raise ValueError("AUC requires at least two evaluation points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("AUC inputs must be finite")
    if not np.all(np.diff(x) > 0):
        raise ValueError("evaluation steps must be strictly increasing")
    if not value_max > value_min:
        raise ValueError("value_max must be greater than value_min")
    scaled = (np.clip(y, value_min, value_max) - value_min) / (
        value_max - value_min
    )
    # ``numpy.trapezoid`` was introduced in NumPy 2.0, while RAIDEN's pinned
    # CUDA container uses NumPy 1.26 for binary compatibility with Torch 2.5.
    # Keep both environments on the identical trapezoidal definition.
    # Do not put ``np.trapz`` in ``getattr``'s default expression: Python
    # evaluates that expression eagerly, and NumPy versions that removed the
    # deprecated alias would fail even though ``np.trapezoid`` exists.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(scaled, x=x) / (x[-1] - x[0]))


def steps_to_threshold(
    steps: Sequence[float],
    values: Sequence[float],
    threshold: float,
) -> Optional[float]:
    """First threshold crossing, linearly interpolated; ``None`` if absent."""

    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("steps and values must be non-empty same-length 1D arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("threshold inputs must be finite")
    if x.size > 1 and not np.all(np.diff(x) > 0):
        raise ValueError("evaluation steps must be strictly increasing")
    reached = np.flatnonzero(y >= threshold)
    if not reached.size:
        return None
    index = int(reached[0])
    if index == 0 or y[index] == y[index - 1]:
        return float(x[index])
    fraction = (threshold - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def aggregate_seed_values(values: Sequence[float]) -> Dict[str, float]:
    """Paper-facing seed statistics with failures kept visible.

    Non-finite seed results count toward ``failure_rate`` and are excluded from
    mean/std.  A run set with no finite result returns NaN for its descriptive
    statistics instead of silently returning zero.
    """

    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("values must be a non-empty 1D sequence")
    finite = raw[np.isfinite(raw)]
    result = {
        "num_seeds": float(raw.size),
        "num_successful_seeds": float(finite.size),
        "failure_rate": float(1.0 - finite.size / raw.size),
    }
    if finite.size:
        result.update({
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "median": float(np.median(finite)),
        })
    else:
        result.update({key: float("nan") for key in ("mean", "std", "min", "max", "median")})
    return result


def aggregate_seed_scorecards(
    scorecards: Mapping[int, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate every common metric across seed-indexed final scorecards."""

    if not scorecards:
        raise ValueError("scorecards must contain at least one seed")
    common = set.intersection(*(set(card) for card in scorecards.values()))
    return {
        metric: aggregate_seed_values(
            [float(scorecards[seed][metric]) for seed in sorted(scorecards)]
        )
        for metric in sorted(common)
    }
