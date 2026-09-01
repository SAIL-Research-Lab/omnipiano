"""Strict JSON serialization helpers for benchmark artifacts.

Python's :mod:`json` encoder emits ``NaN`` and ``Infinity`` by default even
though neither token is part of the JSON standard.  Undefined benchmark
metrics are represented internally as non-finite floats, so every artifact
writer must normalize them to JSON ``null`` before encoding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _sanitize_mapping_key(key: Any) -> str:
    """Return the RFC JSON object-key representation for ``key``.

    JSON object keys are always strings.  This mirrors the useful subset of
    ``json.dumps`` key coercion while also accepting NumPy scalar keys and
    ensuring non-finite numeric keys never introduce non-standard tokens.
    """

    if isinstance(key, np.bool_):
        key = bool(key)
    elif isinstance(key, np.integer):
        key = int(key)
    elif isinstance(key, np.floating):
        key = float(key)
    elif isinstance(key, np.str_):
        key = str(key)
    elif isinstance(key, np.generic):
        key = key.item()
    if isinstance(key, str):
        return key
    if key is None:
        return "null"
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, float):
        return str(key) if math.isfinite(key) else "null"
    raise TypeError(
        "JSON mapping keys must be str, int, float, bool, None, or a NumPy "
        f"scalar; got {type(key).__name__}"
    )


def sanitize_json(value: Any) -> Any:
    """Recursively convert ``value`` into a strict JSON-compatible value.

    ``NaN`` and positive/negative infinity become ``None`` (JSON ``null``).
    NumPy scalars and arrays become their Python scalar/list equivalents.
    Mappings and non-string sequences are traversed recursively.  Unsupported
    leaf values are left for :func:`json.dumps` to reject with its usual clear
    ``TypeError`` rather than being silently stringified.
    """

    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    # Handle NumPy numeric subclasses explicitly.  In particular,
    # ``np.longdouble.item()`` can return another ``np.longdouble`` rather
    # than a Python float, so blindly recurring through ``item()`` would loop.
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return sanitize_json(float(value))
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            sanitized_key = _sanitize_mapping_key(key)
            if sanitized_key in sanitized:
                raise ValueError(
                    "mapping contains keys that collide after JSON "
                    f"normalization: {sanitized_key!r}"
                )
            sanitized[sanitized_key] = sanitize_json(item)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [sanitize_json(item) for item in value]
    return value


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize ``value`` as standards-compliant JSON.

    ``allow_nan`` is deliberately forced to ``False`` so a future unsupported
    code path cannot silently reintroduce non-standard numeric constants.
    """

    kwargs["allow_nan"] = False
    return json.dumps(sanitize_json(value), **kwargs)
