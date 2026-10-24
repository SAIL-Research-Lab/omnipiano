"""Pure, version-pinned Safety RL algorithm parameter profiles.

The reference values are derived from OmniSafe 0.5.0's own YAML files.  The
JSON snapshot deliberately omits experiment protocol, task budget, and
algorithm-identity fields that the OmniPiano runner owns (for example gamma,
cost_limit, and use_cost).  Everything left in the profile is safe to present
as an editable algorithm/model override.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping


_PROFILE_PATH = Path(__file__).with_name("parameter_defaults.json")
_PROFILES = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))["algorithms"]


def parameter_template(algorithm: str) -> dict[str, Any]:
    """Return a detached, complete editable profile for ``algorithm``."""
    try:
        return deepcopy(_PROFILES[algorithm])
    except KeyError as exc:
        raise ValueError(f"No Safety RL parameter profile for {algorithm!r}") from exc


def configurable_sections(algorithm: str) -> tuple[str, ...]:
    """Top-level JSON keys accepted by the unified experiment contract."""
    sections = tuple(parameter_template(algorithm))
    # Preserve the original flat spelling accepted by the first adapter.  New
    # UI documents use ``algo_cfgs.size`` instead.
    return (*sections, "buffer_size") if "size" in _PROFILES[algorithm]["algo_cfgs"] else sections


def _validate_keys(
    supplied: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    path: str,
) -> None:
    unknown = sorted(set(supplied) - set(reference))
    if unknown:
        raise ValueError(f"unsupported Safety RL parameter fields at {path}: {unknown}")
    for key, value in supplied.items():
        expected = reference[key]
        if isinstance(expected, Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}.{key} must be a JSON object")
            _validate_keys(value, expected, path=f"{path}.{key}")


def _deep_merge(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def apply_parameter_overrides(
    config: Mapping[str, Any],
    algorithm: str,
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and merge UI parameters into an OmniSafe ``custom_cfgs`` map."""
    result = deepcopy(dict(config))
    supplied = deepcopy(dict(parameters or {}))
    legacy_size = supplied.pop("buffer_size", None)
    reference = parameter_template(algorithm)
    _validate_keys(supplied, reference, path="algorithm.parameters")
    _deep_merge(result, supplied)
    if legacy_size is not None:
        if "size" not in reference.get("algo_cfgs", {}):
            raise ValueError("buffer_size is only meaningful for off-policy algorithms")
        if "size" in supplied.get("algo_cfgs", {}):
            raise ValueError("set either buffer_size or algo_cfgs.size, not both")
        result.setdefault("algo_cfgs", {})["size"] = legacy_size
    return result
