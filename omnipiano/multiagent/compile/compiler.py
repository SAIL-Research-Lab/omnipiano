"""Pure configuration compiler for OmniPiano MARL experiments.

This module reads and validates versioned JSON, selects algorithm-specific
overrides, and returns a :class:`ResolvedExperiment`. It never imports or
starts Ray, MuJoCo, PettingZoo, or W&B and performs no runtime side effects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from omnipiano.multiagent.compile.schema import (
    CONFIG_FIELDS,
    TRAIN_CONFIG_SCHEMA_VERSION,
    TASK_CONFIG_SCHEMA_VERSION,
    ExperimentRequest,
    ResolvedAgent,
    ResolvedExperiment,
    ResolvedHand,
    ResolvedTask,
)
from omnipiano.multiagent.compile.presets import (
    DEFAULT_ASSIGNMENTS, LAYOUTS, SONGS,
)


DEFAULT_TRAIN_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "marl_train_config_default.json"
)


def resolve_train_config_path(path: os.PathLike[str] | str) -> Path:
    """Resolve a config path exactly once and require a regular file."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"MARL training config does not exist: {candidate}")
    return candidate


def compile_experiment(
    path: os.PathLike[str] | str,
    *,
    registered_algorithms: Iterable[str],
    algo_override: Optional[str] = None,
) -> ResolvedExperiment:
    """Compile legacy v1 or a v2 task request into trainer-ready defaults.

    ``registered_algorithms`` is injected by the caller so this module stays
    independent of the algorithm registry and its optional ML dependencies.
    """
    config_path = resolve_train_config_path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: top-level JSON value must be an object")
    source_snapshot = deepcopy(raw)
    version = raw.get("schema_version")
    if type(version) is not int or version not in (
        TRAIN_CONFIG_SCHEMA_VERSION, TASK_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(
            f"{config_path}: unsupported schema_version "
            f"{raw.get('schema_version')!r}; expected "
            f"1 (legacy env_id) or 2 (custom task)"
        )

    allowed_top = {
        "schema_version",
        "description",
        *CONFIG_FIELDS,
        "native",
        "smoke_test_overrides",
        "algorithm_overrides",
    }
    if version == TASK_CONFIG_SCHEMA_VERSION:
        allowed_top.update({"task", "extends"})
    unknown_top = sorted(set(raw) - allowed_top)
    if unknown_top:
        raise ValueError(f"{config_path}: unknown top-level keys {unknown_top}")

    task = None
    if version == TASK_CONFIG_SCHEMA_VERSION:
        task = compile_task(raw.get("task"))
        if raw.get("experiment", {}).get("env_id") is not None:
            raise ValueError("schema v2: use task, not experiment.env_id")
        # A compact request inherits the archived/current full v1 defaults.
        # Only explicit user sections override them; source JSON is kept intact.
        parent = raw.get("extends")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ValueError("extends must be a non-empty path string")
        parent_path = (config_path.parent / parent if parent is not None
                       else DEFAULT_TRAIN_CONFIG_PATH)
        parent_compiled = compile_experiment(
            parent_path, registered_algorithms=registered_algorithms,
        ) if resolve_train_config_path(parent_path) != config_path else None
        if parent_compiled is None or parent_compiled.request.schema_version != 1:
            raise ValueError("schema v2 extends must reference a schema-v1 config")
        inherited = parent_compiled.request.snapshot()
        for section in CONFIG_FIELDS:
            if section in raw:
                if not isinstance(raw[section], dict):
                    raise ValueError(f"{section} must be an object")
                inherited.setdefault(section, {}).update(raw[section])
        for key in (
            "description", "smoke_test_overrides", "algorithm_overrides", "native"
        ):
            if key in raw:
                if key == "description":
                    inherited[key] = raw[key]
                elif key == "native":
                    if not isinstance(raw[key], dict):
                        raise ValueError("native must be an object")
                    inherited.setdefault(key, {}).update(raw[key])
                else:
                    if not isinstance(raw[key], dict):
                        raise ValueError(f"{key} must be an object")
                    # Per-algorithm section dictionaries merge without dropping
                    # unrelated inherited defaults (e.g. RLlib value clipping).
                    target = inherited.setdefault(key, {})
                    for item, value in raw[key].items():
                        if key == "algorithm_overrides" and isinstance(value, dict):
                            algo_sections = target.setdefault(item, {})
                            for section, fields in value.items():
                                if not isinstance(fields, dict):
                                    raise ValueError(f"{key}.{item}.{section} must be an object")
                                algo_sections.setdefault(section, {}).update(fields)
                        else:
                            target[item] = value
        raw = inherited
        digest = hashlib.sha256(
            json.dumps(task.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:10]
        raw["experiment"]["env_id"] = f"OmniPiano-Custom-{task.name}-{digest}-v0"

    defaults: Dict[str, Any] = {}

    def _merge_sections(container: Mapping[str, Any], *, label: str) -> None:
        unknown_sections = sorted(set(container) - set(CONFIG_FIELDS))
        if unknown_sections:
            raise ValueError(
                f"{config_path}: {label} has unknown sections {unknown_sections}"
            )
        for section, values in container.items():
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"{config_path}: {label}.{section} must be an object"
                )
            field_map = CONFIG_FIELDS[section]
            unknown_fields = sorted(set(values) - set(field_map))
            if unknown_fields:
                raise ValueError(
                    f"{config_path}: {label}.{section} has unknown fields "
                    f"{unknown_fields}"
                )
            if label == "defaults":
                missing_fields = sorted(set(field_map) - set(values))
                if missing_fields:
                    raise ValueError(
                        f"{config_path}: defaults.{section} is missing fields "
                        f"{missing_fields}"
                    )
            for key, value in values.items():
                destination = field_map[key]
                if (
                    label.startswith("algorithm_overrides")
                    and destination == "algo"
                ):
                    raise ValueError(
                        f"{config_path}: an algorithm override cannot change algo"
                    )
                defaults[destination] = value

    base_sections: Dict[str, Any] = {}
    for section in CONFIG_FIELDS:
        if section not in raw:
            # Reward shaping post-dates schema v1. Its absence means the exact
            # historical reward rather than the current canonical coefficient.
            if section == "reward":
                base_sections[section] = {
                    "inter_agent_collision_penalty_coef": 0.0,
                }
                continue
            raise ValueError(f"{config_path}: missing required section {section!r}")
        base_sections[section] = raw[section]
    _merge_sections(base_sections, label="defaults")

    algorithms = tuple(sorted(set(registered_algorithms)))
    configured_algo = str(defaults.get("algo", ""))
    if configured_algo not in algorithms:
        raise ValueError(
            f"{config_path}: unknown configured algorithm {configured_algo!r}; "
            f"registered algorithms are {algorithms}"
        )
    selected_algo = str(algo_override or configured_algo)
    if selected_algo not in algorithms:
        raise ValueError(
            f"unknown --algo {selected_algo!r}; registered algorithms are "
            f"{algorithms}"
        )

    algorithm_overrides = raw.get("algorithm_overrides", {})
    if not isinstance(algorithm_overrides, Mapping):
        raise ValueError(f"{config_path}: algorithm_overrides must be an object")
    unknown_algorithms = sorted(set(algorithm_overrides) - set(algorithms))
    if unknown_algorithms:
        raise ValueError(
            f"{config_path}: overrides reference unknown algorithms "
            f"{unknown_algorithms}"
        )
    selected_overrides = algorithm_overrides.get(selected_algo, {})
    if not isinstance(selected_overrides, Mapping):
        raise ValueError(
            f"{config_path}: algorithm_overrides.{selected_algo} must be an object"
        )
    selected_native = selected_overrides.get("native", {})
    if not isinstance(selected_native, Mapping):
        raise ValueError(
            f"{config_path}: algorithm_overrides.{selected_algo}.native "
            "must be an object"
        )
    selected_training_overrides = {
        key: value for key, value in selected_overrides.items() if key != "native"
    }
    _merge_sections(
        selected_training_overrides,
        label=f"algorithm_overrides.{selected_algo}",
    )
    defaults["algo"] = selected_algo
    if task is not None and defaults["env_id"] != raw["experiment"]["env_id"]:
        raise ValueError("algorithm_overrides cannot change env_id for a custom task")

    smoke_overrides = raw.get("smoke_test_overrides", {})
    if not isinstance(smoke_overrides, Mapping):
        raise ValueError(f"{config_path}: smoke_test_overrides must be an object")
    valid_destinations = {
        destination
        for fields in CONFIG_FIELDS.values()
        for destination in fields.values()
    }
    unknown_smoke = sorted(set(smoke_overrides) - valid_destinations)
    if unknown_smoke:
        raise ValueError(
            f"{config_path}: smoke_test_overrides has unknown fields {unknown_smoke}"
        )
    if task is not None and "env_id" in smoke_overrides:
        raise ValueError("smoke_test_overrides cannot change env_id for a custom task")

    shared_native = raw.get("native", {})
    if not isinstance(shared_native, Mapping):
        raise ValueError(f"{config_path}: native must be an object")
    native_options = dict(shared_native)
    native_options.update(selected_native)

    request = ExperimentRequest(
        schema_version=version,
        description=str(raw.get("description", "")),
        sections={
            section: dict(raw[section])
            for section in CONFIG_FIELDS
            if section in raw
        },
        smoke_test_overrides=dict(smoke_overrides),
        algorithm_overrides=dict(algorithm_overrides),
        _snapshot=source_snapshot,
    )
    return ResolvedExperiment(
        request=request,
        config_path=config_path,
        algorithm=selected_algo,
        values=dict(defaults),
        smoke_test_overrides=dict(smoke_overrides),
        native_options=native_options,
        task=task,
    )


def compile_task(raw: Any) -> ResolvedTask:
    """Resolve ownership, independent observations, and overlapping wrist ranges.

    No physical model is built here. Explicit hand IDs may be non-contiguous;
    ownership never changes the physical hand/action/global-state order.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("task must be an object")
    allowed = {"name", "song", "num_hands", "num_agents", "assignment",
               "agents", "sustain_owner"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"task has unknown fields {unknown}")
    n = raw.get("num_hands")
    if type(n) is not int or n not in LAYOUTS:
        raise ValueError("task.num_hands must be 3, 4 or 5 (existing layouts)")
    layout, hand_names, buckets = LAYOUTS[n]
    song = raw.get("song")
    if not isinstance(song, str) or not re.fullmatch(r"[A-Za-z0-9]+", song):
        raise ValueError("task.song must be a RoboPianist repertoire token")
    name = raw.get("name", f"{song}-{n}hand")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("task.name must contain only letters, digits, '_' or '-'")
    mode = raw.get("assignment", "explicit" if "agents" in raw else "default")
    if mode not in {"default", "balanced", "explicit"}:
        raise ValueError("task.assignment must be default, balanced or explicit")
    if mode == "default":
        groups = DEFAULT_ASSIGNMENTS[n]
        m = raw.get("num_agents", len(groups))
        if type(m) is not int or m != len(groups):
            raise ValueError(
                "task.num_agents conflicts with default assignment; "
                "use balanced/explicit"
            )
    elif mode == "balanced":
        m = raw.get("num_agents")
        if type(m) is not int or not 1 <= m <= n:
            raise ValueError("task.num_agents must be an integer in [1, num_hands]")
        q, remainder = divmod(n, m)
        cursor = 0
        groups_list = []
        for i in range(m):
            size = q + (i < remainder)
            groups_list.append((f"agent_{i + 1}", tuple(range(cursor, cursor + size))))
            cursor += size
        groups = tuple(groups_list)
    else:
        entries = raw.get("agents")
        if not isinstance(entries, list) or not entries:
            raise ValueError("explicit assignment requires a non-empty task.agents list")
        groups = tuple((a.get("name"), a.get("hand_ids"))
                       if isinstance(a, Mapping) else (None, None) for a in entries)
        m = raw.get("num_agents", len(groups))
        if type(m) is not int or m != len(groups):
            raise ValueError("task.num_agents must match task.agents length")
        if m < 1:
            raise ValueError("a compiled control task requires at least one agent")

    entries = raw.get("agents", [])
    if not isinstance(entries, list):
        raise ValueError("task.agents must be a list")
    overrides = {}
    agent_fields = {"name", "hand_ids", "action_key_range", "observation_key_range",
                    "visible_teammate_hands"}
    for a in entries:
        if not isinstance(a, Mapping) or set(a) - agent_fields:
            raise ValueError(f"invalid task.agents entry; allowed fields: {sorted(agent_fields)}")
        if not isinstance(a.get("name"), str) or a["name"] in overrides:
            raise ValueError("task.agents names must be present and unique")
        if mode != "explicit" and "hand_ids" in a:
            raise ValueError("hand_ids requires assignment=explicit")
        overrides[a["name"]] = a
    names = [g[0] for g in groups]
    if any(not isinstance(a, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", a)
           or a in {"__common__", "_global_"} for a in names) or len(set(names)) != m:
        raise ValueError("agent names must be unique identifiers starting with a letter")
    if set(overrides) - set(names):
        raise ValueError(
            "agent overrides reference unknown names "
            f"{sorted(set(overrides) - set(names))}"
        )
    ownership = {}
    normalized = []
    for agent, ids in groups:
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError(f"agent {agent}: hand_ids must be non-empty")
        for h in ids:
            if type(h) is not int or not 0 <= h < n or h in ownership:
                raise ValueError(
                    "hand_ids must be unique zero-based IDs in [0, num_hands); "
                    "every hand has one owner"
                )
            ownership[h] = agent
        normalized.append((agent, tuple(sorted(ids))))
    if set(ownership) != set(range(n)):
        raise ValueError("every hand must be assigned exactly once")
    # Stable policy/global one-hot order, independent of JSON list ordering.
    normalized.sort(key=lambda item: min(item[1]))
    sustain = raw.get("sustain_owner", normalized[0][0])
    if sustain not in names:
        raise ValueError("task.sustain_owner must name exactly one existing agent")

    def key_range(value, label):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(x) is not int for x in value)
                or not 1 <= value[0] < value[1] <= 88):
            raise ValueError(f"{label} must be an inclusive [lo, hi] in 1..88 with lo < hi")
        return value[0] - 1, value[1] - 1

    agents = []
    for agent, ids in normalized:
        a = overrides.get(agent, {})
        action = (
            key_range(a["action_key_range"], f"{agent}.action_key_range")
            if "action_key_range" in a
            else (min(buckets[h][0] for h in ids),
                  max(buckets[h][1] for h in ids))
        )
        observation = (
            key_range(
                a["observation_key_range"], f"{agent}.observation_key_range"
            ) if "observation_key_range" in a else action
        )
        visible = a.get("visible_teammate_hands", "boundary")
        if visible == "none":
            visible_names = ()
        elif visible == "all":
            visible_names = tuple(hand_names[h] for h in range(n) if h not in ids)
        elif visible == "boundary":
            adjacent = {j for h in ids for j in (h - 1, h + 1)
                        if 0 <= j < n and ownership[j] != agent}
            visible_names = tuple(hand_names[h] for h in sorted(adjacent))
        elif isinstance(visible, list):
            invalid = any(
                type(h) is not int or not 0 <= h < n or h in ids
                for h in visible
            )
            if invalid or len(set(visible)) != len(visible):
                raise ValueError(
                    f"{agent}.visible_teammate_hands must contain unique "
                    "other-agent hand IDs"
                )
            visible_names = tuple(hand_names[h] for h in sorted(visible))
        else:
            raise ValueError(
                "visible_teammate_hands must be none, boundary, all or a "
                "hand-ID list"
            )
        agents.append(ResolvedAgent(
            agent, ids, action, observation, visible_names, agent == sustain
        ))
    return ResolvedTask(
        name=name, song=song,
        base_env_name=SONGS.get(
            song, f"RoboPianist-repertoire-150-{song}-v0"
        ), layout=layout,
        assignment_mode=mode,
        hands=tuple(ResolvedHand(h, hand_names[h], buckets[h]) for h in range(n)),
        agents=tuple(agents),
    )


__all__ = [
    "DEFAULT_TRAIN_CONFIG_PATH",
    "compile_experiment",
    "compile_task",
    "resolve_train_config_path",
]
