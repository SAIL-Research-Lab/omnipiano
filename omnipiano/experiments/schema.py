"""Load and validate the versioned, backend-neutral experiment contract."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from omnipiano.experiments.models import (
    AlgorithmRequest,
    DesignRequest,
    ExperimentMetadata,
    ExperimentRequest,
    RunRequest,
    TaskRequest,
)


CONTRACT_SCHEMA_VERSION = 3
PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_DIR / "contract" / "experiment-v3.schema.json"


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


class ExperimentValidationError(ValueError):
    """One or more structural or cross-field contract violations."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        details = "\n".join(
            f"  {issue.path}: {issue.message}" for issue in self.issues
        )
        super().__init__(f"invalid experiment configuration:\n{details}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except FileNotFoundError as exc:
        raise ExperimentValidationError(
            [ValidationIssue("$", f"file not found: {path}")]
        ) from exc
    except json.JSONDecodeError as exc:
        raise ExperimentValidationError(
            [
                ValidationIssue(
                    "$", f"invalid JSON at line {exc.lineno}, column {exc.colno}"
                )
            ]
        ) from exc
    except ValueError as exc:
        raise ExperimentValidationError(
            [ValidationIssue("$", str(exc))]
        ) from exc
    if not isinstance(value, dict):
        raise ExperimentValidationError(
            [ValidationIssue("$", "top-level JSON value must be an object")]
        )
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    schema = _read_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    return schema


def _json_path(parts: Sequence[Any]) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def _semantic_issues(document: Mapping[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    task = document["task"]

    if task["source"] == "custom":
        embodiment = task["embodiment"]
        hands = embodiment["hands"]
        if embodiment["num_hands"] != len(hands):
            issues.append(
                ValidationIssue(
                    "$.task.embodiment.num_hands",
                    "must equal the number of entries in hands",
                )
            )
        hand_ids = [hand["id"] for hand in hands]
        if len(hand_ids) != len(set(hand_ids)):
            issues.append(
                ValidationIssue(
                    "$.task.embodiment.hands", "hand IDs must be unique"
                )
            )
        for index, hand in enumerate(hands):
            key_range = hand.get("initial_key_range")
            if key_range is not None and key_range[0] > key_range[1]:
                issues.append(
                    ValidationIssue(
                        f"$.task.embodiment.hands[{index}].initial_key_range",
                        "range start must not exceed range end",
                    )
                )

    tracking = document["run"].get("tracking", {})
    if tracking.get("provider", "none") == "none" and tracking.get(
        "mode", "disabled"
    ) != "disabled":
        issues.append(
            ValidationIssue(
                "$.run.tracking.mode",
                "must be disabled when tracking.provider is none",
            )
        )

    video = document["run"].get("video", {})
    if video.get("enabled", False) and video.get("frequency", 0) < 1:
        issues.append(
            ValidationIssue(
                "$.run.video.frequency",
                "must be positive when video recording is enabled",
            )
        )

    return issues


def validate_document(document: Mapping[str, Any]) -> None:
    """Validate structure and backend-neutral cross-field invariants."""
    validator = Draft202012Validator(_schema())
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    issues = [
        ValidationIssue(_json_path(list(error.absolute_path)), error.message)
        for error in errors
    ]
    if not issues:
        issues.extend(_semantic_issues(document))
    if issues:
        raise ExperimentValidationError(issues)


def parse_request(document: Mapping[str, Any]) -> ExperimentRequest:
    """Build immutable core objects from a validated JSON mapping."""
    validate_document(document)

    experiment = document["experiment"]
    task = document["task"]
    design = document["design"]
    algorithm = document["algorithm"]
    run = document["run"]

    task_payload = dict(task)
    task_payload.pop("source")
    environment_id = task_payload.pop("environment_id", None)

    run_options = dict(run)
    total_steps = run_options.pop("total_steps")
    gamma = run_options.pop("gamma")

    return ExperimentRequest(
        schema_version=document["schema_version"],
        experiment=ExperimentMetadata(
            name=experiment["name"],
            description=experiment.get("description", ""),
            seeds=tuple(experiment["seeds"]),
            tags=tuple(experiment.get("tags", [])),
        ),
        task=TaskRequest(
            source=task["source"],
            environment_id=environment_id,
            payload=task_payload,
        ),
        design=DesignRequest(
            type=design["type"],
            config=deepcopy(design.get("config", {})),
        ),
        algorithm=AlgorithmRequest(
            name=algorithm["name"],
            preset=algorithm.get("preset", "default"),
            parameters=deepcopy(algorithm.get("parameters", {})),
        ),
        run=RunRequest(
            total_steps=total_steps,
            gamma=gamma,
            options=deepcopy(run_options),
        ),
        _document=deepcopy(dict(document)),
    )


def load_request(path: str | Path) -> ExperimentRequest:
    """Read and validate one canonical experiment JSON file."""
    return parse_request(_read_json(Path(path).expanduser().resolve()))
