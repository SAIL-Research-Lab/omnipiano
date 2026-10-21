"""Pure, JSON-serializable data structures for the unified control plane."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


JsonMapping = Mapping[str, Any]


def _copy(value: Any) -> Any:
    """Return a detached JSON-compatible value."""
    return deepcopy(value)


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str
    description: str
    seeds: Tuple[int, ...]
    tags: Tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "seeds": list(self.seeds),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class TaskRequest:
    """A registered environment reference or a future custom task request."""

    source: str
    environment_id: Optional[str]
    payload: JsonMapping

    def to_dict(self) -> dict[str, Any]:
        result = {"source": self.source, **_copy(dict(self.payload))}
        if self.environment_id is not None:
            result["environment_id"] = self.environment_id
        return result


@dataclass(frozen=True)
class DesignRequest:
    type: str
    config: JsonMapping

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "config": _copy(dict(self.config))}


@dataclass(frozen=True)
class AlgorithmRequest:
    name: str
    preset: str
    parameters: JsonMapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preset": self.preset,
            "parameters": _copy(dict(self.parameters)),
        }


@dataclass(frozen=True)
class RunRequest:
    total_steps: int
    gamma: float
    options: JsonMapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "gamma": self.gamma,
            **_copy(dict(self.options)),
        }


@dataclass(frozen=True)
class ExperimentRequest:
    """Validated user intent before defaults and backend translation."""

    schema_version: int
    experiment: ExperimentMetadata
    task: TaskRequest
    design: DesignRequest
    algorithm: AlgorithmRequest
    run: RunRequest
    _document: JsonMapping = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the original validated request without applying defaults."""
        return _copy(dict(self._document))


@dataclass(frozen=True)
class ResolvedExperiment:
    """A request with all control-plane defaults made explicit."""

    request: ExperimentRequest
    track: str
    _document: JsonMapping = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return _copy(dict(self._document))


@dataclass(frozen=True)
class RunPlan:
    """One deterministic track/algorithm/task/seed unit.

    Phase two intentionally leaves launch binding empty.  Backend adapters add
    the exact command and backend configuration in a later phase.
    """

    run_id: str
    experiment_name: str
    track: str
    algorithm: str
    seed: int
    runtime_profile: str
    output_dir: str
    launch_status: str = "unbound"
    backend: Optional[str] = None
    command: Tuple[str, ...] = ()
    environment: JsonMapping = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "track": self.track,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "runtime_profile": self.runtime_profile,
            "output_dir": self.output_dir,
            "launch": {
                "status": self.launch_status,
                "backend": self.backend,
                "command": list(self.command),
                "environment": _copy(dict(self.environment)),
            },
        }


@dataclass(frozen=True)
class PlanBundle:
    """Resolved experiment plus its seed-expanded, non-executable plans."""

    plan_version: int
    resolved_experiment: ResolvedExperiment
    runs: Tuple[RunPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "track": self.resolved_experiment.track,
            "run_count": len(self.runs),
            "resolved_experiment": self.resolved_experiment.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }
