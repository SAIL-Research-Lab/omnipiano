"""Shared helpers for side-effect-free backend command adapters."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from omnipiano.experiments.models import PlanBundle, RunPlan


class AdapterError(ValueError):
    """A valid unified request cannot be represented by its current backend."""


def option_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def append_option(command: list[str], name: str, value: Any) -> None:
    """Append one argparse-compatible scalar/list/boolean option."""
    flag = option_name(name)
    if isinstance(value, bool):
        command.append(flag if value else "--no-" + name.replace("_", "-"))
    elif isinstance(value, (list, tuple)):
        command.extend((flag, ",".join(str(item) for item in value)))
    elif value is not None:
        command.extend((flag, str(value)))


def reject_unknown(
    values: Mapping[str, Any], allowed: Iterable[str], *, context: str
) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise AdapterError(f"{context} has unsupported fields: {unknown}")


def require_registered_task(document: Mapping[str, Any], *, track: str) -> str:
    task = document["task"]
    if task["source"] != "registered":
        raise AdapterError(f"the {track} adapter requires a registered task")
    overrides = task.get("overrides", {})
    if overrides:
        raise AdapterError(
            f"the {track} backend cannot apply registered-task overrides: "
            f"{sorted(overrides)}"
        )
    return str(task["environment_id"])


def require_single_agent_environment(env_id: str, *, track: str) -> None:
    from omnipiano.envs.registration import _registry

    if env_id not in _registry:
        raise AdapterError(
            f"{env_id!r} is not registered for the current {track} backend"
        )


def require_default_preset(document: Mapping[str, Any], *, track: str) -> None:
    preset = document["algorithm"]["preset"]
    if preset != "default":
        raise AdapterError(
            f"the {track} adapter does not define algorithm preset {preset!r}; "
            "use 'default' and explicit parameters"
        )


def reject_compute_fields(
    compute: Mapping[str, Any], fields: Iterable[str], *, track: str
) -> None:
    unsupported = sorted(set(compute) & set(fields))
    if unsupported:
        raise AdapterError(
            f"the {track} backend cannot apply compute fields: {unsupported}"
        )


def require_evaluation(document: Mapping[str, Any], *, track: str) -> Mapping[str, Any]:
    evaluation = document["run"]["evaluation"]
    if not evaluation["enabled"]:
        raise AdapterError(
            f"the {track} backend always evaluates and cannot disable evaluation"
        )
    return evaluation


def backend_device(value: str, *, auto: str) -> str:
    if value == "auto":
        return auto
    if value not in {"cpu", "cuda"}:
        raise AdapterError(
            f"unsupported compute device {value!r}; use auto, cpu, or cuda"
        )
    return value


def replace_runs(bundle: PlanBundle, runs: Sequence[RunPlan]) -> PlanBundle:
    return replace(bundle, runs=tuple(runs))
