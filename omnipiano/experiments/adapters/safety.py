"""Adapter for the existing manifested OmniSafe Safety RL runner."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any, Mapping

from omnipiano.experiments.adapters.common import (
    AdapterError,
    backend_device,
    reject_compute_fields,
    reject_unknown,
    require_default_preset,
    require_evaluation,
    require_registered_task,
)
from omnipiano.experiments.models import ResolvedExperiment, RunPlan


BACKEND = "omnisafe"
ENTRYPOINT = "omnipiano.safety.run"
GROUPS = ("main", "hands", "budget", "extensions")


def _suite_tasks() -> dict[str, tuple[Any, ...]]:
    # Lazy import keeps the shared request/compiler layer free of MuJoCo.
    from omnipiano.safety.suite import EXTENSIONS, HANDS, MAIN, THRESHOLDS

    return {
        "main": tuple(MAIN),
        "hands": tuple(HANDS),
        "budget": tuple(THRESHOLDS),
        "extensions": tuple(EXTENSIONS),
    }


def _resolve_task(env_id: str, config: Mapping[str, Any]) -> tuple[str, int]:
    reject_unknown(config, {"group", "task_index"}, context="safety design.config")
    suites = _suite_tasks()
    group = config.get("group")
    task_index = config.get("task_index")
    if group is not None:
        if group not in GROUPS:
            raise AdapterError(f"safety design.config.group must be one of {GROUPS}")
        if type(task_index) is not int or not 0 <= task_index < len(suites[group]):
            raise AdapterError("safety design.config.task_index is outside its group")
        actual = suites[group][task_index].env_id
        if actual != env_id:
            raise AdapterError(
                "safety task mismatch: environment_id does not match "
                f"{group}[{task_index}] ({actual})"
            )
        return group, task_index
    if task_index is not None:
        raise AdapterError("safety task_index requires design.config.group")

    matches = [
        (name, index)
        for name, tasks in suites.items()
        for index, item in enumerate(tasks)
        if item.env_id == env_id
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AdapterError(
            f"{env_id!r} is not a task in the current Safety RL suite"
        )
    raise AdapterError(
        f"{env_id!r} occurs in multiple safety groups; specify group and task_index"
    )


def bind_run(resolved: ResolvedExperiment, run: RunPlan) -> RunPlan:
    document = resolved.to_dict()
    require_default_preset(document, track="safety")
    env_id = require_registered_task(document, track="safety")
    evaluation = require_evaluation(document, track="safety")
    group, task_index = _resolve_task(env_id, document["design"]["config"])
    if document["run"]["gamma"] != 0.8:
        raise AdapterError("the current Safety RL backend fixes gamma and cost_gamma at 0.8")
    if document["run"]["video"]["enabled"]:
        raise AdapterError("the current Safety RL backend does not record video")
    steps = document["run"]["total_steps"]
    if steps % 20_000:
        raise AdapterError("Safety RL total_steps must be a multiple of 20,000")
    parameters: Mapping[str, Any] = document["algorithm"]["parameters"]
    from omnipiano.safety.parameters import (
        apply_parameter_overrides,
        configurable_sections,
    )

    algorithm = document["algorithm"]["name"]
    allowed_parameters = set(configurable_sections(algorithm))
    reject_unknown(
        parameters, allowed_parameters, context="safety algorithm.parameters"
    )
    try:
        apply_parameter_overrides({}, algorithm, parameters)
    except ValueError as exc:
        raise AdapterError(str(exc)) from exc

    compute = document["run"]["compute"]
    reject_compute_fields(compute, {"cpus", "gpus"}, track="safety")
    if compute["num_workers"] != 1 or compute["num_envs"] != 1:
        raise AdapterError(
            "the current Safety RL runner requires num_workers=1 and num_envs=1"
        )
    if document["run"]["tracking"]["provider"] not in {"none", "tensorboard"}:
        raise AdapterError(
            "the current Safety RL runner supports its built-in TensorBoard logs only"
        )
    device = backend_device(compute["device"], auto="cpu")
    command = [
        "python", "-m", ENTRYPOINT,
        "--group", group,
        "--algorithms", algorithm,
        "--task-index", str(task_index),
        "--seeds", str(run.seed),
        "--steps", str(steps),
        "--eval-episodes", str(evaluation["episodes"]),
        "--device", device,
        "--out", run.output_dir,
    ]
    if parameters:
        command.extend((
            "--algorithm-parameters-json",
            json.dumps(parameters, separators=(",", ":"), sort_keys=True),
        ))
    command.append("--execute")
    return replace(
        run,
        launch_status="bound",
        backend=BACKEND,
        command=tuple(command),
    )
