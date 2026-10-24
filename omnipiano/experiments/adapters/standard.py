"""Adapter for the existing Stable-Baselines3 Standard RL entrypoint."""

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
    require_single_agent_environment,
)
from omnipiano.experiments.models import ResolvedExperiment, RunPlan
from omnipiano.experiments.parameter_profiles import STANDARD_PARAMETER_TEMPLATES


BACKEND = "stable_baselines3"
ENTRYPOINT = "examples/run_sb3_baseline.py"
ALGORITHM_PARAMETERS = {
    name: frozenset(values) for name, values in STANDARD_PARAMETER_TEMPLATES.items()
}


def bind_run(resolved: ResolvedExperiment, run: RunPlan) -> RunPlan:
    document = resolved.to_dict()
    require_default_preset(document, track="standard")
    reject_unknown(document["design"]["config"], set(), context="standard design.config")
    env_id = require_registered_task(document, track="standard")
    require_single_agent_environment(env_id, track="standard")
    evaluation = require_evaluation(document, track="standard")
    algorithm = document["algorithm"]["name"]
    parameters: Mapping[str, Any] = document["algorithm"]["parameters"]
    allowed_parameters = ALGORITHM_PARAMETERS[algorithm]
    reject_unknown(
        parameters, allowed_parameters, context="standard algorithm.parameters"
    )

    compute = document["run"]["compute"]
    reject_compute_fields(compute, {"cpus", "gpus"}, track="standard")
    if compute["num_workers"] != 1:
        raise AdapterError(
            "Standard RL uses run.compute.num_envs; num_workers must remain 1"
        )
    tracking = document["run"]["tracking"]
    if tracking["provider"] not in {"none", "tensorboard"}:
        raise AdapterError(
            "the Standard RL backend supports its built-in TensorBoard/CSV logs only"
        )
    if document["run"]["video"]["enabled"]:
        raise AdapterError(
            "the Standard RL backend records final evaluation video only; "
            "periodic unified video settings are unsupported"
        )
    command = [
        "python", ENTRYPOINT,
        "--algo", algorithm,
        "--env", env_id,
        "--experiment-name", run.run_id,
        "--logs", run.output_dir,
        "--seed", str(run.seed),
        "--total-steps", str(document["run"]["total_steps"]),
        "--gamma", str(document["run"]["gamma"]),
        "--eval-freq", str(evaluation["frequency"]),
        "--num-eval-eps", str(evaluation["episodes"]),
        "--n-envs", str(compute["num_envs"]),
        "--device", backend_device(compute["device"], auto="auto"),
    ]
    if parameters:
        command.extend((
            "--algorithm-parameters-json",
            json.dumps(parameters, separators=(",", ":"), sort_keys=True),
        ))
    return replace(
        run,
        launch_status="bound",
        backend=BACKEND,
        command=tuple(command),
    )
