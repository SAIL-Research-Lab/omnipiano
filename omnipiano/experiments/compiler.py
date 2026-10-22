"""Resolve defaults and expand a unified request into one plan per seed."""

from __future__ import annotations

from copy import deepcopy
from pathlib import PurePath
from typing import Any

from omnipiano.experiments.catalog import validate_request_capabilities
from omnipiano.experiments.models import (
    ExperimentRequest,
    PlanBundle,
    ResolvedExperiment,
    RunPlan,
)
from omnipiano.experiments.schema import parse_request, validate_document


PLAN_VERSION = 2


def resolve_experiment(request: ExperimentRequest) -> ResolvedExperiment:
    """Make control-plane defaults explicit without touching any backend."""
    validate_request_capabilities(request)
    document = request.to_dict()

    document["experiment"].setdefault("description", "")
    document["experiment"].setdefault("tags", [])

    task = document["task"]
    if task["source"] == "registered":
        task.setdefault("overrides", {})
    else:
        music = task["music"]
        music.setdefault("piece", None)
        music.setdefault("midi_path", None)
        music.setdefault("trim_silence", True)
        music.setdefault("stretch_factor", 1.0)
        music.setdefault("shift_factor", 0)
        music.setdefault("fingering_mode", "optimal_transport")
        task["embodiment"].setdefault("mobility", "unrestricted")
        task.setdefault("environment", {})
        task.setdefault("reward", {})

    document["design"].setdefault("config", {})
    document["algorithm"].setdefault("preset", "default")
    document["algorithm"].setdefault("parameters", {})

    run = document["run"]
    run.setdefault(
        "evaluation", {"enabled": True, "frequency": 50_000, "episodes": 1}
    )
    run["evaluation"].setdefault("seed_offset", 10_000)
    run.setdefault(
        "video",
        {
            "enabled": False,
            "frequency": 250_000,
            "camera": "piano/back",
            "width": 640,
            "height": 480,
        },
    )
    run.setdefault(
        "compute", {"device": "auto", "num_workers": 1, "num_envs": 1}
    )
    run.setdefault("tracking", {"provider": "none", "mode": "disabled"})
    run["tracking"].setdefault("project", None)
    run["tracking"].setdefault("group", None)
    run["tracking"].setdefault("name", None)
    run["tracking"].setdefault("tags", [])
    run.setdefault("output", {"root": "runs"})

    # Re-validate the resolved form so defaults cannot drift from the contract.
    validate_document(document)
    return ResolvedExperiment(
        request=request,
        track=request.design.type,
        _document=deepcopy(document),
    )


def _output_root(document: dict[str, Any]) -> str:
    return document["run"]["output"]["root"]


def compile_plan(request: ExperimentRequest) -> PlanBundle:
    """Return a deterministic, explicitly non-executable seed fan-out."""
    resolved = resolve_experiment(request)
    document = resolved.to_dict()
    name = request.experiment.name
    track = resolved.track
    algorithm = request.algorithm.name
    root = _output_root(document)

    runs = tuple(
        RunPlan(
            run_id=f"{name}--{track}--seed-{seed}",
            experiment_name=name,
            track=track,
            algorithm=algorithm,
            seed=seed,
            runtime_profile=track,
            output_dir=str(PurePath(root) / name / track / f"seed-{seed}"),
        )
        for seed in request.experiment.seeds
    )
    return PlanBundle(
        plan_version=PLAN_VERSION,
        resolved_experiment=resolved,
        runs=runs,
    )


def compile_document(document: dict[str, Any]) -> PlanBundle:
    """Convenience entrypoint for callers that already hold decoded JSON."""
    return compile_plan(parse_request(document))


def bind_plan(request: ExperimentRequest) -> PlanBundle:
    """Compile and bind every seed to its existing backend without I/O."""
    from omnipiano.experiments.adapters import bind_plan as bind_compiled_plan

    return bind_compiled_plan(compile_plan(request))


def bind_document(document: dict[str, Any]) -> PlanBundle:
    """Validate, compile, and bind an already decoded JSON document."""
    return bind_plan(parse_request(document))
