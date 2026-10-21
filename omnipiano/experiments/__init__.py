"""Backend-neutral experiment requests and deterministic run planning.

This package is an additive control plane.  It deliberately does not import
or modify any Standard, Robust, Safety, or MARL training implementation.
"""

from omnipiano.experiments.compiler import compile_plan, resolve_experiment
from omnipiano.experiments.models import (
    AlgorithmRequest,
    DesignRequest,
    ExperimentMetadata,
    ExperimentRequest,
    PlanBundle,
    ResolvedExperiment,
    RunPlan,
    RunRequest,
    TaskRequest,
)
from omnipiano.experiments.schema import (
    CONTRACT_SCHEMA_VERSION,
    ExperimentValidationError,
    load_request,
    parse_request,
    validate_document,
)

__all__ = [
    "AlgorithmRequest",
    "CONTRACT_SCHEMA_VERSION",
    "DesignRequest",
    "ExperimentMetadata",
    "ExperimentRequest",
    "ExperimentValidationError",
    "PlanBundle",
    "ResolvedExperiment",
    "RunPlan",
    "RunRequest",
    "TaskRequest",
    "compile_plan",
    "load_request",
    "parse_request",
    "resolve_experiment",
    "validate_document",
]
