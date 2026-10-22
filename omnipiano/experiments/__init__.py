"""Backend-neutral experiment requests and deterministic run planning.

This package is an additive control plane. Validation and planning do not load
training frameworks; explicit binding lazily inspects existing backend metadata
but never modifies a backend, writes run artifacts, or starts training.
"""

from omnipiano.experiments.adapters import AdapterError
from omnipiano.experiments.compiler import (
    bind_document,
    bind_plan,
    compile_plan,
    resolve_experiment,
)
from omnipiano.experiments.catalog import (
    CATALOG_VERSION,
    CapabilityError,
    catalog_dict,
    get_track_capability,
    validate_request_capabilities,
)
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
    "AdapterError",
    "CATALOG_VERSION",
    "CapabilityError",
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
    "bind_document",
    "bind_plan",
    "compile_plan",
    "catalog_dict",
    "get_track_capability",
    "load_request",
    "parse_request",
    "resolve_experiment",
    "validate_document",
    "validate_request_capabilities",
]
