"""Unified requests, deterministic planning, and explicit local execution.

Validation, planning, and binding stay side-effect free. Materialization and
launch are separate opt-in operations implemented without modifying any of the
four existing training backends.
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
from omnipiano.experiments.execution import (
    ExecutionError,
    MaterializedBundle,
    MaterializedRun,
    discover_runs,
    launch_bundle,
    materialize_bundle,
    process_status,
    stop_process,
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
    "ExecutionError",
    "MaterializedBundle",
    "MaterializedRun",
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
    "discover_runs",
    "launch_bundle",
    "load_request",
    "parse_request",
    "materialize_bundle",
    "process_status",
    "resolve_experiment",
    "validate_document",
    "validate_request_capabilities",
    "stop_process",
]
