"""Multi-agent (PettingZoo ParallelEnv) layer over OmniPiano N-hand envs.

This package wraps the registered single-agent N-hand envs into PettingZoo
``ParallelEnv`` instances. The SA dm_env chain is reused 1:1 (minus
``ConcatObservationWrapper``); see ``multi_agent_design.md`` § 4 and
``ma_territorial_impl_plan.md`` § 10(a) for the chain rationale.

Public entrypoints:
    make_parallel(env_id, **kwargs) → pettingzoo.ParallelEnv
    register_parallel(id, *, sa_env_id, morphology)        → registers an MA env id
    AGENT_ASSIGNMENTS                                       → morphology → agent layout
"""

from omnipiano.multiagent.assignment import (
    AGENT_ASSIGNMENTS,
    AgentDef,
    MorphologyAssignment,
    compute_agent_territory,
    compute_boundary_hands,
    compute_inter_agent_boundaries,
)
from omnipiano.multiagent.registration import (
    list_parallel_envs,
    make_parallel,
    register_parallel,
)

_LAZY_ATTRS = {
    "make_parallel":     "omnipiano.multiagent",
    "register_parallel": "omnipiano.multiagent",
    "list_parallel_envs":"omnipiano.multiagent",
}


def __getattr__(name: str):
    """Lazily expose the multi-agent entrypoints promised by error messages.

    ``omnipiano.envs.registration.make()`` tells users to call
    ``omnipiano.make_parallel(...)``, and ``multi_agent_design.md`` § 4 documents
    it as a top-level parallel entrypoint.  Importing it eagerly would make
    PettingZoo a hard dependency of ``import omnipiano``, even though it is an
    optional ``[marl]`` extra, so resolve it on first attribute access instead.
    """
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'omnipiano' has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"omnipiano.{name} requires the multi-agent extra. "
            f"Install it with: pip install -e '.[marl]'"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value          # cache; __getattr__ is not called again
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))

__all__ = [
    "AGENT_ASSIGNMENTS",
    "AgentDef",
    "MorphologyAssignment",
    "compute_agent_territory",
    "compute_boundary_hands",
    "compute_inter_agent_boundaries",
    "list_parallel_envs",
    "make_parallel",
    "register_parallel",
]
