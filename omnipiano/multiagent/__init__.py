"""Multi-agent (PettingZoo ParallelEnv) layer over OmniPiano N-hand envs.

This package wraps the registered single-agent N-hand envs into PettingZoo
``ParallelEnv`` instances. The SA dm_env chain is reused 1:1 (minus
``ConcatObservationWrapper``); see ``multi_agent_design.md`` § 4 and
``ma_territorial_impl_plan.md`` § 10(a) for the chain rationale.

Public entrypoints:
    make_parallel(env_id, **kwargs) → pettingzoo.ParallelEnv
    make_parallel_from_task(resolved_task, **kwargs) → pettingzoo.ParallelEnv
    register_parallel(id, *, sa_env_id, morphology)        → registers an MA env id
    AGENT_ASSIGNMENTS                                       → morphology → agent layout
"""

from omnipiano.multiagent.compile.env_runtime.topology import (
    AGENT_ASSIGNMENTS,
    AgentDef,
    MorphologyAssignment,
    compute_agent_territory,
    compute_boundary_hands,
    compute_inter_agent_boundaries,
)
from omnipiano.multiagent.compile.env_runtime.registry import (
    list_parallel_envs,
    register_parallel,
)
from omnipiano.multiagent.compile.environment import (
    make_parallel,
    make_parallel_from_task,
)

__all__ = [
    "AGENT_ASSIGNMENTS",
    "AgentDef",
    "MorphologyAssignment",
    "compute_agent_territory",
    "compute_boundary_hands",
    "compute_inter_agent_boundaries",
    "list_parallel_envs",
    "make_parallel",
    "make_parallel_from_task",
    "register_parallel",
]
