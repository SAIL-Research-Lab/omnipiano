"""Named-task registry adapter for multi-agent environments.

Registered IDs are now translated to ``ResolvedTask`` and use the same
builder as schema-v2 user tasks.  This module deliberately owns only the
public registry surface; task expansion and runtime construction live under
``omnipiano.multiagent.compile``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple

from omnipiano.multiagent.compile.env_runtime.topology import AGENT_ASSIGNMENTS
from omnipiano.multiagent.compile.environment import (
    make_parallel_from_task,
    resolve_registered_task,
)

if TYPE_CHECKING:
    from omnipiano.multiagent.compile.env_runtime.parallel_env import (
        OmniPianoParallelEnv,
    )


_RUNTIME_BYPASS_FIELDS = frozenset({
    "seed", "record_dir", "record_every", "record_resolution", "camera_id",
})
_MA_RUNTIME_KWARGS = frozenset({
    "obs_visibility", "reward_mode",
    "inter_agent_collision_penalty_coef", "flatten_obs", "sustain_owner",
    "include_global_state",
})


@dataclass(frozen=True)
class MATaskSpec:
    sa_env_id: str
    morphology: str


_ma_registry: Dict[str, MATaskSpec] = {}


def register_parallel(id: str, *, sa_env_id: str, morphology: str) -> None:
    """Register a historical MA id backed by one registered SA task."""
    if morphology not in AGENT_ASSIGNMENTS:
        raise ValueError(
            f"unknown morphology {morphology!r}; "
            f"valid: {sorted(AGENT_ASSIGNMENTS.keys())}"
        )
    _ma_registry[id] = MATaskSpec(sa_env_id=sa_env_id, morphology=morphology)


def list_parallel_envs() -> Tuple[str, ...]:
    return tuple(sorted(_ma_registry.keys()))


def make_parallel(env_id: str, **kwargs) -> "OmniPianoParallelEnv":
    """Build a registered task through the canonical resolved-task path."""
    allowed = _RUNTIME_BYPASS_FIELDS | _MA_RUNTIME_KWARGS
    illegal = set(kwargs) - allowed
    if illegal:
        raise ValueError(
            f"make_parallel(**kwargs) only accepts {sorted(allowed)}. "
            f"Got illegal overrides: {sorted(illegal)}."
        )
    task = resolve_registered_task(env_id)
    return make_parallel_from_task(task, **kwargs)


__all__ = [
    "MATaskSpec",
    "list_parallel_envs",
    "make_parallel",
    "register_parallel",
]
