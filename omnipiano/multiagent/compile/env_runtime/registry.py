"""Process-local registry for named multi-agent environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from omnipiano.multiagent.compile.env_runtime.topology import AGENT_ASSIGNMENTS


@dataclass(frozen=True)
class MATaskSpec:
    """Reference from a named MA environment to its SA task and topology."""

    sa_env_id: str
    morphology: str


_ma_registry: Dict[str, MATaskSpec] = {}


def register_parallel(id: str, *, sa_env_id: str, morphology: str) -> None:
    """Register a named MA environment backed by one registered SA task."""
    if morphology not in AGENT_ASSIGNMENTS:
        raise ValueError(
            f"unknown morphology {morphology!r}; "
            f"valid: {sorted(AGENT_ASSIGNMENTS.keys())}"
        )
    _ma_registry[id] = MATaskSpec(sa_env_id=sa_env_id, morphology=morphology)


def list_parallel_envs() -> Tuple[str, ...]:
    """Return all registered MA environment ids in stable order."""
    return tuple(sorted(_ma_registry))


__all__ = ["MATaskSpec", "list_parallel_envs", "register_parallel"]
