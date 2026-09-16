"""Backward-compatible imports for the relocated shared PPO module.

The canonical implementation lives in :mod:`omnipiano.multiagent.algos.ppo_module`.
This shim is intentionally kept so older scripts and serialized references that
use ``omnipiano.multiagent._ctde_module`` continue to resolve.
"""

from omnipiano.multiagent.algos.ppo_module import (
    CRITIC_INPUTS,
    CtdePPOTorchRLModule,
    RunningValueNorm,
    build_ctde_module_spec,
    build_multi_module_spec,
)

__all__ = [
    "CRITIC_INPUTS",
    "CtdePPOTorchRLModule",
    "RunningValueNorm",
    "build_ctde_module_spec",
    "build_multi_module_spec",
]
