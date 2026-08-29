"""MAPPO: per-agent actor, one centralized critic input (CTDE)."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo

_REF = ("Yu et al. (2022), The surprising effectiveness of PPO in cooperative "
        "multi-agent games, NeurIPS D&B")

register_algo(
    AlgoSpec(
        name="mappo",
        display_name="MAPPO (per-agent actor pi(a_i|o_i), centralized critic "
                     "V(s) over the agent-specific global state; CTDE)",
        critic_input="global",
        needs_global_state=True,
        rl_module="ctde",
        reference=_REF,
        notes="Identical to ippo in every respect except the critic's input "
              "slice. MAPPO is NOT guaranteed to beat IPPO: the global state is "
              "higher-dimensional and harder to fit, and under a shared team "
              "reward the local observation may already be near-sufficient.",
    )
)

register_algo(
    AlgoSpec(
        name="mappo-own-critic",
        display_name="MAPPO ablation: the centralized state is present in the "
                     "observation but NOT read by the critic (numerically "
                     "equivalent to IPPO)",
        critic_input="own",
        needs_global_state=True,
        rl_module="ctde",
        reference=_REF,
        notes="Validation-only. Because no network reads the extra bytes, this "
              "must reproduce `ippo` to floating-point precision. Any deviation "
              "means something observation-dimension-dependent (e.g. a mean/std "
              "filter) is silently in the pipeline.",
    )
)