"""IPPO: independent PPO, one decentralized actor/critic per agent."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo

register_algo(
    AlgoSpec(
        name="ippo",
        display_name="IPPO (independent PPO; per-agent actor pi(a_i|o_i) and "
                     "decentralized critic V(o_i))",
        critic_input="own",
        needs_global_state=False,
        rl_module="ctde",
        reference="de Witt et al. (2020), Is independent learning all you need "
                  "in the StarCraft multi-agent challenge? arXiv:2011.09533",
        notes="Non-stationarity is not addressed in theory; empirically strong "
              "because PPO's clip keeps every agent's policy a slow-moving "
              "target for the others.",
    )
)

register_algo(
    AlgoSpec(
        name="ippo-rllib-module",
        display_name="IPPO using RLlib's built-in PPO RLModule (sanity anchor, "
                     "NOT architecturally matched to the CTDE baselines)",
        critic_input="own",
        needs_global_state=False,
        rl_module="rllib_default",
        reference="same as ippo",
        notes="Only for debugging our custom RLModule. Do not mix its numbers "
              "with ippo/mappo results: the network is a different architecture.",
    )
)