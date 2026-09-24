"""HAPPO: strict sequential per-agent PPO on aligned joint transitions."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo

_REF = ("Kuba et al. (2022), Trust region policy optimisation in multi-agent "
        "reinforcement learning, ICLR (arXiv:2109.11251)")

register_algo(
    AlgoSpec(
        name="happo",
        display_name="HAPPO (per-agent actor pi(a_i|o_i); agents updated "
                     "SEQUENTIALLY in a random permutation, each reweighting the "
                     "advantage by the compound ratio of already-updated agents)",
        family="on_policy_ppo_sequential",
        critic_input="global",
        critic_head="v",
        needs_global_state=True,
        rl_module="ctde",
        execution="decentralized",
        backend="native",
        status="supported",
        blocking="",
        reference=_REF,
        notes=("Uses the repository's native joint-transition backend because "
               "RLlib's multi-module Learner computes every module loss before "
               "stepping any optimizer and therefore cannot implement strict "
               "HAPPO ordering. Actors have independent parameters and receive "
               "only their local observations at execution. For each rollout a "
               "fresh random agent permutation is sampled; one actor completes "
               "all PPO epochs before its exact post-update probability ratio is "
               "folded into the compound factor used by the next actor. The "
               "factor is not clipped beyond PPO's own current-agent ratio clip. "
               "A single shared V(s) and ValueNorm are used for OmniPiano's "
               "environment-provided global state and shared team reward, "
               "matching HARL's EP HAPPO path. Networks and PPO hyperparameters "
               "remain the shared OmniPiano benchmark settings; gamma=0.8 is the "
               "task-specific protocol choice."),
    )
)
