"""Single-agent PPO on the monolithic four-hand control interface.

This is NOT a MARL algorithm and is not meant to be. It is the control group
that makes the whole paper falsifiable: one policy commands every hand, so if it
matches or beats the MARL baselines, the multi-agent framing is not justified.

The task JSON explicitly assigns every physical hand to one environment agent.
Consequently the existing RLlib IPPO recipe degenerates exactly to ordinary
continuous-action PPO: one actor, one critic, one shared reward and one 89-D
joint action.  Keeping ownership in the task JSON (rather than silently
overriding an MA task inside the algorithm) makes every artifact auditable.
"""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo

register_algo(
    AlgoSpec(
        name="ppo-monolithic",
        display_name="PPO (single policy commanding ALL hands; the "
                     '"is MARL necessary?" control, not a MARL baseline)',
        family="on_policy_ppo",
        critic_input="own",
        critic_head="v",
        needs_global_state=False,
        rl_module="ctde",
        execution="centralized",
        backend="rllib",
        required_num_agents=1,
        status="supported",
        reference="Schulman et al. (2017), Proximal policy optimization, "
                  "arXiv:1707.06347",
        notes=("Execution is centralized by construction: the single agent owns "
               "all four hands, observes their complete joint state plus the full "
               "88-key goal/state, and emits the complete joint action. This is a "
               "privileged upper-bound control, not decentralized execution. The "
               "cross-agent collision penalty and coordination rates are undefined "
               "with one logical agent; compare base_team_return and musical "
               "precision/recall/F1 against MARL, not the shaped total return."),
    )
)
