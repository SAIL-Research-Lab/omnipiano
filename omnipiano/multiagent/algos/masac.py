# algos/masac.py
"""MASAC: multi-agent SAC -- per-agent stochastic actor, centralised Q critic."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo
from omnipiano.multiagent.algos.facmac import _OFF_POLICY_BLOCKING

register_algo(
    AlgoSpec(
        name="masac",
        display_name="MASAC (per-agent squashed-Gaussian actor, centralised "
                     "Q(s, a_joint) critic, entropy-regularised; off-policy)",
        family="off_policy_sac",
        critic_input="global_joint_action",
        critic_head="q",
        needs_global_state=True,
        rl_module="joint_actor_critic",
        execution="decentralized",
        rl_module_class="omnipiano.multiagent.algos._facmac_mixer:FactoredMixer",
        status="planned",
        blocking=_OFF_POLICY_BLOCKING + (
            " Additionally: twin Q critics, automatic entropy-temperature "
            "tuning, and a decision on whether the temperature is shared across "
            "agents or per agent (it changes the exploration coupling)."),
        reference="Haarnoja et al. (2018), Soft actor-critic, ICML "
                  "(arXiv:1801.01290), with a centralised critic after "
                  "Lowe et al. (2017), MADDPG (arXiv:1706.02275)",
        notes=("The intended head-to-head partner for MAPPO on SAMPLE "
               "EFFICIENCY: same CTDE information structure, same decentralized "
               "execution, opposite data-reuse regime. The comparison is only "
               "meaningful on the env-step axis with a matched budget, and the "
               "wall-clock cost must be reported alongside it or the 'more "
               "sample efficient' claim is misleading. Unlike FACMAC the critic "
               "is unfactored, so this is also the control that isolates what "
               "FACMAC's factorisation buys."),
    )
)