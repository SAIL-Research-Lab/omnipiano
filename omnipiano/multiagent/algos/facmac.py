# algos/facmac.py
"""FACMAC: factored centralised critic + centralised deterministic policy gradient."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo

_OFF_POLICY_BLOCKING = (
    "The current trainer builds a PPOConfig; there is no replay buffer, no "
    "target network and no off-policy update loop. Blocking items: "
    "(1) an off-policy trainer path (RLlib's new API stack multi-agent + "
    "replay-buffer combination has open issues, and old-stack MADDPG lives in "
    "the deprecated rllib_contrib -- budget for either fixing this or running "
    "these two baselines in HARL/EPyMARL against the same env wrapper); "
    "(2) target networks with soft updates; (3) an exploration schedule for a "
    "deterministic (FACMAC) or squashed-Gaussian (MASAC) policy; "
    "(4) a joint forward across agents so the critic can be evaluated at the "
    "CURRENT policies' joint action; (5) an env-step budget decision -- at "
    "UTD=1 these do ~50x the gradient work of PPO per env step, so 10M env "
    "steps is 1.5-4 days per run, and 2M is the defensible budget.")

register_algo(
    AlgoSpec(
        name="facmac",
        display_name="FACMAC (factored centralised critic Q_tot = mix(Q_i, s), "
                     "centralised deterministic policy gradient over the joint "
                     "action; off-policy)",
        family="off_policy_ddpg",
        critic_input="global_joint_action",
        critic_head="q_mixed",
        needs_global_state=True,
        rl_module="joint_actor_critic",
        execution="decentralized",
        rl_module_class="omnipiano.multiagent.algos._facmac_mixer:FactoredMixer",
        status="planned",
        blocking=_OFF_POLICY_BLOCKING,
        reference="Peng et al. (2021), FACMAC: Factored multi-agent centralised "
                  "policy gradients, NeurIPS (arXiv:2003.06709)",
        notes=("Targets the C and S domains: factorisation keeps the critic's "
               "input from growing with the number of agents (S), and the "
               "centralised joint-action gradient lets both agents move together "
               "instead of each assuming the other is frozen (C) -- which is "
               "exactly the failure mode behind duplicate presses in the common "
               "area. Execution stays decentralized (deterministic per-agent "
               "actors), so unlike MAT it IS comparable to IPPO/MAPPO at "
               "execution time -- but only on the env-step axis, since its "
               "wall-clock per env step is ~50x PPO's."),
    )
)