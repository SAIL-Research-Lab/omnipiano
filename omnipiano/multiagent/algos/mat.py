# algos/mat.py
"""MAT: MARL as sequence modeling (encoder-decoder, autoregressive actions)."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo

register_algo(
    AlgoSpec(
        name="mat",
        display_name="MAT (Multi-Agent Transformer: joint encoder over all "
                     "observations, decoder emitting actions AUTOREGRESSIVELY "
                     "across agents)",
        family="on_policy_ppo_sequential",
        critic_input="global",
        critic_head="v",
        needs_global_state=True,
        rl_module="joint_autoregressive",
        execution="sequential_autoregressive",
        rl_module_class="omnipiano.multiagent.algos._mat_module:MATEncoder",
        learner_class="omnipiano.multiagent.algos._happo_learner:HAPPOTorchLearner",
        status="planned",
        blocking=(
            "The network exists and is unit tested; the RLlib integration does "
            "not. Blocking items: (1) a MultiRLModule holding ONE shared "
            "encoder+decoder instead of per-agent modules; (2) a ConnectorV2 "
            "that performs n sequential decoder passes per env step so agent m "
            "conditions on a_1..a_{m-1} (RLlib's per-module "
            "forward_exploration cannot express this); (3) a decision on and "
            "logging of the agent ordering; (4) a joint-batch layout so the "
            "learner sees (B, n_agents, ...) instead of a per-module dict."),
        reference="Wen et al. (2022), Multi-agent reinforcement learning is a "
                  "sequence modeling problem, NeurIPS (arXiv:2205.14953)",
        notes=("EXECUTION IS NOT DECENTRALIZED. At inference, agent m consumes "
               "the sampled actions of agents 1..m-1, i.e. inter-agent "
               "communication at execution time. With 2 agents this means agent "
               "2 knows exactly which key agent 1 is about to press, which is "
               "strictly MORE information than IPPO/MAPPO receive; MAT winning "
               "the C (coupling) domain is therefore expected and is NOT "
               "evidence about architecture. Report MAT in a separate execution "
               "class. Also: MAT is sensitive to the agent ordering, an "
               "uncontrolled hyperparameter that matters most in the H domain, "
               "where the 1:3 hand split makes 'who decides first' meaningful."),
    )
)