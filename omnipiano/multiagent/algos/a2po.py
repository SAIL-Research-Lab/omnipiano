"""A2PO: agent-by-agent PPO with preceding-agent correction."""

from __future__ import annotations

from omnipiano.multiagent.algos.base import AlgoSpec, register_algo


_REF = (
    "Wang et al. (2023), Order Matters: Agent-by-agent Policy Optimization, "
    "ICLR (arXiv:2302.06205)"
)


register_algo(
    AlgoSpec(
        name="a2po",
        display_name=(
            "A2PO (independent decentralized actors; agent-by-agent updates "
            "with PreOPC, semi-greedy order and adaptive joint-ratio clipping)"
        ),
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
        notes=(
            "Uses the native aligned-joint-transition backend. Each agent has "
            "an independent actor and an independent V_i(s) critic, matching "
            "the official non-parameter-sharing continuous-control path while "
            "supporting OmniPiano's heterogeneous observation/action sizes. "
            "For every rollout, agents are ordered semi-greedily; before an "
            "agent update, the return is recomputed with preceding-agent "
            "off-policy correction (PreOPC), followed by the paper's critic-only "
            "stage and actor/critic stage. The actor clips the product of the "
            "already-updated agents' ratios and then clips the resulting joint "
            "ratio with an order-adaptive PPO radius. Action-density ratios are "
            "products over continuous action dimensions (summed log-probability), "
            "matching the effective upstream continuous-action path, where "
            "FixedNormal sums dimensions before its aggregation switch. "
            "Gamma=0.8 and the shared PPO/network settings remain OmniPiano "
            "protocol choices."
        ),
    )
)
