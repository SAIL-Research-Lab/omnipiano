"""HAPPO: per-agent actor, sequential updates with a compound policy ratio."""

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
        learner_class="omnipiano.multiagent.algos._happo_learner:HAPPOTorchLearner",
        status="experimental",
        blocking=("Sequential update is implemented and unit tested, but never "
                  "validated end-to-end on this benchmark. Before quoting "
                  "numbers, run the two validation gates in algos_test.py and "
                  "confirm happo/compound_factor_clamp_rate stays < 0.05."),
        training_overrides={
            # The compound factor pairs agent A's row t with agent B's row t.
            # RLlib shuffles per module independently, which would destroy that
            # pairing and silently turn M into noise.
            "shuffle_batch_per_epoch": False,
        },
        reference=_REF,
        notes=("Deliberate deviations from the paper, both recorded in "
               "run_config.json: (1) we keep MAPPO's PER-AGENT critic instead of "
               "the paper's single shared V(s), so that HAPPO vs MAPPO differs "
               "ONLY in the actor update rule -- a one-variable ablation; "
               "(2) minibatch shuffling is disabled, so a fair MAPPO control "
               "should disable it too. Expected to help the H (heterogeneity) "
               "domain, where the 1:3 hand split makes the agents asymmetric and "
               "parameter sharing hurts. NOT guaranteed to beat MAPPO: the "
               "compound ratio is gradient-redundant relative to a per-agent "
               "ratio with reweighted advantage and has strictly higher variance "
               "(see the canonical-form analysis literature)."),
    )
)

register_algo(
    AlgoSpec(
        name="happo-m1-control",
        display_name="HAPPO validation control: the entire HAPPO code path with "
                     "the compound factor forced to M=1 (must reproduce "
                     "mappo-noshuffle)",
        family="on_policy_ppo",          # M == 1 => no sequencing whatsoever
        critic_input="global",
        critic_head="v",
        needs_global_state=True,
        rl_module="ctde",
        execution="decentralized",
        learner_class=(
            "omnipiano.multiagent.algos._happo_learner:"
            "HAPPOCompoundDisabledLearner"),
        status="experimental",
        blocking=("Validation-only. It exists to be COMPARED, not reported; it "
                  "must never appear in a results table."),
        training_overrides={"shuffle_batch_per_epoch": False},
        reference=_REF,
        notes=("Runs the permutation, the per-module bookkeeping and the extra "
               "forward pass, but with exp(0)=1 as the advantage weight, so the "
               "mathematics collapses to MAPPO's. Because no network sees a "
               "different number, this MUST track mappo-noshuffle. It is the only "
               "decisive test that the surrogate-delta trick in "
               "HAPPOTorchLearner.compute_loss_for_module did not corrupt the "
               "loss -- the M=1 identity inside happo_surrogate holds by "
               "construction and therefore proves nothing about the delta."),
    )
)