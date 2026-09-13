"""Single-agent PPO on the monolithic control interface.

This is NOT a MARL algorithm and is not meant to be. It is the control group
that makes the whole paper falsifiable: one policy commands every hand, so if it
matches or beats the MARL baselines, the multi-agent framing is not justified.

It is implemented as the SAME multi-agent environment with num_agents=1 rather
than as the standalone single-agent OmniPiano task, because that keeps the
observation construction, reward shaping, episode length, evaluation protocol
and coordination-metric bookkeeping bit-identical to the MARL runs. The only
thing that changes is how many policies the hands are partitioned across.
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
        num_agents_override=1,
        status="planned",                      # was "experimental"
        blocking=("num_agents is NOT an env_config knob: make_env_for_rllib, "
                  "probe_agent_spaces, make_parallel and evaluate_marl would all "
                  "have to thread it, and the env id encodes 'Duet' so the hand "
                  "partition is baked into registration. Setting "
                  "env_config['num_agents'] is a SILENT NO-OP -- you would train "
                  "2 agents and label the artifact 1. Implement as a separate "
                  "registered 1-agent env id instead."),
        reference="Schulman et al. (2017), Proximal policy optimization, "
                  "arXiv:1707.06347",
        notes=("Execution is centralized BY CONSTRUCTION -- one policy sees every "
               "hand and emits every command -- so it is an upper bound on "
               "information and a lower bound on scalability. Two consequences "
               "that must be stated in the paper: (1) the inter-agent duplicate-"
               "press penalty can never fire (there is only one agent), so its "
               "RETURN IS NOT COMPARABLE to the MARL runs and only musical F1 / "
               "F1-AUC may be compared; (2) both coordination diagnostics are "
               "identically zero, which is the trivially optimal value and must "
               "not be reported as a win."),
    )
)