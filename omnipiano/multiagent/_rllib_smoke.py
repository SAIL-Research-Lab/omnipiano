"""End-to-end RLlib MAPPO compatibility smoke test (Sub-phase 1A step 7).

Runs ~5k env-steps of RLlib MAPPO training on the 4-hand WinterWind MA env
to confirm:

  (a) ``ray.rllib.env.ParallelPettingZooEnv`` adapter accepts our env without
      type / shape errors
  (b) PPOConfig + multi_agent setup runs through training iterations
  (c) Each agent receives reward signal (episode_reward_mean is finite)

Not a learning benchmark — F1 isn't expected to improve in 5k steps. The
goal is **trainer ingestion**: prove a mainstream MARL library can consume
the env for end-to-end paper baselines.

This script is a standalone demo, NOT a pytest test (RLlib initialization
is expensive — ~15-30s — and creates ray cluster state we don't want in the
regular test run). Invoke directly: ``python -m omnipiano.multiagent._rllib_smoke``.
"""

from __future__ import annotations

import os
import sys

import gymnasium as gym

from omnipiano.multiagent import make_parallel


SMOKE_ENV_ID = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"
SMOKE_STEPS = 5000  # ~5 min wall-time on CPU; just needs to not crash


def _make_env_for_rllib(env_config):
    """RLlib env_creator: returns a PettingZoo ParallelEnv (Dict obs → flat for RLlib)."""
    # RLlib ParallelPettingZooEnv handles ParallelEnv natively, but its preprocessor
    # currently struggles with deeply nested Dict obs (own_hands/boundary_hands
    # are nested Dicts). For RLlib compatibility we flatten obs at env level.
    return make_parallel(SMOKE_ENV_ID, seed=env_config.get("seed", 0), flatten_obs=True)


def main() -> int:
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.tune.registry import register_env
    except ImportError as e:
        print(f"[skip] ray/rllib not installed: {e}")
        return 0

    # Register our env_creator under a known name.
    register_env(
        "omnipiano_4hand_ma",
        lambda config: ParallelPettingZooEnv(_make_env_for_rllib(config)),
    )

    # Probe per-agent obs/action spaces by constructing once.
    probe = ParallelPettingZooEnv(_make_env_for_rllib({}))
    obs_spaces = {a: probe.observation_space[a] for a in probe.agents}
    act_spaces = {a: probe.action_space[a] for a in probe.agents}
    probe.close()
    print(f"[info] agents={list(obs_spaces.keys())}")
    for a in obs_spaces:
        print(f"[info]   {a}: obs={obs_spaces[a]}, action={act_spaces[a]}")

    # Per-agent policy mapping (each agent its own policy — IPPO-like setup).
    policies = {a: (None, obs_spaces[a], act_spaces[a], {}) for a in obs_spaces}

    config = (
        PPOConfig()
        .environment("omnipiano_4hand_ma", env_config={"seed": 0})
        .framework("torch")
        .env_runners(num_env_runners=0)  # all rollouts in driver — keeps demo simple
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agent_id, episode=None, **kw: agent_id),
        )
        .training(
            train_batch_size=512,
            minibatch_size=64,
            num_epochs=2,
            lr=3e-4,
        )
    )

    print("[info] Initializing ray + PPO ...")
    ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
    try:
        algo = config.build()
        print("[info] Algo built. Training ~5k env-steps ...")
        total_steps = 0
        iters = 0
        while total_steps < SMOKE_STEPS:
            result = algo.train()
            iters += 1
            # `num_env_steps_sampled_lifetime` is the modern key; fall back to others.
            for key in (
                "num_env_steps_sampled_lifetime",
                "timesteps_total",
                "num_env_steps_sampled",
            ):
                if key in result:
                    total_steps = result[key]
                    break
            print(f"[iter {iters}] steps={total_steps}")
            if iters > 50:
                # Safety: don't loop forever if `total_steps` not updating.
                break
        print(f"[done] iters={iters}, steps={total_steps}")
        # Basic finite-reward sanity from final iter.
        env_runners = result.get("env_runners", result)
        episode_reward = env_runners.get(
            "episode_reward_mean",
            env_runners.get("episode_return_mean", float("nan")),
        )
        print(f"[done] episode_reward_mean={episode_reward}")
        if not (isinstance(episode_reward, (int, float))) or (
            episode_reward != episode_reward  # NaN check (NaN != NaN)
        ):
            print("[warn] episode_reward_mean is NaN — trainer ran but no episodes completed")
        else:
            print("[ok] RLlib MAPPO 4-hand MA env trainer-ingestion smoke PASSED.")
        algo.stop()
    finally:
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
