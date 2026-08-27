"""End-to-end RLlib IPPO compatibility smoke test (Sub-phase 1A step 7).

Runs ~5k env-steps of RLlib independent PPO on the 4-hand WinterWind MA env
to confirm:

  (a) ``ray.rllib.env.ParallelPettingZooEnv`` adapter accepts our env without
      type / shape errors
  (b) PPOConfig + multi_agent setup runs through training iterations
  (c) RLlib reports a finite completed-episode return

Not a learning benchmark — F1 isn't expected to improve in 5k steps. The
goal is **trainer ingestion**: prove a mainstream MARL library can consume
the env for end-to-end paper baselines.

This script is a standalone demo, NOT a pytest test (RLlib initialization
is expensive — ~15-30s — and creates ray cluster state we don't want in the
regular test run). Invoke directly: ``python -m omnipiano.multiagent._rllib_smoke``.
"""

from __future__ import annotations

import math
import sys

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent import make_parallel
from omnipiano.multiagent._ippo_common import (
    RLLIB_ENV_NAME,
    extract_env_steps,
    wrap_parallel_env_for_rllib,
)


SMOKE_ENV_ID = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"
SMOKE_STEPS = 5000  # ~5 min wall-time on CPU; just needs to not crash


def _make_env_for_rllib(env_config):
    """RLlib env_creator: returns a PettingZoo ParallelEnv (Dict obs → flat for RLlib)."""
    # RLlib ParallelPettingZooEnv handles ParallelEnv natively, but its preprocessor
    # currently struggles with deeply nested Dict obs (own_hands/boundary_hands
    # are nested Dicts). For RLlib compatibility we flatten obs at env level.
    parallel_env = make_parallel(
        SMOKE_ENV_ID, seed=env_config.get("seed", 0), flatten_obs=True
    )
    return wrap_parallel_env_for_rllib(parallel_env)


def main() -> int:
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as e:
        print(
            f"[error] ray/rllib not installed: {e}; install with "
            "pip install -e '.[marl]'",
            file=sys.stderr,
        )
        return 1

    # Register our env_creator under a known name.
    register_env(
        RLLIB_ENV_NAME,
        lambda config: _make_env_for_rllib(config),
    )

    # Probe per-agent obs/action spaces by constructing once.
    probe = _make_env_for_rllib({})
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
        .environment(RLLIB_ENV_NAME, env_config={"seed": 0})
        .framework("torch")
        .env_runners(num_env_runners=0)  # all rollouts in driver — keeps demo simple
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agent_id, episode=None, **kw: agent_id),
            count_steps_by="env_steps",
        )
        .training(
            gamma=BenchmarkProtocolConfig().gamma,
            train_batch_size_per_learner=512,
            minibatch_size=64,
            num_epochs=2,
            lr=3e-4,
        )
        .debugging(seed=0)
    )

    print("[info] Initializing ray + PPO ...")
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    algo = None
    try:
        build = getattr(config, "build_algo", None)
        algo = build() if callable(build) else config.build()
        print("[info] Algo built. Training ~5k env-steps ...")
        total_steps = 0
        iters = 0
        while total_steps < SMOKE_STEPS:
            result = algo.train()
            iters += 1
            total_steps = extract_env_steps(result)
            print(f"[iter {iters}] steps={total_steps}")
            if iters > 50:
                raise RuntimeError(
                    "smoke test exceeded 50 iterations without reaching its "
                    "environment-step target"
                )
        print(f"[done] iters={iters}, steps={total_steps}")
        # Basic finite-reward sanity from final iter.
        env_runners = result.get("env_runners", result)
        episode_reward = env_runners.get(
            "episode_return_mean",
            env_runners.get("episode_reward_mean", float("nan")),
        )
        print(f"[done] rllib_agent_sum_return_mean={episode_reward}")
        if not isinstance(episode_reward, (int, float)) or not math.isfinite(
            float(episode_reward)
        ):
            raise RuntimeError(
                "RLlib agent-sum episode return is not finite; no valid episode "
                "was reported by RLlib"
            )
        print("[ok] RLlib IPPO 4-hand MA env trainer-ingestion smoke PASSED.")
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
