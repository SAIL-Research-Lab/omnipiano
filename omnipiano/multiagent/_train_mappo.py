"""Train a MAPPO policy on a 4-hand MA Territorial env via RLlib.

Usage:
    MUJOCO_GL=egl python -m omnipiano.multiagent._train_mappo \
        --env-id OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0 \
        --total-steps 5000000 \
        --num-workers 4 \
        --save-dir /tmp/ma_4hand_winter_wind_ckpt

Saves an RLlib checkpoint dir at --save-dir suitable for loading via
``Algorithm.from_checkpoint(...)``. The matching eval+video script is
``_eval_mappo.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _make_env_for_rllib(env_config):
    """RLlib env_creator: returns flatten_obs=True ParallelEnv (Box obs per agent)."""
    from omnipiano.multiagent import make_parallel
    env_id = env_config.get("env_id", "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0")
    return make_parallel(env_id, seed=env_config.get("seed", 0), flatten_obs=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RLlib MAPPO trainer for 4-hand MA Territorial.")
    parser.add_argument("--env-id", default="OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="/tmp/ma_4hand_winter_wind_ckpt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every-iters", type=int, default=5)
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] save_dir = {save_dir}")
    print(f"[info] env_id = {args.env_id}")
    print(f"[info] total_steps = {args.total_steps:,}")
    print(f"[info] num_workers = {args.num_workers}")

    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.tune.registry import register_env
    except ImportError as e:
        print(f"[error] ray/rllib not installed: {e}", file=sys.stderr)
        return 1

    register_env(
        "omnipiano_4hand_ma",
        lambda config: ParallelPettingZooEnv(_make_env_for_rllib(config)),
    )

    # Probe spaces (single-construct, then close).
    probe = ParallelPettingZooEnv(_make_env_for_rllib({"env_id": args.env_id, "seed": args.seed}))
    obs_spaces = dict(probe.observation_space)
    act_spaces = dict(probe.action_space)
    agents = list(obs_spaces.keys())
    probe.close()
    print(f"[info] agents = {agents}")
    for a in agents:
        print(f"[info]   {a}: obs={obs_spaces[a]}, action={act_spaces[a]}")

    policies = {a: (None, obs_spaces[a], act_spaces[a], {}) for a in agents}

    # local_mode=False so num_env_runners > 0 actually spawns workers.
    ray.init(local_mode=(args.num_workers == 0), ignore_reinit_error=True, log_to_driver=False)

    config = (
        PPOConfig()
        .environment(
            "omnipiano_4hand_ma",
            env_config={"env_id": args.env_id, "seed": args.seed},
        )
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agent_id, episode=None, **kw: agent_id),
        )
        .training(
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )
    )

    algo = config.build()
    print("[info] PPO built. Starting training ...")
    progress_file = save_dir / "progress.jsonl"
    progress_f = progress_file.open("w")

    total_steps = 0
    iters = 0
    t0 = time.time()
    try:
        while total_steps < args.total_steps:
            result = algo.train()
            iters += 1
            for key in (
                "num_env_steps_sampled_lifetime",
                "timesteps_total",
                "num_env_steps_sampled",
            ):
                if key in result:
                    total_steps = int(result[key])
                    break

            er = result.get("env_runners", result)
            ep_reward = er.get("episode_reward_mean", er.get("episode_return_mean", float("nan")))
            ep_len = er.get("episode_len_mean", float("nan"))

            row = {
                "iter": iters,
                "steps": total_steps,
                "wall_s": round(time.time() - t0, 1),
                "ep_reward_mean": ep_reward,
                "ep_len_mean": ep_len,
            }
            progress_f.write(json.dumps(row) + "\n")
            progress_f.flush()
            if iters % args.log_every_iters == 0 or iters == 1:
                print(
                    f"[iter {iters:4d}] steps={total_steps:>9,}  "
                    f"reward={ep_reward:.2f}  ep_len={ep_len:.1f}  "
                    f"wall={time.time()-t0:6.0f}s"
                )

        wall = time.time() - t0
        print(f"[done] iters={iters}, steps={total_steps:,}, wall={wall:.1f}s ({wall/60:.1f} min)")

        ckpt_path = algo.save(str(save_dir))
        if hasattr(ckpt_path, "checkpoint"):
            ckpt_path = ckpt_path.checkpoint.path
        print(f"[done] checkpoint saved at: {ckpt_path}")
        (save_dir / "checkpoint_path.txt").write_text(str(ckpt_path) + "\n")
        algo.stop()
    finally:
        progress_f.close()
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
