"""SB3 PPO trainer for OmniPiano — canonical CLI-parameterized version.

This file is **the** PPO entry point. Per-experiment shell wrappers in
``examples/runs/`` pin the env + a few overrides and call this script.
Run with ``--help`` for the full CLI surface.

What this demonstrates
----------------------
How a SB3 user plugs into OmniPiano WITHOUT needing any OmniPiano-side
training/eval machinery. The benchmark contract on OmniPiano's side is:

  1. ``omnipiano.make(env_id, ...)`` returns a gymnasium env with the
     canonical MetricsWrapper + SafetyWrapper chain. Those wrappers emit
     ``info["episode_task/f1"]``, ``info["episode_safety/cost_total"]``,
     etc., at terminal steps. That is the entire metric contract.
  2. ``BenchmarkProtocolConfig`` (``omnipiano/configs/__init__.py``) is
     the single source of truth for protocol constants: env-step budget,
     scalar seed, num_eval_eps.

Everything else — rollout loops, periodic eval, final eval, report
formatting — is the framework's job. SB3 ships ``EvalCallback`` for
periodic eval; we use a small inline rollout loop for the final
benchmark eval to collect terminal ``info`` dicts.

Defaults (RoboPianist paper Fig 8 + Path C baseline):
- ``gamma = 0.9``        — Fig 8 finds 0.84-0.92 optimal; SB3's 0.99 collapses PPO on piano
- ``net_arch = [256,256,256]`` — paper Table 3
- ``n_steps = 2048``     — SB3 default
- ``batch_size = 64``    — SB3 default
- ``n_envs = 24``        — algorithm-side throughput knob

For paper-style N-seed replication, run this script N times with
distinct ``--seed`` values and aggregate ``eval_summary.json`` offline.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv

from omnipiano import make
from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.integrations.sb3 import TrainIterationSummaryCallback


def _parse_net_arch(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- experiment ---
    p.add_argument(
        "--env",
        default="OmniPiano-ForElise-FingeringAnn-v0",
        help="Registered OmniPiano env id.",
    )
    p.add_argument(
        "--experiment-name",
        default="ppo_sb3_template",
        help="Prefix for log_dir under examples/logs/.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny budget for pipeline validation only (NOT a benchmark result). "
        "Forces total_timesteps=2M and num_eval_eps=1.",
    )

    # --- protocol ---
    proto = BenchmarkProtocolConfig()
    p.add_argument("--total-steps", type=int, default=proto.total_env_steps)
    p.add_argument("--seed", type=int, default=proto.seed)
    p.add_argument("--num-eval-eps", type=int, default=proto.num_eval_eps,
                   help="Number of episodes for the final benchmark eval.")

    # --- vec env ---
    p.add_argument("--n-envs", type=int, default=24,
                   help="Parallel envs in SubprocVecEnv. Pure throughput knob.")
    p.add_argument("--eval-freq", type=int, default=2048,
                   help="Periodic-eval cadence in vec-env steps. With n_steps=2048 "
                   "this fires once per PPO rollout (~102 evals over 5M).")

    # --- PPO hparams ---
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--net-arch", default="256,256,256",
                   help="Comma-separated MLP hidden dims.")
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])

    return p


def _env_creator(env_name: str, log_dir: str, split: str, **extra):
    """Build one OmniPiano env. Shared by training, periodic eval, final eval."""
    def _make():
        return make(env_name, log_dir=log_dir, mode=split, **extra)
    return _make


def _final_eval(
    model: PPO,
    env_name: str,
    run_dir: str,
    eval_seed: int,
    num_eval_eps: int,
) -> Dict[str, Any]:
    """Final benchmark eval: deterministic rollout, collect terminal info.

    Not using SB3's ``evaluate_policy`` because we need terminal ``info``
    keys (``episode_task/*``, ``episode_safety/*``).
    """
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    env = make(
        env_name,
        log_dir=run_dir,
        mode="eval",
        record_dir=video_dir,
        seed=eval_seed,
    )
    episodes: List[Dict[str, Any]] = []
    try:
        for ep_idx in range(num_eval_eps):
            obs, _ = env.reset(seed=eval_seed + ep_idx * 10_000)
            ep_return, ep_length, terminal_info = 0.0, 0, {}
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_length += 1
                terminal_info = info
                done = bool(terminated) or bool(truncated)
            metrics = {
                k: float(v)
                for k, v in terminal_info.items()
                if isinstance(k, str)
                and k.startswith("episode_")
                and isinstance(v, (bool, int, float, np.integer, np.floating))
            }
            episodes.append(
                dict(
                    episode_index=int(ep_idx),
                    episode_return=ep_return,
                    episode_length=int(ep_length),
                    metrics=metrics,
                )
            )
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    summary: Dict[str, float] = {
        "return_mean": float(np.mean([ep["episode_return"] for ep in episodes])),
        "return_std": float(np.std([ep["episode_return"] for ep in episodes])),
        "length_mean": float(np.mean([ep["episode_length"] for ep in episodes])),
    }
    metric_keys = sorted({k for ep in episodes for k in ep["metrics"]})
    for k in metric_keys:
        vals = [ep["metrics"][k] for ep in episodes if k in ep["metrics"]]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))

    return {"env_id": env_name, "eval_seed": int(eval_seed), "episodes": episodes, "summary": summary}


def main():
    args = _build_arg_parser().parse_args()

    if args.smoke_test:
        args.total_steps = 2_000_000
        args.num_eval_eps = 1

    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_root, exist_ok=True)
    next_run_id = get_latest_run_id(logs_root, args.experiment_name) + 1
    experiment_id = f"{args.experiment_name}_{next_run_id}"
    log_dir = os.path.join(logs_root, experiment_id)
    os.makedirs(log_dir, exist_ok=True)

    # Eval seed offset past n_envs so eval rollouts don't collide with
    # training sub-env RNG streams (make_vec_env assigns seed+0..seed+n_envs-1).
    eval_seed = args.seed + args.n_envs + 1

    print(f"Env: {args.env}")
    print(
        f"Mode: {'SMOKE_TEST' if args.smoke_test else 'canonical'} | "
        f"seed={args.seed} eval_seed={eval_seed} num_eval_eps={args.num_eval_eps} "
        f"total_timesteps={args.total_steps:,} n_envs={args.n_envs}"
    )

    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    # CRITICAL: vec_env_cls=SubprocVecEnv. SB3's make_vec_env defaults to
    # DummyVecEnv (single-process serial), which drops fps ~10× on the
    # OmniPiano env chain.
    train_env = make_vec_env(
        _env_creator(args.env, log_dir, "train"),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )
    periodic_eval_env = make_vec_env(
        _env_creator(args.env, log_dir, "eval"),
        n_envs=1,
        seed=args.seed + 10_000,
    )
    periodic_eval = EvalCallback(
        periodic_eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(net_arch=_parse_net_arch(args.net_arch))
    print(f"Initializing PPO (policy=MlpPolicy, net_arch={policy_kwargs['net_arch']})...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        gamma=args.gamma,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir,
        seed=args.seed,
        device=args.device,
    )

    print(f"Training for {args.total_steps:,} timesteps...")
    iteration_summary = TrainIterationSummaryCallback(log_dir=log_dir)
    callbacks = CallbackList([periodic_eval, iteration_summary])
    model.learn(total_timesteps=args.total_steps, callback=callbacks)

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    print(f"Saved: {model_path}.zip")

    print("Running final benchmark eval...")
    result = _final_eval(model, args.env, log_dir, eval_seed, args.num_eval_eps)
    proto = BenchmarkProtocolConfig()
    result["seed"] = int(args.seed)
    result["total_env_steps"] = int(args.total_steps)
    result["protocol_version"] = proto.protocol_version
    result["algorithm"] = "PPO (SB3)"
    result["smoke_test"] = bool(args.smoke_test)
    # Source-script audit field — lets a downstream analysis script tell
    # template runs apart from ``run_sb3_baseline.py`` runs without relying
    # on log-dir naming convention. ``os.path.basename(__file__)`` records the
    # actual entry-point filename rather than a hardcoded constant.
    result["source_script"] = os.path.basename(__file__)
    result["hparams"] = {
        "n_envs": args.n_envs,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "net_arch": _parse_net_arch(args.net_arch),
    }

    out_path = os.path.join(log_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"eval summary → {out_path}")
    print(f"summary: {json.dumps(result['summary'], indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
