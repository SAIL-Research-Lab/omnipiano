"""SB3 SAC trainer for OmniPiano — canonical CLI-parameterized version.

This file is **the** SAC entry point. Per-experiment shell wrappers in
``examples/runs/`` pin the env + a few overrides (e.g., ``n_envs``,
``gradient_steps``) and call this script. Run with ``--help`` for the
full CLI surface.

Defaults reproduce the paper's SAC config from ``robopianist-rl/``:
- ``hidden_dims = (256, 256, 256)``  — ``sac.py:57``
- ``gamma = 0.8``                    — ``run.sh --discount 0.8``
- ``batch_size = 256``               — ``train.py:29``
- ``buffer_size = 1_000_000``        — ``train.py:32``
- ``learning_starts = 5000``         — ``run.sh --warmstart-steps 5000``
- ``tau = 0.005``                    — ``sac.py:62``
- ``learning_rate = 3e-4``           — ``robopianist-rl`` default
- ``total_timesteps = 5_000_000``    — ``BenchmarkProtocolConfig``
- ``n_envs = 4`` (paper uses 1; 4 is a mild parallelism win)
- ``train_freq = 1, gradient_steps = 1`` (UTD = 1 / n_envs)

For paper-matched UTD = 1.0 across n_envs, set ``--gradient-steps`` =
``--n-envs`` (this is what the 3-hand wrappers do).

What we do NOT replicate from paper (would require SB3 subclassing):
- ``critic_dropout_rate = 0.01``     — DroQ
- ``critic_layer_norm = True``       — DroQ

These omissions cap F1 below paper's 0.78-0.80 ceiling on 2-hand pieces.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import SubprocVecEnv

from omnipiano import make
from omnipiano.configs import BenchmarkProtocolConfig


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
        default="OmniPiano-ForElise-FingeringOT-v0",
        help="Registered OmniPiano env id.",
    )
    p.add_argument(
        "--experiment-name",
        default="sac_sb3_template",
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
    p.add_argument("--n-envs", type=int, default=4,
                   help="Parallel envs in SubprocVecEnv. Off-policy: speeds rollout, "
                   "doesn't affect sample efficiency. Pair with --gradient-steps to keep UTD.")
    p.add_argument("--eval-interval-env-steps", type=int, default=10_000,
                   help="Periodic-eval cadence in env-steps (matches RoboPianist's eval_interval).")

    # --- SAC hparams ---
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--train-freq", type=int, default=1,
                   help="Trigger training every N vec-env steps.")
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Gradient updates per training trigger. UTD = gradient_steps / n_envs. "
                   "Set equal to n_envs for paper-matched UTD=1.")
    p.add_argument("--net-arch", default="256,256,256",
                   help="Comma-separated MLP hidden dims.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return p


def _env_creator(env_name: str, log_dir: str, split: str, **extra):
    """Build one OmniPiano env. Shared by training, periodic eval, final eval."""
    def _make():
        return make(env_name, log_dir=log_dir, mode=split, **extra)
    return _make


def _final_eval(
    model: SAC,
    env_name: str,
    run_dir: str,
    eval_seed: int,
    num_eval_eps: int,
) -> Dict[str, Any]:
    """Final benchmark eval: deterministic rollout, collect terminal info."""
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

    eval_seed = args.seed + args.n_envs + 1

    print(f"Env: {args.env}")
    print(
        f"Mode: {'SMOKE_TEST' if args.smoke_test else 'canonical'} | "
        f"seed={args.seed} eval_seed={eval_seed} num_eval_eps={args.num_eval_eps} "
        f"total_timesteps={args.total_steps:,} n_envs={args.n_envs} "
        f"UTD={args.gradient_steps / args.n_envs:.3f}"
    )

    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    # CRITICAL: vec_env_cls=SubprocVecEnv. SB3's make_vec_env defaults to
    # DummyVecEnv (single-process serial). For OmniPiano env chain, that
    # drops fps ~10×.
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
    eval_freq = max(args.eval_interval_env_steps // args.n_envs, 1)
    periodic_eval = EvalCallback(
        periodic_eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(net_arch=_parse_net_arch(args.net_arch))
    print(f"Initializing SAC (policy=MlpPolicy, net_arch={policy_kwargs['net_arch']})...")
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        learning_rate=args.learning_rate,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir,
        seed=args.seed,
        device=args.device,
    )

    print(f"Training for {args.total_steps:,} timesteps...")
    model.learn(total_timesteps=args.total_steps, callback=periodic_eval)

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    print(f"Saved: {model_path}.zip")

    print("Running final benchmark eval...")
    result = _final_eval(model, args.env, log_dir, eval_seed, args.num_eval_eps)
    proto = BenchmarkProtocolConfig()
    result["seed"] = int(args.seed)
    result["total_env_steps"] = int(args.total_steps)
    result["protocol_version"] = proto.protocol_version
    result["algorithm"] = "SAC (SB3, paper-matched hyperparameters)"
    result["smoke_test"] = bool(args.smoke_test)
    result["hparams"] = {
        "n_envs": args.n_envs,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "tau": args.tau,
        "learning_rate": args.learning_rate,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "net_arch": _parse_net_arch(args.net_arch),
        "utd": args.gradient_steps / args.n_envs,
    }

    out_path = os.path.join(log_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"eval summary → {out_path}")
    print(f"summary: {json.dumps(result['summary'], indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
