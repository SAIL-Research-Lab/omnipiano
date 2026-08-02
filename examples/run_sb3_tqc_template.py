"""SB3-contrib TQC trainer for OmniPiano — TQC counterpart of run_sb3_sac_template.py.

This file is **the** TQC entry point. Per-experiment shell wrappers in
``examples/runs/`` pin the env + a few overrides (e.g., ``n_envs``,
``gradient_steps``) and call this script. Run with ``--help`` for the
full CLI surface.

Why TQC over SAC for OmniPiano N-hand tasks:
- SAC's twin-critic min-of-two underestimates Q on long-horizon dense-reward
  tasks (314+ steps with cumulative key_press / sustain rewards). TQC replaces
  that with **distributional critics + truncation of top quantiles**, which
  gives a tunable bias-variance knob and consistently beats SAC on DM Control
  / locomotion benchmarks (Kuznetsov et al., 2020, https://arxiv.org/abs/2005.04269).
- Drop-in replacement for SAC's API: same {policy, gamma, tau, lr, buffer,
  train_freq, gradient_steps} surface; only adds {n_quantiles, n_critics,
  top_quantiles_to_drop_per_net} on top.

Defaults match TQC paper / SB3-contrib defaults except where SAC's paper-matched
choices apply (gamma, batch_size, lr, buffer, etc. — same as SAC template):
- ``hidden_dims = (256, 256, 256)``        — paper SAC convention
- ``gamma = 0.8``                          — RoboPianist paper
- ``batch_size = 256, buffer = 1M``        — paper
- ``learning_starts = 5000``               — paper
- ``tau = 0.005``                          — paper
- ``learning_rate = 3e-4``                 — paper
- ``n_quantiles = 25``                     — TQC paper / SB3 default
- ``n_critics = 2``                        — SB3 default (paper uses 5; we keep 2 for compute parity with SAC)
- ``top_quantiles_to_drop_per_net = 2``    — TQC paper / SB3 default
- ``total_timesteps = 5_000_000``          — BenchmarkProtocolConfig

For paper-matched UTD = 1.0 across n_envs, set ``--gradient-steps`` =
``--n-envs`` (same convention as SAC template).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as sb3_configure_logger
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
        default="tqc_sb3_template",
        help="Prefix for log_dir under examples/logs/.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny budget for pipeline validation only (NOT a benchmark result). "
        "Forces total_timesteps=20k and num_eval_eps=1.",
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
    p.add_argument("--eval-interval-env-steps", type=int,
                   default=proto.eval_freq_env_steps,
                   help="Periodic-eval cadence in env-steps. Defaults to the "
                        "protocol constant so this template's eval grid lines "
                        "up with run_sb3_baseline.py's; a different value puts "
                        "the curves on a grid that cannot be overlaid.")

    # --- shared SAC-family hparams ---
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

    # --- TQC-specific hparams ---
    p.add_argument("--n-quantiles", type=int, default=25,
                   help="Number of quantiles per critic head. TQC paper / SB3 default = 25.")
    p.add_argument("--n-critics", type=int, default=2,
                   help="Number of critic networks. SB3 default = 2; TQC paper uses 5. "
                   "n_critics=2 keeps compute parity with SAC's twin critics.")
    p.add_argument("--top-quantiles-to-drop-per-net", type=int, default=2,
                   help="Top-K quantiles truncated per critic before averaging. "
                   "Higher = more pessimistic Q-target. TQC default = 2.")

    return p


def _env_creator(env_name: str, log_dir: str, split: str, **extra):
    """Build one OmniPiano env. Shared by training, periodic eval, final eval."""
    def _make():
        return make(env_name, log_dir=log_dir, mode=split, **extra)
    return _make


def _final_eval(
    model: TQC,
    env_name: str,
    run_dir: str,
    eval_seed: int,
    num_eval_eps: int,
) -> Dict[str, Any]:
    """Final benchmark eval: deterministic rollout, collect terminal info.

    Identical logic to the SAC template's _final_eval — TQC inherits
    BaseAlgorithm.predict() so the rollout loop is unchanged.
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
        args.total_steps = 20_000
        args.num_eval_eps = 1
        # Shrink the eval cadence too, or the protocol's 50k interval would
        # never fire inside a 20k budget and the smoke test would silently
        # skip the EvalCallback -> SafeRecordEpisodeStatistics path it exists
        # to exercise. Matches run_sb3_baseline.py's smoke block.
        args.eval_interval_env_steps = 10_000

    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_root, exist_ok=True)
    next_run_id = get_latest_run_id(logs_root, args.experiment_name) + 1
    experiment_id = f"{args.experiment_name}_{next_run_id}"
    log_dir = os.path.join(logs_root, experiment_id)
    os.makedirs(log_dir, exist_ok=True)

    eval_seed = args.seed + args.n_envs + 1

    print(f"Env: {args.env}")
    print(
        f"Algo: TQC | "
        f"Mode: {'SMOKE_TEST' if args.smoke_test else 'canonical'} | "
        f"seed={args.seed} eval_seed={eval_seed} num_eval_eps={args.num_eval_eps} "
        f"total_timesteps={args.total_steps:,} n_envs={args.n_envs} "
        f"UTD={args.gradient_steps / args.n_envs:.3f} "
        f"n_quantiles={args.n_quantiles} n_critics={args.n_critics} "
        f"drop={args.top_quantiles_to_drop_per_net}"
    )

    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

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

    # TQC's policy_kwargs accepts SAC-family keys (net_arch) PLUS TQC-specific
    # head config (n_quantiles, n_critics). top_quantiles_to_drop_per_net is a
    # top-level TQC arg, not policy_kwargs.
    policy_kwargs = dict(
        net_arch=_parse_net_arch(args.net_arch),
        n_quantiles=args.n_quantiles,
        n_critics=args.n_critics,
    )
    print(
        f"Initializing TQC (policy=MlpPolicy, "
        f"net_arch={policy_kwargs['net_arch']}, "
        f"n_quantiles={args.n_quantiles}, n_critics={args.n_critics}, "
        f"drop_per_net={args.top_quantiles_to_drop_per_net})..."
    )
    model = TQC(
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
        top_quantiles_to_drop_per_net=args.top_quantiles_to_drop_per_net,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir,
        seed=args.seed,
        device=args.device,
    )

    # ``tensorboard_log=`` in the constructor above installs a TB-only logger,
    # which leaves no progress.csv on disk. Replace it with the same
    # multi-format logger run_sb3_baseline.py uses, so this template emits the
    # native training-rollout telemetry (rollout/ep_rew_mean, train/*, time/*)
    # that the paper's reward curves are read from.
    model.set_logger(sb3_configure_logger(
        log_dir, ["stdout", "csv", "tensorboard"]
    ))

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
    result["algorithm"] = "TQC (sb3-contrib)"
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
        "n_quantiles": args.n_quantiles,
        "n_critics": args.n_critics,
        "top_quantiles_to_drop_per_net": args.top_quantiles_to_drop_per_net,
    }

    out_path = os.path.join(log_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"eval summary → {out_path}")
    print(f"summary: {json.dumps(result['summary'], indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
