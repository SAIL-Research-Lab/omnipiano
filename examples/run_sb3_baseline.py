"""Unified SB3 baseline trainer for the OmniPiano benchmark.

ONE script for PPO / SAC / TQC at **library-default hyperparameters**. This is
the "out-of-the-box" baseline used to populate the benchmark's main results
table; per-algorithm tuning is intentionally NOT applied here.

Usage
-----
    python examples/run_sb3_baseline.py --algo sac --seed 0
    python examples/run_sb3_baseline.py --algo ppo --seed 1
    python examples/run_sb3_baseline.py --algo tqc --seed 2

What is held identical across algos
-----------------------------------
- env_id, total_timesteps, eval_freq (in env-steps), n_eval_episodes,
  ``deterministic=True`` at eval, no VecNormalize, MlpPolicy.

What differs per algo (deliberately — these are the library defaults)
---------------------------------------------------------------------
- PPO: on-policy, defaults to ``n_envs=16`` on this host (= physical core
  count of the 16C/32T box; PPO needs parallel envs for diverse advantage
  estimates; HT-doubled n_envs=32 is slower due to cache contention).
- SAC: off-policy, ``n_envs=24`` (2026-06-12 calibration, see registry
  comment); TQC: off-policy, default ``n_envs=1`` (replay-buffer based).
- All other hparams (lr, gamma, batch_size, net_arch, ent_coef, ...) are
  whatever the installed sb3 / sb3-contrib version ships as defaults.

Pin sb3 + sb3-contrib versions in the paper for reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List, Tuple, Type

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import TQC

import stable_baselines3
import sb3_contrib

from omnipiano import make
from omnipiano.configs import BenchmarkProtocolConfig


# algo -> (class, policy, default n_envs on this host, default constructor extras)
#
# PPO: n_envs=16 = physical core count on this 16C/32T host. Hyperthreads
#   don't help MuJoCo (CPU-bound, cache-sensitive); going to 32 contends.
#   No constructor extras — all hparams left at SB3 library defaults.
#
# SAC: n_envs=24 chosen empirically (2026-06-12 calibration on 4-hand
#   PicturesGreatKiev StaticPartition, seed 0, 150K steps): n_envs=24
#   delivered the best ent_coef stability AND the second-fastest wallclock,
#   safely between the two failure modes we measured:
#     - n_envs ≤ 8  → ent_coef collapse (auto-tuned to ~0.006, policy → 'do nothing')
#     - n_envs ≥ 32 → ent_coef explosion (auto-tuned to ~5.2 by 150K, entropy
#                     bonus drowns task signal)
#   gradient_steps=-1 paired so UTD = gradient_steps / (train_freq*n_envs)
#   stays at 1.0 (matches SAC paper / RoboPianist convention).
#
# TQC: n_envs left at SB3 default 1 pending calibration (TQC's pessimistic
#   Q targets are expected to be more robust at higher n_envs than SAC's
#   auto-α, but unverified empirically).
ALGO_REGISTRY: Dict[str, Tuple[Type[BaseAlgorithm], str, int, Dict[str, Any]]] = {
    "ppo": (PPO, "MlpPolicy", 16, {}),
    "sac": (SAC, "MlpPolicy", 24, {"gradient_steps": -1}),
    "tqc": (TQC, "MlpPolicy", 1, {}),
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    proto = BenchmarkProtocolConfig()

    p.add_argument("--algo", required=True, choices=sorted(ALGO_REGISTRY))
    p.add_argument(
        "--env",
        default="OmniPiano-PicturesGreatKiev-ThreeHand-StaticPartition-v0",
        help="Registered OmniPiano env id.",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Log dir prefix under examples/logs/. Default: "
        "'{algo}_baseline_{env_short}_seed{seed}'.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=proto.seed,
        help=f"Training seed. Default = BenchmarkProtocolConfig.seed "
        f"({proto.seed}) = seeds[0], i.e. replicate #1 of the reported "
        f"replication set {proto.seeds}. Paper numbers require the full "
        "set — run this script once per seed and aggregate offline.",
    )
    p.add_argument(
        "--eval-seed-offset",
        type=int,
        default=proto.eval_seed_offset,
        help="Evaluation episode i uses training seed + this offset + i; "
        "the stream is independent of n_envs and algorithm.",
    )
    p.add_argument(
        "--total-steps",
        type=int,
        default=proto.total_env_steps,
        help=f"Total env steps (vec-env-aggregated). Default {proto.total_env_steps:,}.",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=proto.eval_freq_env_steps,
        help=f"Periodic eval cadence in env-steps. Default = "
        f"BenchmarkProtocolConfig.eval_freq_env_steps "
        f"({proto.eval_freq_env_steps:,}) — unified across SB3 / OmniSafe / "
        "future framework backends so learning-curve samples align. "
        "Actual eval interval snaps to the algo's rollout boundary "
        "(SB3 PPO @ rollout=32K → ~65K effective).",
    )
    p.add_argument(
        "--num-eval-eps",
        type=int,
        default=proto.num_eval_eps,
        help=f"Episodes per periodic-eval AND for the final benchmark eval. "
        f"Default = BenchmarkProtocolConfig.num_eval_eps "
        f"({proto.num_eval_eps}); env is deterministic at reset, so 1 is "
        "sufficient.",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override library-default n_envs (PPO=16, SAC=24, TQC=1). "
        "Off-policy algos: changing this affects sample efficiency "
        "interpretation.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny budget for pipeline validation (total_steps=20_000).",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="SAC/TQC only: gradient_steps per train_freq. ``-1`` = match "
        "transitions collected per vec-step (preserves UTD=1.0 when scaling "
        "n_envs). Library default = 1 (UTD = 1/n_envs). Ignored for PPO.",
    )
    p.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="SAC/TQC only: number of warmup transitions to collect with a "
        "uniform random policy before any gradient update. SB3 default = 100; "
        "RoboPianist paper / robopianist-rl uses 5000. Critical for SAC "
        "auto-α stability — too small a warmup leaves the critic learning from "
        "a tiny biased buffer, which can trigger auto-α explosion on long "
        "training horizons.",
    )
    p.add_argument(
        "--ent-coef",
        default=None,
        help='SAC/TQC only: entropy coefficient α. Default "auto" tunes α via '
        "dual gradient ascent (Haarnoja 2019). Pass a float (e.g. ``0.1``) to "
        "fix α — recommended for N-hand StaticPartition tasks where auto-α can "
        "diverge (see ``static_partition_design.md`` §7).",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=proto.gamma,
        help=f"Discount factor. OmniPiano protocol default = {proto.gamma} "
        "(treated as a *task property*, applied uniformly across SAC / PPO / "
        "TQC / future algorithms — see ``BenchmarkProtocolConfig.gamma`` for "
        "rationale and ablation evidence). Library defaults would be 0.99 for "
        "both SAC and PPO; pass ``--gamma 0.99`` explicitly to reproduce "
        "library-default behavior for ablation.",
    )
    p.add_argument(
        "--checkpoint-at",
        default="",
        help="Comma-separated env-step counts at which to save an extra "
        "checkpoint, e.g. ``--checkpoint-at 5000000``. Saved as "
        "``checkpoints/checkpoint_<N>.zip`` in the log_dir. Independent of "
        "best_model.zip (always written by EvalCallback) and "
        "final_model.zip (written at end of training).",
    )
    return p


def _postprocess_eval_csv(log_dir: str) -> None:
    """Join SB3's authoritative training-step axis into the periodic eval CSV.

    OmniPiano's ``logger_wrapper`` records ``env_step_count`` — a per-eval-env
    local step counter — which is NOT the training step. SB3's ``EvalCallback``
    writes the true training step at each eval event into
    ``evaluations.npz['timesteps']``. This function does a 1:1 row-by-row join
    so the CSV gains a ``training_step`` column as its leftmost field.

    Only operates on the periodic in-training eval CSV (row count matches
    ``len(timesteps)``); skips final-eval CSVs (1 row, no match) and any CSV
    that already has a ``training_step`` column.
    """
    npz_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        print(f"[postprocess] no evaluations.npz in {log_dir}, skipping")
        return
    ts = np.load(npz_path)["timesteps"]

    for path in sorted(glob.glob(os.path.join(log_dir, "eval_episode_metrics_*.csv"))):
        with open(path, "r", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = [h.strip() for h in rows[0]]
        if "training_step" in header:
            print(f"[postprocess] {os.path.basename(path)} already has training_step, skipping")
            continue
        data_rows = rows[1:]
        if len(data_rows) != len(ts):
            print(f"[postprocess] {os.path.basename(path)}: {len(data_rows)} data rows vs {len(ts)} "
                  f"npz timesteps — not the periodic eval CSV, skipping")
            continue
        new_header = ["training_step"] + header
        new_rows = [[str(int(ts[i]))] + r for i, r in enumerate(data_rows)]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(new_header)
            w.writerows(new_rows)
        print(f"[postprocess] added training_step to {os.path.basename(path)} ({len(data_rows)} rows)")


class StepCheckpointCallback(BaseCallback):
    """Save a model snapshot at specific env-step counts.

    SB3's ``CheckpointCallback`` only supports a fixed period. We want a
    single snapshot at, e.g., 5M when total_steps=8M so the 5M state is
    archived in case 5M ends up being the canonical budget. The check
    triggers on the first ``_on_step`` where
    ``self.num_timesteps >= target`` (so the saved model has at least the
    target's worth of training, never less).
    """

    def __init__(self, save_at: List[int], save_dir: str, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.save_at = sorted(set(int(x) for x in save_at))
        self.save_dir = save_dir
        self._next_idx = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        while (
            self._next_idx < len(self.save_at)
            and self.num_timesteps >= self.save_at[self._next_idx]
        ):
            target = self.save_at[self._next_idx]
            path = os.path.join(self.save_dir, f"checkpoint_{target}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[StepCheckpointCallback] saved {path} (num_timesteps={self.num_timesteps:,})")
            self._next_idx += 1
        return True


def _env_creator(env_name: str, log_dir: str, split: str, **extra):
    def _make():
        return make(env_name, log_dir=log_dir, mode=split, **extra)
    return _make


def _short_env_token(env_id: str) -> str:
    """OmniPiano-PicturesGreatKiev-ThreeHand-StaticPartition-v0
       -> picturesgreatkiev_threehand_staticpartition"""
    parts = env_id.split("-")
    # drop 'OmniPiano' prefix and trailing 'v0'
    body = [p for p in parts if p and p != "OmniPiano" and not p.startswith("v")]
    return "_".join(s.lower() for s in body)


def _record_default_hparams(model: BaseAlgorithm) -> Dict[str, Any]:
    """Snapshot the library-default hparams actually in effect."""
    out: Dict[str, Any] = {}
    for key in (
        "gamma", "learning_rate", "batch_size", "n_steps", "n_epochs",
        "gae_lambda", "clip_range", "ent_coef", "vf_coef",
        "buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps",
        "target_entropy", "target_update_interval", "use_sde",
    ):
        if not hasattr(model, key):
            continue
        val = getattr(model, key)
        try:
            if callable(val):
                val = "schedule_or_callable"
            elif hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                val = list(val)
            json.dumps(val)
            out[key] = val
        except (TypeError, ValueError):
            out[key] = repr(val)
    # net arch — accessible via policy
    policy = getattr(model, "policy", None)
    if policy is not None:
        na = getattr(policy, "net_arch", None)
        if na is not None:
            try:
                json.dumps(na)
                out["net_arch"] = na
            except (TypeError, ValueError):
                out["net_arch"] = repr(na)
    return out


def _final_eval(
    model: BaseAlgorithm,
    env_name: str,
    run_dir: str,
    eval_seed: int,
    num_eval_eps: int,
) -> Dict[str, Any]:
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    env = make(
        env_name,
        log_dir=run_dir,
        mode="eval",   # unified with other templates; tools/postprocess_eval_csv.py
                       # renames the 1-row final-eval CSV to final_eval_*.csv post-hoc
        record_dir=video_dir,
        seed=eval_seed,
    )
    episodes: List[Dict[str, Any]] = []
    try:
        for ep_idx in range(num_eval_eps):
            obs, _ = env.reset(seed=eval_seed + ep_idx)
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
    return {"env_id": env_name, "eval_seed": int(eval_seed),
            "episodes": episodes, "summary": summary}


def main():
    args = _build_arg_parser().parse_args()
    if args.eval_seed_offset < 0:
        raise ValueError("--eval-seed-offset must be non-negative")

    AlgoCls, policy_name, default_n_envs, default_extras = ALGO_REGISTRY[args.algo]
    n_envs = args.n_envs if args.n_envs is not None else default_n_envs

    if args.smoke_test:
        args.total_steps = 20_000
        args.eval_freq = 10_000
        args.num_eval_eps = 1

    logs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_root, exist_ok=True)
    # Fallback experiment name carries the ``sb3_baseline`` marker so log dirs
    # are self-documenting about which entry script produced them (the
    # per-algorithm ``run_sb3_{sac,tqc}_template.py`` use their own prefixes).
    # Override explicitly with ``--experiment-name`` if a different prefix is
    # desired.
    exp_name = args.experiment_name or (
        f"{args.algo}_sb3_baseline_{_short_env_token(args.env)}_seed{args.seed}"
    )
    next_id = get_latest_run_id(logs_root, exp_name) + 1
    experiment_id = f"{exp_name}_{next_id}"
    log_dir = os.path.join(logs_root, experiment_id)
    os.makedirs(log_dir, exist_ok=True)

    eval_seed = args.seed + args.eval_seed_offset

    print(f"[run_baseline] algo={args.algo.upper()}  env={args.env}")
    print(
        f"[run_baseline] mode={'SMOKE' if args.smoke_test else 'benchmark'}  "
        f"seed={args.seed}  eval_seed={eval_seed}  n_envs={n_envs}  "
        f"total_steps={args.total_steps:,}  eval_freq={args.eval_freq:,} env-steps  "
        f"num_eval_eps={args.num_eval_eps}"
    )
    print(f"[run_baseline] log_dir={log_dir}")

    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        _env_creator(args.env, log_dir, "train"),
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
    )
    eval_env = make_vec_env(
        _env_creator(args.env, log_dir, "eval"),
        n_envs=1,
        seed=eval_seed,
    )

    # EvalCallback's eval_freq is vec-env-step count, not aggregate env steps.
    eval_freq_vec = max(args.eval_freq // n_envs, 1)
    periodic_eval = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq_vec,
        n_eval_episodes=args.num_eval_eps,
        deterministic=True,
        render=False,
    )

    extra_hparams: Dict[str, Any] = dict(default_extras)
    if args.algo in ("sac", "tqc") and args.gradient_steps is not None:
        # CLI overrides the algo-default gradient_steps in ALGO_REGISTRY.
        extra_hparams["gradient_steps"] = args.gradient_steps
    if args.algo in ("sac", "tqc") and args.learning_starts is not None:
        extra_hparams["learning_starts"] = args.learning_starts
    if args.algo in ("sac", "tqc") and args.ent_coef is not None:
        # Accept "auto" or float string.
        try:
            extra_hparams["ent_coef"] = float(args.ent_coef)
        except ValueError:
            extra_hparams["ent_coef"] = args.ent_coef
    # gamma is unified across algorithms via BenchmarkProtocolConfig (task
    # property, see configs/__init__.py). args.gamma is always set (defaults
    # to proto.gamma); pass it through to override the library default.
    extra_hparams["gamma"] = args.gamma

    print(
        f"[run_baseline] Instantiating {AlgoCls.__name__} with LIBRARY DEFAULTS "
        f"(sb3={stable_baselines3.__version__}, sb3-contrib={sb3_contrib.__version__})"
        + (f"  + overrides={extra_hparams}" if extra_hparams else "")
    )
    model = AlgoCls(
        policy_name,
        train_env,
        verbose=1,
        tensorboard_log=tensorboard_dir,
        seed=args.seed,
        device=args.device,
        **extra_hparams,
    )

    # Add CSV output to SB3's logger so the run produces ``progress.csv``
    # alongside the TensorBoard event file. progress.csv is one row per
    # logger.dump() (= per rollout for PPO, per train_freq for SAC), with
    # ``time/total_timesteps`` + ``rollout/ep_rew_mean`` + ``train/*`` losses
    # — schema-parallel to OmniSafe's ``progress.csv``, so the training-
    # rollout curve plotted in the paper appendix can overlay SB3 and
    # OmniSafe baselines on the same axes.
    #
    # ``tensorboard_log=...`` in the AlgoCls constructor above already
    # configured a TB-only logger; we replace it with a multi-format logger
    # that ALSO writes csv (and keeps stdout/tensorboard for parity).
    model.set_logger(sb3_configure_logger(
        log_dir, ["stdout", "csv", "tensorboard"]
    ))

    default_hparams = _record_default_hparams(model)
    print(f"[run_baseline] default hparams in effect: {json.dumps(default_hparams, sort_keys=True)}")

    callbacks: List[BaseCallback] = [periodic_eval]
    save_at = [int(x) for x in args.checkpoint_at.split(",") if x.strip()]
    if save_at:
        ckpt_cb = StepCheckpointCallback(
            save_at=save_at,
            save_dir=os.path.join(log_dir, "checkpoints"),
        )
        callbacks.append(ckpt_cb)
        print(f"[run_baseline] extra checkpoints at env_steps={save_at}")
    callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    print(f"[run_baseline] Training for {args.total_steps:,} timesteps...")
    model.learn(total_timesteps=args.total_steps, callback=callback)

    model_path = os.path.join(log_dir, "final_model")
    model.save(model_path)
    print(f"[run_baseline] saved {model_path}.zip")

    print("[run_baseline] Postprocess: join SB3 evaluations.npz timesteps into eval CSV...")
    _postprocess_eval_csv(log_dir)

    print("[run_baseline] Final benchmark eval (deterministic)...")
    result = _final_eval(model, args.env, log_dir, eval_seed, args.num_eval_eps)
    proto = BenchmarkProtocolConfig()
    result["algorithm"] = f"{args.algo.upper()} (SB3, library defaults)"
    result["seed"] = int(args.seed)
    result["eval_seed_offset"] = int(args.eval_seed_offset)
    result["total_env_steps"] = int(args.total_steps)
    result["protocol_version"] = proto.protocol_version
    result["metrics_protocol_version"] = proto.metrics_protocol_version
    result["smoke_test"] = bool(args.smoke_test)
    # Source-script audit field — lets a downstream analysis script tell runs
    # of this script apart from the per-algorithm template scripts' runs
    # without relying on log-dir naming convention. ``os.path.basename(__file__)``
    # records the actual entry-point filename rather than a hardcoded constant
    # that drifts after a rename.
    result["source_script"] = os.path.basename(__file__)
    result["library_versions"] = {
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }
    result["effective_config"] = {
        "n_envs": n_envs,
        "eval_seed_offset": int(args.eval_seed_offset),
        "eval_freq_env_steps": int(args.eval_freq),
        "default_hparams": default_hparams,
        "overrides": extra_hparams,
        "vec_normalize": False,
    }

    out_path = os.path.join(log_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"[run_baseline] eval summary → {out_path}")
    print(f"[run_baseline] summary: {json.dumps(result['summary'], indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
