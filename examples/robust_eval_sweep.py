"""robust_eval_sweep.py — SB3 robustness sweep harness (Phase 1, §5.2).

Sibling of ``examples/checkpoint_replay_eval.py``, but for **SB3** policies on
robust tasks. Robust tasks carry no safety cost → they are SB3 territory; do
**NOT** copy checkpoint_replay's ``omnisafe.Evaluator`` path here (that is
OmniSafe-specific).

Given one trained SB3 checkpoint and a registered robust env id, this evaluates
the SAME policy over a grid of eval noise magnitudes (``eval_noise_scale``) and
produces the robustness curve's data points:

    for scale in scales:
        env = make(env_id, mode="eval", eval_noise_scale=scale)
        run N deterministic episodes   # env's own SafeRecordEpisodeStatistics
                                        # writes the schema-locked per-ep CSV
    concat per-scale CSVs → robust_sweep_<env>.csv

Reuse, not reimplement: each per-scale eval env is built with ``log_dir`` set,
so its ``SafeRecordEpisodeStatistics`` writes the full 24-column CSV
(``ep_return`` / ``ep_cost`` / ``ep_f1`` / decomposition / ``eval_noise_scale``
/ ``ep_return_true`` / ``ep_noise_*``) — identical schema to every other eval
CSV, including the robust-noise accumulation and ``ep_return_true``.

Correctness (§5.2):
  * Structure held constant by env_id: eval env == training env in task / MIDI
    / wrapper chain / obs+action space; the ONLY difference across the sweep is
    the noise magnitude (``scale``). ``scale=1.0`` reproduces training noise.
  * Fixed anchor seed: all scales reset with the same
    ``seed_base = train_seed + EVAL_SEED_OFFSET`` (+ ep index), so every scale
    runs the same batch of episodes (same MIDI / init) and the curve reflects
    noise only, not env-init drift.
  * Decision 11: reward is symmetric with action/obs — the scale grid runs on
    ALL channels (A/O/R) identically; no Clean-v0 switch, no channel branching.
  * Raw obs (no VecNormalize, §11) → the policy runs directly on the env from
    make(); no normalization stats to transfer.

Example:
    MUJOCO_GL=egl python examples/robust_eval_sweep.py \
        --ckpt runs/clairdelune_a_gauss_p05/best_model.zip \
        --env  OmniPiano-ClairDeLune-A-Gauss-P05-v0 \
        --algo ppo --scales 0.0 0.5 1.0 2.0 4.0 --num-eval-eps 10
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

# --- repo-root import shim (mirrors checkpoint_replay_eval.py) -------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import omnipiano  # noqa: E402  (triggers env registration side-effect)
from omnipiano.envs.registration import make  # noqa: E402

# Match the SB3 / checkpoint_replay convention (examples/checkpoint_replay_eval.py:48).
EVAL_SEED_OFFSET = 10_000


# ---------------------------------------------------------------------------
# SB3 model loading — by algo (robust = no cost = SB3; PPO/SAC/TQC)
# ---------------------------------------------------------------------------
def _load_model(algo: str, ckpt: str, device: str, framework: str = "sb3"):
    if framework == "torchrl":
        from omnipiano.integrations.torch_rl.model.policy_adapter import (
            TorchRLPolicyAdapter,
        )
        return TorchRLPolicyAdapter.load(ckpt, device=device)
    algo = algo.lower()
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(ckpt, device=device)
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC.load(ckpt, device=device)
    if algo == "tqc":
        from sb3_contrib import TQC
        return TQC.load(ckpt, device=device)
    raise ValueError(f"--algo must be one of ppo/sac/tqc, got {algo!r}")


# ---------------------------------------------------------------------------
# One scale: run N deterministic episodes; SafeRecord writes the per-ep CSV
# ---------------------------------------------------------------------------
def _run_one_scale(model, env_id, scale, scale_dir, num_eval_eps, seed_base):
    os.makedirs(scale_dir, exist_ok=True)
    env = make(
        env_id,
        log_dir=scale_dir,           # → SafeRecordEpisodeStatistics attaches
        mode="eval",
        eval_noise_scale=scale,
        seed=seed_base,
    )
    try:
        for ep_idx in range(num_eval_eps):
            # Fixed anchor seed, scale-independent → same episode batch per scale.
            obs, _ = env.reset(seed=seed_base + ep_idx)
            done = False
            while not done:
                prediction = model.predict(obs, deterministic=True)
                action = prediction[0] if isinstance(prediction, tuple) else prediction
                obs, _reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated) or bool(truncated)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    # SafeRecord wrote exactly one CSV in scale_dir (env_id=None → uuid name).
    hits = glob.glob(os.path.join(scale_dir, "eval_episode_metrics_*.csv"))
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one eval CSV in {scale_dir}, found {hits}")
    return hits[0]


# ---------------------------------------------------------------------------
# Concatenate per-scale CSVs → one sweep CSV (prepend env_id + ckpt tag)
# ---------------------------------------------------------------------------
def _concat_csvs(per_scale_csvs, combined_path, env_id, ckpt_tag):
    header = None
    rows = []
    for path in per_scale_csvs:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            elif file_header != header:
                raise RuntimeError(
                    f"schema drift between per-scale CSVs:\n{header}\nvs\n{file_header}")
            rows.extend(reader)
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["env_id", "ckpt"] + header)
        for row in rows:
            writer.writerow([env_id, ckpt_tag] + row)
    return combined_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="SB3 .zip or TorchRL checkpoint directory.")
    p.add_argument("--env", required=True,
                   help="Registered robust env id (the robustness-curve eval env).")
    p.add_argument("--framework", default="sb3", choices=["sb3", "torchrl"])
    p.add_argument("--algo", default="ppo",
                   choices=["ppo", "sac", "tqc", "eppo", "a2p_sac", "ompo", "scpo"],
                   help="Algorithm name; TorchRL checkpoint carries its network specification.")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 4.0],
                   help="eval_noise_scale grid (0=clean, 1=matched/training, >1=stress).")
    p.add_argument("--num-eval-eps", type=int, default=10,
                   help="Deterministic episodes per scale.")
    p.add_argument("--train-seed", type=int, default=0,
                   help="Training seed; eval seeds anchor at train_seed + eval_seed_offset.")
    p.add_argument("--eval-seed-offset", type=int, default=EVAL_SEED_OFFSET,
                   help=f"Anchor offset (default {EVAL_SEED_OFFSET}, SB3 convention).")
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: <ckpt_dir>/robust_sweep).")
    p.add_argument("--device", default="auto",
                   help="Torch device for SB3 .load (auto/cpu/cuda).")
    args = p.parse_args()

    if any(s < 0 for s in args.scales):
        p.error("--scales must be non-negative (eval_noise_scale >= 0).")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "robust_sweep")
    os.makedirs(out_dir, exist_ok=True)

    model = _load_model(args.algo, args.ckpt, args.device, args.framework)
    seed_base = args.train_seed + args.eval_seed_offset
    ckpt_tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    per_scale_csvs = []
    for scale in args.scales:
        scale_dir = os.path.join(out_dir, f"scale_{scale:g}")
        print(f"[robust_eval_sweep] {args.env}  scale={scale:g}  "
              f"({args.num_eval_eps} eps) → {scale_dir}")
        per_scale_csvs.append(
            _run_one_scale(model, args.env, scale, scale_dir,
                           args.num_eval_eps, seed_base))

    safe_env = args.env.replace("/", "_")
    combined = os.path.join(out_dir, f"robust_sweep_{safe_env}.csv")
    _concat_csvs(per_scale_csvs, combined, args.env, ckpt_tag)
    print(f"[robust_eval_sweep] wrote combined sweep CSV → {combined}")


if __name__ == "__main__":
    main()
