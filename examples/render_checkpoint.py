"""Render a recorded video for a saved SB3 checkpoint.

Usage:
    python examples/render_checkpoint.py \
        --ckpt examples/logs/sac_3hand_forelise_1/best_model.zip \
        --env  OmniPiano-ForElise-ThreeHandPrototype-v0

Loads the checkpoint, builds the env with ``record_dir=<ckpt_dir>/videos``
(triggers ``PianoSoundVideoWrapper`` at the dm_env layer — saves an MP4 +
WAV per episode), runs one deterministic episode, prints terminal metrics.

The script auto-detects the algorithm class (PPO vs SAC) from the
checkpoint's zip contents and verifies that the env's observation space
matches what the model expects, so a checkpoint/env id mismatch fails
fast with a clear error instead of producing silent shape errors deep
in PyTorch. Since OmniPiano now enforces registry-only env construction,
``args.env`` must be the same registered id used during training (or a
new id whose env_config matches that training-time configuration).
"""
from __future__ import annotations

import argparse
import os
import zipfile
from typing import Optional, Type

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from OmniPiano import make


_ALGOS: dict[str, Type[BaseAlgorithm]] = {"PPO": PPO, "SAC": SAC}


def _detect_algo(ckpt_path: str) -> Optional[str]:
    """Peek into the SB3 zip to identify SAC vs PPO.

    Heuristic: SB3's ``BaseAlgorithm.save()`` packs algorithm-specific
    optimizer state files. SAC ships ``actor.optimizer.pth`` (separate
    actor/critic optimizers + entropy coefficient optimizer); PPO ships
    a single ``policy.optimizer.pth``. Reading the zip's filename list
    is faster and more robust than parsing the ``data`` json blob and
    cheaper than try-loading the whole checkpoint twice.
    """
    try:
        with zipfile.ZipFile(ckpt_path) as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, FileNotFoundError):
        return None
    if "actor.optimizer.pth" in names or "ent_coef_optimizer.pth" in names:
        return "SAC"
    if "policy.optimizer.pth" in names:
        return "PPO"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--env", required=True, help="Registered OmniPiano env id")
    parser.add_argument(
        "--algo",
        default=None,
        choices=list(_ALGOS),
        help="SB3 algo class (PPO/SAC). Auto-detected from the checkpoint "
        "zip contents if omitted; pass explicitly to override.",
    )
    parser.add_argument("--seed", type=int, default=67, help="Eval seed (default 67)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write videos. Default: <ckpt_dir>/videos",
    )
    parser.add_argument(
        "--camera",
        default="piano/topdown",
        help="Camera id (piano/topdown shows full keyboard; "
        "piano/back is the paper default; also available: "
        "piano/closeup, piano/left, piano/right, piano/egocentric).",
    )
    args = parser.parse_args()

    # ---- algo selection -----------------------------------------------------
    algo_name = args.algo or _detect_algo(args.ckpt)
    if algo_name is None:
        raise SystemExit(
            f"Could not auto-detect algorithm class from {args.ckpt}. "
            "Pass --algo PPO or --algo SAC explicitly."
        )
    if args.algo and args.algo != _detect_algo(args.ckpt):
        print(
            f"Warning: --algo {args.algo} overrides auto-detected "
            f"{_detect_algo(args.ckpt)}. Loading may fail."
        )

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_dir = args.out_dir or os.path.join(ckpt_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {algo_name} from: {args.ckpt}")
    model = _ALGOS[algo_name].load(args.ckpt, device="cpu")

    print(f"Building env: {args.env}  (record_dir={out_dir}, camera={args.camera})")
    env = make(
        args.env,
        seed=args.seed,
        record_dir=out_dir,
        camera_id=args.camera,
    )

    # ---- shape sanity check before rolling out ------------------------------
    expected = tuple(model.observation_space.shape)
    actual = tuple(env.observation_space.shape)
    if expected != actual:
        env.close()
        raise SystemExit(
            f"\nObservation-shape mismatch:\n"
            f"  model expects: {expected}\n"
            f"  env produces:  {actual}\n"
            f"Common causes:\n"
            f"  - --env points at a different morphology than training (e.g., 2-hand vs 3-hand)\n"
            f"  - --action-reward-observation toggle disagrees with training\n"
            f"  - registry's BenchmarkEnvConfig changed since the checkpoint was saved"
        )

    # ---- rollout ------------------------------------------------------------
    obs, _ = env.reset(seed=args.seed)
    ep_return, ep_length = 0.0, 0
    terminal_info: dict = {}
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
    print(f"\nEpisode finished — return={ep_return:.2f}  length={ep_length}")
    for k in sorted(metrics):
        print(f"  {k:48s} {metrics[k]:.4f}")

    env.close()
    print(f"\nVideo saved under: {out_dir}")


if __name__ == "__main__":
    main()
