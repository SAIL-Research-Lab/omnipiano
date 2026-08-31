"""Render a recorded video for a saved SB3 / sb3-contrib checkpoint.

Usage:
    python examples/render_checkpoint.py \
        --ckpt examples/logs/sac_3hand_forelise_1/best_model.zip \
        --env  OmniPiano-ForElise-ThreeHandPrototype-v0

Loads the checkpoint, builds the env with ``record_dir=<ckpt_dir>/videos``
(triggers ``PianoSoundVideoWrapper`` at the dm_env layer — saves an MP4 +
WAV per episode), runs one deterministic episode, prints terminal metrics.

Algorithm auto-detection
------------------------
The script identifies the algorithm class by parsing the ``data`` JSON
embedded inside the SB3 zip (key: ``policy_class``). This handles three
distinct algorithms:

* **PPO** — `stable_baselines3.PPO`
* **SAC** — `stable_baselines3.SAC`
* **TQC** — `sb3_contrib.TQC`  (requires ``pip install sb3-contrib``)

Falling back to filename heuristics (older approach: detecting
``policy.optimizer.pth`` vs ``actor.optimizer.pth``) cannot distinguish
SAC from TQC because TQC ships the same files as SAC. The
``policy_class`` JSON path is the reliable disambiguator.

If TQC is detected but ``sb3_contrib`` is not importable in the current
Python environment, the script fails fast with an actionable error
pointing the user at the right install command and conda env hint.

Note (project-internal usage)
-----------------------------
This script is a convenience wrapper specific to the SB3 / sb3-contrib
saved-zip format. Other RL frameworks (RLlib, CleanRL, Mava, MARLlib,
JaxMARL, ...) use different checkpoint layouts and need their own
rendering scripts. Treat this file as a local utility for the SB3
templates under `examples/` rather than a canonical public-API entry
point of the benchmark.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from typing import Optional, Type

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from omnipiano import make


# ---------------------------------------------------------------------------
# Algorithm registry — TQC entry is lazy-loaded so this module can still be
# imported (and PPO/SAC checkpoints rendered) when sb3-contrib is missing.
# ---------------------------------------------------------------------------

_BUILTIN_ALGOS: dict[str, Type[BaseAlgorithm]] = {"PPO": PPO, "SAC": SAC}
_VALID_ALGO_CHOICES = ("PPO", "SAC", "TQC")


def _resolve_algo_class(name: str) -> Type[BaseAlgorithm]:
    """Return the algorithm class for a detected name, lazy-loading TQC.

    Raises a clear actionable error if the user asks for TQC but
    ``sb3_contrib`` is not importable.
    """
    if name in _BUILTIN_ALGOS:
        return _BUILTIN_ALGOS[name]
    if name == "TQC":
        try:
            from sb3_contrib import TQC  # noqa: WPS433 (intentional lazy import)
        except ImportError as e:  # pragma: no cover (rendered at runtime)
            raise SystemExit(
                "TQC checkpoint detected but `sb3_contrib` is not installed in "
                "the current Python environment.\n"
                f"  Active interpreter: {os.path.realpath(os.sys.executable)}\n"
                f"  Install with:       pip install sb3-contrib\n"
                f"  Or activate the matching conda env (e.g., `conda activate "
                f"pianist`) and re-run this script.\n"
                f"  Original ImportError: {e}"
            )
        return TQC
    raise SystemExit(f"Unknown algorithm name: {name!r}; expected one of {_VALID_ALGO_CHOICES}")


# ---------------------------------------------------------------------------
# Algorithm detection from the saved zip
# ---------------------------------------------------------------------------


def _detect_algo(ckpt_path: str) -> Optional[str]:
    """Identify the saving algorithm class by reading the SB3 zip's ``data`` JSON.

    SB3's ``BaseAlgorithm.save()`` packs a ``data`` member that contains a
    JSON-encoded ``policy_class`` string of the form
    ``"<class 'stable_baselines3.sac.policies.SACPolicy'>"`` or
    ``"<class 'sb3_contrib.tqc.policies.TQCPolicy'>"``. This is more
    reliable than filename heuristics because TQC ships the same
    ``actor.optimizer.pth`` / ``ent_coef_optimizer.pth`` files as SAC.

    Returns one of ``"PPO"``, ``"SAC"``, ``"TQC"``, or ``None`` if the
    zip does not match a recognized SB3 layout.
    """
    try:
        with zipfile.ZipFile(ckpt_path) as zf:
            if "data" not in zf.namelist():
                return None
            with zf.open("data") as f:
                blob = f.read().decode("utf-8")
        data = json.loads(blob)
    except (zipfile.BadZipFile, FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

    # ``policy_class`` is serialized in two possible forms by SB3:
    #   - simple string:   "<class 'stable_baselines3.sac.policies.SACPolicy'>"
    #   - dict object:     {":serialized:": "...", ":type:": "<class '...'>", ...}
    policy_class = data.get("policy_class") or data.get("policy_kwargs", {}).get("policy_class")
    candidate_strs: list[str] = []
    if isinstance(policy_class, str):
        candidate_strs.append(policy_class)
    elif isinstance(policy_class, dict):
        for v in policy_class.values():
            if isinstance(v, str):
                candidate_strs.append(v)

    for s in candidate_strs:
        sl = s.lower()
        if "sb3_contrib.tqc" in sl or "tqcpolicy" in sl:
            return "TQC"
        if "stable_baselines3.sac" in sl or "sacpolicy" in sl:
            return "SAC"
        if "stable_baselines3.ppo" in sl or "actorcriticpolicy" in sl:
            return "PPO"

    # ----- Fall back to filename heuristics if JSON path inconclusive -----
    try:
        with zipfile.ZipFile(ckpt_path) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return None
    if "actor.optimizer.pth" in names or "ent_coef_optimizer.pth" in names:
        # Cannot disambiguate SAC vs TQC from filenames alone — return SAC
        # as the safer default; user can override via --algo TQC.
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
        choices=list(_VALID_ALGO_CHOICES),
        help="Algorithm class (PPO/SAC/TQC). Auto-detected from the "
        "checkpoint's `data` JSON (`policy_class` field) when omitted; "
        "pass explicitly to override (useful when filename heuristics "
        "must disambiguate SAC vs TQC and JSON detection is inconclusive).",
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
    parser.add_argument(
        "--resolution",
        default="480x640",
        help="Render size HEIGHTxWIDTH (default 480x640, paper-same). "
        "e.g. 960x1280 for a sharper full-keyboard topdown clip.",
    )
    args = parser.parse_args()
    rec_h, rec_w = (int(v) for v in args.resolution.lower().split("x"))

    # ---- algo selection -----------------------------------------------------
    detected = _detect_algo(args.ckpt)
    algo_name = args.algo or detected
    if algo_name is None:
        raise SystemExit(
            f"Could not auto-detect algorithm class from {args.ckpt}. "
            f"Pass --algo {{{','.join(_VALID_ALGO_CHOICES)}}} explicitly."
        )
    if args.algo and detected and args.algo != detected:
        print(
            f"Warning: --algo {args.algo} overrides auto-detected "
            f"{detected}. Loading may fail if the override is wrong."
        )

    algo_cls = _resolve_algo_class(algo_name)

    ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_dir = args.out_dir or os.path.join(ckpt_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {algo_name} ({algo_cls.__module__}.{algo_cls.__name__}) from: {args.ckpt}")
    model = algo_cls.load(args.ckpt, device="cpu")

    print(f"Building env: {args.env}  (record_dir={out_dir}, camera={args.camera})")
    env = make(
        args.env,
        seed=args.seed,
        record_dir=out_dir,
        camera_id=args.camera,
        record_resolution=(rec_h, rec_w),
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
