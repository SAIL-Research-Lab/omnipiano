"""Checkpoint-replay evaluation for OmniSafe baselines.

OmniSafe lacks an inline periodic-evaluation callback comparable to SB3's
``EvalCallback``. This tool fills the gap by loading each saved checkpoint
from a completed OmniSafe training run, rolling out N deterministic eval
episodes per checkpoint, and writing the result as an SB3-compatible
learning-curve trajectory.

Paper-language: "post-training periodic evaluation via checkpoint replay."

Produces, under ``<log_dir>/``:
    * ``evaluations.npz`` — SB3 schema (``timesteps``, ``results``,
      ``ep_lengths``) extended with ``ep_costs`` for SafeRL.
    * ``eval_episode_metrics_<uuid>.csv`` — per-episode detail with
      ``ep_f1`` / ``ep_precision`` / ``ep_recall`` / sustain_* / reward
      decomposition; mirrors the schema written by SB3 EvalCallback +
      ``SafeRecordEpisodeStatistics``.

These artefacts let ``tools/plot_eval_metrics.py`` plot OmniSafe and SB3
learning curves on the same axes without code changes.

Cost: ~5 min for a 5M PPOLag run (100 ckpts × 1 ep × ~720 steps @ ~400
FPS on CPU; num_eval_eps=1 default — same heuristic as the ETA print). Output is idempotent — re-running
overwrites both files.

Usage
-----
::

    python examples/checkpoint_replay_eval.py \\
        --log-dir examples/logs/ppolag_forelise_wristlimit_seed0_1 \\
        --env OmniPiano-ForElise-WristLimit-v0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict, List, Tuple

# Eval RNG decoupled from training by a large offset:
# replay seed = train_seed + EVAL_SEED_OFFSET + ep_i. NOTE: this is NOT
# the SB3 templates' scheme (run_sb3_baseline.py:_final_eval uses
# eval_seed = seed + n_envs + 1, then eval_seed + ep*10_000) — eval
# episodes are not draw-matched across the two CSV families.
EVAL_SEED_OFFSET = 10_000

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import omnisafe

from omnipiano.utils.info_keys import EpisodeInfoKeys

# Import the OmniPianoCMDP adapter via the template — it has the
# ``@env_register`` side-effect that exposes OmniPiano envs to OmniSafe.
# Without this import OmniSafe will fail to construct the env at
# evaluator.load_saved() time.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))
import run_omnisafe_template  # noqa: F401 — triggers @env_register


# ---------------------------------------------------------------------------
# Path discovery: OmniSafe writes a 2-level subdir tree under our log_dir
# (see run_omnisafe_template.py:373+). We need the inner ``seed-NNN-<ts>/``
# dir to feed Evaluator.load_saved.
# ---------------------------------------------------------------------------

def _latest_subdir(d: str) -> str:
    kids = [os.path.join(d, x) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    if not kids:
        raise RuntimeError(f"No subdirectory under {d} — did OmniSafe training run?")
    return max(kids, key=os.path.getmtime)


def _find_omnisafe_save_dir(log_dir: str) -> str:
    """Return ``<log_dir>/<Algo>-{<env>}/seed-NNN-<timestamp>/``."""
    algo_dir = _latest_subdir(log_dir)
    return _latest_subdir(algo_dir)


def _read_train_seed(save_dir: str) -> int:
    """OmniSafe writes the training seed to ``<save_dir>/config.json``.

    Reading it here lets us derive a deterministic eval seed
    (``train_seed + EVAL_SEED_OFFSET + ep_idx``) so the per-checkpoint
    learning curve is reproducible without coupling eval RNG to whatever
    state the env happens to be in after ``Evaluator.load_saved``.
    """
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return int(cfg["seed"])


def _assert_no_active_noise(save_dir: str) -> str:
    """Refuse to replay a robust (noise-injecting) env. Returns the env id.

    WHY THIS GUARD EXISTS
    ---------------------
    This script does NOT build its env through ``omnipiano.make()``. It takes
    whatever ``omnisafe.Evaluator.load_saved()`` reconstructs from the training
    config (see ``evaluator._env`` below). That path therefore **never applies
    eval-time semantics** — in particular it never multiplies the RobustConfig
    magnitudes by ``eval_noise_scale``.

    Consequence for a robust env: every checkpoint would be evaluated under the
    **training-level** perturbation, no matter what scale the caller intended.
    A clean eval (scale 0) and a stress eval (scale 2) would both silently
    produce the matched (scale 1) number, and nothing in the output would
    reveal it — the CSV's ``eval_noise_scale`` column is written as a constant
    1.0 further down. That is a silent-wrong-number failure, the worst kind for
    a benchmark, so we fail fast instead.

    Safety-only envs (the entire current OmniSafe roster) are unaffected: with
    no active noise channel there is nothing for ``eval_noise_scale`` to scale,
    so the replay path and ``make(mode="eval")`` agree exactly.

    If you need a robustness curve, use ``examples/robust_eval_sweep.py``,
    which builds each eval env via ``make(env_id, mode="eval",
    eval_noise_scale=...)`` and thus does inherit the scaling.
    """
    with open(os.path.join(save_dir, "config.json")) as f:
        env_id = json.load(f)["env_id"]

    from omnipiano.envs.registration import _registry

    spec = _registry.get(env_id)
    robust = getattr(spec, "robust_config", None) if spec else None
    if robust is None:
        return env_id
    active = [c for c in ("action", "obs", "reward") if robust.is_channel_active(c)]
    if active:
        raise ValueError(
            f"checkpoint_replay_eval refuses to replay {env_id!r}: it has "
            f"active robust channel(s) {active}. This script's env comes from "
            f"omnisafe.Evaluator.load_saved(), which bypasses omnipiano.make() "
            f"and therefore never applies eval_noise_scale — every checkpoint "
            f"would be scored under TRAINING-level noise while the output "
            f"claims eval_noise_scale=1.0, whatever scale you meant. Use "
            f"examples/robust_eval_sweep.py instead (it builds the eval env "
            f"via make(mode='eval', eval_noise_scale=...))."
        )
    return env_id


def _epoch_num(fname: str) -> int:
    """Numeric extraction so ``epoch-5000.pt`` sorts after ``epoch-900.pt``."""
    m = re.match(r"epoch-(\d+)\.pt$", fname)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Per-episode CSV writer (mirrors SafeRecordEpisodeStatistics schema so
# tools/plot_eval_metrics.py reads it unchanged).
# ---------------------------------------------------------------------------

_CSV_HEADER = [
    "env_step_count", "episode", "time_elapsed",
    "ep_return", "ep_length",
    "ep_cost", "ep_violations",
    "ep_f1", "ep_precision", "ep_recall",
    "ep_sustain_f1", "ep_sustain_precision", "ep_sustain_recall",
    "energy_reward", "fingering_reward", "ot_fingering_reward",
    "forearm_reward", "key_press_reward", "sustain_reward",
]
# Robust-eval columns (§0.6, decision 11) — schema-locked with
# SafeRecordEpisodeStatistics. checkpoint_replay is OmniSafe/safety-only (no
# robust noise), so these are nominal: eval_noise_scale=1.0 (matched training
# config), ep_noise_*=0.0, and ep_return_true == the received ep_return.
_ROBUST_EVAL_COLS = [
    "eval_noise_scale", "ep_return_true",
    "ep_noise_action_l2", "ep_noise_obs_l2", "ep_noise_reward",
]
_CSV_HEADER = _CSV_HEADER + _ROBUST_EVAL_COLS

_INFO_KEY_BY_CSV_COL = {
    "ep_cost":                EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL,
    "ep_violations":          EpisodeInfoKeys.EPISODE_SAFETY_VIOLATIONS,
    "ep_f1":                  EpisodeInfoKeys.EPISODE_TASK_F1,
    "ep_precision":           EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION,
    "ep_recall":              EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL,
    "ep_sustain_f1":          EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_F1,
    "ep_sustain_precision":   EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_PRECISION,
    "ep_sustain_recall":      EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_RECALL,
    "energy_reward":          EpisodeInfoKeys.EPISODE_TASK_ENERGY_REWARD,
    "fingering_reward":       EpisodeInfoKeys.EPISODE_TASK_FINGERING_REWARD,
    "ot_fingering_reward":    EpisodeInfoKeys.EPISODE_TASK_OT_FINGERING_REWARD,
    "forearm_reward":         EpisodeInfoKeys.EPISODE_TASK_FOREARM_REWARD,
    "key_press_reward":       EpisodeInfoKeys.EPISODE_TASK_KEY_PRESS_REWARD,
    "sustain_reward":         EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_REWARD,
}


# ---------------------------------------------------------------------------
# Per-checkpoint deterministic rollout
# ---------------------------------------------------------------------------

def _replay_one_checkpoint(
    evaluator: "omnisafe.Evaluator",
    save_dir: str,
    ckpt_name: str,
    num_eval_eps: int,
    train_seed: int,
    eval_seed_offset: int = EVAL_SEED_OFFSET,
) -> Tuple[List[float], List[float], List[int], List[Dict[str, Any]]]:
    """Load ckpt and run num_eval_eps deterministic episodes.

    Each episode resets with ``seed = train_seed + eval_seed_offset + ep_i``.
    Using the same anchor seed across all checkpoints in a run means the
    learning curve is not noised by env init drift — only policy evolution
    can move the eval metric. (See the ``EVAL_SEED_OFFSET`` note at module
    top: the SB3 templates use a different per-episode scheme.)

    Returns (returns, costs, lengths, terminal_infos).
    """
    evaluator.load_saved(save_dir=save_dir, model_name=ckpt_name)
    env = evaluator._env
    actor = evaluator._actor
    if env is None or actor is None:
        raise RuntimeError(f"Evaluator.load_saved did not initialize env+actor for {ckpt_name}")

    returns: List[float] = []
    costs:   List[float] = []
    lengths: List[int]   = []
    terminal_infos: List[Dict[str, Any]] = []

    for ep_i in range(num_eval_eps):
        obs, _ = env.reset(seed=train_seed + eval_seed_offset + ep_i)
        ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        terminal_info: Dict[str, Any] = {}
        done = False
        while not done:
            with torch.no_grad():
                act = actor.predict(obs, deterministic=True)
            obs, rew, cost, term, trunc, info = env.step(act)
            ep_ret += float(rew.item() if hasattr(rew, "item") else rew)
            ep_cost += float(cost.item() if hasattr(cost, "item") else cost)
            ep_len += 1
            done = bool(
                (term.item() if hasattr(term, "item") else term)
                or (trunc.item() if hasattr(trunc, "item") else trunc)
            )
            if done:
                # info dict propagates through OmniPianoCMDP unchanged —
                # contains all EpisodeInfoKeys.* that MetricsWrapper + SafetyWrapper put there.
                terminal_info = info if isinstance(info, dict) else {}

        returns.append(ep_ret)
        costs.append(ep_cost)
        lengths.append(ep_len)
        terminal_infos.append(terminal_info)
    return returns, costs, lengths, terminal_infos


# ---------------------------------------------------------------------------
# Main: iterate ckpts, write npz + csv
# ---------------------------------------------------------------------------

def replay_all_checkpoints(
    log_dir: str,
    env_id: str,
    num_eval_eps: int = 10,
    steps_per_epoch: int = 20_000,
    eval_seed_offset: int = EVAL_SEED_OFFSET,
) -> None:
    save_dir = _find_omnisafe_save_dir(log_dir)
    # Fail before any checkpoint is loaded: a robust env cannot be scored
    # correctly on this path (see _assert_no_active_noise).
    trained_env_id = _assert_no_active_noise(save_dir)
    train_seed = _read_train_seed(save_dir)
    torch_save_dir = os.path.join(save_dir, "torch_save")
    ckpts = sorted(
        (f for f in os.listdir(torch_save_dir) if f.endswith(".pt")),
        key=_epoch_num,
    )
    if not ckpts:
        raise RuntimeError(f"No checkpoints under {torch_save_dir}")

    print(f"[checkpoint_replay_eval] log_dir = {log_dir}")
    print(f"[checkpoint_replay_eval] save_dir = {save_dir}")
    # The trained env id read from the saved config is authoritative; --env is
    # only informational and is not used to construct anything.
    print(f"[checkpoint_replay_eval] env (from saved config) = {trained_env_id}")
    if env_id and env_id != trained_env_id:
        print(f"[checkpoint_replay_eval] NOTE: --env {env_id!r} differs from the "
              f"trained env id above; the saved config wins.")
    print(f"[checkpoint_replay_eval] train_seed = {train_seed}  "
          f"eval_seed_base = {train_seed + eval_seed_offset}")
    print(f"[checkpoint_replay_eval] found {len(ckpts)} checkpoints, "
          f"num_eval_eps={num_eval_eps}, steps_per_epoch={steps_per_epoch}")
    eta_s = len(ckpts) * num_eval_eps * 720 / 400.0
    print(f"[checkpoint_replay_eval] ETA: ~{eta_s/60:.1f} min "
          f"(assumes 400 FPS CPU; scales with episode_length & num_eval_eps)")

    # One Evaluator instance, reused via repeated load_saved() — saves env reconstruction overhead.
    evaluator = omnisafe.Evaluator()

    csv_uuid = uuid.uuid4().hex[:8]
    csv_path = os.path.join(log_dir, f"eval_episode_metrics_{csv_uuid}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_HEADER)

    timesteps: List[int] = []
    results_2d: List[List[float]] = []
    ep_lengths_2d: List[List[int]] = []
    ep_costs_2d: List[List[float]] = []
    cumulative_episode = 0
    t0 = time.perf_counter()

    for i, ckpt in enumerate(ckpts):
        epoch = _epoch_num(ckpt)
        env_step = epoch * steps_per_epoch
        t_ckpt_start = time.perf_counter()
        returns, costs, lengths, terminal_infos = _replay_one_checkpoint(
            evaluator, save_dir, ckpt, num_eval_eps,
            train_seed=train_seed, eval_seed_offset=eval_seed_offset,
        )
        t_ckpt = time.perf_counter() - t_ckpt_start

        timesteps.append(env_step)
        results_2d.append(returns)
        ep_lengths_2d.append(lengths)
        ep_costs_2d.append(costs)

        # Append per-episode rows to csv
        time_elapsed_s = time.perf_counter() - t0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for ep_i in range(num_eval_eps):
                cumulative_episode += 1
                info = terminal_infos[ep_i]
                row = [
                    env_step,
                    cumulative_episode,
                    round(time_elapsed_s, 2),
                    returns[ep_i],
                    lengths[ep_i],
                ]
                # info-key columns (ep_cost ... sustain_reward); the last 5 of
                # _CSV_HEADER are the appended robust-eval cols, handled below.
                for col_after_length in _CSV_HEADER[5:-len(_ROBUST_EVAL_COLS)]:
                    info_key = _INFO_KEY_BY_CSV_COL[col_after_length]
                    row.append(info.get(info_key, ""))
                # clean/denoised return = sum of the reward-decomposition terms
                # just appended (the last 6 entries of row); == received return
                # here since safety tasks carry no reward noise.
                ep_return_true = sum(float(x) for x in row[-6:] if x != "")
                # robust-eval cols: safety-only → scale 1.0 (matched), no noise
                row.extend([1.0, ep_return_true, 0.0, 0.0, 0.0])
                writer.writerow(row)

        print(f"  [{i+1:3d}/{len(ckpts)}] ckpt {ckpt:>20s} "
              f"env_step={env_step:>8d}  ret={np.mean(returns):8.2f}  "
              f"cost={np.mean(costs):7.2f}  len={np.mean(lengths):6.1f}  "
              f"({t_ckpt:.1f}s)")

    # Write evaluations.npz (SB3 schema + ep_costs extension)
    npz_path = os.path.join(log_dir, "evaluations.npz")
    np.savez(
        npz_path,
        timesteps=np.array(timesteps, dtype=np.int64),
        results=np.array(results_2d, dtype=np.float64),
        ep_lengths=np.array(ep_lengths_2d, dtype=np.int64),
        ep_costs=np.array(ep_costs_2d, dtype=np.float64),
    )
    total_s = time.perf_counter() - t0
    print(f"\n[checkpoint_replay_eval] wrote {npz_path}")
    print(f"[checkpoint_replay_eval] wrote {csv_path}")
    print(f"[checkpoint_replay_eval] total wallclock = {total_s/60:.1f} min")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--log-dir", required=True,
                   help="Run directory under examples/logs/ that contains "
                        "<Algo>-{<env>}/seed-.../torch_save/.")
    p.add_argument("--env", required=True,
                   help="Env id used during training (informational; "
                        "Evaluator infers env from saved config).")
    p.add_argument("--num-eval-eps", type=int, default=1,
                   help="Episodes per checkpoint. Default 1 matches "
                        "BenchmarkProtocolConfig.num_eval_eps + RoboPianist "
                        "convention; cross-train-seed runs (3) provide "
                        "the paper-level variance, not within-run eps.")
    p.add_argument("--steps-per-epoch", type=int, default=20_000,
                   help="Must match the trained algorithm's "
                        "algo_cfgs.steps_per_epoch (PPOLag default 20000).")
    p.add_argument("--eval-seed-offset", type=int, default=EVAL_SEED_OFFSET,
                   help=f"Per-episode seed = train_seed + this + ep_idx. "
                        f"Default {EVAL_SEED_OFFSET} matches SB3 convention.")
    args = p.parse_args()
    replay_all_checkpoints(
        log_dir=args.log_dir,
        env_id=args.env,
        num_eval_eps=args.num_eval_eps,
        steps_per_epoch=args.steps_per_epoch,
        eval_seed_offset=args.eval_seed_offset,
    )


if __name__ == "__main__":
    main()
