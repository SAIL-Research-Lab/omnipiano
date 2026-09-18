"""Framework-neutral benchmark evaluation, scheduling, and artifacts."""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from omnipiano.utils.info_keys import InfoKeys


def evaluate_policy(
    policy, env_id: str, env, eval_seed: int, num_episodes: int = 1,
    config=None,
    *, true_reward: bool = True,
):
    """Evaluate a policy, instantiating it first when config is provided."""
    if config is not None:
        policy = policy(config)
    episodes = []
    for index in range(num_episodes):
        obs, _ = env.reset(seed=eval_seed + index * 10_000)
        ep_return, ep_length, terminal_info = 0.0, 0, {}
        done = False
        while not done:
            prediction = policy.predict(obs, deterministic=True)
            action = prediction[0] if isinstance(prediction, tuple) else prediction
            obs, reward, terminated, truncated, terminal_info = env.step(action)
            ep_return += float(
                terminal_info[InfoKeys.TASK_TRUE_REWARD] if true_reward else reward
            )
            ep_length += 1
            done = bool(terminated or truncated)
        metrics = {
            key: float(value)
            for key, value in terminal_info.items()
            if isinstance(key, str)
            and key.startswith("episode_")
            and isinstance(value, (bool, int, float, np.integer, np.floating))
        }
        episodes.append({
            "episode_index": int(index),
            "episode_return": ep_return,
            "episode_length": int(ep_length),
            "metrics": metrics,
        })

    summary = {
        "return_mean": float(np.mean([episode["episode_return"] for episode in episodes])),
        "return_std": float(np.std([episode["episode_return"] for episode in episodes])),
        "length_mean": float(np.mean([episode["episode_length"] for episode in episodes])),
    }
    for key in sorted({key for episode in episodes for key in episode["metrics"]}):
        values = [episode["metrics"][key] for episode in episodes if key in episode["metrics"]]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return {"env_id": env_id, "eval_seed": int(eval_seed),
            "episodes": episodes, "summary": summary}


def write_eval_summary(result: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def postprocess_eval_csv(log_dir: str | Path) -> None:
    """Add the authoritative training-step axis to the periodic eval CSV."""
    log_dir = os.fspath(log_dir)
    npz_path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        print(f"[postprocess] no evaluations.npz in {log_dir}, skipping")
        return
    archive = np.load(npz_path)
    timesteps = archive["timesteps"]
    row_timesteps = np.repeat(timesteps, archive["results"].shape[1])
    for path in sorted(glob.glob(os.path.join(log_dir, "eval_episode_metrics_*.csv"))):
        with open(path, newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            continue
        header, data = [value.strip() for value in rows[0]], rows[1:]
        if "training_step" in header or len(data) != len(row_timesteps):
            continue
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["training_step", *header])
            writer.writerows([[str(int(row_timesteps[index])), *row]
                              for index, row in enumerate(data)])
        print(f"[postprocess] added training_step to {os.path.basename(path)} ({len(data)} rows)")


class EvalCallback:
    """Schedule evaluation without importing a training framework."""

    def __init__(
        self, env_id, env, run_dir, eval_seed, protocol,
        policy, save_best_fn,
    ):
        self.env_id = env_id
        self.env = env
        self.run_dir = Path(run_dir)
        self.eval_seed = int(eval_seed)
        self.protocol = protocol
        self.policy = policy
        self.save_best_fn = save_best_fn
        self.next_eval = protocol.eval_freq_env_steps
        self.best_mean_reward = -np.inf
        self.timesteps = []
        self.results = []
        self.ep_lengths = []

    def on_step(self, env_steps: int):
        stats = {
            "eval/mean_reward": float("nan"),
            "eval/mean_f1": float("nan"),
            "eval/mean_ep_length": float("nan"),
        }
        if env_steps < self.next_eval:
            return stats, False

        result = evaluate_policy(
            self.policy(), self.env_id, self.env, self.eval_seed,
            self.protocol.num_eval_eps, true_reward=False,
        )
        returns = [episode["episode_return"] for episode in result["episodes"]]
        lengths = [episode["episode_length"] for episode in result["episodes"]]
        self.timesteps.append(env_steps)
        self.results.append(returns)
        self.ep_lengths.append(lengths)
        np.savez(
            self.run_dir / "evaluations.npz",
            timesteps=np.asarray(self.timesteps, dtype=np.int64),
            results=np.asarray(self.results, dtype=float),
            ep_lengths=np.asarray(self.ep_lengths, dtype=np.int64),
        )
        mean_reward = result["summary"]["return_mean"]
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.save_best_fn(self.run_dir / "best_model")
        stats.update({
            "eval/mean_reward": mean_reward,
            "eval/mean_f1": result["summary"].get("episode_task/f1_mean", float("nan")),
            "eval/mean_ep_length": result["summary"]["length_mean"],
        })
        while self.next_eval <= env_steps:
            self.next_eval += self.protocol.eval_freq_env_steps
        return stats, True
