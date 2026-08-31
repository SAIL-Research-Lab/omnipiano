"""Evaluation through the native Gymnasium environment and CSV logger."""

from __future__ import annotations

import numpy as np

from omnipiano.utils.info_keys import InfoKeys


def evaluate_policy(
    policy,
    env_id: str,
    env,
    seed: int,
    num_episodes: int = 1,
):
    episodes = []
    for index in range(num_episodes):
        obs, _ = env.reset(seed=seed + index)
        total, length, done = 0.0, 0, False
        terminal_info = {}
        while not done:
            action = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, terminal_info = env.step(action)
            total += float(terminal_info[InfoKeys.TASK_TRUE_REWARD])
            length += 1
            done = bool(terminated or truncated)
        metrics = {
            key: float(value)
            for key, value in terminal_info.items()
            if key.startswith("episode_") and np.isscalar(value)
        }
        episodes.append(
            {"episode_index": index, "episode_return": total,
             "episode_length": length, "metrics": metrics}
        )
    returns = [item["episode_return"] for item in episodes]
    result = {
        "env_id": env_id,
        "eval_seed": seed,
        "episodes": episodes,
        "summary": {
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "length_mean": float(np.mean([x["episode_length"] for x in episodes])),
        },
    }
    return result


def write_eval_summary(result: dict, path) -> None:
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
