"""Iteration-level benchmark logging callback for SB3 PPO.

This callback directly consumes SB3 step events (`infos` + done flags),
collects episodes that finished during the current PPO iteration
(one rollout collection), and appends one summary row per iteration to
`train_iteration_summary.csv`.
"""


import csv
import math
import os
import statistics
from typing import Dict, List, Optional
import warnings

from stable_baselines3.common.callbacks import BaseCallback
from OmniPiano.utils.info_keys import EpisodeInfoKeys

def _parse_finite_float(value: str) -> Optional[float]:
    """Parse a numeric metric field and drop invalid/non-finite values.

    Returning None (instead of 0.0) avoids biasing benchmark summaries when fields are
    missing or malformed.
    """
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _mean(values: List[Optional[float]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return statistics.fmean(valid)


def _std(values: List[Optional[float]]) -> float:
    """Sample std (ddof=1) over valid values; 0.0 when undefined."""
    valid = [v for v in values if v is not None]
    if len(valid) <= 1:
        return 0.0
    return statistics.stdev(valid)


def _mean_or_empty(values: List[Optional[float]], has_episode: bool):
    if not has_episode:
        return ""
    return _mean(values)


def _std_or_empty(values: List[Optional[float]], has_episode: bool):
    if not has_episode:
        return ""
    return _std(values)


class TrainIterationSummaryCallback(BaseCallback):
    """Append one summary row from episodes completed in this PPO iteration."""

    def __init__(
        self,
        log_dir: str,
        summary_filename: str = "train_iteration_summary.csv",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_dir = log_dir
        self.summary_path = os.path.join(log_dir, summary_filename)
        self.iteration_idx = 0
        self.completed_episodes_in_iteration: List[Dict[str, float]] = []

        self.header = [
            "iteration",
            "sb3_num_timesteps",
            "expected_env_steps_per_iteration",
            "num_new_episodes",
            "num_envs",
            "ep_return_mean",
            "ep_return_std",
            "ep_length_mean",
            "ep_cost_mean",
            "ep_violations_mean",
            "ep_f1_mean",
            "ep_precision_mean",
            "ep_recall_mean",
            "ep_sustain_f1_mean",
            "episode_energy_reward_mean",
            "episode_fingering_reward_mean",
            "episode_ot_fingering_reward_mean",
            "episode_forearm_reward_mean",
            "episode_key_press_reward_mean",
            "episode_sustain_reward_mean",
        ]

    def _on_training_start(self) -> None:
        # Always start a fresh iteration summary for this run.
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.summary_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def _on_rollout_end(self) -> None:
        # SB3 PPO calls this once per iteration (i.e., one rollout collection).
        self.iteration_idx += 1
        rows = self.completed_episodes_in_iteration

        # Iteration definition: n_steps * num_envs environment steps.
        num_envs = int(getattr(self.training_env, "num_envs", 1))
        expected_env_steps = int(self.model.n_steps) * num_envs

        returns = [_parse_finite_float(r.get("ep_return", "")) for r in rows]
        lengths = [_parse_finite_float(r.get("ep_length", "")) for r in rows]
        costs = [_parse_finite_float(r.get("ep_cost", "")) for r in rows]
        violations = [_parse_finite_float(r.get("ep_violations", "")) for r in rows]
        f1s = [_parse_finite_float(r.get("ep_f1", "")) for r in rows]
        precisions = [_parse_finite_float(r.get("ep_precision", "")) for r in rows]
        recalls = [_parse_finite_float(r.get("ep_recall", "")) for r in rows]
        sustain_f1s = [_parse_finite_float(r.get("ep_sustain_f1", "")) for r in rows]
        energy_rewards = [
            _parse_finite_float(r.get("episode_energy_reward", ""))
            for r in rows
        ]
        fingering_rewards = [
            _parse_finite_float(r.get("episode_fingering_reward", ""))
            for r in rows
        ]
        ot_fingering_rewards = [
            _parse_finite_float(r.get("episode_ot_fingering_reward", ""))
            for r in rows
        ]
        forearm_rewards = [
            _parse_finite_float(r.get("episode_forearm_reward", ""))
            for r in rows
        ]
        key_press_rewards = [
            _parse_finite_float(r.get("episode_key_press_reward", ""))
            for r in rows
        ]
        sustain_rewards = [
            _parse_finite_float(r.get("episode_sustain_reward", ""))
            for r in rows
        ]
        has_episode = len(rows) > 0

        row = [
            self.iteration_idx,
            int(self.num_timesteps),
            expected_env_steps,
            len(rows),
            num_envs,
            _mean_or_empty(returns, has_episode),
            _std_or_empty(returns, has_episode),
            _mean_or_empty(lengths, has_episode),
            _mean_or_empty(costs, has_episode),
            _mean_or_empty(violations, has_episode),
            _mean_or_empty(f1s, has_episode),
            _mean_or_empty(precisions, has_episode),
            _mean_or_empty(recalls, has_episode),
            _mean_or_empty(sustain_f1s, has_episode),
            _mean_or_empty(energy_rewards, has_episode),
            _mean_or_empty(fingering_rewards, has_episode),
            _mean_or_empty(ot_fingering_rewards, has_episode),
            _mean_or_empty(forearm_rewards, has_episode),
            _mean_or_empty(key_press_rewards, has_episode),
            _mean_or_empty(sustain_rewards, has_episode),
        ]

        # Append one line per PPO iteration.
        with open(self.summary_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.completed_episodes_in_iteration = []

    def _on_step(self) -> bool:
        # Consume per-step events emitted by SB3/VecEnv and keep only episodes that
        # finished during the current PPO iteration.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not infos:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            episode_dict = info.get("episode", {})
            if not episode_dict:
                raise ValueError(
                    "Done episode found without info['episode']. "
                    "The TrainIterationSummaryCallback strictly relies on Stable Baselines 3's "
                    "Monitor or VecMonitor wrapper to provide 'r' (return) and 'l' (length) "
                    "in info['episode']. Please ensure your environment is wrapped with Monitor."
                )
            completed_episode = {
                # Monitor/VecMonitor keys.
                "ep_return": episode_dict.get("r", 0.0),
                "ep_length": episode_dict.get("l", 0.0),
                # Safety/musical metrics injected by wrappers.
                "ep_cost": info.get(EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL, 0.0),
                "ep_violations": info.get(EpisodeInfoKeys.EPISODE_SAFETY_VIOLATIONS, 0.0),
                "ep_f1": info.get(EpisodeInfoKeys.EPISODE_TASK_F1, ""),
                "ep_precision": info.get(EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION, ""),
                "ep_recall": info.get(EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL, ""),
                "ep_sustain_f1": info.get(EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_F1, ""),
                "episode_energy_reward": info.get(
                    EpisodeInfoKeys.EPISODE_TASK_ENERGY_REWARD, ""
                ),
                "episode_fingering_reward": info.get(
                    EpisodeInfoKeys.EPISODE_TASK_FINGERING_REWARD, ""
                ),
                "episode_ot_fingering_reward": info.get(
                    EpisodeInfoKeys.EPISODE_TASK_OT_FINGERING_REWARD, ""
                ),
                "episode_forearm_reward": info.get(
                    EpisodeInfoKeys.EPISODE_TASK_FOREARM_REWARD, ""
                ),
                "episode_key_press_reward": info.get(
                    EpisodeInfoKeys.EPISODE_TASK_KEY_PRESS_REWARD, ""
                ),
                "episode_sustain_reward": info.get(
                    EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_REWARD, ""
                ),
            }
            self.completed_episodes_in_iteration.append(completed_episode)
        return True