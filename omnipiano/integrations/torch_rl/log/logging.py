"""SB3-schema-aligned CSV logging for TorchRL training."""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

from omnipiano.utils.info_keys import InfoKeys


class TrainingEpisodeTracker:
    """Track SB3-style rolling episode return/length from collector batches."""

    def __init__(self, window_size: int = 100):
        self._partial_returns = {}
        self._partial_lengths = {}
        self._recent_returns = deque(maxlen=window_size)
        self._recent_lengths = deque(maxlen=window_size)

    def update(self, batch) -> dict:
        traj_ids = batch["collector", "traj_ids"].detach().reshape(-1).cpu().tolist()
        rewards = (batch["next", InfoKeys.TASK_TRUE_REWARD].detach().reshape(-1).cpu().tolist())
        dones = batch["next", "done"].detach().reshape(-1).cpu().tolist()
        if not (len(traj_ids) == len(rewards) == len(dones)):
            raise RuntimeError(
                "Collector traj_ids/reward/done lengths do not match: "
                f"{len(traj_ids)}, {len(rewards)}, {len(dones)}"
            )

        for traj_id, reward, done in zip(traj_ids, rewards, dones):
            key = int(traj_id)
            self._partial_returns[key] = (
                self._partial_returns.get(key, 0.0) + float(reward)
            )
            self._partial_lengths[key] = self._partial_lengths.get(key, 0) + 1
            if bool(done):
                self._recent_returns.append(self._partial_returns.pop(key))
                self._recent_lengths.append(self._partial_lengths.pop(key))

        return {
            "rollout/ep_rew_mean": self._mean_or_nan(self._recent_returns),
            "rollout/ep_len_mean": self._mean_or_nan(self._recent_lengths),
        }

    @staticmethod
    def _mean_or_nan(values):
        return sum(values) / len(values) if values else float("nan")


class ProgressLogger:
    _FIELD_MAPS = {
        "ppo": {
            "loss_objective": "policy_gradient_loss",
            "loss_actor": "policy_gradient_loss",
            "loss_critic": "value_loss",
            "loss_entropy": "entropy_loss",
            "kl_approx": "approx_kl",
        },
        "sac": {
            "loss_actor": "actor_loss",
            "loss_qvalue": "critic_loss",
            "loss_critic": "critic_loss",
            "loss_alpha": "ent_coef_loss",
            "alpha": "ent_coef",
        },
    }

    def __init__(
        self,
        path: str | Path,
        min_interval_steps: int = 5_000,
        sb3_family: str | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields = None
        self._min_interval_steps = int(min_interval_steps)
        if sb3_family not in (None, *self._FIELD_MAPS):
            raise ValueError(f"Unknown SB3 logging family: {sb3_family!r}")
        self._field_map = self._FIELD_MAPS.get(sb3_family, {})
        self._last_written_step = None
        self._pending_row = None
        self._stream = None
        self._writer = None
        self._start_time = time.perf_counter()
        self._iterations = 0

    def write(self, row: dict, force: bool = False) -> None:
        """Record the latest row, writing at most once per step interval."""
        self._iterations += 1
        self._pending_row = self._to_sb3_schema(row)
        env_steps = int(self._pending_row["time/total_timesteps"])
        if (
            not force
            and self._last_written_step is not None
            and env_steps - self._last_written_step < self._min_interval_steps
        ):
            return

        if force:
            self._print_progress(self._pending_row)
        self._write_pending_row()

    @staticmethod
    def _print_progress(row: dict) -> None:
        values = [
            ("timesteps", f'{row["time/total_timesteps"]:,}'),
            ("fps", str(row["time/fps"])),
            *((key.removeprefix("train/"), f"{value:.4g}")
              for key, value in row.items()
              if key.startswith("train/") and "loss" in key),
            ("eval reward", f'{row["eval/mean_reward"]:.4g}'),
        ]
        width = max(len(name) for name, _ in values)
        border = "-" * (width + 19)
        print(border)
        for name, value in values:
            print(f"| {name:<{width}} | {value:>14} |")
        print(border, flush=True)

    def _to_sb3_schema(self, row: dict) -> dict:
        env_steps = int(row["env_steps"])
        elapsed = max(time.perf_counter() - self._start_time, 1e-9)
        aligned = {
            "time/fps": int(env_steps / elapsed),
            "time/iterations": self._iterations,
            "time/time_elapsed": int(elapsed),
        }
        for key in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
            aligned[key] = row.get(key, float("nan"))
        aligned["time/total_timesteps"] = env_steps
        for key, value in row.items():
            if key == "env_steps" or key.startswith("rollout/"):
                continue
            mapped_key = self._field_map.get(key, key)
            aligned[
                mapped_key if "/" in mapped_key else f"train/{mapped_key}"
            ] = value
        return aligned

    def _write_pending_row(self) -> None:
        row = self._pending_row
        if row is None:
            return

        if self._fields is None:
            self._fields = list(row)
        if self._stream is None:
            self._stream = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._stream, fieldnames=self._fields
            )
            if self._stream.tell() == 0:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._stream.flush()
        self._last_written_step = int(row["time/total_timesteps"])
        self._pending_row = None

    def close(self) -> None:
        """Write the final pending row and release the persistent handle."""
        if self._pending_row is not None:
            self._write_pending_row()
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._writer = None
