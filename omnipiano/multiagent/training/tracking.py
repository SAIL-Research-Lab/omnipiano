"""Weights & Biases integration for the OmniPiano multi-agent baselines.

Three constraints shape this module:

  1. A full benchmark run must never die because of a W&B or network problem. Any
     failure *after* a successful init degrades to a printed warning.
  2. If W&B was explicitly requested, a credential problem must fail loudly at
     init, *before* GPU hours are spent. Discovering after ten hours that
     nothing was uploaded is worse than crashing in the first second.
  3. The x-axis is always lifetime ENVIRONMENT steps -- never RLlib iterations
     and never agent steps -- so IPPO / MAPPO / single-agent curves overlay
     directly. See ``training.runtime.extract_env_steps`` for why that distinction
     decides whether a nominal budget counts physical or per-agent interactions.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

# ``vf_explained_var`` is the single most diagnostic number in this benchmark:
# it answers "is the critic learning, or just predicting a constant?"
_LEARNER_KEYS: Sequence[str] = (
    "policy_loss", "vf_loss", "vf_loss_unclipped", "vf_explained_var",
    "entropy", "mean_kl_loss", "curr_kl_coeff", "curr_entropy_coeff",
    "total_loss", "gradients_default_optimizer_global_norm",
    "gradients_actor_global_norm", "gradients_critic_global_norm",
    "value_norm_mean", "value_norm_std", "value_clip_fraction",
)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def learner_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten RLlib per-module Learner stats into ``learner/<module>/<key>``."""
    out: Dict[str, float] = {}
    learners = result.get("learners")
    if not isinstance(learners, Mapping):
        return out
    for module_id, stats in learners.items():
        if not isinstance(stats, Mapping):
            continue
        label = "__all__" if module_id == "__all_modules__" else str(module_id)
        for key in _LEARNER_KEYS:
            number = _finite(stats.get(key))
            if number is not None:
                out[f"learner/{label}/{key}"] = number
    return out


def _has_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    for candidate in (Path.home() / ".netrc",
                      Path.home() / "_netrc",
                      Path.home() / ".config" / "wandb" / "settings"):
        if candidate.exists():
            return True
    return False


class WandbRun:
    """Failure-tolerant wrapper around one ``wandb.Run``."""

    def __init__(
        self, *,
        mode: str = "online",
        entity: Optional[str] = "omnipiano",
        project: str = "multiagent",
        name: Optional[str] = None,
        group: Optional[str] = None,
        job_type: str = "train",
        tags: Sequence[str] = (),
        notes: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        run_dir: Optional[Path] = None,
        resume_id: Optional[str] = None,
    ) -> None:
        self._run = None
        self._wandb = None
        self._degraded = False
        self._last_step = -1
        self.mode = str(mode)
        configured_target = (config or {}).get(
            "requested_total_env_steps", (config or {}).get("total_env_steps")
        )
        try:
            self._target_env_steps = int(configured_target)
        except (TypeError, ValueError):
            self._target_env_steps = None

        if self.mode == "disabled":
            print("[wandb] disabled (--wandb-mode disabled)")
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "--wandb-mode is not 'disabled' but wandb is not installed.\n"
                "  pip install wandb    (or pass --wandb-mode disabled)"
            ) from exc
        self._wandb = wandb

        if self.mode == "online" and not _has_credentials():
            raise RuntimeError(
                "--wandb-mode online but no W&B credentials were found.\n"
                "  wandb login                 # interactive, once per machine\n"
                "  export WANDB_API_KEY=<key>  # headless / cluster\n"
                "  --wandb-mode offline        # log locally, `wandb sync` later\n"
                "  --wandb-mode disabled       # no W&B at all\n"
                "  Key: https://wandb.ai/authorize"
            )

        # NOTE: do NOT pass wandb.Settings(start_method=...). That field was
        # removed when W&B moved Settings to pydantic (>= 0.18), and passing it
        # raises `extra_forbidden`. Users who need it can set WANDB_START_METHOD.
        init_kwargs: Dict[str, Any] = dict(
            mode=self.mode, entity=entity, project=project, name=name,
            group=group, job_type=job_type, tags=list(tags), notes=notes,
            config=dict(config or {}),
            # Keep W&B's staging dir inside the run dir so one directory is the
            # complete, self-contained record of the run.
            dir=str(run_dir) if run_dir is not None else None,
        )
        if resume_id:
            init_kwargs.update(id=resume_id, resume="allow")
        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"wandb.init failed ({type(exc).__name__}: {exc}).\n"
                "  Use --wandb-mode offline or --wandb-mode disabled to proceed."
            ) from exc

        try:
            self._run.define_metric("env_steps")
            for family in ("train/*", "eval/*", "learner/*", "time/*", "final/*"):
                self._run.define_metric(family, step_metric="env_steps")
        except Exception as exc:  # non-fatal cosmetics
            print(f"[wandb warning] define_metric failed: {exc}")

        print(f"[wandb] mode={self.mode} url={self.url}")

    # ---------------- properties ----------------

    @property
    def active(self) -> bool:
        return self._run is not None and not self._degraded

    @property
    def url(self) -> Optional[str]:
        return getattr(self._run, "url", None) if self._run is not None else None

    @property
    def run_id(self) -> Optional[str]:
        return getattr(self._run, "id", None) if self._run is not None else None

    # ---------------- logging ----------------

    def _log(self, payload: Mapping[str, Any], env_steps: int) -> None:
        if not self.active:
            return
        row = {k: v for k, v in payload.items() if v is not None}
        if not row:
            return
        step = int(env_steps)
        # W&B requires a non-decreasing step; a regression would silently drop
        # the row instead of raising.
        step = max(step, self._last_step)
        row["env_steps"] = int(env_steps)
        try:
            self._run.log(row, step=step)
            self._last_step = step
        except Exception as exc:
            self._degraded = True
            print(f"[wandb warning] logging failed ({type(exc).__name__}: {exc}); "
                  "continuing without W&B. Local JSONL artifacts are unaffected.")

    def log_train(self, env_steps: int, progress_row: Mapping[str, Any],
                  result: Optional[Mapping[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "train/rllib_agent_sum_return_mean":
                progress_row.get("rllib_agent_sum_return_mean"),
            "train/episode_length_mean": progress_row.get("episode_length_mean"),
            "train/iteration": progress_row.get("iteration"),
            "time/wall_seconds": progress_row.get("wall_seconds"),
        }
        wall = _finite(progress_row.get("wall_seconds"))
        if wall and wall > 0:
            payload["time/env_steps_per_second"] = env_steps / wall
            if self._target_env_steps is not None:
                remaining = max(0, self._target_env_steps - env_steps)
                payload["time/eta_hours_to_target"] = (
                    remaining / (env_steps / wall) / 3600.0
                    if env_steps > 0 else None
                )
        if result is not None:
            payload.update(learner_metrics(result))
        self._log(payload, env_steps)

    def log_eval(self, env_steps: int, evaluation: Mapping[str, Any], *,
                 scheduled_env_step: Optional[int] = None) -> None:
        summary = evaluation.get("summary")
        if not isinstance(summary, Mapping):
            return
        payload: Dict[str, Any] = {
            "eval/team_return_mean": summary.get("team_return_mean"),
            "eval/team_return_std": summary.get("team_return_std"),
            "eval/length_mean": summary.get("length_mean"),
            "eval/musical_f1": summary.get("episode_task/musical_f1_mean"),
            "eval/musical_f1_std": summary.get("episode_task/musical_f1_std"),
            "eval/musical_precision": summary.get("episode_task/musical_precision_mean"),
            "eval/musical_recall": summary.get("episode_task/musical_recall_mean"),
            "eval/sustain_f1": summary.get("episode_task/sustain_f1_mean"),
            "eval/coordination/common_area_success_rate": summary.get(
                "episode_coordination/common_area_success_rate_mean"
            ),
            "eval/coordination/common_area_success_rate_std": summary.get(
                "episode_coordination/common_area_success_rate_std"
            ),
            "eval/coordination/common_area_duplicate_press_rate": summary.get(
                "episode_coordination/common_area_duplicate_press_rate_mean"
            ),
            "eval/coordination/common_area_duplicate_press_rate_std": summary.get(
                "episode_coordination/common_area_duplicate_press_rate_std"
            ),
            "eval/coordination/inter_agent_collision_step_rate": summary.get(
                "episode_coordination/inter_agent_collision_step_rate_mean"
            ),
            "eval/coordination/inter_agent_collision_step_rate_std": summary.get(
                "episode_coordination/inter_agent_collision_step_rate_std"
            ),
            "eval/reward/base_team_return": summary.get(
                "episode_reward/base_team_return_mean"
            ),
            "eval/reward/inter_agent_collision_penalty_return": summary.get(
                "episode_reward/inter_agent_collision_penalty_return_mean"
            ),
            "eval/reward/shaped_team_return": summary.get(
                "episode_reward/shaped_team_return_mean"
            ),
            "eval/reward/inter_agent_collision_penalty_coef": summary.get(
                "episode_reward/inter_agent_collision_penalty_coef_mean"
            ),
        }
        if scheduled_env_step is not None:
            payload["eval/scheduled_env_step"] = int(scheduled_env_step)
        self._log(payload, env_steps)

    def log_eval_videos(
        self,
        env_steps: int,
        video_paths: Sequence[Path],
        *,
        scheduled_env_step: Optional[int] = None,
    ) -> None:
        """Upload completed evaluation MP4s immediately as W&B media.

        This is intentionally separate from the optional end-of-run artifact
        upload: videos become visible while a long training run is still active.
        """
        if not self.active:
            return
        paths = [Path(path) for path in video_paths]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"cannot upload missing evaluation video(s): {missing}"
            )
        if not paths:
            raise ValueError("cannot upload an empty evaluation video list")
        label = (
            int(scheduled_env_step)
            if scheduled_env_step is not None
            else int(env_steps)
        )
        payload: Dict[str, Any] = {}
        for index, path in enumerate(paths):
            key = "eval/video" if index == 0 else f"eval/video_episode_{index}"
            payload[key] = self._wandb.Video(
                str(path),
                format="mp4",
                caption=(
                    f"deterministic evaluation at scheduled env step {label:,}; "
                    f"{path.name}"
                ),
            )
        self._log(payload, env_steps)

    def log_final(self, env_steps: int, eval_summary: Mapping[str, Any]) -> None:
        """Also write scalar summary fields (these become Runs-table columns)."""
        if not self.active:
            return
        summary = eval_summary.get("summary")
        if isinstance(summary, Mapping):
            self._log({f"final/{k}": v for k, v in summary.items()
                       if _finite(v) is not None}, env_steps)
        try:
            if isinstance(summary, Mapping):
                for key in ("team_return_mean",
                            "episode_task/musical_f1_mean",
                            "episode_task/musical_precision_mean",
                            "episode_task/musical_recall_mean",
                            "episode_task/sustain_f1_mean",
                            "episode_coordination/common_area_success_rate_mean",
                            "episode_coordination/common_area_duplicate_press_rate_mean",
                            "episode_coordination/inter_agent_collision_step_rate_mean"):
                    if key in summary:
                        self._run.summary[f"final/{key}"] = float(summary[key])
            self._run.summary["actual_total_env_steps"] = int(
                eval_summary.get("actual_total_env_steps", env_steps))
            for key in ("algo", "seed", "env_id", "protocol_training_compliant"):
                if key in eval_summary:
                    self._run.summary[key] = eval_summary[key]
        except Exception as exc:
            self._degraded = True
            print(f"[wandb warning] summary update failed: {exc}")

    def log_artifact_dir(self, path: Path, *, name: str,
                         artifact_type: str = "run-artifacts") -> None:
        if not self.active:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            self._degraded = True
            print(f"[wandb warning] artifact upload failed: {exc}")

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as exc:
            print(f"[wandb warning] finish failed: {exc}")
        finally:
            self._run = None
