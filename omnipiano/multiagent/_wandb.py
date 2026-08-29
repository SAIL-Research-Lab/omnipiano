# ===== omnipiano/multiagent/_wandb.py（新建）=====
"""Weights & Biases integration for the OmniPiano multi-agent baselines.

Three design constraints drive this module:

  1. A 10M-step run must never die because of a W&B or network problem.  Any
     failure *after* a successful init degrades to a printed warning.
  2. If W&B was explicitly requested, a login/permission problem must fail
     loudly at init, *before* GPU hours are spent.  Discovering after ten hours
     that nothing was uploaded is worse than crashing in the first second.
  3. The x-axis is always lifetime ENVIRONMENT steps, never RLlib iterations
     and never agent steps, so IPPO / MAPPO / single-agent curves overlay
     directly.  See ``_ippo_common.extract_env_steps`` for why the distinction
     matters in a two-agent benchmark.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

# RLlib Learner-result keys we surface.  ``vf_explained_var`` is the single most
# diagnostic number in this whole benchmark: it answers "is the critic actually
# learning, or is it just predicting a constant?"
_LEARNER_KEYS: Sequence[str] = (
    "policy_loss",
    "vf_loss",
    "vf_loss_unclipped",
    "vf_explained_var",
    "entropy",
    "mean_kl_loss",
    "curr_kl_coeff",
    "curr_entropy_coeff",
    "total_loss",
    "gradients_default_optimizer_global_norm",
)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def learner_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Flatten RLlib's per-module Learner stats into ``learner/<module>/<key>``."""
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


class WandbRun:
    """Thin, failure-tolerant wrapper around one ``wandb.Run``."""

    def __init__(
        self,
        *,
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
        self._degraded = False
        self.mode = str(mode)
        if self.mode == "disabled":
            print("[wandb] disabled by --wandb-mode disabled")
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "--wandb-mode is not 'disabled' but wandb is not installed. "
                "Run: pip install wandb   (or pass --wandb-mode disabled)"
            ) from exc
        self._wandb = wandb

        if self.mode == "online" and not (
            os.environ.get("WANDB_API_KEY")
            or (Path.home() / ".netrc").exists()
            or (Path.home() / ".config" / "wandb" / "settings").exists()
        ):
            raise RuntimeError(
                "--wandb-mode online but no W&B credentials were found.\n"
                "  Fix with any of:\n"
                "    wandb login                       # interactive, once per machine\n"
                "    export WANDB_API_KEY=<key>        # headless / cluster\n"
                "    --wandb-mode offline              # log locally, `wandb sync` later\n"
                "    --wandb-mode disabled             # no W&B at all\n"
                "  Get a key at https://wandb.ai/authorize"
            )

        # W&B writes its own staging dir; keep it next to the run artifacts so a
        # single directory is the complete, self-contained record of the run.
        wandb_dir = str(run_dir) if run_dir is not None else None
        try:
            self._run = wandb.init(
                mode=self.mode,
                entity=entity,
                project=project,
                name=name,
                group=group,
                job_type=job_type,
                tags=list(tags),
                notes=notes,
                config=dict(config or {}),
                dir=wandb_dir,
                id=resume_id,
                resume="allow" if resume_id else None,
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"wandb.init failed ({type(exc).__name__}: {exc}). "
                "Use --wandb-mode offline or --wandb-mode disabled to proceed."
            ) from exc

        # Make lifetime env-steps the explicit x-axis for every metric family.
        self._run.define_metric("env_steps")
        for family in ("train/*", "eval/*", "learner/*", "time/*", "final/*"):
            self._run.define_metric(family, step_metric="env_steps")

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
        row["env_steps"] = int(env_steps)
        try:
            self._run.log(row, step=int(env_steps))
        except Exception as exc:  # never kill a long run over telemetry
            self._degraded = True
            print(
                f"[wandb warning] logging failed ({type(exc).__name__}: {exc}); "
                "continuing without W&B. Local JSONL artifacts are unaffected."
            )

    def log_train(
        self,
        env_steps: int,
        progress_row: Mapping[str, Any],
        result: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "train/rllib_agent_sum_return_mean": progress_row.get(
                "rllib_agent_sum_return_mean"
            ),
            "train/episode_length_mean": progress_row.get("episode_length_mean"),
            "train/iteration": progress_row.get("iteration"),
            "time/wall_seconds": progress_row.get("wall_seconds"),
        }
        wall = _finite(progress_row.get("wall_seconds"))
        if wall and wall > 0:
            payload["time/env_steps_per_second"] = env_steps / wall
        if result is not None:
            payload.update(learner_metrics(result))
        self._log(payload, env_steps)

    def log_eval(
        self,
        env_steps: int,
        evaluation: Mapping[str, Any],
        *,
        scheduled_env_step: Optional[int] = None,
    ) -> None:
        summary = evaluation.get("summary")
        if not isinstance(summary, Mapping):
            return
        payload: Dict[str, Any] = {
            "eval/team_return_mean": summary.get("team_return_mean"),
            "eval/team_return_std": summary.get("team_return_std"),
            "eval/length_mean": summary.get("length_mean"),
            "eval/musical_f1": summary.get("episode_task/musical_f1_mean"),
            "eval/musical_f1_std": summary.get("episode_task/musical_f1_std"),
            "eval/musical_precision": summary.get(
                "episode_task/musical_precision_mean"
            ),
            "eval/musical_recall": summary.get("episode_task/musical_recall_mean"),
            "eval/sustain_f1": summary.get("episode_task/sustain_f1_mean"),
        }
        if scheduled_env_step is not None:
            payload["eval/scheduled_env_step"] = int(scheduled_env_step)
        self._log(payload, env_steps)

    def log_final(self, env_steps: int, eval_summary: Mapping[str, Any]) -> None:
        """Write scalar summary fields (these show up in the W&B runs table)."""
        if not self.active:
            return
        summary = eval_summary.get("summary")
        if isinstance(summary, Mapping):
            self._log(
                {
                    f"final/{k[len('episode_task/'):] if k.startswith('episode_task/') else k}": v
                    for k, v in summary.items()
                    if _finite(v) is not None
                },
                env_steps,
            )
        try:
            for key in (
                "team_return_mean",
                "episode_task/musical_f1_mean",
                "episode_task/musical_precision_mean",
                "episode_task/musical_recall_mean",
                "episode_task/sustain_f1_mean",
            ):
                if isinstance(summary, Mapping) and key in summary:
                    self._run.summary[f"final/{key}"] = float(summary[key])
            self._run.summary["actual_total_env_steps"] = int(
                eval_summary.get("actual_total_env_steps", env_steps)
            )
        except Exception as exc:
            self._degraded = True
            print(f"[wandb warning] summary update failed: {exc}")

    def log_artifact_dir(
        self, path: Path, *, name: str, artifact_type: str = "run-artifacts"
    ) -> None:
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