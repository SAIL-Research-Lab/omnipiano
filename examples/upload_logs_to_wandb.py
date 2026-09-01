#!/usr/bin/env python3
"""Import the historical ``logs/`` tree into organized W&B projects.

The importer is deliberately conservative:

* no network write happens unless ``--upload`` is supplied;
* checkpoints, videos, figures, TensorBoard events, and operation logs are
  never uploaded;
* deterministic run IDs make repeated invocations discoverable;
* an already-completed import is skipped instead of duplicated;
* evaluation CSV rows are placed on a training-step axis only when they can
  be matched exactly to ``evaluations.npz``.

Run from the repository root with the dedicated environment:

    source .venv-wandb/bin/activate
    python examples/upload_logs_to_wandb.py --dry-run
    python examples/upload_logs_to_wandb.py --upload
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IMPORTER_VERSION = "1.0"
DEFAULT_ENTITY = "omnipiano"
DEFAULT_PROJECTS = ("general", "safe", "robust")


ROBUST_CONDITION_NAMES = {
    "clean": "clean",
    "clean_nooar": "clean-no-oar",
    "a_gauss_p15": "action-gaussian-0p15",
    "a_uniform_p15": "action-uniform-0p15",
    "a_shift_p15": "action-shift-plus0p15",
    "a_shift_n15": "action-shift-minus0p15",
    "o_gauss_p15": "observation-gaussian-0p15",
    "o_uniform_p15": "observation-uniform-0p15",
    "o_shift_p15": "observation-shift-plus0p15",
    "o_shift_n15": "observation-shift-minus0p15",
    "r_gauss_p50": "reward-gaussian-0p50",
    "r_uniform_p50": "reward-uniform-0p50",
    "r_shift_p50": "reward-shift-plus0p50",
    "r_shift_n50": "reward-shift-minus0p50",
    "r_shift_p50_nooar": "reward-shift-plus0p50-no-oar",
    "r_shift_n50_nooar": "reward-shift-minus0p50-no-oar",
    "ao_gauss_p15": "action-observation-gaussian-0p15",
    "ar_uniform_a15_r50": "action-reward-uniform-action0p15-reward0p50",
    "or_shift_on15_rn50": (
        "observation-reward-shift-observation-minus0p15-reward-minus0p50"
    ),
}


@dataclass
class RunSpec:
    path: Path
    source_name: str
    project: str
    algorithm: str
    algorithm_label: str
    framework: str
    env_id: str
    seed: int
    intended_steps: int
    observed_steps: int
    source_status: str
    group: str
    run_id: str
    run_name: str = ""
    attempt: int | None = None
    train_points: list[dict[str, float | int]] = field(default_factory=list)
    eval_points: list[dict[str, float | int]] = field(default_factory=list)
    final_metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    resolved_config: dict[str, Any] | None = None
    periodic_eval_files: list[str] = field(default_factory=list)
    final_eval_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "historical_import": {
                "version": IMPORTER_VERSION,
                "source_directory": self.source_name,
                "source_status": self.source_status,
                "observed_env_steps": self.observed_steps,
                "periodic_eval_files": self.periodic_eval_files,
                "final_eval_files": self.final_eval_files,
            },
            "algorithm": self.algorithm,
            "algorithm_label": self.algorithm_label,
            "framework": self.framework,
            "env_id": self.env_id,
            "seed": self.seed,
            "total_env_steps": self.intended_steps,
            "wandb_group": self.group,
            "source_metadata": self.source_metadata,
        }
        if self.attempt is not None:
            result["historical_import"]["attempt"] = self.attempt
        if self.resolved_config is not None:
            result["resolved_config"] = self.resolved_config
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument(
        "--projects",
        nargs="+",
        choices=DEFAULT_PROJECTS,
        default=list(DEFAULT_PROJECTS),
        help="Only process these logical projects.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--upload",
        action="store_true",
        help="Perform W&B writes. Without this flag the command is a dry-run.",
    )
    mode.add_argument("--dry-run", action="store_true", help="Parse and validate only.")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Do not upload the four source runs that ended before completion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N runs after filtering (useful for testing).",
    )
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path("."),
        help="Base directory for W&B's local cache (defaults to ./wandb).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [(field or "").strip() for field in (reader.fieldnames or [])]
        rows: list[dict[str, str]] = []
        for source_row in reader:
            row: dict[str, str] = {}
            for key, value in source_row.items():
                clean_key = (key or "").strip()
                row[clean_key] = (value or "").strip()
            rows.append(row)
    return fields, rows


def number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
    if not math.isfinite(parsed):
        return None
    if parsed.is_integer() and abs(parsed) <= 2**53:
        return int(parsed)
    return parsed


def snake(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_").lower()


def slug(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "-", text)
    return text.strip("-").lower()


def env_slug(env_id: str) -> str:
    value = env_id
    if value.startswith("OmniPiano-"):
        value = value[len("OmniPiano-") :]
    return slug(value)


def format_number_slug(value: float | int) -> str:
    parsed = float(value)
    if parsed.is_integer():
        return str(int(parsed))
    return (f"{parsed:g}").replace("-", "minus").replace(".", "p")


def algorithm_slug(label: str) -> str:
    lower = label.lower()
    if "ppolag" in lower:
        return "ppolag"
    if "tqc" in lower:
        return "tqc"
    if "sac" in lower:
        return "sac"
    if "ppo" in lower:
        return "ppo"
    return slug(label)


def classify_project(name: str) -> str:
    if name.startswith("omnisafe_"):
        return "safe"
    if name.startswith("ppo_sb3_baseline_clairdelune_"):
        return "robust"
    return "general"


def parse_seed_from_name(name: str) -> int:
    match = re.search(r"_seed(\d+)(?:_|$)", name)
    if match:
        return int(match.group(1))
    # One legacy general run omits the seed from its directory name. Its
    # eval_summary.json is authoritative, so reaching here is an error.
    raise ValueError(f"Cannot infer seed from {name}")


def infer_incomplete_general_env(name: str) -> str:
    if name.startswith("sac_4hand_winterwind_proto_"):
        return "OmniPiano-WinterWind-FourHandPrototype-v0"
    raise ValueError(f"Cannot infer environment for incomplete run {name}")


def framework_for(project: str, algorithm: str) -> str:
    if project == "safe":
        return "omnisafe"
    if algorithm == "tqc":
        return "sb3-contrib"
    return "stable-baselines3"


def resolved_omnisafe_config(run_dir: Path) -> dict[str, Any] | None:
    configs = sorted(run_dir.rglob("config.json"))
    if not configs:
        return None
    if len(configs) != 1:
        raise ValueError(f"Expected one OmniSafe config in {run_dir}, got {len(configs)}")
    return load_json(configs[0])


def normalize_train_metric(column: str) -> str | None:
    known = {
        "rollout/ep_rew_mean": "train/episode_return",
        "rollout/ep_len_mean": "train/episode_length",
        "Metrics/EpRet": "train/episode_return",
        "Metrics/EpCost": "train_safety/episode_cost",
        "Metrics/EpLen": "train/episode_length",
        "Metrics/TestEpRet": None,
        "Metrics/TestEpCost": None,
        "Metrics/TestEpLen": None,
        "Metrics/LagrangeMultiplier": "train_safety/lagrange_multiplier",
        "Train/Epoch": "train/epoch",
        "Train/LR": "train/learning_rate",
        "Train/Entropy": "train/entropy",
        "Train/KL": "train/kl",
        "Train/PolicyStd": "train/policy_std",
        "Loss/Loss_pi": "train_loss/policy",
        "Loss/Loss_reward_critic": "train_loss/reward_critic",
        "Loss/Loss_cost_critic": "train_loss/cost_critic",
        "Value/reward": "train_value/reward",
        "Value/cost": "train_value/cost",
        "Value/reward_critic": "train_value/reward_critic",
    }
    if column in known:
        return known[column]
    if column in {"TotalEnvSteps", "time/total_timesteps"}:
        return None
    if column.startswith("eval/"):
        # evaluations.npz is the common, framework-independent source.
        return None
    if column.startswith("train/"):
        return "train/" + snake(column.split("/", 1)[1])
    if column.startswith("time/"):
        return "train_time/" + snake(column.split("/", 1)[1])
    if column.startswith("rollout/"):
        return "train/" + snake(column.split("/", 1)[1])
    if column.startswith("Train/"):
        return "train/" + snake(column.split("/", 1)[1])
    if column.startswith("Loss/"):
        return "train_loss/" + snake(column.split("/", 1)[1])
    if column.startswith("Value/"):
        return "train_value/" + snake(column.split("/", 1)[1])
    if column.startswith("Time/"):
        return "train_time/" + snake(column.split("/", 1)[1])
    if column.startswith("Metrics/LagrangeMultiplier/"):
        suffix = snake(column.removeprefix("Metrics/LagrangeMultiplier/"))
        return "train_safety/lagrange_multiplier_" + suffix
    if column.startswith("Metrics/"):
        return "train/" + snake(column.split("/", 1)[1])
    return "source_train/" + snake(column)


def load_train_points(run_dir: Path) -> tuple[list[dict[str, float | int]], str | None]:
    progress_files = sorted(run_dir.rglob("progress.csv"))
    if not progress_files:
        return [], None
    if len(progress_files) != 1:
        raise ValueError(f"Expected at most one progress.csv in {run_dir}")
    path = progress_files[0]
    fields, rows = read_csv(path)
    step_column = None
    for candidate in ("TotalEnvSteps", "time/total_timesteps"):
        if candidate in fields:
            step_column = candidate
            break
    if step_column is None:
        raise ValueError(f"No environment-step column in {path}")

    by_step: dict[int, dict[str, float | int]] = {}
    for source_row in rows:
        raw_step = number(source_row.get(step_column))
        if raw_step is None:
            continue
        step = int(raw_step)
        point = by_step.setdefault(step, {"train/env_step": step})
        for column, raw_value in source_row.items():
            metric = normalize_train_metric(column)
            if metric is None:
                continue
            parsed = number(raw_value)
            if parsed is not None:
                point[metric] = parsed
    return [by_step[key] for key in sorted(by_step)], str(path.relative_to(run_dir))


RICH_EVAL_METRICS = {
    "ep_cost": "eval_safety/cost",
    "ep_violations": "eval_safety/violations",
    "ep_f1": "eval/f1",
    "ep_precision": "eval/precision",
    "ep_recall": "eval/recall",
    "ep_sustain_f1": "eval/sustain_f1",
    "ep_sustain_precision": "eval/sustain_precision",
    "ep_sustain_recall": "eval/sustain_recall",
    "energy_reward": "eval_reward/energy",
    "fingering_reward": "eval_reward/fingering",
    "ot_fingering_reward": "eval_reward/ot_fingering",
    "forearm_reward": "eval_reward/forearm",
    "key_press_reward": "eval_reward/key_press",
    "sustain_reward": "eval_reward/sustain",
    "eval_noise_scale": "eval_noise/scale",
    "ep_return_true": "eval/return_true",
    "ep_noise_action_l2": "eval_noise/action_l2",
    "ep_noise_obs_l2": "eval_noise/observation_l2",
    "ep_noise_reward": "eval_noise/reward",
}


def mean_std(values: Iterable[float | int]) -> tuple[float, float] | None:
    clean = np.asarray(list(values), dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not clean.size:
        return None
    return float(clean.mean()), float(clean.std())


def add_rich_eval_stats(
    point: dict[str, float | int], rows: list[dict[str, str]]
) -> None:
    for source_key, target_key in RICH_EVAL_METRICS.items():
        values = [parsed for row in rows if (parsed := number(row.get(source_key))) is not None]
        stats = mean_std(values)
        if stats is None:
            continue
        point[target_key] = stats[0]
        point[target_key + "_std"] = stats[1]


def final_csv_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    temporary: dict[str, float | int] = {}
    add_rich_eval_stats(temporary, rows)
    result: dict[str, float | int] = {"final_eval/num_episodes": len(rows)}
    for key, value in temporary.items():
        section, metric = key.split("/", 1)
        if section == "eval":
            result[f"final_eval/{metric}"] = value
        elif section == "eval_safety":
            result[f"final_eval_safety/{metric}"] = value
        elif section == "eval_reward":
            result[f"final_eval_reward/{metric}"] = value
        elif section == "eval_noise":
            result[f"final_eval_noise/{metric}"] = value
    return result


def array_row_stats(array: np.ndarray, index: int) -> tuple[float, float]:
    row = np.asarray(array[index], dtype=np.float64).reshape(-1)
    row = row[np.isfinite(row)]
    if not row.size:
        return float("nan"), float("nan")
    return float(row.mean()), float(row.std())


def load_eval_points(
    run_dir: Path,
) -> tuple[
    list[dict[str, float | int]],
    dict[str, float | int],
    list[str],
    list[str],
    list[str],
]:
    npz_path = run_dir / "evaluations.npz"
    if not npz_path.is_file():
        raise ValueError(f"Missing {npz_path}")

    warnings: list[str] = []
    with np.load(npz_path, allow_pickle=False) as archive:
        timesteps = np.asarray(archive["timesteps"]).reshape(-1)
        results = np.asarray(archive["results"])
        lengths = np.asarray(archive["ep_lengths"])
        costs = np.asarray(archive["ep_costs"]) if "ep_costs" in archive.files else None

        if results.shape[0] != len(timesteps) or lengths.shape[0] != len(timesteps):
            raise ValueError(f"Inconsistent evaluations.npz arrays in {run_dir}")
        if costs is not None and costs.shape[0] != len(timesteps):
            raise ValueError(f"Inconsistent ep_costs array in {run_dir}")

        by_step: dict[int, dict[str, float | int]] = {}
        for index, raw_step in enumerate(timesteps):
            step = int(raw_step)
            point: dict[str, float | int] = {"eval/env_step": step}
            return_mean, return_std = array_row_stats(results, index)
            length_mean, length_std = array_row_stats(lengths, index)
            point.update(
                {
                    "eval/return_mean": return_mean,
                    "eval/return_std": return_std,
                    "eval/length_mean": length_mean,
                    "eval/length_std": length_std,
                }
            )
            if costs is not None:
                cost_mean, cost_std = array_row_stats(costs, index)
                point["eval_safety/cost"] = cost_mean
                point["eval_safety/cost_std"] = cost_std
            by_step[step] = point

        n_points = len(timesteps)
        if results.ndim >= 2:
            episodes_per_point = int(np.prod(results.shape[1:]))
        else:
            episodes_per_point = 1

    periodic_files: list[str] = []
    final_files: list[str] = []
    final_metrics: dict[str, float | int] = {}

    for csv_path in sorted(run_dir.glob("eval_episode_metrics_*.csv")):
        fields, rows = read_csv(csv_path)
        relative_name = csv_path.name
        groups: list[tuple[int, list[dict[str, str]]]] | None = None

        if "training_step" in fields:
            grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                parsed = number(row.get("training_step"))
                if parsed is not None:
                    grouped[int(parsed)].append(row)
            if len(grouped) == n_points and set(grouped) == set(by_step):
                groups = [(step, grouped[step]) for step in sorted(grouped)]
            else:
                warnings.append(
                    f"{relative_name}: training_step values do not exactly match evaluations.npz"
                )
        elif len(rows) == n_points * episodes_per_point:
            groups = []
            steps = sorted(by_step)
            for index, step in enumerate(steps):
                start = index * episodes_per_point
                groups.append((step, rows[start : start + episodes_per_point]))

        if groups is not None:
            periodic_files.append(relative_name)
            for step, group_rows in groups:
                add_rich_eval_stats(by_step[step], group_rows)
        else:
            final_files.append(relative_name)
            final_metrics.update(final_csv_metrics(rows))

    return (
        [by_step[key] for key in sorted(by_step)],
        final_metrics,
        periodic_files,
        final_files,
        warnings,
    )


SUMMARY_CANONICAL_KEYS = {
    "return_mean": "final/return_mean",
    "return_std": "final/return_std",
    "cost_mean": "final/cost_mean",
    "cost_std": "final/cost_std",
    "length_mean": "final/length_mean",
    "episode_task/f1_mean": "final/f1",
    "episode_task/f1_std": "final/f1_std",
    "episode_task/key_precision_mean": "final/precision",
    "episode_task/key_precision_std": "final/precision_std",
    "episode_task/key_recall_mean": "final/recall",
    "episode_task/key_recall_std": "final/recall_std",
    "episode_task/sustain_f1_mean": "final/sustain_f1",
    "episode_task/sustain_f1_std": "final/sustain_f1_std",
    "episode_safety/cost_total_mean": "final/cost_mean",
    "episode_safety/cost_total_std": "final/cost_std",
    "episode_safety/violations_mean": "final/violations_mean",
    "episode_safety/violations_std": "final/violations_std",
}


def metrics_from_summary(summary_json: dict[str, Any]) -> dict[str, float | int]:
    block = summary_json.get("summary")
    if not isinstance(block, dict):
        return {}
    result: dict[str, float | int] = {}
    for source_key, raw_value in block.items():
        parsed = number(raw_value)
        if parsed is None:
            continue
        source_metric = "source_summary/" + snake(source_key)
        result[source_metric] = parsed
        canonical = SUMMARY_CANONICAL_KEYS.get(source_key)
        if canonical is not None:
            result[canonical] = parsed
    return result


def robust_condition(source_name: str) -> str:
    prefix = "ppo_sb3_baseline_clairdelune_"
    match = re.fullmatch(r"(.+)_seed\d+_\d+", source_name.removeprefix(prefix))
    if not match:
        raise ValueError(f"Cannot parse robust run name {source_name}")
    encoded = match.group(1)
    try:
        return ROBUST_CONDITION_NAMES[encoded]
    except KeyError as exc:
        raise ValueError(f"Unknown robust condition {encoded!r}") from exc


def group_for(
    project: str,
    source_name: str,
    algorithm: str,
    algorithm_label: str,
    env_id: str,
    intended_steps: int,
    resolved_config: dict[str, Any] | None,
) -> str:
    if project == "robust":
        return f"ppo-clair-de-lune-{robust_condition(source_name)}"

    base = f"{algorithm}-{env_slug(env_id)}"
    if project == "safe" and algorithm == "ppolag":
        cost_limit = None
        if resolved_config is not None:
            cost_limit = (resolved_config.get("lagrange_cfgs") or {}).get("cost_limit")
        if cost_limit is None:
            match = re.search(r"_cl(\d+(?:p\d+)?)_", source_name)
            if match:
                cost_limit = float(match.group(1).replace("p", "."))
        if cost_limit is None:
            raise ValueError(f"Missing cost limit for {source_name}")
        return f"{base}-cl{format_number_slug(cost_limit)}"

    if project == "general":
        label_lower = algorithm_label.lower()
        if "gamma08_ablation" in source_name:
            base += "-library-defaults-gamma0p8"
        elif "paper-matched" in label_lower or source_name.startswith(
            "sac_4hand_winterwind_proto_"
        ):
            base += "-paper-matched"
        elif algorithm == "ppo" and "gamma08" in source_name:
            base += "-gamma0p8"
        if intended_steps != 5_000_000:
            base += f"-steps{format_number_slug(intended_steps / 1_000_000)}m"
    return base


def source_metadata(summary_json: dict[str, Any]) -> dict[str, Any]:
    excluded = {"summary", "episodes", "returns", "costs"}
    return {key: value for key, value in summary_json.items() if key not in excluded}


def build_spec(run_dir: Path) -> RunSpec:
    source_name = run_dir.name
    project = classify_project(source_name)
    summary_path = run_dir / "eval_summary.json"
    summary_json = load_json(summary_path) if summary_path.is_file() else {}
    omnisafe_config = resolved_omnisafe_config(run_dir) if project == "safe" else None

    algorithm_label = str(summary_json.get("algorithm") or "")
    if not algorithm_label and omnisafe_config is not None:
        algorithm_label = f"{omnisafe_config.get('algo', 'unknown')} (OmniSafe)"
    if not algorithm_label and source_name.startswith("sac_4hand_winterwind_proto_"):
        algorithm_label = "SAC (SB3, paper-matched hyperparameters)"
    algorithm = algorithm_slug(algorithm_label)

    env_id = str(summary_json.get("env_id") or summary_json.get("env") or "")
    if not env_id and omnisafe_config is not None:
        env_id = str(omnisafe_config.get("env_id") or "")
    if not env_id:
        env_id = infer_incomplete_general_env(source_name)

    seed_value = summary_json.get("seed")
    if seed_value is None and omnisafe_config is not None:
        seed_value = omnisafe_config.get("seed")
    seed = int(seed_value) if seed_value is not None else parse_seed_from_name(source_name)

    intended_value = summary_json.get("total_env_steps")
    if intended_value is None and omnisafe_config is not None:
        intended_value = (omnisafe_config.get("train_cfgs") or {}).get("total_steps")
    intended_steps = int(intended_value or 5_000_000)

    train_points, progress_file = load_train_points(run_dir)
    (
        eval_points,
        csv_final_metrics,
        periodic_files,
        final_files,
        eval_warnings,
    ) = load_eval_points(run_dir)

    observed_candidates = [0]
    observed_candidates.extend(int(point["train/env_step"]) for point in train_points)
    observed_candidates.extend(int(point["eval/env_step"]) for point in eval_points)
    observed_steps = max(observed_candidates)
    source_status = "complete" if summary_path.is_file() else "incomplete"

    group = group_for(
        project,
        source_name,
        algorithm,
        algorithm_label,
        env_id,
        intended_steps,
        omnisafe_config,
    )
    digest = hashlib.sha256(f"{project}/{source_name}".encode()).hexdigest()[:12]
    run_id = f"hist-{digest}"

    metadata = source_metadata(summary_json)
    metadata["progress_file"] = progress_file
    final_metrics = metrics_from_summary(summary_json)
    final_metrics.update(csv_final_metrics)

    warnings = list(eval_warnings)
    if not train_points:
        warnings.append("No progress.csv; uploading evaluation curves only")
    if source_status == "incomplete":
        warnings.append(
            f"Source run is incomplete ({observed_steps:,}/{intended_steps:,} observed env steps)"
        )

    return RunSpec(
        path=run_dir,
        source_name=source_name,
        project=project,
        algorithm=algorithm,
        algorithm_label=algorithm_label,
        framework=framework_for(project, algorithm),
        env_id=env_id,
        seed=seed,
        intended_steps=intended_steps,
        observed_steps=observed_steps,
        source_status=source_status,
        group=group,
        run_id=run_id,
        train_points=train_points,
        eval_points=eval_points,
        final_metrics=final_metrics,
        source_metadata=metadata,
        resolved_config=omnisafe_config,
        periodic_eval_files=periodic_files,
        final_eval_files=final_files,
        warnings=warnings,
    )


def assign_run_names(specs: list[RunSpec]) -> None:
    buckets: dict[tuple[str, str, int], list[RunSpec]] = defaultdict(list)
    for spec in specs:
        buckets[(spec.project, spec.group, spec.seed)].append(spec)
    for bucket in buckets.values():
        bucket.sort(key=lambda spec: spec.source_name)
        for index, spec in enumerate(bucket, start=1):
            base = f"{spec.group}-seed{spec.seed}"
            if len(bucket) > 1:
                spec.attempt = index
                base += f"-attempt{index}"
            if spec.source_status == "incomplete":
                base += "-incomplete"
            spec.run_name = base


def discover_specs(logs_dir: Path) -> list[RunSpec]:
    if not logs_dir.is_dir():
        raise ValueError(f"Logs directory does not exist: {logs_dir}")
    run_dirs = sorted(path for path in logs_dir.iterdir() if path.is_dir())
    specs = [build_spec(run_dir) for run_dir in run_dirs]
    assign_run_names(specs)
    run_ids = [spec.run_id for spec in specs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Deterministic W&B run ID collision")
    return specs


def print_manifest(specs: list[RunSpec], entity: str) -> None:
    counts = Counter(spec.project for spec in specs)
    groups = Counter((spec.project, spec.group) for spec in specs)
    statuses = Counter(spec.source_status for spec in specs)
    print(
        "MANIFEST "
        f"entity={entity} runs={len(specs)} "
        f"general={counts['general']} safe={counts['safe']} robust={counts['robust']} "
        f"groups={len(groups)} complete={statuses['complete']} "
        f"incomplete={statuses['incomplete']}"
    )
    for spec in specs:
        print(
            f"[{spec.project:7}] {spec.run_name}\n"
            f"          group={spec.group}\n"
            f"          id={spec.run_id} source={spec.source_name} "
            f"train_points={len(spec.train_points)} eval_points={len(spec.eval_points)} "
            f"status={spec.source_status}"
        )
        for warning in spec.warnings:
            print(f"          WARNING: {warning}")


def remote_run_or_none(api: Any, path: str) -> Any | None:
    try:
        return api.run(path)
    except Exception as exc:  # W&B has changed its public exception hierarchy.
        message = str(exc).lower()
        if any(token in message for token in ("could not find run", "not found", "404")):
            return None
        raise


def upload_spec(spec: RunSpec, entity: str, wandb_dir: Path, api: Any) -> str:
    import wandb

    remote_path = f"{entity}/{spec.project}/{spec.run_id}"
    existing = remote_run_or_none(api, remote_path)
    if existing is not None:
        completed = bool(existing.summary.get("historical_import/complete"))
        if completed:
            print(f"SKIP already imported: {existing.url}", flush=True)
            return "skipped"
        raise RuntimeError(
            f"Remote run {remote_path} exists but is not marked as a complete import; "
            "refusing to append duplicate history"
        )

    tags = [
        "historical-import",
        spec.project,
        spec.framework,
        spec.algorithm,
        f"seed-{spec.seed}",
        spec.source_status,
    ]
    notes = (
        f"Historical import from logs/{spec.source_name}. "
        "Only scalar curves and metadata were imported; models and media were excluded."
    )
    run = wandb.init(
        entity=entity,
        project=spec.project,
        id=spec.run_id,
        resume="never",
        name=spec.run_name,
        group=spec.group,
        job_type="historical-import",
        tags=tags,
        notes=notes,
        config=spec.config,
        dir=str(wandb_dir.resolve()),
        mode="online",
        force=True,
        save_code=False,
        reinit="finish_previous",
        settings=wandb.Settings(console="off", silent=True),
    )
    if run is None:
        raise RuntimeError(f"wandb.init returned None for {spec.source_name}")

    try:
        run.define_metric("train/env_step")
        for pattern in (
            "train/*",
            "train_safety/*",
            "train_loss/*",
            "train_value/*",
            "train_time/*",
            "source_train/*",
        ):
            run.define_metric(pattern, step_metric="train/env_step")

        run.define_metric("eval/env_step")
        for pattern in (
            "eval/*",
            "eval_safety/*",
            "eval_reward/*",
            "eval_noise/*",
        ):
            run.define_metric(pattern, step_metric="eval/env_step")

        for point in spec.train_points:
            run.log(point)
        for point in spec.eval_points:
            run.log(point)

        for key, value in spec.final_metrics.items():
            run.summary[key] = value
        run.summary["historical_import/complete"] = True
        run.summary["historical_import/source_status"] = spec.source_status
        run.summary["historical_import/train_points"] = len(spec.train_points)
        run.summary["historical_import/eval_points"] = len(spec.eval_points)
        run.summary["historical_import/observed_env_steps"] = spec.observed_steps
        url = run.url
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise

    print(f"UPLOADED {spec.run_name}: {url}", flush=True)
    return "uploaded"


def upload_all(specs: list[RunSpec], entity: str, wandb_dir: Path) -> None:
    import wandb

    wandb_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    # Force an authenticated request before creating any run.
    _ = api.viewer

    results = Counter()
    failures: list[tuple[RunSpec, Exception]] = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"UPLOAD {index}/{len(specs)} {entity}/{spec.project} "
            f"{spec.run_name}",
            flush=True,
        )
        try:
            results[upload_spec(spec, entity, wandb_dir, api)] += 1
        except Exception as exc:
            failures.append((spec, exc))
            results["failed"] += 1
            print(f"FAILED {spec.source_name}: {exc}", file=sys.stderr, flush=True)

    print(
        f"UPLOAD_SUMMARY uploaded={results['uploaded']} skipped={results['skipped']} "
        f"failed={results['failed']}",
        flush=True,
    )
    if failures:
        for spec, exc in failures:
            print(f"  {spec.source_name}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    specs = discover_specs(args.logs_dir.resolve())
    selected_projects = set(args.projects)
    specs = [spec for spec in specs if spec.project in selected_projects]
    if args.skip_incomplete:
        specs = [spec for spec in specs if spec.source_status == "complete"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        specs = specs[: args.limit]

    print_manifest(specs, args.entity)
    if not args.upload:
        print("DRY_RUN_OK: no W&B writes performed")
        return
    upload_all(specs, args.entity, args.wandb_dir)


if __name__ == "__main__":
    main()
