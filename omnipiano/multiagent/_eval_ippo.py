"""Evaluate an RLlib IPPO checkpoint with fail-fast deterministic inference.

The evaluator reports the one-copy shared team return plus terminal musical
F1/precision/recall metrics.  It never substitutes zero actions when checkpoint
loading or policy inference fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent._ippo_common import (
    LEGACY_RLLIB_ENV_NAME,
    REQUIRED_MUSICAL_METRICS,
    RLLIB_ENV_NAME,
    evaluate_ippo,
    resolve_checkpoint_path,
    write_json,
)
from omnipiano.multiagent._train_ippo import (
    ALGORITHM_NAME,
    RLLIB_TARGET_VERSION,
    _make_env_for_rllib,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    proto = BenchmarkProtocolConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-dir",
        dest="checkpoint",
        required=True,
        help="Direct RLlib checkpoint or a run dir containing checkpoint_path.txt.",
    )
    parser.add_argument(
        "--output-dir",
        "--record-dir",
        dest="output_dir",
        default=None,
        help="Evaluation artifact directory. --record-dir is a compatibility alias.",
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Override env id. Otherwise read it from the training run_config.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluation seed. Otherwise use the recorded eval seed.",
    )
    parser.add_argument(
        "--num-eval-eps", type=int, default=proto.num_eval_eps,
        help="Number of deterministic evaluation episodes.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not record evaluation videos.",
    )
    return parser


def _candidate_run_dirs(requested: Path, checkpoint: Path) -> Sequence[Path]:
    candidates = []
    for path in (requested, checkpoint):
        current = path if path.is_dir() else path.parent
        for _ in range(5):
            if current not in candidates:
                candidates.append(current)
            if current.parent == current:
                break
            current = current.parent
    return candidates


def _load_run_config(requested: Path, checkpoint: Path) -> Dict[str, Any]:
    for directory in _candidate_run_dirs(requested, checkpoint):
        path = directory / "run_config.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise RuntimeError(f"{path} does not contain a JSON object")
            payload["_path"] = str(path)
            return payload
    return {}


def _default_output_dir(requested: Path, run_config: Mapping[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    located_config = run_config.get("_path")
    if isinstance(located_config, str):
        # Prefer where the config was actually found. The recorded absolute
        # run_dir may refer to a different machine after copying an artifact.
        base = Path(located_config).expanduser().resolve().parent
    elif requested.is_dir():
        base = requested
    else:
        base = requested.parent
    return (base / f"standalone_eval_{timestamp}").resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.num_eval_eps <= 0:
        raise ValueError("num_eval_eps must be positive")

    requested = Path(args.checkpoint).expanduser().resolve()
    checkpoint = resolve_checkpoint_path(requested)
    run_config = _load_run_config(requested, checkpoint)

    env_id = args.env_id or run_config.get("env_id")
    if not isinstance(env_id, str) or not env_id:
        raise ValueError(
            "--env-id is required when no run_config.json can be found near the checkpoint"
        )
    if args.seed is not None:
        eval_seed = int(args.seed)
    elif "eval_seed" in run_config:
        eval_seed = int(run_config["eval_seed"])
    else:
        eval_seed = BenchmarkProtocolConfig().seed + 10_000

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(requested, run_config)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_summary.json"
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; choose a fresh output directory"
        )

    print(f"[ippo eval] checkpoint={checkpoint}")
    print(f"[ippo eval] env_id={env_id}")
    print(f"[ippo eval] eval_seed={eval_seed} episodes={args.num_eval_eps}")
    print(f"[ippo eval] output_dir={output_dir}")

    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.tune.registry import register_env
    except ImportError as exc:
        write_json(
            output_dir / "failure.json",
            {"stage": "dependency_import", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise RuntimeError("IPPO evaluation requires ray[rllib]") from exc

    if ray.__version__ != RLLIB_TARGET_VERSION:
        print(
            "[ippo eval warning] evaluator targets "
            f"ray[rllib]=={RLLIB_TARGET_VERSION}, found {ray.__version__}"
        )

    register_env(RLLIB_ENV_NAME, _make_env_for_rllib)
    register_env(LEGACY_RLLIB_ENV_NAME, _make_env_for_rllib)

    algo = None
    ray_started = False
    try:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        ray_started = True
        algo = Algorithm.from_checkpoint(str(checkpoint))
        video_dir = None if args.no_video else str(output_dir / "videos")
        evaluation = evaluate_ippo(
            algo,
            env_id,
            eval_seed=eval_seed,
            num_episodes=args.num_eval_eps,
            record_dir=video_dir,
        )
        config_path = run_config.get("_path")
        training_run_config = {
            key: value for key, value in run_config.items() if key != "_path"
        }
        if isinstance(config_path, str):
            # Interpret relative paths from this eval_summary.json's directory,
            # so copying the complete training run preserves the reference.
            checkpoint_path_reference = os.path.relpath(
                str(checkpoint), start=str(output_dir)
            )
        else:
            checkpoint_path_reference = str(checkpoint)
        result = {
            "schema_version": 1,
            "algorithm": ALGORITHM_NAME,
            "source_script": Path(__file__).name,
            "checkpoint_path": checkpoint_path_reference,
            "checkpoint_path_resolved": str(checkpoint),
            "training_run_config": training_run_config or None,
            "training_seed": run_config.get("seed"),
            "protocol_version": run_config.get("protocol_version"),
            "protocol_replication_seeds": run_config.get(
                "protocol_replication_seeds"
            ),
            "protocol_settings_compliant": run_config.get(
                "protocol_settings_compliant"
            ),
            "requested_total_env_steps": run_config.get(
                "requested_total_env_steps", run_config.get("total_env_steps")
            ),
            "effective_config": run_config.get("effective_config"),
            "ray_version": ray.__version__,
            "rllib_target_version": RLLIB_TARGET_VERSION,
            "required_terminal_metrics": list(REQUIRED_MUSICAL_METRICS),
            "video_recording": not args.no_video,
            **evaluation,
        }
        write_json(output_path, result)
        summary = evaluation["summary"]
        print(
            "[ippo eval] "
            f"team_return={summary['team_return_mean']:.3f}  "
            f"musical_f1={summary['episode_task/musical_f1_mean']:.6f}  "
            f"precision={summary['episode_task/musical_precision_mean']:.6f}  "
            f"recall={summary['episode_task/musical_recall_mean']:.6f}"
        )
        print(f"[ippo eval] summary={output_path}")
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "stage": "checkpoint_load_or_evaluation",
                "checkpoint_path": str(checkpoint),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error: Optional[Exception] = None
        if algo is not None:
            try:
                algo.stop()
            except Exception as exc:
                if not active_exception:
                    cleanup_error = exc
        if ray_started:
            try:
                ray.shutdown()
            except Exception as exc:
                if not active_exception and cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None:
            raise cleanup_error

    return 0


if __name__ == "__main__":
    sys.exit(main())
