"""Algorithm-agnostic deterministic evaluation of a multi-agent checkpoint.

Reads ``run_config.json`` next to the checkpoint to recover the env id, the
evaluation seed and -- critically -- whether the policy was trained on the
CTDE ``[global_state | own]`` observation.  A mismatch there would surface as
an opaque tensor-shape error deep inside inference.

Never substitutes zero actions when checkpoint loading or inference fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent.paths import resolve_run_path
from omnipiano.multiagent._ippo_common import (
    COORDINATION_RATE_METRICS,
    LEGACY_RLLIB_ENV_NAME,
    REQUIRED_MUSICAL_METRICS,
    RLLIB_ENV_NAME,
    evaluate_marl,
    resolve_checkpoint_path,
    write_json,
)
from omnipiano.multiagent.train import RLLIB_TARGET_VERSION, make_env_for_rllib


def build_arg_parser() -> argparse.ArgumentParser:
    proto = BenchmarkProtocolConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", "--checkpoint-dir", dest="checkpoint", required=True,
                   help="Direct RLlib checkpoint, or a run dir containing "
                        "final_checkpoint_path.txt.")
    p.add_argument("--output-dir", "--record-dir", dest="output_dir", default=None)
    p.add_argument("--env-id", default=None,
                   help="Override; otherwise read from run_config.json.")
    p.add_argument("--seed", type=int, default=None, help="Override the eval seed.")
    p.add_argument("--num-eval-eps", type=int, default=proto.num_eval_eps)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--include-global-state", dest="include_global_state",
                   action="store_true", default=None,
                   help="Override the value recorded in run_config.json.")
    return p


def _load_run_config(requested: Path, checkpoint: Path) -> Dict[str, Any]:
    seen: list = []
    for start in (requested, checkpoint):
        cur = start if start.is_dir() else start.parent
        for _ in range(5):
            if cur not in seen:
                seen.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent
    for directory in seen:
        path = directory / "run_config.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise RuntimeError(f"{path} does not contain a JSON object")
            payload["_path"] = str(path)
            return payload
    return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.num_eval_eps <= 0:
        raise ValueError("num_eval_eps must be positive")

    requested = resolve_run_path(args.checkpoint)
    checkpoint = resolve_checkpoint_path(requested)
    run_config = _load_run_config(requested, checkpoint)
    effective = run_config.get("effective_config", {}) or {}

    env_id = args.env_id or run_config.get("env_id")
    if not isinstance(env_id, str) or not env_id:
        raise ValueError("--env-id is required when no run_config.json is found "
                         "near the checkpoint")
    if args.seed is not None:
        eval_seed = int(args.seed)
    elif "eval_seed" in run_config:
        eval_seed = int(run_config["eval_seed"])
    else:
        eval_seed = BenchmarkProtocolConfig().seed + 10_000
    include_global_state = (
        bool(effective.get("include_global_state", False))
        if args.include_global_state is None else bool(args.include_global_state)
    )
    reward_config = effective.get("inter_agent_collision_penalty", {})
    nested_penalty_coef = (
        reward_config.get("coefficient", 0.0)
        if isinstance(reward_config, Mapping)
        else 0.0
    )
    inter_agent_collision_penalty_coef = float(
        effective.get(
            "inter_agent_collision_penalty_coef", nested_penalty_coef
        )
    )
    if (not math.isfinite(inter_agent_collision_penalty_coef)
            or inter_agent_collision_penalty_coef < 0.0):
        raise ValueError(
            "recorded inter_agent_collision_penalty_coef must be finite and "
            "non-negative"
        )
    algo_name = run_config.get("algo") or (
        "mappo" if run_config.get("is_centralized_critic") else "ippo")

    if args.output_dir is not None:
        output_dir = resolve_run_path(args.output_dir).resolve()
    else:
        cfg_path = run_config.get("_path")
        anchor = (Path(cfg_path).parent if cfg_path
                  else (requested if requested.is_dir() else requested.parent))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (anchor / f"standalone_eval_{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_summary.json"
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists; choose a fresh output dir")

    print(f"[eval] algo={algo_name}  env_id={env_id}")
    print(f"[eval] checkpoint={checkpoint}")
    print(f"[eval] eval_seed={eval_seed} episodes={args.num_eval_eps} "
          f"include_global_state={include_global_state}")
    print(
        "[eval] inter_agent_collision_penalty="
        f"-{inter_agent_collision_penalty_coef:g} per collision step"
    )
    print(f"[eval] output_dir={output_dir}")

    try:
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.tune.registry import register_env
    except ImportError as exc:
        write_json(output_dir / "failure.json",
                   {"stage": "dependency_import", "error": f"{type(exc).__name__}: {exc}"})
        raise RuntimeError("evaluation requires ray[rllib]") from exc

    if ray.__version__ != RLLIB_TARGET_VERSION:
        print(f"[eval warning] targets ray[rllib]=={RLLIB_TARGET_VERSION}, "
              f"found {ray.__version__}")
    register_env(RLLIB_ENV_NAME, make_env_for_rllib)
    register_env(LEGACY_RLLIB_ENV_NAME, make_env_for_rllib)

    algo = None
    started = False
    try:
        ray.init(ignore_reinit_error=True, log_to_driver=False)
        started = True
        algo = Algorithm.from_checkpoint(str(checkpoint))
        evaluation = evaluate_marl(
            algo, env_id, eval_seed=eval_seed, num_episodes=args.num_eval_eps,
            record_dir=None if args.no_video else str(output_dir / "videos"),
            include_global_state=include_global_state,
            inter_agent_collision_penalty_coef=(
                inter_agent_collision_penalty_coef
            ),
        )
        write_json(output_path, {
            "schema_version": 2,
            "algo": algo_name,
            "algorithm": run_config.get("algorithm"),
            "is_centralized_critic": run_config.get("is_centralized_critic"),
            "source_script": "evaluate.py",
            "checkpoint_path": os.path.relpath(str(checkpoint), start=str(output_dir)),
            "checkpoint_path_resolved": str(checkpoint),
            "training_run_config": {k: v for k, v in run_config.items() if k != "_path"} or None,
            "training_seed": run_config.get("seed"),
            "protocol_version": run_config.get("protocol_version"),
            "protocol_training_compliant": run_config.get("protocol_training_compliant"),
            "protocol_settings_compliant": run_config.get("protocol_settings_compliant"),
            "effective_config": effective or None,
            "ray_version": ray.__version__,
            "rllib_target_version": RLLIB_TARGET_VERSION,
            "required_terminal_metrics": list(REQUIRED_MUSICAL_METRICS),
            "reported_coordination_metrics": list(COORDINATION_RATE_METRICS),
            "video_recording": not args.no_video,
            **evaluation,
        })
        s = evaluation["summary"]
        print(f"[eval] team_return={s['team_return_mean']:.3f}  "
              f"musical_f1={s['episode_task/musical_f1_mean']:.6f}  "
              f"precision={s['episode_task/musical_precision_mean']:.6f}  "
              f"recall={s['episode_task/musical_recall_mean']:.6f}")
        coordination_summary_keys = [f"{key}_mean" for key in COORDINATION_RATE_METRICS]
        if all(key in s for key in coordination_summary_keys):
            print(
                "[eval coordination] "
                f"common_success={s[coordination_summary_keys[0]]:.6f}  "
                f"duplicate_press={s[coordination_summary_keys[1]]:.6f}  "
                f"inter_agent_collision={s[coordination_summary_keys[2]]:.6f}"
            )
        print(f"[eval] summary={output_path}")
    except Exception as exc:
        write_json(output_dir / "failure.json", {
            "stage": "checkpoint_load_or_evaluation",
            "checkpoint_path": str(checkpoint),
            "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        active = sys.exc_info()[0] is not None
        cleanup: Optional[Exception] = None
        if algo is not None:
            try:
                algo.stop()
            except Exception as exc:
                if not active:
                    cleanup = exc
        if started:
            try:
                ray.shutdown()
            except Exception as exc:
                if not active and cleanup is None:
                    cleanup = exc
        if cleanup is not None:
            raise cleanup
    return 0


if __name__ == "__main__":
    sys.exit(main())
