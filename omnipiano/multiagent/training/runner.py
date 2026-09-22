"""Training orchestration for every registered OmniPiano MARL baseline.

One training loop, one bookkeeping path, one W&B integration.  The algorithm
contributes only an ``AlgoSpec`` (see ``algos/base.py``), so IPPO, MAPPO and any
future baseline are guaranteed to differ *only* where their spec differs.

Canonical invocation::

    python -m omnipiano.multiagent.train --algo mappo --seed 0
    python -m omnipiano.multiagent.train path/to/experiment.json

All numerical defaults come from ``marl_train_config_default.json``. Pass
``--config path/to/variant.json`` for an experiment configuration; explicit
CLI flags override the selected JSON values.

Every deviation from an RLlib default or from ``BenchmarkProtocolConfig`` is
recorded with its rationale in ``run_config.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent.algos import AlgoSpec, algo_table
from omnipiano.multiagent.compile import ResolvedTask
from omnipiano.multiagent.training.config import _parse_args, _resolve_args
from omnipiano.multiagent.training.rllib import (
    RLLIB_TARGET_VERSION,
    build_ppo_config,
    probe_agent_spaces,
    register_rllib_envs,
)
from omnipiano.multiagent.training.paths import logs_root, resolve_run_path
from omnipiano.multiagent.training.runtime import (
    COORDINATION_RATE_METRICS,
    REQUIRED_MUSICAL_METRICS,
    append_jsonl,
    checkpoint_reference,
    crossed_eval_targets,
    evaluate_marl,
    extract_env_steps,
    next_periodic_target,
    save_algorithm_checkpoint,
    write_json,
)



def _install_sigterm_as_interrupt() -> None:
    """Route SIGTERM through the same graceful path as Ctrl-C.

    A multi-hour run is far more likely to be stopped by a launcher script, a
    job scheduler or a container shutdown -- all of which send SIGTERM -- than
    by a keyboard. Python's default SIGTERM handler terminates the process
    outright, so the ``finally`` block never runs, W&B never flushes, and a
    perfectly healthy run is displayed as crashed.

    SIGHUP is deliberately NOT handled: ``nohup`` sets it to SIG_IGN, and
    installing a handler would un-ignore it, killing the run when the terminal
    closes -- the exact opposite of what nohup is for.
    """
    import signal

    def _handler(signum, frame):  # noqa: ANN001
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):  # not the main thread / unsupported platform
        pass


# ===========================================================================
# Artifact helpers
# ===========================================================================


def _env_token(env_id: str) -> str:
    return "_".join(
        part.lower() for part in env_id.split("-")
        if part and part != "OmniPiano" and not part.startswith("v")
    )


def _prepare_run_dir(requested: Optional[str], algo: str, env_id: str, seed: int) -> Path:
    if requested is not None:
        run_dir = resolve_run_path(requested)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = logs_root() / f"{algo}_{_env_token(env_id)}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    protected = (
        "run_config.json", "progress.jsonl", "periodic_eval.jsonl",
        "eval_summary.json", "checkpoint_path.txt", "final_checkpoint_path.txt",
        "latest_checkpoint_path.txt", "checkpoints.jsonl", "videos.jsonl", "videos",
        "recovery", "latest_recovery_checkpoint.json",
    )
    existing = [n for n in protected if (run_dir / n).exists()]
    if existing:
        raise FileExistsError(
            f"run_dir {run_dir} already contains run artifacts {existing}; choose "
            "a fresh directory so two configurations cannot be mixed into one curve"
        )
    return run_dir


def _space_metadata(space: Any) -> Dict[str, Any]:
    return {"type": type(space).__name__, "shape": list(space.shape),
            "dtype": str(space.dtype)}


def _completed_eval_videos(
    video_dir: Path, *, expected_episodes: int
) -> Sequence[Path]:
    """Return finished MP4s and fail if recording silently produced the wrong count."""
    paths = sorted(Path(video_dir).glob("*.mp4"))
    if len(paths) != int(expected_episodes):
        raise RuntimeError(
            f"evaluation video recording in {video_dir} produced {len(paths)} MP4(s); "
            f"expected {expected_episodes}"
        )
    return paths


def _git_metadata() -> Dict[str, Any]:
    from omnipiano.multiagent.training.paths import repo_root

    def _run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=str(repo_root()), check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True).stdout.strip()
    try:
        return {"branch": _run("branch", "--show-current"),
                "commit": _run("rev-parse", "HEAD"),
                "dirty": bool(_run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"branch": None, "commit": None, "dirty": None}


def _protocol_compliance(
    args: argparse.Namespace, proto: BenchmarkProtocolConfig
) -> Dict[str, bool]:
    """Split compliance into what changes the POLICY vs only the reported curve.

    Evaluation cadence does not affect the trained policy, so recording one
    boolean for both would mark a perfectly valid result non-compliant merely
    because it was evaluated less often.
    """
    training = bool(
        args.seed in proto.seeds
        and args.total_steps == proto.total_env_steps
        and args.gamma == proto.gamma
        and not args.smoke_test
    )
    eval_cadence = bool(
        args.eval_freq == proto.eval_freq_env_steps
        and args.num_eval_eps == proto.num_eval_eps
    )
    return {
        "protocol_training_compliant": training,
        "protocol_eval_cadence_compliant": eval_cadence,
        # Historical field: AND of everything, kept for artifact compatibility.
        "protocol_settings_compliant": training and eval_cadence,
    }


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _training_telemetry(result: Mapping[str, Any]) -> Dict[str, Any]:
    from omnipiano.multiagent.training.tracking import learner_metrics

    runners = result.get("env_runners", result)
    if not isinstance(runners, Mapping):
        runners = {}
    # RLlib sums simultaneously-acting agents here. The explicit name prevents
    # misreporting it as the one-copy shared team return (they differ by ~N).
    agent_sum = runners.get("episode_return_mean", runners.get("episode_reward_mean"))
    telemetry = {
        "rllib_agent_sum_return_mean": _finite_or_none(agent_sum),
        "episode_length_mean": _finite_or_none(runners.get("episode_len_mean")),
    }
    # Keep local progress.jsonl as diagnostically complete as W&B.  This is
    # especially important for HAPPO: the compound factor is the observable
    # proof that the sequential update did not silently degrade to MAPPO.
    telemetry.update(learner_metrics(result))
    return telemetry


def _build_run_config(
    args: argparse.Namespace, spec: AlgoSpec,
    proto: BenchmarkProtocolConfig, run_dir: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "algo": spec.name,
        "algo_spec": spec.metadata(),
        "allow_experimental": bool(getattr(args, "allow_experimental", False)),
        "algorithm": spec.display_name,
        "algorithm_reference": spec.reference,
        "algorithm_notes": spec.notes,
        "is_centralized_critic": spec.is_centralized_critic,
        "source_script": "train.py",
        "training_config": {
            "path": args._training_config_path,
            "snapshot": args._training_config_snapshot,
        },
        "env_id": args.env_id,
        "resolved_task": getattr(args, "_resolved_task", None),
        "run_dir": str(run_dir),
        "seed": int(args.seed),
        "eval_seed": int(args.seed + args.eval_seed_offset),
        "protocol_version": proto.protocol_version,
        "protocol_replication_seeds": list(proto.seeds),
        **_protocol_compliance(args, proto),
        "replication_complete": False,
        "replication_status": "single_seed_replicate",
        "total_env_steps": int(args.total_steps),
        "requested_total_env_steps": int(args.total_steps),
        "num_eval_eps": int(args.num_eval_eps),
        "num_eval_episodes": int(args.num_eval_eps),
        "smoke_test": bool(args.smoke_test),
        "effective_config": {
            # --- algorithm identity ---
            "algo": spec.name,
            "backend": spec.backend,
            "critic_input": spec.critic_input,
            "rl_module": spec.rl_module,
            "learner_class": (
                # Placeholder: overwritten in main() with the class actually
                # installed, once spec.learner_class has been resolved.
                "unresolved"
            ),
            "include_global_state": bool(spec.needs_global_state),
            "hidden_sizes": list(args.hidden_sizes_parsed),
            "activation": args.activation,
            "hidden_orthogonal_gain": float(args.hidden_orthogonal_gain),
            "policy_output_gain": float(args.policy_output_gain),
            "value_output_gain": float(args.value_output_gain),
            "initial_log_std": float(args.initial_log_std),
            "log_std_min": float(args.log_std_min),
            "log_std_max": float(args.log_std_max),
            "input_layer_norm": bool(args.input_layer_norm),
            "value_norm": bool(args.value_norm),
            "value_norm_beta": float(args.value_norm_beta),
            "value_norm_epsilon": float(args.value_norm_epsilon),
            "value_norm_variance_floor": float(
                args.value_norm_variance_floor
            ),
            # --- protocol ---
            "gamma": float(args.gamma),
            "eval_freq_env_steps": int(args.eval_freq),
            "count_steps_by": "env_steps",
            "inter_agent_collision_penalty": {
                "coefficient": float(
                    args.inter_agent_collision_penalty_coef
                ),
                "form": "-coefficient * 1[any cross-agent hand-geom contact]",
                "scope": "all active collision geoms owned by different agents",
                "frequency": "once per control step before shared-reward broadcast",
                "aligned_metric": (
                    "episode_coordination/inter_agent_collision_step_rate"
                ),
            },
            # Scalar alias retained for simple checkpoint re-evaluation and
            # command-line audit code.
            "inter_agent_collision_penalty_coef": float(
                args.inter_agent_collision_penalty_coef
            ),
            # --- ppo ---
            "train_batch_size": int(args.train_batch_size),
            "minibatch_size": int(args.minibatch_size),
            "num_epochs": int(args.num_epochs),
            "learning_rate": float(args.lr),
            "critic_learning_rate": float(args.critic_lr),
            "adam_epsilon": float(args.adam_epsilon),
            "separate_actor_critic_optimizers": bool(spec.rl_module == "ctde"),
            "gae_lambda": float(args.gae_lambda),
            "clip_param": float(args.clip_param),
            "vf_clip_param": float(args.vf_clip_param),
            "value_clip_semantics": (
                "old_normalized_prediction_delta_then_max_loss"
                if spec.rl_module == "ctde"
                else "rllib_squared_error_ceiling"
            ),
            "vf_loss_coeff": float(args.vf_loss_coeff),
            "entropy_coeff": float(args.entropy_coeff),
            "use_kl_loss": bool(args.use_kl_loss),
            "grad_clip": float(args.grad_clip),
            "grad_clip_by": args.grad_clip_by,
            "rllib_default_overrides": {
                "lambda_": [1.0, float(args.gae_lambda),
                            "standard PPO/MAPPO GAE setting"],
                "clip_param": [0.3, float(args.clip_param),
                               "PPO paper and MAPPO ablation both favour 0.2"],
                "value_clip_semantics": [
                    "RLlib squared-error ceiling",
                    "old prediction delta clip + pessimistic max loss",
                    "official PPO/MAPPO semantics; preserves critic gradients",
                ],
                "use_kl_loss": [True, bool(args.use_kl_loss),
                                "MAPPO reference implementation uses clipping only"],
                "grad_clip": [None, float(args.grad_clip),
                              "MAPPO uses max_grad_norm=10; applied separately "
                              "to actor and critic"],
                "minibatch_size": [128, int(args.minibatch_size),
                                   "one full-batch minibatch per MAPPO epoch"],
                "num_epochs": [30, int(args.num_epochs),
                               "conservative end of MAPPO's 5-15 epoch guidance"],
                "adam_epsilon": [1e-8, float(args.adam_epsilon),
                                 "official MAPPO optimizer epsilon"],
            },
            # --- env / compute ---
            "flatten_obs": True,
            "reward_mode": "shared",
            "framework": "torch",
            "num_env_runners": int(args.num_workers),
            "num_learners": int(args.num_learners),
            "num_gpus_per_learner": float(args.num_gpus_per_learner),
            "ray_num_cpus": args.ray_num_cpus,
            "sample_timeout_s": float(args.sample_timeout_s),
            "max_sampling_stalls": int(args.max_sampling_stalls),
            "checkpoint_freq_env_steps": int(args.checkpoint_freq),
            "evaluation_video": {
                "enabled": bool(args.video_enabled),
                "freq_env_steps": int(args.video_freq),
                "record_final": bool(args.video_record_final),
                "camera_id": str(args.video_camera_id),
                "resolution": [int(args.video_height), int(args.video_width)],
                "wandb_upload": bool(args.video_wandb_upload),
            },
            "algorithm_seed": int(args.seed),
            "worker_seeding": ("env seed = seed + 1000*worker_index + "
                               "vector_index; torch/module seeding via "
                               "AlgorithmConfig.debugging(seed=...)"),
            "rllib_target_version": RLLIB_TARGET_VERSION,
        },
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "git": _git_metadata(),
    }


# ===========================================================================
# Main
# ===========================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    _install_sigterm_as_interrupt()
    if args.list_algos:
        print(algo_table())
        return 0
    if args.list_envs:
        from omnipiano.multiagent import list_parallel_envs
        print("\n".join(list_parallel_envs()))
        return 0

    spec = _resolve_args(args)
    from omnipiano.multiagent.compile.environment import (
        prepare_task,
        resolve_registered_task,
    )
    if getattr(args, "_resolved_task", None) is None:
        prepared = resolve_registered_task(args.env_id)
    else:
        prepared = prepare_task(ResolvedTask.from_dict(args._resolved_task))
    args._resolved_task = prepared.to_dict()

    if args.dry_run:
        # Environment/task constructors contain a few legacy informational
        # prints. Keep stdout machine-readable JSON and route those messages
        # to stderr during inspection.
        with contextlib.redirect_stdout(sys.stderr):
            agents, obs_spaces, act_spaces, layouts = probe_agent_spaces(
                args.env_id,
                args.seed,
                spec.needs_global_state,
                task=args._resolved_task,
            )
        spec.assert_agent_count(len(agents))
        print(json.dumps({
            "status": "ok",
            "mode": "dry-run",
            "algorithm": spec.metadata(),
            "env_id": args.env_id,
            "resolved_task": args._resolved_task,
            "agents": list(agents),
            "observation_spaces": {
                agent: _space_metadata(obs_spaces[agent]) for agent in agents
            },
            "action_spaces": {
                agent: _space_metadata(act_spaces[agent]) for agent in agents
            },
            "obs_layout": {
                agent: {key: list(value) for key, value in layouts[agent].items()}
                for agent in agents
            },
            "gpu_or_cluster_started": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.eval_native:
        from omnipiano.multiagent.training.native import evaluate_saved
        return evaluate_saved(args)
    proto = BenchmarkProtocolConfig()
    run_dir = _prepare_run_dir(args.run_dir, spec.name, args.env_id, args.seed)
    run_config = _build_run_config(args, spec, proto, run_dir)
    write_json(run_dir / "run_config.json", run_config)
    for name in (
        "progress.jsonl", "periodic_eval.jsonl", "checkpoints.jsonl",
        "videos.jsonl",
    ):
        (run_dir / name).touch(exist_ok=False)

    tag = spec.name
    print(f"[{tag}] algo={spec.display_name}")
    print(f"[{tag}] critic_input={spec.critic_input}  "
          f"global_state={spec.needs_global_state}  rl_module={spec.rl_module}")
    print(f"[{tag}] env_id={args.env_id}")
    print(f"[{tag}] seed={args.seed}  steps={args.total_steps:,}  "
          f"gamma={args.gamma}  eval_every={args.eval_freq:,} env-steps")
    print(
        f"[{tag}] inter_agent_collision_penalty="
        f"-{args.inter_agent_collision_penalty_coef:g} per collision step"
    )
    if args.video_enabled:
        print(
            f"[{tag}] eval_video_every={args.video_freq:,} env-steps  "
            f"camera={args.video_camera_id}  "
            f"resolution={args.video_width}x{args.video_height}  "
            f"wandb_upload={args.video_wandb_upload}"
        )
    else:
        print(f"[{tag}] eval_video=disabled")
    print(f"[{tag}] run_dir={run_dir}")

    # ---- W&B: fail fast here, before any GPU hour is spent ----
    from omnipiano.multiagent.training.tracking import WandbRun

    penalty_token = format(
        float(args.inter_agent_collision_penalty_coef), ".8g"
    ).replace("-", "m").replace(".", "p")
    default_group = f"{spec.name}__{_env_token(args.env_id)}"
    if args.inter_agent_collision_penalty_coef > 0.0:
        default_group += f"__inter_collision_{penalty_token}"
    group = args.wandb_group or default_group
    tags = sorted({t.strip() for t in args.wandb_tags.split(",") if t.strip()} | {
        spec.name,
        f"critic-{spec.critic_input}",
        _env_token(args.env_id),
        f"seed{args.seed}",
        (f"{args.total_steps // 1_000_000}M" if args.total_steps >= 1_000_000
         else f"{args.total_steps // 1_000}k"),
        f"inter-collision-penalty-{penalty_token}",
        *(["smoke-test"] if args.smoke_test else []),
    })
    wandb_run = WandbRun(
        mode=args.wandb_mode, entity=args.wandb_entity, project=args.wandb_project,
        name=args.wandb_name or run_dir.name, group=group,
        job_type="smoke-test" if args.smoke_test else "train",
        tags=tags, notes=args.wandb_notes, config=run_config, run_dir=run_dir,
    )
    run_config["wandb"] = {
        "mode": args.wandb_mode, "entity": args.wandb_entity,
        "project": args.wandb_project, "group": group,
        "run_id": wandb_run.run_id, "url": wandb_run.url, "tags": tags,
    }
    write_json(run_dir / "run_config.json", run_config)

    ray = None
    try:
        import gymnasium
        import pettingzoo
        import torch
        if not spec.is_native:
            import ray as ray_module
            ray = ray_module
    except ImportError as exc:
        write_json(run_dir / "failure.json",
                   {"stage": "dependency_import", "error": f"{type(exc).__name__}: {exc}"})
        wandb_run.finish(exit_code=1)
        dependency = (
            "PettingZoo and torch" if spec.is_native
            else "ray[rllib], PettingZoo and torch"
        )
        raise RuntimeError(
            f"multi-agent {spec.backend} training requires {dependency}: "
            "pip install -e '.[marl]'"
        ) from exc

    if args.num_gpus_per_learner > 0 and not torch.cuda.is_available():
        wandb_run.finish(exit_code=1)
        raise RuntimeError(
            "--num-gpus-per-learner > 0 but torch.cuda.is_available() is False. "
            "Pass --num-gpus-per-learner 0, or check CUDA_VISIBLE_DEVICES."
        )

    if not spec.is_native:
        register_rllib_envs()

    agents, obs_spaces, act_spaces, layouts = probe_agent_spaces(
        args.env_id, args.seed, spec.needs_global_state,
        task=getattr(args, "_resolved_task", None),
    )
    spec.assert_agent_count(len(agents))
    run_config.update({
        "agents": list(agents),
        "policy_mapping": {a: a for a in agents},
        "observation_spaces": {a: _space_metadata(obs_spaces[a]) for a in agents},
        "action_spaces": {a: _space_metadata(act_spaces[a]) for a in agents},
        "obs_layout": {a: {k: list(v) for k, v in layouts[a].items()} for a in agents},
        "library_versions": {
            "ray": (ray.__version__ if ray is not None else None),
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "pettingzoo": getattr(pettingzoo, "__version__", "unknown"),
        },
    })
    if ray is not None and ray.__version__ != RLLIB_TARGET_VERSION:
        print(f"[{tag} warning] targets ray[rllib]=={RLLIB_TARGET_VERSION}, "
              f"found {ray.__version__} (recorded in run_config.json)")
    write_json(run_dir / "run_config.json", run_config)

    if spec.is_native:
        from omnipiano.multiagent.training.native import build_native_config
        config, learner_cls = build_native_config(
            args, spec, agents, obs_spaces, act_spaces, layouts
        )
    else:
        config, learner_cls = build_ppo_config(
            args, spec, agents, obs_spaces, act_spaces, layouts
        )

    # Record what was ACTUALLY installed, not what the branch implies.
    default_learner = (
        "OmniPiano native joint learner"
        if spec.is_native else "RLlib default PPOTorchLearner"
    )
    run_config["effective_config"]["learner_class"] = (
        learner_cls.__name__ if learner_cls is not None else default_learner
    )
    run_config["effective_config"]["learner_class_module"] = (
        learner_cls.__module__ if learner_cls is not None
        else ("omnipiano.multiagent.algos._native" if spec.is_native else "ray.rllib")
    )
    print(f"[{tag}] learner_class={run_config['effective_config']['learner_class']}")
    write_json(run_dir / "run_config.json", run_config)

    algo = None
    ray_started = False
    start = time.time()
    periodic_evals: list = []
    periodic_ckpts: list = []
    total_steps = 0
    iterations = 0
    next_eval = args.eval_freq
    next_ckpt: Optional[int] = args.checkpoint_freq if args.checkpoint_freq > 0 else None
    last_eval_step: Optional[int] = None
    final_eval: Optional[Dict[str, Any]] = None
    final_ckpt: Optional[str] = None
    final_ckpt_ref: Optional[str] = None
    interrupted = False
    last_video_step: Optional[int] = None
    recorded_videos: list = []
    pending_final_video_record: Optional[Dict[str, Any]] = None

    def _evaluate(record_dir: Optional[Path] = None) -> Dict[str, Any]:
        # include_global_state MUST match training: a MAPPO policy expects the
        # wider [global_state | own] vector even though its actor ignores the
        # global block.
        return evaluate_marl(
            algo, args.env_id,
            task=getattr(args, "_resolved_task", None),
            eval_seed=args.seed + args.eval_seed_offset,
            num_episodes=args.num_eval_eps,
            record_dir=str(record_dir) if record_dir is not None else None,
            record_resolution=(args.video_height, args.video_width),
            camera_id=args.video_camera_id,
            include_global_state=bool(spec.needs_global_state),
            inter_agent_collision_penalty_coef=float(
                args.inter_agent_collision_penalty_coef
            ),
        )

    def _finish_video_recording(
        video_dir: Path,
        *,
        scheduled_env_step: Optional[int],
        actual_env_step: int,
    ) -> Dict[str, Any]:
        video_paths = _completed_eval_videos(
            video_dir, expected_episodes=args.num_eval_eps
        )
        record = {
            "scheduled_env_step": (
                int(scheduled_env_step)
                if scheduled_env_step is not None
                else None
            ),
            "actual_env_step": int(actual_env_step),
            "video_dir": str(video_dir.relative_to(run_dir)),
            "video_files": [
                str(path.relative_to(run_dir)) for path in video_paths
            ],
            "wandb_upload_requested": bool(args.video_wandb_upload),
        }
        append_jsonl(run_dir / "videos.jsonl", record)
        recorded_videos.append(record)
        return record

    def _upload_video_record(record: Mapping[str, Any]) -> None:
        if not args.video_wandb_upload:
            return
        wandb_run.log_eval_videos(
            int(record["actual_env_step"]),
            [run_dir / path for path in record["video_files"]],
            scheduled_env_step=record.get("scheduled_env_step"),
        )

    try:
        if ray is not None:
            ray.init(
                ignore_reinit_error=True,
                log_to_driver=bool(args.ray_log_to_driver),
                num_cpus=args.ray_num_cpus,
                include_dashboard=False,
            )
            ray_started = True
        build = getattr(config, "build_algo", None)
        algo = build() if callable(build) else config.build()
        if spec.is_native:
            run_config["effective_config"] = algo.effective_config()
            run_config["effective_config"]["learner_class"] = default_learner
            run_config["effective_config"]["learner_class_module"] = (
                "omnipiano.multiagent.algos._native"
            )
            write_json(run_dir / "run_config.json", run_config)
            print(f"[{tag}] native {spec.display_name} learner built; starting training")
        else:
            print(f"[{tag}] RLlib PPO built; starting training")

        if not spec.is_native and spec.rl_module == "ctde":
            # Snapshot the realized actor/critic parameter split so an
            # IPPO/MAPPO pair can be verified as architecturally matched
            # after the fact.
            try:
                run_config["ctde_modules"] = {
                    a: algo.get_module(a).ctde_spec for a in agents
                }
                write_json(run_dir / "run_config.json", run_config)
                for a in agents:
                    s = run_config["ctde_modules"][a]
                    print(f"[{tag}] module[{a}] actor_params={s['actor_params']:,} "
                          f"critic_params={s['critic_params']:,} "
                          f"critic_in_dim={s['critic_in_dim']}")
            except Exception as exc:
                print(f"[{tag} warning] could not snapshot CTDE specs: {exc}")

        # Native checkpoints restore lifetime counters. Continue periodic
        # schedules after the restored point; restarting them from step zero
        # would make the first post-resume iteration cross every old target.
        restored_steps = (
            int(getattr(algo, "env_steps", 0)) if spec.is_native else 0
        )
        if restored_steps:
            total_steps = restored_steps
            next_eval = next_periodic_target(total_steps, args.eval_freq)
            if next_ckpt is not None:
                next_ckpt = next_periodic_target(
                    total_steps, args.checkpoint_freq
                )

        session_start_steps = total_steps
        previous = total_steps
        stalls = 0
        while total_steps < args.total_steps:
            result = algo.train()
            iterations += 1
            total_steps = extract_env_steps(result)
            if total_steps <= previous:
                stalls += 1
                print(f"[{tag} warning] iteration {iterations} produced no new env "
                      f"steps ({previous} -> {total_steps}); sampling stall "
                      f"{stalls}/{args.max_sampling_stalls}. RLlib discards any env "
                      f"runner slower than --sample-timeout-s "
                      f"({args.sample_timeout_s:.0f}s); raise it or lower "
                      f"--num-workers.")
                if stalls >= args.max_sampling_stalls:
                    raise RuntimeError(
                        f"no new env steps for {args.max_sampling_stalls} consecutive "
                        f"iterations (stuck at {total_steps}). Sampling is broken, "
                        f"not merely slow. Reproduce in-process with "
                        f"--num-workers 0 --ray-log-to-driver to see the real "
                        f"exception.")
                continue
            stalls = 0
            previous = total_steps

            telemetry = _training_telemetry(result)
            row = {"iteration": iterations, "env_steps": total_steps,
                   "wall_seconds": round(time.time() - start, 3), **telemetry}
            should_log = (
                iterations == 1
                or iterations % args.log_every_iters == 0
                or total_steps >= args.total_steps
            )
            if should_log:
                # Shared-filesystem JSONL and W&B calls are observability, not
                # part of the learner.  Respect the configured cadence instead
                # of performing both on every native 24-step update block.
                append_jsonl(run_dir / "progress.jsonl", row)
                wandb_run.log_train(total_steps, row, result)
                elapsed = max(time.time() - start, 1e-9)
                sps = (total_steps - session_start_steps) / elapsed
                eta_h = ((args.total_steps - total_steps) / sps / 3600.0
                         if sps > 0 else float("inf"))
                print(f"[{tag} iter {iterations:5d}] env_steps={total_steps:>10,}  "
                      f"{sps:7.1f} steps/s  wall={elapsed/3600:5.2f}h  "
                      f"eta={eta_h:6.2f}h  "
                      f"agent_sum_return={telemetry['rllib_agent_sum_return_mean']}")

            # Recovery checkpoints BEFORE evaluation: if inference or metric
            # extraction is broken, the policy that triggered it survives.
            if next_ckpt is not None:
                crossed, next_ckpt = crossed_eval_targets(
                    total_steps, next_ckpt, args.checkpoint_freq)
                if len(crossed) > 1:
                    raise RuntimeError("one iteration crossed multiple checkpoint "
                                       f"thresholds {crossed}; raise checkpoint_freq")
                for scheduled in crossed:
                    if total_steps >= args.total_steps:
                        continue  # the final save below supersedes it
                    path = save_algorithm_checkpoint(
                        algo,
                        run_dir / "checkpoints" / f"step_{total_steps:09d}",
                        native_replay_transitions=(0 if spec.is_native else None),
                    )
                    ref = checkpoint_reference(path, run_dir)
                    record = {"scheduled_env_step": int(scheduled),
                              "actual_env_step": int(total_steps),
                              "checkpoint_path": ref}
                    recovery_tail = int(
                        getattr(algo, "options", {}).get(
                            "checkpoint_replay_transitions", 0
                        )
                    )
                    if spec.is_native and recovery_tail > 0:
                        recovery_root = run_dir / "recovery"
                        recovery_path = save_algorithm_checkpoint(
                            algo,
                            recovery_root / f"step_{total_steps:09d}",
                            native_replay_transitions=recovery_tail,
                        )
                        recovery_ref = checkpoint_reference(
                            recovery_path, run_dir
                        )
                        record["recovery_checkpoint_path"] = recovery_ref
                        record["recovery_replay_transitions"] = min(
                            int(getattr(algo.replay, "size", 0)), recovery_tail
                        )
                        write_json(
                            run_dir / "latest_recovery_checkpoint.json",
                            {
                                "actual_env_step": int(total_steps),
                                "checkpoint_path": recovery_ref,
                                "replay_transitions": record[
                                    "recovery_replay_transitions"
                                ],
                            },
                        )
                        resolved_recovery = Path(recovery_path).resolve()
                        for old in recovery_root.glob("step_*"):
                            if old.resolve() != resolved_recovery:
                                shutil.rmtree(old)
                    append_jsonl(run_dir / "checkpoints.jsonl", record)
                    periodic_ckpts.append(record)
                    (run_dir / "latest_checkpoint_path.txt").write_text(ref + "\n",
                                                                       encoding="utf-8")
                    print(f"[{tag} checkpoint] actual={total_steps:,} path={ref}")

            if total_steps >= args.total_steps and final_ckpt is None:
                final_ckpt = save_algorithm_checkpoint(
                    algo,
                    run_dir / "checkpoints" / "final",
                    native_replay_transitions=(0 if spec.is_native else None),
                )
                final_ckpt_ref = checkpoint_reference(final_ckpt, run_dir)
                for name in ("final_checkpoint_path.txt",
                             "latest_checkpoint_path.txt",
                             "checkpoint_path.txt"):
                    (run_dir / name).write_text(final_ckpt_ref + "\n", encoding="utf-8")
                print(f"[{tag} checkpoint] final path={final_ckpt_ref}")

            crossed, next_eval = crossed_eval_targets(
                total_steps, next_eval, args.eval_freq)
            if len(crossed) > 1:
                raise RuntimeError("one iteration crossed multiple protocol eval "
                                   f"thresholds {crossed}; reduce train_batch_size")
            for scheduled in crossed:
                record_video = bool(
                    args.video_enabled and scheduled % args.video_freq == 0
                )
                video_dir = (
                    run_dir / "videos" / f"step_{scheduled:09d}"
                    if record_video else None
                )
                if video_dir is not None:
                    video_dir.mkdir(parents=True, exist_ok=False)
                evaluation = _evaluate(video_dir)
                video_record = None
                if video_dir is not None:
                    video_record = _finish_video_recording(
                        video_dir,
                        scheduled_env_step=scheduled,
                        actual_env_step=total_steps,
                    )
                    last_video_step = total_steps
                    evaluation["video"] = video_record
                append_jsonl(run_dir / "periodic_eval.jsonl",
                             {"scheduled_env_step": int(scheduled),
                              "actual_env_step": int(total_steps), **evaluation})
                wandb_run.log_eval(total_steps, evaluation,
                                   scheduled_env_step=scheduled)
                if video_record is not None:
                    _upload_video_record(video_record)
                periodic_evals.append(evaluation)
                final_eval = evaluation
                last_eval_step = total_steps
                s = evaluation["summary"]
                print(f"[{tag} eval] actual={total_steps:,} "
                      f"team_return={s['team_return_mean']:.3f} "
                      f"musical_f1={s['episode_task/musical_f1_mean']:.6f}")
                coordination_summary_keys = [
                    f"{key}_mean" for key in COORDINATION_RATE_METRICS
                ]
                if all(key in s for key in coordination_summary_keys):
                    print(
                        f"[{tag} coordination] "
                        f"common_success={s[coordination_summary_keys[0]]:.6f} "
                        f"duplicate_press={s[coordination_summary_keys[1]]:.6f} "
                        f"inter_agent_collision={s[coordination_summary_keys[2]]:.6f}"
                    )

        if final_ckpt is None or final_ckpt_ref is None:
            raise RuntimeError("training completed without a final checkpoint")
        needs_final_video = bool(
            args.video_enabled
            and args.video_record_final
            and last_video_step != total_steps
        )
        if last_eval_step != total_steps or needs_final_video:
            final_video_dir = (
                run_dir / "videos" / f"final_step_{total_steps:09d}"
                if needs_final_video else None
            )
            if final_video_dir is not None:
                final_video_dir.mkdir(parents=True, exist_ok=False)
            final_eval = _evaluate(final_video_dir)
            if final_video_dir is not None:
                final_video_record = _finish_video_recording(
                    final_video_dir,
                    scheduled_env_step=None,
                    actual_env_step=total_steps,
                )
                last_video_step = total_steps
                final_eval["video"] = final_video_record
                pending_final_video_record = final_video_record
        if final_eval is None:
            raise RuntimeError("training completed without a final evaluation")

        cumulative_iterations = (
            int(getattr(algo, "iterations", iterations))
            if spec.is_native else int(iterations)
        )
        eval_summary = {
            "schema_version": 2,
            "algo": spec.name,
            "algorithm": spec.display_name,
            "is_centralized_critic": spec.is_centralized_critic,
            "source_script": "train.py",
            "env_id": args.env_id,
            "seed": int(args.seed),
            "eval_seed": int(args.seed + args.eval_seed_offset),
            "protocol_version": proto.protocol_version,
            "protocol_replication_seeds": list(proto.seeds),
            **_protocol_compliance(args, proto),
            "replication_complete": False,
            "replication_status": "single_seed_replicate",
            "total_env_steps": int(args.total_steps),
            "requested_total_env_steps": int(args.total_steps),
            "actual_total_env_steps": int(total_steps),
            "trained_env_steps": int(total_steps),
            "training_iterations": cumulative_iterations,
            "iterations_in_this_process": int(iterations),
            "num_eval_eps": int(args.num_eval_eps),
            "num_eval_episodes": int(args.num_eval_eps),
            "checkpoint_path": final_ckpt_ref,
            "checkpoint_path_resolved": final_ckpt,
            "periodic_eval_file": "periodic_eval.jsonl",
            "periodic_eval_count": len(periodic_evals),
            "periodic_checkpoint_file": "checkpoints.jsonl",
            "periodic_checkpoints": periodic_ckpts,
            "evaluation_video_file": "videos.jsonl",
            "evaluation_videos": recorded_videos,
            "final_evaluation": final_eval,
            "episodes": final_eval["episodes"],
            "summary": final_eval["summary"],
            "wall_seconds": round(time.time() - start, 3),
            "effective_config": run_config["effective_config"],
            "library_versions": run_config["library_versions"],
            "required_terminal_metrics": list(REQUIRED_MUSICAL_METRICS),
            "reported_coordination_metrics": list(COORDINATION_RATE_METRICS),
            "smoke_test": bool(args.smoke_test),
        }
        write_json(run_dir / "eval_summary.json", eval_summary)
        wandb_run.log_final(total_steps, eval_summary)
        if pending_final_video_record is not None:
            _upload_video_record(pending_final_video_record)
        if args.wandb_upload_artifacts:
            for name in ("run_config.json", "eval_summary.json", "progress.jsonl",
                         "periodic_eval.jsonl", "checkpoints.jsonl", "videos.jsonl"):
                wandb_run.log_artifact_dir(
                    run_dir / name,
                    name=f"{run_dir.name}__{name.replace('.', '_')}")
        print(f"[{tag}] final checkpoint: {final_ckpt_ref}")
        print(f"[{tag}] eval summary:    {run_dir / 'eval_summary.json'}")
        if wandb_run.url:
            print(f"[{tag}] wandb:           {wandb_run.url}")
    except KeyboardInterrupt:
        # A deliberate stop is NOT a failure. Write a distinct artifact and let
        # W&B close the run with exit_code=0, so a manually stopped run doesn't
        # sit next to genuinely broken runs wearing the same red "Failed" badge.
        interrupted = True
        write_json(run_dir / "interrupted.json", {
            "stage": "training_or_evaluation",
            "reason": "KeyboardInterrupt (Ctrl-C or SIGTERM)",
            "env_steps": int(total_steps),
            "training_iterations": int(iterations)})
        print(f"\n[{tag}] stopped by user at {total_steps:,} env steps "
              f"({iterations} iterations). Artifacts flushed. NOT a failure.")
    except Exception as exc:
        write_json(run_dir / "failure.json", {
            "stage": "training_or_evaluation",
            "error": f"{type(exc).__name__}: {exc}",
            "env_steps": int(total_steps), "training_iterations": int(iterations)})
        raise
    finally:
        active = sys.exc_info()[0] is not None and not interrupted
        try:
            wandb_run.finish(exit_code=1 if active else 0)
        except Exception:
            pass
        cleanup: Optional[Exception] = None
        if algo is not None:
            try:
                algo.stop()
            except Exception as exc:
                if not active:
                    cleanup = exc
        if ray_started:
            try:
                ray.shutdown()
            except Exception as exc:
                if not active and cleanup is None:
                    cleanup = exc
        if cleanup is not None:
            raise cleanup
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
