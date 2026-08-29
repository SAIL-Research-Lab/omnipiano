"""Generic trainer for every registered OmniPiano multi-agent baseline.

One training loop, one bookkeeping path, one W&B integration.  The algorithm
contributes only an ``AlgoSpec`` (see ``algos/base.py``), so IPPO, MAPPO and any
future baseline are guaranteed to differ *only* where their spec differs.

Canonical invocation::

    python -m omnipiano.multiagent.train --algo mappo \\
        --env-id OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0 \\
        --seed 0 --total-steps 5000000

Every deviation from an RLlib default or from ``BenchmarkProtocolConfig`` is
recorded with its rationale in ``run_config.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent.algos import AlgoSpec, algo_table, get_algo, list_algos
from omnipiano.multiagent.paths import logs_root, resolve_run_path
from omnipiano.multiagent._ippo_common import (
    LEGACY_RLLIB_ENV_NAME,
    REQUIRED_MUSICAL_METRICS,
    RLLIB_ENV_NAME,
    append_jsonl,
    checkpoint_reference,
    crossed_eval_targets,
    evaluate_marl,
    extract_env_steps,
    save_algorithm_checkpoint,
    wrap_parallel_env_for_rllib,
    write_json,
)

DEFAULT_ENV_ID = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"
RLLIB_TARGET_VERSION = "2.55.1"


# ===========================================================================
# CLI
# ===========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    proto = BenchmarkProtocolConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    core = p.add_argument_group("experiment identity")
    core.add_argument("--algo", default="ippo", choices=list_algos(),
                     help="Registered baseline; see --list-algos.")
    core.add_argument("--list-algos", action="store_true",
                     help="Print the algorithm registry and exit.")
    core.add_argument("--list-envs", action="store_true",
                     help="Print every registered multi-agent env id and exit.")
    core.add_argument("--env-id", default=DEFAULT_ENV_ID)
    core.add_argument("--seed", type=int, default=proto.seed,
                     help=f"One training seed. Protocol replication set: {proto.seeds}.")
    core.add_argument("--run-dir", "--save-dir", dest="run_dir", default=None,
                     help="Run artifact directory (repo-relative paths allowed).")

    proto_g = p.add_argument_group("protocol (BenchmarkProtocolConfig)")
    proto_g.add_argument("--total-steps", type=int, default=proto.total_env_steps,
                         help=f"ENVIRONMENT-step budget (protocol {proto.total_env_steps:,}).")
    proto_g.add_argument("--gamma", type=float, default=proto.gamma)
    proto_g.add_argument("--eval-freq", type=int, default=proto.eval_freq_env_steps,
                         help="Deterministic-evaluation cadence in lifetime env steps.")
    proto_g.add_argument("--num-eval-eps", type=int, default=proto.num_eval_eps)
    proto_g.add_argument("--eval-seed-offset", type=int, default=10_000)

    ppo = p.add_argument_group("ppo (RLlib defaults corrected for this benchmark)")
    ppo.add_argument("--train-batch-size", type=int, default=4000)
    ppo.add_argument("--minibatch-size", type=int, default=256)
    ppo.add_argument("--num-epochs", type=int, default=10)
    ppo.add_argument("--lr", type=float, default=3e-4)
    ppo.add_argument("--gae-lambda", type=float, default=0.95,
                     help="RLlib default 1.0 = pure Monte-Carlo advantage.")
    ppo.add_argument("--clip-param", type=float, default=0.2,
                     help="RLlib default 0.3; PPO paper and MAPPO both use 0.2.")
    ppo.add_argument("--vf-clip-param", type=float, default=1000.0,
                     help="RLlib default 10.0 clamps the SQUARED value error, so "
                          "|V-target|>sqrt(10) gets ZERO gradient. Our returns "
                          "are ~600, which would disable critic learning.")
    ppo.add_argument("--vf-loss-coeff", type=float, default=1.0)
    ppo.add_argument("--entropy-coeff", type=float, default=0.0)
    ppo.add_argument("--use-kl-loss", action="store_true", default=False,
                     help="RLlib defaults this ON; MAPPO reference uses clip only.")
    ppo.add_argument("--grad-clip", type=float, default=10.0)

    net = p.add_argument_group("network")
    net.add_argument("--hidden-sizes", default="256,256")
    net.add_argument("--activation", choices=("tanh", "relu"), default="tanh")

    comp = p.add_argument_group("compute")
    comp.add_argument("--num-workers", type=int, default=4,
                      help="RLlib env runners (sampling processes).")
    comp.add_argument("--num-cpus-per-env-runner", type=int, default=1)
    comp.add_argument("--num-learners", type=int, default=1,
                      help="Pinned to 1: total batch size scales with this.")
    comp.add_argument("--num-gpus-per-learner", type=float, default=None,
                      help="Default 1.0 (0.0 under --smoke-test). Select the "
                           "device with CUDA_VISIBLE_DEVICES.")
    comp.add_argument("--ray-num-cpus", type=int, default=None,
                      help="Hard-cap Ray's CPU pool; required when two jobs "
                           "share one node.")
    comp.add_argument("--checkpoint-freq", type=int, default=500_000,
                      help="Recoverable checkpoint cadence in env steps (0=off).")
    comp.add_argument("--log-every-iters", type=int, default=5)
    comp.add_argument("--smoke-test", action="store_true",
                      help="5k-step ingestion/checkpoint/eval test; not a result.")

    wb = p.add_argument_group("weights & biases")
    wb.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                    default="online")
    wb.add_argument("--wandb-entity", default="omnipiano")
    wb.add_argument("--wandb-project", default="marl")
    wb.add_argument("--wandb-group", default=None,
                    help="Defaults to '<algo>__<env-token>' so seeds aggregate.")
    wb.add_argument("--wandb-name", default=None, help="Defaults to run-dir name.")
    wb.add_argument("--wandb-tags", default="")
    wb.add_argument("--wandb-notes", default=None)
    wb.add_argument("--wandb-upload-artifacts", action="store_true")
    return p


def _resolve_args(args: argparse.Namespace) -> AlgoSpec:
    """Apply algorithm-dependent and smoke-test defaults, then validate."""
    spec = get_algo(args.algo)

    if args.smoke_test:
        args.total_steps = 5_000
        args.eval_freq = 5_000
        args.num_eval_eps = 1
        args.num_workers = 0
        args.train_batch_size = 512
        args.minibatch_size = 64
        args.num_epochs = 2
        args.checkpoint_freq = 0
    if args.num_gpus_per_learner is None:
        args.num_gpus_per_learner = 0.0 if args.smoke_test else 1.0

    args.hidden_sizes_parsed = tuple(
        int(t) for t in str(args.hidden_sizes).split(",") if t.strip()
    )
    if not args.hidden_sizes_parsed or any(h <= 0 for h in args.hidden_sizes_parsed):
        raise ValueError(f"invalid --hidden-sizes {args.hidden_sizes!r}")

    positive = {
        "total_steps": args.total_steps, "num_eval_eps": args.num_eval_eps,
        "eval_freq": args.eval_freq, "train_batch_size": args.train_batch_size,
        "minibatch_size": args.minibatch_size, "num_epochs": args.num_epochs,
        "log_every_iters": args.log_every_iters,
    }
    bad = {k: v for k, v in positive.items() if v <= 0}
    if bad:
        raise ValueError(f"these CLI values must be positive: {bad}")
    if args.num_workers < 0 or args.checkpoint_freq < 0:
        raise ValueError("num_workers and checkpoint_freq must be non-negative")
    if not math.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.minibatch_size > args.train_batch_size:
        raise ValueError("minibatch_size cannot exceed train_batch_size")
    if args.train_batch_size > args.eval_freq:
        raise ValueError("train_batch_size > eval_freq: one PPO iteration could "
                         "skip multiple protocol evaluation thresholds")
    if 0 < args.checkpoint_freq < args.train_batch_size:
        raise ValueError("checkpoint_freq must be 0 or >= train_batch_size")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not 0.0 < args.clip_param < 1.0:
        raise ValueError("clip_param must be in (0, 1)")
    if args.vf_clip_param <= 0.0 or args.grad_clip <= 0.0:
        raise ValueError("vf_clip_param and grad_clip must be positive")
    if args.entropy_coeff < 0.0 or args.vf_loss_coeff < 0.0:
        raise ValueError("entropy_coeff and vf_loss_coeff must be non-negative")
    if args.num_learners != 1:
        raise ValueError(
            "this benchmark requires exactly one Learner: total_train_batch_size "
            "scales with num_learners and would silently change the recorded "
            "protocol batch size"
        )
    if args.num_gpus_per_learner < 0:
        raise ValueError("num_gpus_per_learner must be non-negative")
    return spec


# ===========================================================================
# Environment plumbing
# ===========================================================================


def make_env_for_rllib(env_config: Mapping[str, Any]) -> Any:
    """RLlib env creator. ``include_global_state`` arrives via env_config."""
    from omnipiano.multiagent import make_parallel

    parallel_env = make_parallel(
        str(env_config.get("env_id", DEFAULT_ENV_ID)),
        seed=int(env_config.get("seed", 0)),
        flatten_obs=True,
        include_global_state=bool(env_config.get("include_global_state", False)),
    )
    # RLlib permits only ``__common__`` as a non-agent info key; the native
    # PettingZoo env emits ``_global_``. Translate at this adapter boundary.
    return wrap_parallel_env_for_rllib(parallel_env)


def probe_agent_spaces(
    env_id: str, seed: int, include_global_state: bool
) -> Tuple[Sequence[str], Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Tuple[int, int]]]]:
    """Return agents, obs/action spaces, and the flat actor/critic slice layout."""
    from omnipiano.multiagent import make_parallel

    env = make_parallel(
        env_id, seed=seed, flatten_obs=True,
        include_global_state=include_global_state,
    )
    try:
        agents = list(env.possible_agents)
        obs = {a: env.observation_space(a) for a in agents}
        act = {a: env.action_space(a) for a in agents}
        layouts: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for agent in agents:
            layout = env.obs_layout(agent)
            if "own" in layout:
                own, glob = layout["own"], layout.get("global_state", (0, 0))
            else:
                # No global state: the whole vector is the agent's own obs.
                own, glob = (0, int(np.prod(obs[agent].shape))), (0, 0)
            layouts[agent] = {"own": own, "global_state": glob}
        return agents, obs, act, layouts
    finally:
        env.close()


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
        "latest_checkpoint_path.txt", "checkpoints.jsonl",
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


def _git_metadata() -> Dict[str, Any]:
    from omnipiano.multiagent.paths import repo_root

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


def _resolved_ppo_config(config: Any) -> Dict[str, Any]:
    raw = config.to_dict()
    keys = ("gamma", "use_critic", "use_gae", "lambda", "lambda_", "use_kl_loss",
            "kl_coeff", "kl_target", "clip_param", "vf_clip_param", "entropy_coeff",
            "vf_loss_coeff", "grad_clip", "grad_clip_by", "lr", "train_batch_size",
            "train_batch_size_per_learner", "minibatch_size", "num_epochs",
            "shuffle_batch_per_epoch", "rollout_fragment_length", "batch_mode",
            "normalize_actions", "clip_actions", "framework", "framework_str",
            "num_learners")
    out: Dict[str, Any] = {}
    for k in keys:
        if k not in raw:
            continue
        try:
            json.dumps(raw[k])
            out[k] = raw[k]
        except (TypeError, ValueError):
            out[k] = repr(raw[k])
    return out


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _training_telemetry(result: Mapping[str, Any]) -> Dict[str, Any]:
    runners = result.get("env_runners", result)
    if not isinstance(runners, Mapping):
        runners = {}
    # RLlib sums simultaneously-acting agents here. The explicit name prevents
    # misreporting it as the one-copy shared team return (they differ by ~N).
    agent_sum = runners.get("episode_return_mean", runners.get("episode_reward_mean"))
    return {
        "rllib_agent_sum_return_mean": _finite_or_none(agent_sum),
        "episode_length_mean": _finite_or_none(runners.get("episode_len_mean")),
    }


def _build_run_config(
    args: argparse.Namespace, spec: AlgoSpec,
    proto: BenchmarkProtocolConfig, run_dir: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "algo": spec.name,
        "algorithm": spec.display_name,
        "algorithm_reference": spec.reference,
        "algorithm_notes": spec.notes,
        "is_centralized_critic": spec.is_centralized_critic,
        "source_script": "train.py",
        "env_id": args.env_id,
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
            "critic_input": spec.critic_input,
            "rl_module": spec.rl_module,
            "include_global_state": bool(spec.needs_global_state),
            "hidden_sizes": list(args.hidden_sizes_parsed),
            "activation": args.activation,
            # --- protocol ---
            "gamma": float(args.gamma),
            "eval_freq_env_steps": int(args.eval_freq),
            "count_steps_by": "env_steps",
            # --- ppo ---
            "train_batch_size": int(args.train_batch_size),
            "minibatch_size": int(args.minibatch_size),
            "num_epochs": int(args.num_epochs),
            "learning_rate": float(args.lr),
            "gae_lambda": float(args.gae_lambda),
            "clip_param": float(args.clip_param),
            "vf_clip_param": float(args.vf_clip_param),
            "vf_loss_coeff": float(args.vf_loss_coeff),
            "entropy_coeff": float(args.entropy_coeff),
            "use_kl_loss": bool(args.use_kl_loss),
            "grad_clip": float(args.grad_clip),
            "grad_clip_by": "global_norm",
            "rllib_default_overrides": {
                "lambda_": [1.0, float(args.gae_lambda),
                            "GAE lambda=1 is pure Monte-Carlo; the variance is "
                            "unusable over ~1000-step episodes"],
                "clip_param": [0.3, float(args.clip_param),
                               "PPO paper and MAPPO ablation both favour 0.2"],
                "vf_clip_param": [10.0, float(args.vf_clip_param),
                                  "RLlib clamps the SQUARED value error; at "
                                  "return scale ~600 the default zeroes the "
                                  "critic gradient on nearly every sample"],
                "use_kl_loss": [True, bool(args.use_kl_loss),
                                "MAPPO reference implementation uses clipping only"],
                "grad_clip": [None, float(args.grad_clip),
                              "MAPPO uses max_grad_norm=10"],
            },
            # --- env / compute ---
            "flatten_obs": True,
            "reward_mode": "shared",
            "framework": "torch",
            "num_env_runners": int(args.num_workers),
            "num_learners": int(args.num_learners),
            "num_gpus_per_learner": float(args.num_gpus_per_learner),
            "ray_num_cpus": args.ray_num_cpus,
            "checkpoint_freq_env_steps": int(args.checkpoint_freq),
            "algorithm_seed": int(args.seed),
            "worker_seeding": ("RLlib AlgorithmConfig.debugging(seed=...); each "
                               "env runner uses seed + worker_index"),
            "rllib_target_version": RLLIB_TARGET_VERSION,
        },
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "git": _git_metadata(),
    }


# ===========================================================================
# Main
# ===========================================================================


def run_algorithm_entrypoint(
    algo_name: str, argv: Optional[Sequence[str]] = None
) -> int:
    """Run a legacy algorithm-specific module through the canonical trainer.

    ``_train_ippo`` and ``_train_mappo`` are public commands used by existing
    cluster manifests. They must force their advertised algorithm instead of
    accepting a conflicting second ``--algo`` value.
    """
    get_algo(algo_name)
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--algo" or arg.startswith("--algo=") for arg in forwarded):
        raise SystemExit(
            f"this entry point implies --algo {algo_name}; use "
            "python -m omnipiano.multiagent.train to select an algorithm"
        )
    return main(["--algo", algo_name, *forwarded])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.list_algos:
        print(algo_table())
        return 0
    if args.list_envs:
        from omnipiano.multiagent import list_parallel_envs
        print("\n".join(list_parallel_envs()))
        return 0

    spec = _resolve_args(args)
    proto = BenchmarkProtocolConfig()
    run_dir = _prepare_run_dir(args.run_dir, spec.name, args.env_id, args.seed)
    run_config = _build_run_config(args, spec, proto, run_dir)
    write_json(run_dir / "run_config.json", run_config)
    for name in ("progress.jsonl", "periodic_eval.jsonl", "checkpoints.jsonl"):
        (run_dir / name).touch(exist_ok=False)

    tag = spec.name
    print(f"[{tag}] algo={spec.display_name}")
    print(f"[{tag}] critic_input={spec.critic_input}  "
          f"global_state={spec.needs_global_state}  rl_module={spec.rl_module}")
    print(f"[{tag}] env_id={args.env_id}")
    print(f"[{tag}] seed={args.seed}  steps={args.total_steps:,}  "
          f"gamma={args.gamma}  eval_every={args.eval_freq:,} env-steps")
    print(f"[{tag}] run_dir={run_dir}")

    # ---- W&B: fail fast here, before any GPU hour is spent ----
    from omnipiano.multiagent._wandb import WandbRun

    group = args.wandb_group or f"{spec.name}__{_env_token(args.env_id)}"
    tags = sorted({t.strip() for t in args.wandb_tags.split(",") if t.strip()} | {
        spec.name,
        f"critic-{spec.critic_input}",
        _env_token(args.env_id),
        f"seed{args.seed}",
        (f"{args.total_steps // 1_000_000}M" if args.total_steps >= 1_000_000
         else f"{args.total_steps // 1_000}k"),
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

    try:
        import gymnasium
        import pettingzoo
        import ray
        import torch
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.tune.registry import register_env
    except ImportError as exc:
        write_json(run_dir / "failure.json",
                   {"stage": "dependency_import", "error": f"{type(exc).__name__}: {exc}"})
        wandb_run.finish(exit_code=1)
        raise RuntimeError("multi-agent training requires ray[rllib], PettingZoo "
                          "and torch: pip install -e '.[marl]'") from exc

    if args.num_gpus_per_learner > 0 and not torch.cuda.is_available():
        wandb_run.finish(exit_code=1)
        raise RuntimeError(
            "--num-gpus-per-learner > 0 but torch.cuda.is_available() is False. "
            "Pass --num-gpus-per-learner 0, or check CUDA_VISIBLE_DEVICES."
        )

    register_env(RLLIB_ENV_NAME, make_env_for_rllib)
    register_env(LEGACY_RLLIB_ENV_NAME, make_env_for_rllib)

    agents, obs_spaces, act_spaces, layouts = probe_agent_spaces(
        args.env_id, args.seed, spec.needs_global_state
    )
    run_config.update({
        "agents": list(agents),
        "policy_mapping": {a: a for a in agents},
        "observation_spaces": {a: _space_metadata(obs_spaces[a]) for a in agents},
        "action_spaces": {a: _space_metadata(act_spaces[a]) for a in agents},
        "obs_layout": {a: {k: list(v) for k, v in layouts[a].items()} for a in agents},
        "library_versions": {
            "ray": ray.__version__, "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "pettingzoo": getattr(pettingzoo, "__version__", "unknown"),
        },
    })
    if ray.__version__ != RLLIB_TARGET_VERSION:
        print(f"[{tag} warning] targets ray[rllib]=={RLLIB_TARGET_VERSION}, "
              f"found {ray.__version__} (recorded in run_config.json)")
    write_json(run_dir / "run_config.json", run_config)

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .environment(RLLIB_ENV_NAME, env_config={
            "env_id": args.env_id, "seed": args.seed,
            "include_global_state": bool(spec.needs_global_state),
        })
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers,
                     num_cpus_per_env_runner=args.num_cpus_per_env_runner)
        .learners(num_learners=args.num_learners,
                  num_gpus_per_learner=args.num_gpus_per_learner)
        .multi_agent(
            policies={a: (None, obs_spaces[a], act_spaces[a], {}) for a in agents},
            policy_mapping_fn=(lambda agent_id, episode=None, **kw: agent_id),
            count_steps_by="env_steps",
        )
        .training(
            gamma=args.gamma, lr=args.lr,
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size, num_epochs=args.num_epochs,
            lambda_=args.gae_lambda, clip_param=args.clip_param,
            vf_clip_param=args.vf_clip_param, vf_loss_coeff=args.vf_loss_coeff,
            entropy_coeff=args.entropy_coeff, use_kl_loss=args.use_kl_loss,
            grad_clip=args.grad_clip, grad_clip_by="global_norm",
            **dict(spec.training_overrides),
        )
        .debugging(seed=args.seed)
    )
    if spec.rl_module == "ctde":
        from omnipiano.multiagent._ctde_module import (
            build_ctde_module_spec, build_multi_module_spec,
        )
        config = config.rl_module(rl_module_spec=build_multi_module_spec({
            a: build_ctde_module_spec(
                observation_space=obs_spaces[a], action_space=act_spaces[a],
                own_slice=layouts[a]["own"],
                global_state_slice=layouts[a]["global_state"],
                critic_input=spec.critic_input,
                hidden_sizes=args.hidden_sizes_parsed, activation=args.activation,
            ) for a in agents
        }))

    resolved_batch = getattr(config, "total_train_batch_size", None)
    if resolved_batch is None:
        raise RuntimeError("PPOConfig exposes no total_train_batch_size; install "
                           f"the pinned extra (ray[rllib]=={RLLIB_TARGET_VERSION})")
    if int(resolved_batch) != args.train_batch_size:
        raise RuntimeError("RLlib resolved a different total train batch size: "
                           f"requested={args.train_batch_size}, got={resolved_batch}")
    run_config["effective_config"]["resolved_total_train_batch_size"] = int(resolved_batch)
    run_config["effective_config"]["rllib_resolved"] = _resolved_ppo_config(config)
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

    def _evaluate() -> Dict[str, Any]:
        # include_global_state MUST match training: a MAPPO policy expects the
        # wider [global_state | own] vector even though its actor ignores the
        # global block.
        return evaluate_marl(
            algo, args.env_id,
            eval_seed=args.seed + args.eval_seed_offset,
            num_episodes=args.num_eval_eps,
            include_global_state=bool(spec.needs_global_state),
        )

    try:
        ray.init(ignore_reinit_error=True, log_to_driver=False,
                 num_cpus=args.ray_num_cpus)
        ray_started = True
        build = getattr(config, "build_algo", None)
        algo = build() if callable(build) else config.build()
        print(f"[{tag}] RLlib PPO built; starting training")

        if spec.rl_module == "ctde":
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

        previous = -1
        while total_steps < args.total_steps:
            result = algo.train()
            iterations += 1
            total_steps = extract_env_steps(result)
            if total_steps <= previous:
                raise RuntimeError("RLlib lifetime env-step counter did not "
                                   f"advance: {previous} -> {total_steps}")
            previous = total_steps

            telemetry = _training_telemetry(result)
            row = {"iteration": iterations, "env_steps": total_steps,
                   "wall_seconds": round(time.time() - start, 3), **telemetry}
            append_jsonl(run_dir / "progress.jsonl", row)
            wandb_run.log_train(total_steps, row, result)
            if iterations == 1 or iterations % args.log_every_iters == 0:
                print(f"[{tag} iter {iterations:5d}] env_steps={total_steps:>10,}  "
                      f"agent_sum_return={telemetry['rllib_agent_sum_return_mean']}  "
                      f"wall={time.time() - start:.0f}s")

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
                        algo, run_dir / "checkpoints" / f"step_{total_steps:09d}")
                    ref = checkpoint_reference(path, run_dir)
                    record = {"scheduled_env_step": int(scheduled),
                              "actual_env_step": int(total_steps),
                              "checkpoint_path": ref}
                    append_jsonl(run_dir / "checkpoints.jsonl", record)
                    periodic_ckpts.append(record)
                    (run_dir / "latest_checkpoint_path.txt").write_text(ref + "\n",
                                                                       encoding="utf-8")
                    print(f"[{tag} checkpoint] actual={total_steps:,} path={ref}")

            if total_steps >= args.total_steps and final_ckpt is None:
                final_ckpt = save_algorithm_checkpoint(
                    algo, run_dir / "checkpoints" / "final")
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
                evaluation = _evaluate()
                append_jsonl(run_dir / "periodic_eval.jsonl",
                             {"scheduled_env_step": int(scheduled),
                              "actual_env_step": int(total_steps), **evaluation})
                wandb_run.log_eval(total_steps, evaluation,
                                   scheduled_env_step=scheduled)
                periodic_evals.append(evaluation)
                final_eval = evaluation
                last_eval_step = total_steps
                s = evaluation["summary"]
                print(f"[{tag} eval] actual={total_steps:,} "
                      f"team_return={s['team_return_mean']:.3f} "
                      f"musical_f1={s['episode_task/musical_f1_mean']:.6f}")

        if final_ckpt is None or final_ckpt_ref is None:
            raise RuntimeError("training completed without a final checkpoint")
        if last_eval_step != total_steps:
            final_eval = _evaluate()
        if final_eval is None:
            raise RuntimeError("training completed without a final evaluation")

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
            "training_iterations": int(iterations),
            "num_eval_eps": int(args.num_eval_eps),
            "num_eval_episodes": int(args.num_eval_eps),
            "checkpoint_path": final_ckpt_ref,
            "checkpoint_path_resolved": final_ckpt,
            "periodic_eval_file": "periodic_eval.jsonl",
            "periodic_eval_count": len(periodic_evals),
            "periodic_checkpoint_file": "checkpoints.jsonl",
            "periodic_checkpoints": periodic_ckpts,
            "final_evaluation": final_eval,
            "episodes": final_eval["episodes"],
            "summary": final_eval["summary"],
            "wall_seconds": round(time.time() - start, 3),
            "effective_config": run_config["effective_config"],
            "library_versions": run_config["library_versions"],
            "required_terminal_metrics": list(REQUIRED_MUSICAL_METRICS),
            "smoke_test": bool(args.smoke_test),
        }
        write_json(run_dir / "eval_summary.json", eval_summary)
        wandb_run.log_final(total_steps, eval_summary)
        if args.wandb_upload_artifacts:
            for name in ("run_config.json", "eval_summary.json", "progress.jsonl",
                         "periodic_eval.jsonl", "checkpoints.jsonl"):
                wandb_run.log_artifact_dir(
                    run_dir / name,
                    name=f"{run_dir.name}__{name.replace('.', '_')}")
        print(f"[{tag}] final checkpoint: {final_ckpt_ref}")
        print(f"[{tag}] eval summary:    {run_dir / 'eval_summary.json'}")
        if wandb_run.url:
            print(f"[{tag}] wandb:           {wandb_run.url}")
    except Exception as exc:
        write_json(run_dir / "failure.json", {
            "stage": "training_or_evaluation",
            "error": f"{type(exc).__name__}: {exc}",
            "env_steps": int(total_steps), "training_iterations": int(iterations)})
        raise
    finally:
        active = sys.exc_info()[0] is not None
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
