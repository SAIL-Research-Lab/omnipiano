"""Train the reproducible RLlib IPPO baseline for multi-agent OmniPiano.

This is independent PPO (one actor/critic policy per agent) with a shared team
reward.  It is intentionally *not* called MAPPO: neither critic receives a
centralized global state.

The default invocation is one protocol replicate (seed 0).  Paper numbers
require three independent invocations with ``--seed 0``, ``--seed 1``, and
``--seed 2``; the replication set comes from ``BenchmarkProtocolConfig``.

Example::

    MUJOCO_GL=egl python -m omnipiano.multiagent._train_ippo \
        --env-id OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0 \
        --seed 0
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

os.environ.setdefault("MUJOCO_GL", "egl")

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent._ippo_common import (
    LEGACY_RLLIB_ENV_NAME,
    REQUIRED_MUSICAL_METRICS,
    RLLIB_ENV_NAME,
    append_jsonl,
    checkpoint_reference,
    crossed_eval_targets,
    evaluate_ippo,
    extract_env_steps,
    save_algorithm_checkpoint,
    wrap_parallel_env_for_rllib,
    write_json,
)


DEFAULT_ENV_ID = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"
ALGORITHM_NAME = "IPPO (RLlib PPO; one independent actor/critic per agent)"
RLLIB_TARGET_VERSION = "2.55.1"


def _build_arg_parser() -> argparse.ArgumentParser:
    proto = BenchmarkProtocolConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--run-dir",
        "--save-dir",
        dest="run_dir",
        default=None,
        help="Run artifact directory. --save-dir is a compatibility alias.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=proto.total_env_steps,
        help=f"Environment-step budget (protocol default {proto.total_env_steps:,}).",
    )
    parser.add_argument(
        "--seed", type=int, default=proto.seed,
        help=f"One training seed. Protocol replication set: {proto.seeds}.",
    )
    parser.add_argument(
        "--num-eval-eps", type=int, default=proto.num_eval_eps,
        help="Episodes per periodic and final deterministic evaluation.",
    )
    parser.add_argument(
        "--gamma", type=float, default=proto.gamma,
        help=f"Discount factor (protocol task-property default {proto.gamma}).",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=proto.eval_freq_env_steps,
        help="Periodic evaluation cadence in lifetime environment steps.",
    )
    parser.add_argument(
        "--eval-seed-offset", type=int, default=10_000,
        help="Fixed offset from training seed for deterministic evaluations.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every-iters", type=int, default=5)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=500_000,
        help=(
            "Recoverable checkpoint cadence in lifetime environment steps "
            "(0 disables intermediate checkpoints; final is always saved)."
        ),
    )
    # ---------------- PPO hyperparameters (MA review N2-N5) ----------------
    # RLlib's PPO defaults are tuned for small-return classic-control tasks and
    # are actively harmful at this benchmark's return scale (~600 per episode).
    ppo = parser.add_argument_group("ppo (corrected RLlib defaults)")
    ppo.add_argument(
        "--gae-lambda", type=float, default=0.95,
        help="GAE lambda. RLlib default is 1.0 (pure Monte-Carlo advantage, "
             "very high variance over ~1000-step episodes). PPO/MAPPO use 0.95.",
    )
    ppo.add_argument(
        "--clip-param", type=float, default=0.2,
        help="PPO policy clip epsilon. RLlib default 0.3; PPO paper and MAPPO "
             "ablations both favour 0.2.",
    )
    ppo.add_argument(
        "--vf-clip-param", type=float, default=1000.0,
        help="Value-loss clip. RLlib default 10.0 clamps the SQUARED value "
             "error, so any |V - target| > sqrt(10) ~ 3.16 gets ZERO gradient. "
             "With episode returns ~600 that disables critic learning almost "
             "entirely. Set well above the return scale.",
    )
    ppo.add_argument("--vf-loss-coeff", type=float, default=1.0)
    ppo.add_argument("--entropy-coeff", type=float, default=0.0)
    ppo.add_argument(
        "--use-kl-loss", action="store_true", default=False,
        help="Add PPO's adaptive KL penalty on top of clipping. RLlib defaults "
             "this ON; the MAPPO reference implementation uses clipping only.",
    )
    ppo.add_argument(
        "--grad-clip", type=float, default=10.0,
        help="Global-norm gradient clip. RLlib default is None; MAPPO uses 10.",
    )
    
    algo_group = parser.add_argument_group("algorithm")
    algo_group.add_argument(
        "--algo", choices=("ippo", "mappo"), default="ippo",
        help="ippo: decentralized critic V(o_i). "
             "mappo: centralized critic V(s) over the CTDE global state. "
             "Everything else is identical.",
    )
    algo_group.add_argument(
        "--rl-module", choices=("default", "ctde"), default="ctde",
        help="'ctde' uses the shared CtdePPOTorchRLModule so IPPO and MAPPO are "
             "architecturally identical apart from the critic input. 'default' "
             "keeps RLlib's built-in PPO module (IPPO only) as a sanity anchor.",
    )
    algo_group.add_argument(
        "--critic-input", choices=("own", "global"), default=None,
        help="Override the critic input. Defaults to 'own' for --algo ippo and "
             "'global' for --algo mappo. Use '--algo mappo --critic-input own' "
             "for the IPPO-equivalence validation run.",
    )
    algo_group.add_argument("--hidden-sizes", default="256,256")
    algo_group.add_argument("--activation", choices=("tanh", "relu"), default="tanh")
    algo_group.add_argument(
        "--include-global-state", dest="include_global_state",
        action="store_true", default=None,
        help="Force the env to emit the global state. Implied by --algo mappo.",
    )

    # ---------------- compute placement ----------------
    compute = parser.add_argument_group("compute")
    compute.add_argument(
        "--num-learners", type=int, default=1,
        help="Learner actors. Keep at 1 for a reproducible benchmark run.",
    )
    compute.add_argument(
        "--num-gpus-per-learner", type=float, default=1.0,
        help="Set 0 for CPU-only. Pin a specific device with CUDA_VISIBLE_DEVICES.",
    )
    compute.add_argument("--num-cpus-per-env-runner", type=int, default=1)
    compute.add_argument(
        "--ray-num-cpus", type=int, default=None,
        help="Hard-cap Ray's CPU pool. Essential when two jobs share one node.",
    )

    # ---------------- Weights & Biases ----------------
    wb = parser.add_argument_group("weights & biases")
    wb.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                    default="online")
    wb.add_argument("--wandb-entity", default="omnipiano")
    wb.add_argument("--wandb-project", default="multiagent")
    wb.add_argument("--wandb-group", default=None,
                    help="Defaults to '<algo>_<env-token>' so seeds aggregate.")
    wb.add_argument("--wandb-name", default=None, help="Defaults to run-dir name.")
    wb.add_argument("--wandb-tags", default="", help="Comma-separated.")
    wb.add_argument("--wandb-notes", default=None)
    wb.add_argument("--wandb-upload-artifacts", action="store_true",
                    help="Also upload run_dir JSONL/JSON artifacts at the end.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 5k-step ingestion/checkpoint/evaluation test; not a benchmark result.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "total_steps": args.total_steps,
        "num_eval_eps": args.num_eval_eps,
        "eval_freq": args.eval_freq,
        "train_batch_size": args.train_batch_size,
        "minibatch_size": args.minibatch_size,
        "num_epochs": args.num_epochs,
        "log_every_iters": args.log_every_iters,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"positive CLI values required, got {invalid}")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.checkpoint_freq < 0:
        raise ValueError("checkpoint_freq must be non-negative")
    if not math.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("lr must be positive")
    if args.minibatch_size > args.train_batch_size:
        raise ValueError("minibatch_size cannot exceed train_batch_size")
    if args.train_batch_size > args.eval_freq:
        raise ValueError(
            "train_batch_size cannot exceed eval_freq: one PPO iteration could "
            "skip multiple protocol evaluation thresholds"
        )
    if 0 < args.checkpoint_freq < args.train_batch_size:
        raise ValueError(
            "checkpoint_freq must be 0 or at least train_batch_size so one PPO "
            "iteration cannot cross multiple checkpoint thresholds"
        )
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not 0.0 < args.clip_param < 1.0:
        raise ValueError("clip_param must be in (0, 1)")
    if args.vf_clip_param <= 0.0:
        raise ValueError("vf_clip_param must be positive")
    if args.entropy_coeff < 0.0 or args.vf_loss_coeff < 0.0:
        raise ValueError("entropy_coeff and vf_loss_coeff must be non-negative")
    if args.grad_clip is not None and args.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive (or omitted)")
    if args.num_learners != 1:
        # total_train_batch_size = train_batch_size_per_learner * num_learners,
        # so anything other than 1 silently changes the effective batch size and
        # breaks comparability with the recorded protocol.
        raise ValueError("this benchmark requires exactly one Learner")
    if args.num_gpus_per_learner < 0:
        raise ValueError("num_gpus_per_learner must be non-negative")


def _short_env_token(env_id: str) -> str:
    body = [
        part.lower()
        for part in env_id.split("-")
        if part and part != "OmniPiano" and not part.startswith("v")
    ]
    return "_".join(body)


def _default_run_dir(env_id: str, seed: int) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"ippo_rllib_{_short_env_token(env_id)}_seed{seed}_{timestamp}"
    return repo_root / "examples" / "logs" / name


def _prepare_run_dir(requested: Optional[str], env_id: str, seed: int) -> Path:
    run_dir = (
        Path(requested).expanduser().resolve()
        if requested is not None
        else _default_run_dir(env_id, seed).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    protected = (
        "run_config.json",
        "progress.jsonl",
        "periodic_eval.jsonl",
        "eval_summary.json",
        "checkpoint_path.txt",
        "final_checkpoint_path.txt",
        "latest_checkpoint_path.txt",
        "checkpoints.jsonl",
    )
    existing = [name for name in protected if (run_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"run_dir {run_dir} already contains run artifacts {existing}; "
            "choose a fresh directory to avoid mixing experiments"
        )
    return run_dir


def _make_env_for_rllib(env_config: Mapping[str, Any]) -> Any:
    from omnipiano.multiagent import make_parallel

    env_id = str(env_config.get("env_id", DEFAULT_ENV_ID))
    parallel_env = make_parallel(
        env_id,
        seed=int(env_config.get("seed", 0)),
        flatten_obs=True,
        include_global_state=bool(env_config.get("include_global_state", False)),
    )
    return wrap_parallel_env_for_rllib(parallel_env)


def _probe_agent_spaces(env_id: str, seed: int, include_global_state: bool):
    """Return agents, obs/action spaces, and the flat CTDE slice layout."""
    from omnipiano.multiagent import make_parallel

    env = make_parallel(
        env_id, seed=seed, flatten_obs=True,
        include_global_state=include_global_state,
    )
    try:
        agents = list(env.possible_agents)
        observations = {a: env.observation_space(a) for a in agents}
        actions = {a: env.action_space(a) for a in agents}
        layouts: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for agent in agents:
            layout = env.obs_layout(agent)
            if "own" in layout:
                own = layout["own"]
                glob = layout.get("global_state", (0, 0))
            else:
                # No global state: every component belongs to the agent's own
                # observation, so the actor slice is the whole vector.
                own = (0, int(np.prod(observations[agent].shape)))
                glob = (0, 0)
            layouts[agent] = {"own": own, "global_state": glob}
        return agents, observations, actions, layouts
    finally:
        env.close()


def _space_metadata(space: Any) -> Dict[str, Any]:
    return {
        "type": type(space).__name__,
        "shape": list(space.shape),
        "dtype": str(space.dtype),
    }


def _git_metadata() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]

    def _run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "branch": _run("branch", "--show-current"),
            "commit": _run("rev-parse", "HEAD"),
            "dirty": bool(_run("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"branch": None, "commit": None, "dirty": None}


def _protocol_settings_compliant(
    args: argparse.Namespace, proto: BenchmarkProtocolConfig
) -> bool:
    return bool(
        args.seed in proto.seeds
        and args.total_steps == proto.total_env_steps
        and args.num_eval_eps == proto.num_eval_eps
        and args.gamma == proto.gamma
        and args.eval_freq == proto.eval_freq_env_steps
        and not args.smoke_test
    )


def _resolved_ppo_config(config: Any) -> Dict[str, Any]:
    """Snapshot core RLlib defaults that materially affect PPO results."""
    raw = config.to_dict()
    keys = (
        "gamma",
        "use_critic",
        "use_gae",
        "lambda",
        "lambda_",
        "use_kl_loss",
        "kl_coeff",
        "kl_target",
        "clip_param",
        "vf_clip_param",
        "entropy_coeff",
        "vf_loss_coeff",
        "grad_clip",
        "lr",
        "train_batch_size",
        "train_batch_size_per_learner",
        "minibatch_size",
        "num_epochs",
        "shuffle_batch_per_epoch",
        "rollout_fragment_length",
        "batch_mode",
        "normalize_actions",
        "clip_actions",
        "observation_filter",
        "model",
        "model_config",
        "framework",
        "framework_str",
        "num_learners",
    )
    resolved: Dict[str, Any] = {}
    for key in keys:
        if key not in raw:
            continue
        value = raw[key]
        try:
            json.dumps(value)
            resolved[key] = value
        except (TypeError, ValueError):
            resolved[key] = repr(value)
    return resolved


def _training_telemetry(result: Mapping[str, Any]) -> Dict[str, Any]:
    runners = result.get("env_runners", result)
    if not isinstance(runners, Mapping):
        runners = {}
    # RLlib commonly sums simultaneous agents here.  The explicit name avoids
    # misreporting this quantity as the one-copy shared team return.
    agent_sum_return = runners.get(
        "episode_return_mean", runners.get("episode_reward_mean")
    )
    episode_length = runners.get("episode_len_mean")

    def _finite_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    return {
        "rllib_agent_sum_return_mean": _finite_or_none(agent_sum_return),
        "episode_length_mean": _finite_or_none(episode_length),
    }


def _build_run_config(
    args: argparse.Namespace,
    proto: BenchmarkProtocolConfig,
    run_dir: Path,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": ALGORITHM_NAME,
        "is_centralized_critic": False,
        "source_script": Path(__file__).name,
        "env_id": args.env_id,
        "run_dir": str(run_dir),
        "seed": int(args.seed),
        "eval_seed": int(args.seed + args.eval_seed_offset),
        "protocol_version": proto.protocol_version,
        "protocol_replication_seeds": list(proto.seeds),
        "protocol_settings_compliant": _protocol_settings_compliant(args, proto),
        "replication_complete": False,
        "replication_status": "single_seed_replicate",
        "total_env_steps": int(args.total_steps),
        "requested_total_env_steps": int(args.total_steps),
        "num_eval_eps": int(args.num_eval_eps),
        "num_eval_episodes": int(args.num_eval_eps),
        "effective_config": {
            "gamma": float(args.gamma),
            "eval_freq_env_steps": int(args.eval_freq),
            "num_env_runners": int(args.num_workers),
            "train_batch_size": int(args.train_batch_size),
            "minibatch_size": int(args.minibatch_size),
            "num_epochs": int(args.num_epochs),
            "learning_rate": float(args.lr),
            "checkpoint_freq_env_steps": int(args.checkpoint_freq),
            "framework": "torch",
            "count_steps_by": "env_steps",
            "flatten_obs": True,
            "reward_mode": "shared",
            "algorithm_seed": int(args.seed),
            "worker_seeding": (
                "RLlib AlgorithmConfig.debugging(seed=...); environment creator "
                "receives the base seed before RLlib-managed reset seeding"
            ),
            "rllib_target_version": RLLIB_TARGET_VERSION,
            # PPO hyperparameters. Recorded explicitly because several deviate
            # from RLlib defaults for documented, return-scale reasons.
            "gae_lambda": float(args.gae_lambda),
            "clip_param": float(args.clip_param),
            "vf_clip_param": float(args.vf_clip_param),
            "vf_loss_coeff": float(args.vf_loss_coeff),
            "entropy_coeff": float(args.entropy_coeff),
            "use_kl_loss": bool(args.use_kl_loss),
            "grad_clip": float(args.grad_clip),
            "grad_clip_by": "global_norm",
            "rllib_default_overrides": {
                # value: [rllib_default, ours, why]
                "lambda_": [1.0, float(args.gae_lambda),
                            "GAE lambda=1 is pure Monte-Carlo; variance is "
                            "unusable over ~1000-step episodes"],
                "clip_param": [0.3, float(args.clip_param),
                               "PPO paper / MAPPO ablation both favour 0.2"],
                "vf_clip_param": [10.0, float(args.vf_clip_param),
                                  "RLlib clamps the squared value error; at "
                                  "return scale ~600 the default zeroes the "
                                  "critic gradient on nearly every sample"],
                "use_kl_loss": [True, bool(args.use_kl_loss),
                                "MAPPO reference uses clipping only"],
                "grad_clip": [None, float(args.grad_clip),
                              "MAPPO uses max_grad_norm=10"],
            },
            "num_learners": int(args.num_learners),
            "num_gpus_per_learner": float(args.num_gpus_per_learner),
        },
        "smoke_test": bool(args.smoke_test),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "git": _git_metadata(),
    }


def _resolve_algo(args: argparse.Namespace) -> None:
    """Fill in algorithm-dependent defaults and reject incoherent combinations."""
    if args.critic_input is None:
        args.critic_input = "own" if args.algo == "ippo" else "global"
    if args.include_global_state is None:
        # MAPPO needs it. IPPO does not, but allowing it enables the equivalence
        # control run in which the global state is present yet never read.
        args.include_global_state = args.algo == "mappo"

    if args.critic_input == "global" and not args.include_global_state:
        raise ValueError(
            "--critic-input global requires the env to emit the global state; "
            "pass --include-global-state (implied by --algo mappo)"
        )
    if args.rl_module == "default":
        if args.critic_input != "own":
            raise ValueError(
                "--rl-module default only supports --critic-input own "
                "(RLlib's built-in PPO module has no centralized critic)"
            )
        if args.include_global_state:
            raise ValueError(
                "--rl-module default cannot consume a global state; it would be "
                "fed to the actor as well, breaking decentralized execution"
            )
    if args.algo == "mappo" and args.rl_module != "ctde":
        raise ValueError("--algo mappo requires --rl-module ctde")

    args.hidden_sizes_parsed = tuple(
        int(tok) for tok in str(args.hidden_sizes).split(",") if tok.strip()
    )
    if not args.hidden_sizes_parsed or any(h <= 0 for h in args.hidden_sizes_parsed):
        raise ValueError(f"invalid --hidden-sizes {args.hidden_sizes!r}")

    args.algorithm_name = {
        ("ippo", "own"): "IPPO (RLlib PPO; one independent actor/critic per agent)",
        ("mappo", "global"): (
            "MAPPO (RLlib PPO; per-agent actor over o_i, centralized critic "
            "over the agent-specific global state s; CTDE)"
        ),
        ("mappo", "own"): (
            "MAPPO-ablation (centralized state present in the observation but "
            "NOT read by the critic; numerically equivalent to IPPO)"
        ),
    }[(args.algo, args.critic_input)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.smoke_test:
        args.total_steps = 5_000
        args.eval_freq = 5_000
        args.num_eval_eps = 1
        args.num_workers = 0
        args.train_batch_size = 512
        args.minibatch_size = 64
        args.num_epochs = 2
    _validate_args(args)

    proto = BenchmarkProtocolConfig()
    run_dir = _prepare_run_dir(args.run_dir, args.env_id, args.seed)
    run_config = _build_run_config(args, proto, run_dir)
    write_json(run_dir / "run_config.json", run_config)
    # Pre-create append-only telemetry files so their existence is guaranteed
    # even for short ablations that finish before the first periodic eval.
    (run_dir / "progress.jsonl").touch(exist_ok=False)
    (run_dir / "periodic_eval.jsonl").touch(exist_ok=False)
    (run_dir / "checkpoints.jsonl").touch(exist_ok=False)

    print(f"[ippo] run_dir={run_dir}")
    print(f"[ippo] env_id={args.env_id}")
    print(
        f"[ippo] seed={args.seed} (protocol seeds={proto.seeds})  "
        f"steps={args.total_steps:,}  gamma={args.gamma}  "
        f"eval_every={args.eval_freq:,} env-steps"
    )
    
    from omnipiano.multiagent._wandb import WandbRun

    wandb_group = args.wandb_group or f"ippo_{_short_env_token(args.env_id)}"
    wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_tags = sorted(set(wandb_tags) | {
        "ippo",
        _short_env_token(args.env_id),
        f"seed{args.seed}",
        f"{args.total_steps // 1_000_000}M" if args.total_steps >= 1_000_000
        else f"{args.total_steps // 1_000}k",
        *(["smoke-test"] if args.smoke_test else []),
    })
    wandb_run = WandbRun(
        mode=args.wandb_mode,
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_name or run_dir.name,
        group=wandb_group,
        job_type="smoke-test" if args.smoke_test else "train",
        tags=wandb_tags,
        notes=args.wandb_notes,
        config=run_config,
        run_dir=run_dir,
    )
    # Make the W&B run discoverable from the local artifacts and vice versa.
    run_config["wandb"] = {
        "mode": args.wandb_mode,
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "group": wandb_group,
        "run_id": wandb_run.run_id,
        "url": wandb_run.url,
        "tags": wandb_tags,
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
        write_json(
            run_dir / "failure.json",
            {"stage": "dependency_import", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise RuntimeError(
            "IPPO training requires ray[rllib], PettingZoo, torch, and OmniPiano's "
            "runtime dependencies"
        ) from exc

    # Register both names so checkpoints from the previous misnamed trainer can
    # still be restored by the standalone evaluator.
    register_env(RLLIB_ENV_NAME, _make_env_for_rllib)
    register_env(LEGACY_RLLIB_ENV_NAME, _make_env_for_rllib)

    
    agents, obs_spaces, act_spaces, obs_layouts = _probe_agent_spaces(
        args.env_id, args.seed, args.include_global_state
    )
    policies = {
        agent: (None, obs_spaces[agent], act_spaces[agent], {})
        for agent in agents
    }
    run_config["agents"] = list(agents)
    run_config["policy_mapping"] = {agent: agent for agent in agents}
    run_config["observation_spaces"] = {
        agent: _space_metadata(obs_spaces[agent]) for agent in agents
    }
    run_config["action_spaces"] = {
        agent: _space_metadata(act_spaces[agent]) for agent in agents
    }
    run_config["library_versions"] = {
        "ray": ray.__version__,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "pettingzoo": getattr(pettingzoo, "__version__", "unknown"),
    }
    if ray.__version__ != RLLIB_TARGET_VERSION:
        print(
            "[ippo warning] this baseline targets "
            f"ray[rllib]=={RLLIB_TARGET_VERSION}, found {ray.__version__}; "
            "the exact version is recorded in run_config.json"
        )
    write_json(run_dir / "run_config.json", run_config)
    config = (
        PPOConfig()
        .api_stack(
            # Pinned explicitly: the CTDE RLModule added in the MAPPO commit
            # requires the new stack, and we want the two baselines to share
            # exactly one code path.
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            RLLIB_ENV_NAME,
            env_config={
                "env_id": args.env_id,
                "seed": args.seed,
                "include_global_state": bool(args.include_global_state),
            },
        )
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agent_id, episode=None, **kw: agent_id),
            count_steps_by="env_steps",
        )
        .training(
            gamma=args.gamma,
            lr=args.lr,
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            # ---- corrected RLlib defaults; see MA review N2-N5 ----
            lambda_=args.gae_lambda,          # RLlib default 1.0  -> 0.95
            clip_param=args.clip_param,       # RLlib default 0.3  -> 0.2
            vf_clip_param=args.vf_clip_param, # RLlib default 10.0 -> 1000.0
            vf_loss_coeff=args.vf_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            use_kl_loss=args.use_kl_loss,     # RLlib default True -> False
            grad_clip=args.grad_clip,         # RLlib default None -> 10.0
            grad_clip_by="global_norm",
        )
        .debugging(seed=args.seed)
    )
    if args.rl_module == "ctde":
        from omnipiano.multiagent._ctde_module import (
            build_ctde_module_spec,
            build_multi_module_spec,
        )

        module_specs = {
            agent: build_ctde_module_spec(
                observation_space=obs_spaces[agent],
                action_space=act_spaces[agent],
                own_slice=obs_layouts[agent]["own"],
                global_state_slice=obs_layouts[agent]["global_state"],
                critic_input=args.critic_input,
                hidden_sizes=args.hidden_sizes_parsed,
                activation=args.activation,
            )
            for agent in agents
        }
        config = config.rl_module(rl_module_spec=build_multi_module_spec(module_specs))
    configured_batch_size = getattr(config, "total_train_batch_size", None)
    if configured_batch_size is None:
        raise RuntimeError(
            "target RLlib PPOConfig exposes no total_train_batch_size; install "
            f"the pinned marl extra (ray[rllib]=={RLLIB_TARGET_VERSION})"
        )
    if int(configured_batch_size) != args.train_batch_size:
        raise RuntimeError(
            "RLlib resolved a different total train batch size: "
            f"requested={args.train_batch_size}, resolved={configured_batch_size}"
        )
    run_config["effective_config"]["rllib_resolved"] = _resolved_ppo_config(config)
    run_config["effective_config"]["resolved_total_train_batch_size"] = int(
        configured_batch_size
    )
    
    run_config["algorithm"] = args.algorithm_name
    run_config["is_centralized_critic"] = args.critic_input == "global"
    run_config["effective_config"].update({
        "algo": args.algo,
        "rl_module": args.rl_module,
        "critic_input": args.critic_input,
        "include_global_state": bool(args.include_global_state),
        "hidden_sizes": list(args.hidden_sizes_parsed),
        "activation": args.activation,
        "obs_layout": {
            agent: {k: list(v) for k, v in layout.items()}
            for agent, layout in obs_layouts.items()
        },
    })
    if args.rl_module == "ctde":
        # Record the realized parameter split so an IPPO/MAPPO pair can be
        # verified as architecturally matched after the fact.
        try:
            run_config["ctde_modules"] = {
                agent: algo.get_module(agent).ctde_spec for agent in agents
            }
        except Exception as exc:
            print(f"[warning] could not snapshot CTDE module specs: {exc}")
    
    write_json(run_dir / "run_config.json", run_config)

    algo = None
    ray_started = False
    start_time = time.time()
    periodic_evaluations = []
    periodic_checkpoints = []
    total_steps = 0
    iterations = 0
    next_eval_step = args.eval_freq
    next_checkpoint_step: Optional[int] = (
        args.checkpoint_freq if args.checkpoint_freq > 0 else None
    )
    last_eval_actual_step: Optional[int] = None
    final_evaluation: Optional[Dict[str, Any]] = None
    final_checkpoint_path: Optional[str] = None
    final_checkpoint_reference: Optional[str] = None

    try:
        ray.init(
            ignore_reinit_error=True,
            log_to_driver=False,
        )
        ray_started = True
        build = getattr(config, "build_algo", None)
        algo = build() if callable(build) else config.build()
        print("[ippo] RLlib PPO built; starting independent-policy training")

        previous_steps = -1
        while total_steps < args.total_steps:
            result = algo.train()
            iterations += 1
            total_steps = extract_env_steps(result)
            if total_steps <= previous_steps:
                raise RuntimeError(
                    "RLlib lifetime env-step counter did not advance: "
                    f"previous={previous_steps}, current={total_steps}"
                )
            previous_steps = total_steps

            telemetry = _training_telemetry(result)
            progress_row = {
                "iteration": iterations,
                "env_steps": total_steps,
                "wall_seconds": round(time.time() - start_time, 3),
                **telemetry,
            }
            append_jsonl(run_dir / "progress.jsonl", progress_row)
            wandb_run.log_train(total_steps, progress_row, result)
            if iterations == 1 or iterations % args.log_every_iters == 0:
                print(
                    f"[ippo iter {iterations:4d}] env_steps={total_steps:>9,}  "
                    f"rllib_agent_sum_return={telemetry['rllib_agent_sum_return_mean']}  "
                    f"wall={time.time() - start_time:.0f}s"
                )

            # Save recovery state before evaluation. If inference or metric
            # extraction then fails, the policy that triggered the failure is
            # still available for diagnosis.
            if next_checkpoint_step is not None:
                crossed_checkpoints, next_checkpoint_step = crossed_eval_targets(
                    total_steps, next_checkpoint_step, args.checkpoint_freq
                )
                if len(crossed_checkpoints) > 1:
                    raise RuntimeError(
                        "one RLlib train iteration crossed multiple checkpoint "
                        f"thresholds {crossed_checkpoints}; increase checkpoint_freq"
                    )
                for scheduled_step in crossed_checkpoints:
                    # The final save below supersedes a periodic save at the
                    # terminal budget, avoiding two copies of the same policy.
                    if total_steps >= args.total_steps:
                        continue
                    saved_path = save_algorithm_checkpoint(
                        algo,
                        run_dir
                        / "checkpoints"
                        / f"step_{int(total_steps):09d}",
                    )
                    saved_reference = checkpoint_reference(saved_path, run_dir)
                    checkpoint_record = {
                        "scheduled_env_step": int(scheduled_step),
                        "actual_env_step": int(total_steps),
                        "checkpoint_path": saved_reference,
                    }
                    append_jsonl(run_dir / "checkpoints.jsonl", checkpoint_record)
                    periodic_checkpoints.append(checkpoint_record)
                    (run_dir / "latest_checkpoint_path.txt").write_text(
                        saved_reference + "\n", encoding="utf-8"
                    )
                    print(
                        f"[ippo checkpoint] scheduled={scheduled_step:,} "
                        f"actual={total_steps:,} path={saved_reference}"
                    )

            # Persist the terminal policy before its fail-fast evaluation. If
            # inference or terminal metric extraction is broken, the completed
            # training state remains available for diagnosis instead of
            # silently falling back to the previous periodic checkpoint.
            if total_steps >= args.total_steps and final_checkpoint_path is None:
                final_checkpoint_path = save_algorithm_checkpoint(
                    algo, run_dir / "checkpoints" / "final"
                )
                final_checkpoint_reference = checkpoint_reference(
                    final_checkpoint_path, run_dir
                )
                for sidecar_name in (
                    "final_checkpoint_path.txt",
                    "latest_checkpoint_path.txt",
                    # Historical evaluator compatibility.
                    "checkpoint_path.txt",
                ):
                    (run_dir / sidecar_name).write_text(
                        final_checkpoint_reference + "\n", encoding="utf-8"
                    )
                print(
                    "[ippo checkpoint] final "
                    f"actual={total_steps:,} path={final_checkpoint_reference}"
                )

            crossed, next_eval_step = crossed_eval_targets(
                total_steps, next_eval_step, args.eval_freq
            )
            if len(crossed) > 1:
                raise RuntimeError(
                    "one RLlib train iteration crossed multiple protocol eval "
                    f"thresholds {crossed}; reduce train_batch_size"
                )
            for scheduled_step in crossed:
                evaluation = evaluate_ippo(
                    algo,
                    args.env_id,
                    eval_seed=args.seed + args.eval_seed_offset,
                    num_episodes=args.num_eval_eps,
                )
                eval_record = {
                    "scheduled_env_step": int(scheduled_step),
                    "actual_env_step": int(total_steps),
                    **evaluation,
                }
                append_jsonl(run_dir / "periodic_eval.jsonl", eval_record)
                wandb_run.log_eval(
                    total_steps, evaluation, scheduled_env_step=scheduled_step
                )
                periodic_evaluations.append(eval_record)
                final_evaluation = evaluation
                last_eval_actual_step = total_steps
                f1 = evaluation["summary"][
                    "episode_task/musical_f1_mean"
                ]
                team_return = evaluation["summary"]["team_return_mean"]
                print(
                    f"[ippo eval] scheduled={scheduled_step:,} "
                    f"actual={total_steps:,} team_return={team_return:.3f} "
                    f"musical_f1={f1:.6f}"
                )

        if final_checkpoint_path is None or final_checkpoint_reference is None:
            raise RuntimeError("training completed without a final checkpoint")

        # If the final PPO update did not coincide with an evaluation crossing,
        # evaluate the exact final policy once.  Deterministic seeds are shared
        # with periodic evals, so values remain directly comparable.
        if last_eval_actual_step != total_steps:
            final_evaluation = evaluate_ippo(
                algo,
                args.env_id,
                eval_seed=args.seed + args.eval_seed_offset,
                num_episodes=args.num_eval_eps,
            )

        if final_evaluation is None:  # Defensive: total_steps is validated > 0.
            raise RuntimeError("training completed without a final evaluation")

        eval_summary = {
            "schema_version": 1,
            "algorithm": ALGORITHM_NAME,
            "source_script": Path(__file__).name,
            "env_id": args.env_id,
            "seed": int(args.seed),
            "eval_seed": int(args.seed + args.eval_seed_offset),
            "protocol_version": proto.protocol_version,
            "protocol_replication_seeds": list(proto.seeds),
            "protocol_settings_compliant": _protocol_settings_compliant(args, proto),
            "replication_complete": False,
            "replication_status": "single_seed_replicate",
            "total_env_steps": int(args.total_steps),
            "requested_total_env_steps": int(args.total_steps),
            "actual_total_env_steps": int(total_steps),
            "trained_env_steps": int(total_steps),
            "training_iterations": int(iterations),
            "num_eval_eps": int(args.num_eval_eps),
            "num_eval_episodes": int(args.num_eval_eps),
            "checkpoint_path": final_checkpoint_reference,
            "checkpoint_path_resolved": final_checkpoint_path,
            "periodic_eval_file": "periodic_eval.jsonl",
            "periodic_eval_count": len(periodic_evaluations),
            "periodic_checkpoint_file": "checkpoints.jsonl",
            "periodic_checkpoint_count": len(periodic_checkpoints),
            "periodic_checkpoints": periodic_checkpoints,
            "final_evaluation": final_evaluation,
            "episodes": final_evaluation["episodes"],
            "summary": final_evaluation["summary"],
            "wall_seconds": round(time.time() - start_time, 3),
            "effective_config": run_config["effective_config"],
            "library_versions": run_config["library_versions"],
            "required_terminal_metrics": list(REQUIRED_MUSICAL_METRICS),
            "smoke_test": bool(args.smoke_test),
        }
        write_json(run_dir / "eval_summary.json", eval_summary)
        print(f"[ippo] final checkpoint: {final_checkpoint_reference}")
        print(f"[ippo] eval summary: {run_dir / 'eval_summary.json'}")
        wandb_run.log_final(total_steps, eval_summary)
        if args.wandb_upload_artifacts:
            for artifact_name in (
                "run_config.json", "eval_summary.json",
                "progress.jsonl", "periodic_eval.jsonl", "checkpoints.jsonl",
            ):
                wandb_run.log_artifact_dir(
                    run_dir / artifact_name,
                    name=f"{run_dir.name}__{artifact_name.replace('.', '_')}",
                    artifact_type="run-artifacts",
                )
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {
                "stage": "training_or_evaluation",
                "error": f"{type(exc).__name__}: {exc}",
                "env_steps": int(total_steps),
                "training_iterations": int(iterations),
            },
        )
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            wandb_run.finish(exit_code=1 if active_exception else 0)
        except Exception:
            pass
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
