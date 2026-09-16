"""Generic trainer for every registered OmniPiano multi-agent baseline.

One training loop, one bookkeeping path, one W&B integration.  The algorithm
contributes only an ``AlgoSpec`` (see ``algos/base.py``), so IPPO, MAPPO and any
future baseline are guaranteed to differ *only* where their spec differs.

Canonical invocation::

    python -m omnipiano.multiagent.train --algo mappo --seed 0
    python -m omnipiano.multiagent.train path/to/experiment.json

All numerical defaults come from ``marl_train_config.json``. Pass
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
from omnipiano.multiagent.compile import (
    DEFAULT_TRAIN_CONFIG_PATH,
    ResolvedTask,
    compile_experiment,
    resolve_train_config_path,
)
from omnipiano.multiagent.training.paths import logs_root, resolve_run_path
from omnipiano.multiagent.training.runtime import (
    COORDINATION_RATE_METRICS,
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

RLLIB_TARGET_VERSION = "2.55.1"


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
# CLI
# ===========================================================================


def build_arg_parser(
    config_path: Optional[os.PathLike[str] | str] = None,
    *,
    algo_override: Optional[str] = None,
) -> argparse.ArgumentParser:
    compiled = compile_experiment(
        config_path or DEFAULT_TRAIN_CONFIG_PATH,
        registered_algorithms=list_algos(),
        algo_override=algo_override,
    )
    defaults = dict(compiled.values)
    proto = BenchmarkProtocolConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.set_defaults(
        _smoke_test_overrides=dict(compiled.smoke_test_overrides),
        _training_config_snapshot=compiled.request.snapshot(),
        _training_config_path=str(compiled.config_path),
        _resolved_task=(compiled.task.to_dict() if compiled.task is not None else None),
    )
    p.add_argument(
        "experiment_config", nargs="?", metavar="EXPERIMENT.json",
        help="Optional positional spelling of --config for one-command launches.",
    )

    core = p.add_argument_group("experiment identity")
    core.add_argument("--config", default=str(compiled.config_path),
                     help="MARL JSON config. CLI flags override its selected values.")
    core.add_argument("--algo", default=defaults["algo"], choices=list_algos(),
                     help="Registered baseline; see --list-algos.")
    core.add_argument("--list-algos", action="store_true",
                     help="Print the algorithm registry and exit.")
    core.add_argument("--list-envs", action="store_true",
                     help="Print every registered multi-agent env id and exit.")
    core.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve and construct one environment, print its complete task "
            "and spaces, then exit before Ray, W&B, GPU use, or run artifacts."
        ),
    )
    core.add_argument("--env-id", default=defaults["env_id"])
    core.add_argument("--seed", type=int, default=defaults["seed"],
                     help=f"One training seed. Protocol replication set: {proto.seeds}.")
    core.add_argument("--run-dir", "--save-dir", dest="run_dir",
                     default=defaults["run_dir"],
                     help="Run artifact directory (repo-relative paths allowed).")

    proto_g = p.add_argument_group("protocol (BenchmarkProtocolConfig)")
    proto_g.add_argument("--total-steps", type=int, default=defaults["total_steps"],
                         help=f"ENVIRONMENT-step budget (protocol {proto.total_env_steps:,}).")
    proto_g.add_argument("--gamma", type=float, default=defaults["gamma"])
    proto_g.add_argument("--eval-freq", type=int, default=defaults["eval_freq"],
                         help="Deterministic-evaluation cadence in lifetime env steps.")
    proto_g.add_argument("--num-eval-eps", type=int,
                         default=defaults["num_eval_eps"])
    proto_g.add_argument("--eval-seed-offset", type=int,
                         default=defaults["eval_seed_offset"])

    reward = p.add_argument_group("multi-agent reward shaping")
    reward.add_argument(
        "--inter-agent-collision-penalty-coef",
        type=float,
        default=defaults["inter_agent_collision_penalty_coef"],
        help=(
            "Subtract coef once per control step when any active collision "
            "geoms owned by different agents touch; 0 disables the term."
        ),
    )

    ppo = p.add_argument_group("ppo (RLlib defaults corrected for this benchmark)")
    ppo.add_argument("--train-batch-size", type=int,
                     default=defaults["train_batch_size"])
    ppo.add_argument("--minibatch-size", type=int,
                     default=defaults["minibatch_size"],
                     help="Full-batch PPO by default (official MAPPO uses one "
                          "mini-batch per epoch).")
    ppo.add_argument("--num-epochs", type=int, default=defaults["num_epochs"])
    ppo.add_argument("--lr", type=float, default=defaults["lr"])
    ppo.add_argument("--critic-lr", type=float, default=defaults["critic_lr"],
                     help="Critic Adam learning rate; defaults to --lr.")
    ppo.add_argument("--adam-epsilon", type=float,
                     default=defaults["adam_epsilon"],
                     help="Epsilon for the independent actor/critic Adam optimizers.")
    ppo.add_argument("--gae-lambda", type=float, default=defaults["gae_lambda"],
                     help="Shared PPO/MAPPO GAE setting.")
    ppo.add_argument("--clip-param", type=float, default=defaults["clip_param"],
                     help="RLlib default 0.3; PPO paper and MAPPO both use 0.2.")
    ppo.add_argument("--vf-clip-param", type=float,
                     default=defaults["vf_clip_param"],
                     help="Maximum change from the rollout-time normalized value "
                          "prediction (official PPO/MAPPO semantics).")
    ppo.add_argument("--vf-loss-coeff", type=float,
                     default=defaults["vf_loss_coeff"])
    ppo.add_argument("--entropy-coeff", type=float,
                     default=defaults["entropy_coeff"])
    ppo.add_argument("--use-kl-loss", action=argparse.BooleanOptionalAction,
                     default=defaults["use_kl_loss"],
                     help="RLlib defaults this ON; MAPPO reference uses clip only.")
    ppo.add_argument("--grad-clip", type=float, default=defaults["grad_clip"])
    ppo.add_argument("--grad-clip-by",
                     choices=("value", "norm", "global_norm"),
                     default=defaults["grad_clip_by"])

    net = p.add_argument_group("network")
    net.add_argument("--hidden-sizes", default=defaults["hidden_sizes"])
    net.add_argument("--activation", choices=("tanh", "relu"),
                     default=defaults["activation"])
    net.add_argument("--hidden-orthogonal-gain", type=float,
                     default=defaults["hidden_orthogonal_gain"])
    net.add_argument("--policy-output-gain", type=float,
                     default=defaults["policy_output_gain"])
    net.add_argument("--value-output-gain", type=float,
                     default=defaults["value_output_gain"])
    net.add_argument("--initial-log-std", type=float,
                     default=defaults["initial_log_std"])
    net.add_argument("--log-std-min", type=float,
                     default=defaults["log_std_min"])
    net.add_argument("--log-std-max", type=float,
                     default=defaults["log_std_max"])
    net.add_argument("--input-layer-norm", action=argparse.BooleanOptionalAction,
                     default=defaults["input_layer_norm"],
                     help="MAPPO-style feature LayerNorm on each actor/critic input.")
    net.add_argument("--value-norm", action=argparse.BooleanOptionalAction,
                     default=defaults["value_norm"],
                     help="Bias-corrected running normalization of critic targets.")
    net.add_argument("--value-norm-beta", type=float,
                     default=defaults["value_norm_beta"])
    net.add_argument("--value-norm-epsilon", type=float,
                     default=defaults["value_norm_epsilon"])
    net.add_argument("--value-norm-variance-floor", type=float,
                     default=defaults["value_norm_variance_floor"])

    comp = p.add_argument_group("compute")
    comp.add_argument("--num-workers", type=int, default=defaults["num_workers"],
                      help="RLlib env runners (sampling processes).")
    comp.add_argument("--num-cpus-per-env-runner", type=int,
                      default=defaults["num_cpus_per_env_runner"])
    comp.add_argument("--num-learners", type=int,
                      default=defaults["num_learners"],
                      help="Pinned to 1: total batch size scales with this.")
    comp.add_argument("--num-gpus-per-learner", type=float,
                      default=defaults["num_gpus_per_learner"],
                      help="Configured per learner (0.0 under canonical smoke test). "
                           "Select the "
                           "device with CUDA_VISIBLE_DEVICES.")
    comp.add_argument("--ray-num-cpus", type=int, default=defaults["ray_num_cpus"],
                      help="Hard-cap Ray's CPU pool; required when two jobs "
                           "share one node.")
    comp.add_argument("--sample-timeout-s", type=float,
                      default=defaults["sample_timeout_s"],
                      help="Remote-sampling timeout per iteration. RLlib's 60s "
                           "default is far too small here: every env runner must "
                           "compile MJCF TWICE (reach probe env + real env) before "
                           "returning its first fragment, and RLlib DISCARDS the "
                           "result of any runner that exceeds this timeout.")
    comp.add_argument("--max-sampling-stalls", type=int,
                      default=defaults["max_sampling_stalls"],
                      help="Abort after this many consecutive empty sample iterations.")
    comp.add_argument("--ray-log-to-driver",
                      action=argparse.BooleanOptionalAction,
                      default=defaults["ray_log_to_driver"],
                      help="Forward env-runner stdout/stderr to the driver. "
                           "Required to see why sampling fails.")
    comp.add_argument("--checkpoint-freq", type=int,
                      default=defaults["checkpoint_freq"],
                      help="Recoverable checkpoint cadence in env steps (0=off).")
    comp.add_argument("--log-every-iters", type=int,
                      default=defaults["log_every_iters"])
    comp.add_argument("--smoke-test", action=argparse.BooleanOptionalAction,
                      default=defaults["smoke_test"],
                      help="5k-step ingestion/checkpoint/eval test; not a result.")

    video = p.add_argument_group("evaluation video")
    video.add_argument("--eval-video", dest="video_enabled",
                       action=argparse.BooleanOptionalAction,
                       default=defaults["video_enabled"],
                       help="Record deterministic evaluation videos during training.")
    video.add_argument("--video-freq", type=int, default=defaults["video_freq"],
                       help="Video cadence in lifetime env steps; must be a multiple "
                            "of --eval-freq.")
    video.add_argument("--video-record-final", dest="video_record_final",
                       action=argparse.BooleanOptionalAction,
                       default=defaults["video_record_final"],
                       help="Always record the final policy if its step was not already "
                            "a periodic video target.")
    video.add_argument("--video-camera-id", default=defaults["video_camera_id"])
    video.add_argument("--video-height", type=int, default=defaults["video_height"])
    video.add_argument("--video-width", type=int, default=defaults["video_width"])
    video.add_argument("--video-wandb-upload", dest="video_wandb_upload",
                       action=argparse.BooleanOptionalAction,
                       default=defaults["video_wandb_upload"],
                       help="Upload each completed periodic video to the active W&B run.")

    wb = p.add_argument_group("weights & biases")
    wb.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                    default=defaults["wandb_mode"])
    wb.add_argument("--wandb-entity", default=defaults["wandb_entity"])
    wb.add_argument("--wandb-project", default=defaults["wandb_project"])
    wb.add_argument("--wandb-group", default=defaults["wandb_group"],
                    help="Defaults to '<algo>__<env-token>' so seeds aggregate.")
    wb.add_argument("--wandb-name", default=defaults["wandb_name"],
                    help="Defaults to run-dir name.")
    wb.add_argument("--wandb-tags", default=defaults["wandb_tags"])
    wb.add_argument("--wandb-notes", default=defaults["wandb_notes"])
    wb.add_argument("--wandb-upload-artifacts",
                    action=argparse.BooleanOptionalAction,
                    default=defaults["wandb_upload_artifacts"])
    
    gate = p.add_argument_group("safety gates")
    gate.add_argument(
        "--allow-experimental", action="store_true",
        help="Permit an algorithm whose AlgoSpec.status is 'experimental'. "
             "Deliberately CLI-only and NOT settable from JSON: a config file "
             "must never be able to silently promote an unvalidated algorithm.")

    return p


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Two-phase parse so --config and --algo select the correct JSON defaults."""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    # Positional config is deliberately first-token only. Letting argparse's
    # parse_known_args see an unknown option such as ``--total-steps 5000``
    # can otherwise mistake that option's value for the positional path.
    positional_config = (
        tokens[0] if tokens and not tokens[0].startswith("-") else None
    )
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_TRAIN_CONFIG_PATH))
    bootstrap.add_argument("--algo", default=None)
    selected, _ = bootstrap.parse_known_args(
        tokens[1:] if positional_config is not None else tokens
    )
    selected_config = positional_config or selected.config
    parser = build_arg_parser(
        selected_config,
        algo_override=selected.algo,
    )
    args = parser.parse_args(tokens)
    if positional_config and any(
        token == "--config" or token.startswith("--config=") for token in tokens
    ):
        raise ValueError(
            "pass the experiment JSON positionally or with --config, not both"
        )
    args.config = str(resolve_train_config_path(selected_config))
    explicit_destinations = set()
    for token in tokens:
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            explicit_destinations.add(action.dest)
    args._explicit_cli_destinations = explicit_destinations
    return args


def _resolve_args(args: argparse.Namespace) -> AlgoSpec:
    """Apply algorithm-dependent and smoke-test defaults, then validate."""
    spec = get_algo(args.algo)
    spec.assert_launchable(
        allow_experimental=bool(getattr(args, "allow_experimental", False)))
    if (getattr(args, "_resolved_task", None) is not None
            and "env_id" in getattr(args, "_explicit_cli_destinations", set())):
        raise ValueError(
            "--env-id cannot override a schema-v2 task; edit the task block "
            "or use a schema-v1 registered-environment config"
        )

    if args.smoke_test:
        for destination, value in args._smoke_test_overrides.items():
            if destination not in getattr(args, "_explicit_cli_destinations", set()):
                setattr(args, destination, value)
    if args.num_gpus_per_learner is None:
        args.num_gpus_per_learner = 0.0 if args.smoke_test else 1.0
    if args.critic_lr is None:
        args.critic_lr = args.lr

    if isinstance(args.hidden_sizes, str):
        args.hidden_sizes_parsed = tuple(
            int(t) for t in args.hidden_sizes.split(",") if t.strip()
        )
    elif isinstance(args.hidden_sizes, (list, tuple)):
        args.hidden_sizes_parsed = tuple(int(t) for t in args.hidden_sizes)
    else:
        raise ValueError(
            "hidden_sizes must be a JSON array or comma-separated CLI string, "
            f"got {type(args.hidden_sizes).__name__}"
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
    if args.num_workers < 0 or args.checkpoint_freq < 0 or args.video_freq < 0:
        raise ValueError(
            "num_workers, checkpoint_freq and video_freq must be non-negative"
        )
    if args.num_cpus_per_env_runner <= 0:
        raise ValueError("num_cpus_per_env_runner must be positive")
    if args.ray_num_cpus is not None and args.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be null or positive")
    if not math.isfinite(args.gamma) or not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if (not math.isfinite(args.inter_agent_collision_penalty_coef)
            or args.inter_agent_collision_penalty_coef < 0.0):
        raise ValueError(
            "inter_agent_collision_penalty_coef must be finite and non-negative"
        )
    if (not math.isfinite(args.lr) or args.lr <= 0.0
            or not math.isfinite(args.critic_lr) or args.critic_lr <= 0.0):
        raise ValueError("lr and critic_lr must be positive")
    if not math.isfinite(args.adam_epsilon) or args.adam_epsilon <= 0.0:
        raise ValueError("adam_epsilon must be positive")
    if args.minibatch_size > args.train_batch_size:
        raise ValueError("minibatch_size cannot exceed train_batch_size")
    if args.train_batch_size > args.eval_freq:
        raise ValueError("train_batch_size > eval_freq: one PPO iteration could "
                         "skip multiple protocol evaluation thresholds")
    if 0 < args.checkpoint_freq < args.train_batch_size:
        raise ValueError("checkpoint_freq must be 0 or >= train_batch_size")
    if args.video_enabled:
        if args.video_freq <= 0:
            raise ValueError("video_freq must be positive when eval video is enabled")
        if args.video_freq % args.eval_freq != 0:
            raise ValueError(
                "video_freq must be an integer multiple of eval_freq so recording "
                "reuses a scheduled deterministic evaluation"
            )
        if args.video_height <= 0 or args.video_width <= 0:
            raise ValueError("video_height and video_width must be positive")
        if not str(args.video_camera_id).strip():
            raise ValueError("video_camera_id must be non-empty")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not 0.0 < args.clip_param < 1.0:
        raise ValueError("clip_param must be in (0, 1)")
    if args.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if spec.rl_module == "ctde" and not 0.0 < args.vf_clip_param < 1.0:
        raise ValueError(
            "CTDE vf_clip_param must be in (0, 1); stale vf_clip_param=1000 "
            "arguments are invalid under prediction-delta clipping"
        )
    if spec.rl_module == "rllib_default" and args.vf_clip_param <= 0.0:
        raise ValueError("RLlib-default vf_clip_param must be positive")
    if not 0.0 <= args.value_norm_beta < 1.0:
        raise ValueError("value_norm_beta must be in [0, 1)")
    if args.value_norm_epsilon <= 0.0:
        raise ValueError("value_norm_epsilon must be positive")
    positive_network = {
        "hidden_orthogonal_gain": args.hidden_orthogonal_gain,
        "policy_output_gain": args.policy_output_gain,
        "value_output_gain": args.value_output_gain,
        "value_norm_variance_floor": args.value_norm_variance_floor,
    }
    if any(not math.isfinite(v) or v <= 0.0 for v in positive_network.values()):
        raise ValueError(f"network gains and variance floor must be positive: "
                         f"{positive_network}")
    if not all(math.isfinite(v) for v in (
        args.initial_log_std, args.log_std_min, args.log_std_max
    )) or args.log_std_min >= args.log_std_max:
        raise ValueError("log_std bounds must be finite and min < max")
    if not args.log_std_min <= args.initial_log_std <= args.log_std_max:
        raise ValueError("initial_log_std must lie within [log_std_min, log_std_max]")
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
    if args.sample_timeout_s <= 0:
        raise ValueError("sample_timeout_s must be positive")
    if args.max_sampling_stalls <= 0:
        raise ValueError("max_sampling_stalls must be positive")
    return spec


# ===========================================================================
# Environment plumbing
# ===========================================================================


def make_env_for_rllib(env_config: Mapping[str, Any]) -> Any:
    """RLlib env creator. ``include_global_state`` arrives via env_config.

    Each env runner gets a DISTINCT env seed. ``EnvContext`` carries
    worker_index / vector_index; a plain dict (evaluation path) does not, so
    both fall back to 0 and the driver-side probe stays reproducible.
    """
    from omnipiano.multiagent.compile.environment import (
        make_parallel_from_task,
        resolve_registered_task,
    )

    base_seed = int(env_config.get("seed", 0))
    worker_index = int(getattr(env_config, "worker_index", 0) or 0)
    vector_index = int(getattr(env_config, "vector_index", 0) or 0)

    task = env_config.get("task")
    if task is None:
        task = resolve_registered_task(str(env_config["env_id"])).to_dict()
    parallel_env = make_parallel_from_task(
        task,
        seed=base_seed + 1_000 * worker_index + vector_index,
        flatten_obs=True,
        include_global_state=bool(env_config.get("include_global_state", False)),
        inter_agent_collision_penalty_coef=float(
            env_config.get("inter_agent_collision_penalty_coef", 0.0)
        ),
    )
    # RLlib permits only ``__common__`` as a non-agent info key; the native
    # PettingZoo env emits ``_global_``. Translate at this adapter boundary.
    return wrap_parallel_env_for_rllib(parallel_env)


def probe_agent_spaces(
    env_id: str, seed: int, include_global_state: bool,
    task: Optional[Mapping[str, Any]] = None,
) -> Tuple[Sequence[str], Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Tuple[int, int]]]]:
    """Return agents, obs/action spaces, and the flat actor/critic slice layout."""
    from omnipiano.multiagent.compile.environment import (
        make_parallel_from_task,
        resolve_registered_task,
    )

    if task is None:
        task = resolve_registered_task(env_id).to_dict()
    env = make_parallel_from_task(
        task, seed=seed, flatten_obs=True,
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
        "latest_checkpoint_path.txt", "checkpoints.jsonl", "videos.jsonl", "videos",
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


def _resolved_ppo_config(config: Any) -> Dict[str, Any]:
    raw = config.to_dict()
    keys = ("gamma", "use_critic", "use_gae", "lambda", "lambda_", "use_kl_loss",
            "kl_coeff", "kl_target", "clip_param", "vf_clip_param", "entropy_coeff",
            "vf_loss_coeff", "grad_clip", "grad_clip_by", "lr", "train_batch_size",
            "train_batch_size_per_learner", "minibatch_size", "num_epochs",
            "shuffle_batch_per_epoch", "rollout_fragment_length", "batch_mode",
            "normalize_actions", "clip_actions", "framework", "framework_str",
            "num_learners", "learner_config_dict")
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
        args.env_id, args.seed, spec.needs_global_state,
        task=getattr(args, "_resolved_task", None),
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
            "task": getattr(args, "_resolved_task", None),
            "include_global_state": bool(spec.needs_global_state),
            "inter_agent_collision_penalty_coef": float(
                args.inter_agent_collision_penalty_coef
            ),
        })
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers,
                     num_cpus_per_env_runner=args.num_cpus_per_env_runner,
                     sample_timeout_s=args.sample_timeout_s)
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
            grad_clip=args.grad_clip, grad_clip_by=args.grad_clip_by,
        )
        .debugging(seed=args.seed)
    )
    # An algorithm's declared overrides are applied in a SEPARATE .training()
    # call, never as **kwargs inside the call above: an override that names an
    # explicit kwarg there (e.g. clip_param) would raise
    # "got multiple values for keyword argument". A second call cannot collide
    # and always wins, because AlgorithmConfig.training() only writes the keys
    # it is given.
    if spec.training_overrides:
        config = config.training(**dict(spec.training_overrides))

    # num_agents is NOT an env_config knob in this codebase; see
    # ppo_monolithic.py. Refuse loudly rather than write a label we did not honour.
    if spec.num_agents_override is not None:
        raise NotImplementedError(
            f"--algo {spec.name} declares num_agents_override="
            f"{spec.num_agents_override}, but make_env_for_rllib / "
            f"probe_agent_spaces / make_parallel / evaluate_marl do not thread it. "
            f"Setting env_config['num_agents'] would be a SILENT NO-OP and the "
            f"artifacts would misstate the agent count."
        )

    # ---- the learner class has exactly ONE resolution site ----
    # Getting this wrong is the single failure mode that produces plausible but
    # mislabelled curves, so the choice is made once and recorded.
    if spec.rl_module == "ctde":
        from omnipiano.multiagent.algos.ppo_module import (
            build_ctde_module_spec, build_multi_module_spec,
        )
        from omnipiano.multiagent.algos.ppo_learner import OmniPianoPPOTorchLearner

        learner_cls = spec.resolve_learner_class() or OmniPianoPPOTorchLearner
        # A CTDE algorithm that does NOT inherit our learner would silently lose
        # the separate actor/critic Adam optimizers, critic_lr, adam_epsilon,
        # value normalisation and prediction-delta vf clipping -- i.e. it would
        # differ from MAPPO in five ways instead of one, and the ablation would
        # be meaningless.
        if not issubclass(learner_cls, OmniPianoPPOTorchLearner):
            raise TypeError(
                f"--algo {spec.name}: learner_class {learner_cls.__name__} must "
                f"subclass OmniPianoPPOTorchLearner for rl_module='ctde', or the "
                f"comparison against ippo/mappo is not architecturally matched."
            )
        config = (
            config.learners(
                learner_class=learner_cls,
                learner_config_dict={
                    "critic_lr": float(args.critic_lr),
                    "adam_epsilon": float(args.adam_epsilon),
                },
            )
            .rl_module(rl_module_spec=build_multi_module_spec({
                a: build_ctde_module_spec(
                    observation_space=obs_spaces[a], action_space=act_spaces[a],
                    own_slice=layouts[a]["own"],
                    global_state_slice=layouts[a]["global_state"],
                    critic_input=spec.critic_input,
                    hidden_sizes=args.hidden_sizes_parsed,
                    activation=args.activation,
                    hidden_orthogonal_gain=args.hidden_orthogonal_gain,
                    policy_output_gain=args.policy_output_gain,
                    value_output_gain=args.value_output_gain,
                    initial_log_std=args.initial_log_std,
                    log_std_min=args.log_std_min,
                    log_std_max=args.log_std_max,
                    input_layer_norm=args.input_layer_norm,
                    value_norm=args.value_norm,
                    value_norm_beta=args.value_norm_beta,
                    value_norm_epsilon=args.value_norm_epsilon,
                    value_norm_variance_floor=args.value_norm_variance_floor,
                ) for a in agents
            }))
        )
    else:
        learner_cls = spec.resolve_learner_class()
        if learner_cls is not None:
            config = config.learners(learner_class=learner_cls)

    # Record what was ACTUALLY installed, not what the branch implies.
    run_config["effective_config"]["learner_class"] = (
        learner_cls.__name__ if learner_cls is not None
        else "RLlib default PPOTorchLearner"
    )
    run_config["effective_config"]["learner_class_module"] = (
        learner_cls.__module__ if learner_cls is not None else "ray.rllib"
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
        ray.init(ignore_reinit_error=True,
                 log_to_driver=bool(args.ray_log_to_driver),
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
            append_jsonl(run_dir / "progress.jsonl", row)
            wandb_run.log_train(total_steps, row, result)
            if iterations == 1 or iterations % args.log_every_iters == 0:
                elapsed = max(time.time() - start, 1e-9)
                sps = total_steps / elapsed
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
