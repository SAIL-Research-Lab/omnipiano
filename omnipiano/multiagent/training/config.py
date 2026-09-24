"""CLI construction, precedence resolution, and validation for MARL training."""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional, Sequence

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent.algos import AlgoSpec, get_algo, list_algos
from omnipiano.multiagent.compile import (
    DEFAULT_TRAIN_CONFIG_PATH,
    compile_experiment,
    resolve_train_config_path,
)


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
        _robust_config=(
            dict(compiled.robust_config)
            if compiled.robust_config is not None else None
        ),
        _native_options=dict(compiled.native_options),
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
    core.add_argument(
        "--resume-native", default=None,
        help=("Resume a trusted native-backend checkpoint into a fresh run "
              "directory. Learner/replay/RNG state is restored; the exact "
              "MuJoCo episode state is not."),
    )
    core.add_argument(
        "--eval-native", default=None,
        help=("Evaluate a trusted native-backend checkpoint and exit without "
              "creating training artifacts."),
    )

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
    if (args.resume_native or args.eval_native) and not spec.is_native:
        raise ValueError(
            "--resume-native/--eval-native require an algorithm with "
            "backend='native'"
        )
    if args.resume_native and args.eval_native:
        raise ValueError("--resume-native and --eval-native are mutually exclusive")
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
    if not spec.is_native and args.minibatch_size > args.train_batch_size:
        raise ValueError("minibatch_size cannot exceed train_batch_size")
    if not spec.is_native and args.train_batch_size > args.eval_freq:
        raise ValueError("train_batch_size > eval_freq: one PPO iteration could "
                         "skip multiple protocol evaluation thresholds")
    if (not spec.is_native
            and 0 < args.checkpoint_freq < args.train_batch_size):
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
