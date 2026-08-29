"""TorchRL PPO / SAC trainer for OmniPiano Phase 0.

This is intentionally a small framework-native entry point. It shares the
benchmark protocol and output conventions with the SB3 trainer templates, but
does not import or reuse their training helpers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import torch

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from omnipiano.configs import BenchmarkProtocolConfig  # noqa: E402
from omnipiano.integrations.torch_rl.env.evaluator import (  # noqa: E402
    evaluate_policy,
    write_eval_summary,
)
from omnipiano.integrations.torch_rl.env.factory import (  # noqa: E402
    make_eval_env,
    make_parallel_env,
)
from omnipiano.integrations.torch_rl.algo.ppo import make_config as make_ppo_config, train_ppo  # noqa: E402
from omnipiano.integrations.torch_rl.algo.sac import make_config as make_sac_config, train_sac  # noqa: E402
from omnipiano.integrations.torch_rl.algo.eppo import make_config as make_eppo_config, train_eppo  # noqa: E402
from omnipiano.integrations.torch_rl.algo.a2p_sac import make_config as make_a2p_sac_config, train_a2p_sac  # noqa: E402
from omnipiano.integrations.torch_rl.algo.ompo import make_config as make_ompo_config, train_ompo  # noqa: E402


def _parse_net_arch(value: str) -> tuple[int, ...]:
    return tuple(int(width) for width in value.split(",") if width)


def _next_run_id(logs_root: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    ids = [
        int(match.group(1))
        for path in logs_root.iterdir()
        if path.is_dir() and (match := pattern.match(path.name))
    ]
    return max(ids, default=0) + 1


def _short_env_token(env_id: str) -> str:
    """Match the naming convention used by the existing baseline scripts."""
    parts = env_id.split("-")
    body = [part for part in parts if part and part != "OmniPiano" and not part.startswith("v")]
    return "_".join(part.lower() for part in body)


ALGO_REGISTRY = {
    "ppo": (make_ppo_config, train_ppo),
    "sac": (make_sac_config, train_sac),
    "eppo": (make_eppo_config, train_eppo),
    "a2p_sac": (make_a2p_sac_config, train_a2p_sac),
    "ompo": (make_ompo_config, train_ompo),
}


def _build_arg_parser() -> argparse.ArgumentParser:
    proto = BenchmarkProtocolConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- experiment ---
    parser.add_argument("--algo", required=True, choices=sorted(ALGO_REGISTRY))
    parser.add_argument(
        "--env", default="OmniPiano-ClairDeLune-Clean-v0",
        help="Registered OmniPiano env id.",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="Log-dir prefix under examples/logs/. Default includes algo, env, and seed.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Pipeline validation only: 20K environment steps, 10K eval cadence, 1 eval episode.",
    )

    # --- shared benchmark protocol ---
    parser.add_argument("--total-steps", type=int, default=proto.total_env_steps)
    parser.add_argument("--seed", type=int, default=proto.seed)
    parser.add_argument("--num-eval-eps", type=int, default=proto.num_eval_eps)
    parser.add_argument(
        "--eval-freq", type=int, default=proto.eval_freq_env_steps,
        help="Periodic evaluation cadence in aggregate environment steps.",
    )
    parser.add_argument(
        "--gamma", type=float, default=proto.gamma,
        help="OmniPiano task discount from BenchmarkProtocolConfig.",
    )

    # --- common network / collector settings ---
    parser.add_argument(
        "--n-envs", type=int, default=None,
        help="Override the algorithm default (PPO=16, SAC=24).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--checkpoint-at", default="",
        help="Comma-separated environment-step checkpoints, e.g. 5000000,3000000.",
    )
    parser.add_argument(
        "--net-arch", default="256,256",
        help="Comma-separated hidden-layer widths for actor and critic networks.",
    )

    # --- PPO settings ---
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--critic-coeff", type=float, default=0.5)

    # --- EPPO settings ---
    parser.add_argument("--eppo-mode", choices=["cor", "ind", "mean"], default="cor")
    parser.add_argument("--kappa", type=float, default=0.01)
    parser.add_argument("--evidential-reg", type=float, default=0.01)

    # --- A2P-SAC settings ---
    parser.add_argument(
        "--a2p-epsilon-mode", choices=["adaptive", "fixed", "none"], default="adaptive"
    )
    parser.add_argument("--a2p-epsilon", type=float, default=0.1)
    parser.add_argument("--adversary-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adversary-update-freq", type=int, default=10)
    parser.add_argument("--a2p-warmup-steps", type=int, default=None)

    # --- OMPO settings ---
    parser.add_argument("--ompo-local-buffer-size", type=int, default=1_000)
    parser.add_argument("--ompo-updates-per-step", type=int, default=3)
    parser.add_argument(
        "--ompo-discriminator-observation", choices=["dynamic", "full"], default="dynamic"
    )
    parser.add_argument("--ompo-exponent", type=float, default=1.5)
    parser.add_argument("--ompo-correction-coeff", type=float, default=0.001)
    parser.add_argument("--ompo-reward-max", type=float, default=1.0)

    # --- SAC settings ---
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--utd", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    proto = BenchmarkProtocolConfig()

    if args.smoke_test:
        args.total_steps = 20_000
        args.eval_freq = 10_000
        args.num_eval_eps = 1

    protocol = replace(
        proto,
        total_env_steps=args.total_steps,
        gamma=args.gamma,
        eval_freq_env_steps=args.eval_freq,
        num_eval_eps=args.num_eval_eps,
    )
    device = args.device

    checkpoint_steps = tuple(
        int(value) for value in args.checkpoint_at.split(",") if value.strip()
    )
    logs_root = Path(__file__).resolve().parent / "logs"
    logs_root.mkdir(exist_ok=True)
    experiment_name = args.experiment_name or (
        f"{args.algo}_torchrl_baseline_{_short_env_token(args.env)}_seed{args.seed}"
    )
    run_dir = logs_root / f"{experiment_name}_{_next_run_id(logs_root, experiment_name)}"
    run_dir.mkdir()
    hidden_sizes = _parse_net_arch(args.net_arch)

    config_builder, trainer = ALGO_REGISTRY[args.algo]
    config = config_builder(args, device)
    if args.n_envs is not None:
        config.n_envs = args.n_envs

    eval_seed = args.seed + config.n_envs + 1
    print(
        f"Env: {args.env}\n"
        f"Mode: {'SMOKE_TEST' if args.smoke_test else 'canonical'} | "
        f"seed={args.seed} eval_seed={eval_seed} n_envs={config.n_envs} "
        f"device={device} total_steps={args.total_steps:,}"
    )
    train_env = make_parallel_env(args.env, args.seed, config.n_envs)
    periodic_eval_env = make_eval_env(args.env, eval_seed, str(run_dir))
    print(f"Training for {args.total_steps:,} environment steps...")
    policy, _ = trainer(
        args.env, args.seed, run_dir, protocol, config,
        train_env, periodic_eval_env, hidden_sizes, checkpoint_steps,
    )
    periodic_eval_env.close()
    print("Running final benchmark eval...")
    final_eval_env = make_eval_env(args.env, eval_seed, str(run_dir))
    result = evaluate_policy(policy, args.env, final_eval_env, eval_seed, args.num_eval_eps)
    final_eval_env.close()
    result.update(
        {
            "framework": "torchrl",
            "algorithm": args.algo.upper(),
            "seed": args.seed,
            "total_env_steps": args.total_steps,
            "protocol_version": protocol.protocol_version,
            "smoke_test": args.smoke_test,
            "hparams": {**vars(config), "net_arch": list(hidden_sizes)},
        }
    )
    out_path = run_dir / "eval_summary.json"
    write_eval_summary(result, out_path)
    print(f"eval summary -> {out_path}")
    print(f"summary: {json.dumps(result['summary'], indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
