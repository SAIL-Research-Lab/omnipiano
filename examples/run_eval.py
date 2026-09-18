"""Run standalone LLM evaluation for OmniPiano.

The evaluation loop and metric aggregation live in
``omnipiano.integrations.eval_callback.evaluate_policy``. This script only
loads a policy, creates ``make(..., mode="eval")``, and writes artifacts.

Set OPENAI_API_KEY, then run::

    python examples/run_eval.py \\
        --env OmniPiano-ClairDeLune-Clean-v0

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnipiano import make
from omnipiano.integrations.eval_callback import (
    evaluate_policy,
    write_eval_summary,
)
from omnipiano.integrations.llm import LLMConfig, LLMPolicy
from omnipiano.integrations.policy_registry import (
    register_policy,
    registered_policies,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", required=True, help="Registered OmniPiano env id.")
    parser.add_argument("--num-eval-eps", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--eval-noise-scale",
        type=float,
        default=1.0,
        help="0=clean, 1=matched, >1=stress evaluation.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to examples/logs/eval.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record every evaluation episode under <out-dir>/videos.",
    )
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir or "examples/logs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "mode": "eval",
        "log_dir": str(out_dir),
        "eval_noise_scale": args.eval_noise_scale,
        "seed": args.eval_seed,
    }
    if args.record:
        video_dir = out_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        env_kwargs["record_dir"] = str(video_dir)

    env = make(args.env, **env_kwargs)
    try:
        # Register every policy that should be evaluated in this run.
        llm_config = LLMConfig(action_space=env.action_space)
        register_policy("llm", LLMPolicy, llm_config)

        for name, (policy, config) in registered_policies():
            result = evaluate_policy(
                policy,
                args.env,
                env,
                args.eval_seed,
                args.num_eval_eps,
                config,
            )
            result["algorithm"] = name
            output_path = out_dir / f"eval_summary_{name}.json"
            write_eval_summary(result, output_path)
            print(f"eval summary -> {output_path}")
            print(json.dumps(result["summary"], indent=2, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    main()
