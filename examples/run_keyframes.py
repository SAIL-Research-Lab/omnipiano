"""Let DeepSeek write, practice, and freeze a keyframe controller.

Each practice attempt runs the generated play.py in a separate Python process,
then returns F1, sounded-note timing, mismatches, and rendered frames to the
same model conversation. Formal evaluation remains in run_eval.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omnipiano import make
from omnipiano.integrations.llm import KeyframeOptimizer, KeyframeOptimizerConfig


def _short_env_token(env_id):
    return "_".join(
        part.lower() for part in env_id.split("-")
        if part and part != "OmniPiano" and not part.startswith("v")
    )


def _run_dir(root, name):
    run_ids = [
        int(path.name.rsplit("_", 1)[1])
        for path in root.glob(f"{name}_[0-9]*")
        if path.name.rsplit("_", 1)[1].isdigit()
    ]
    return root / f"{name}_{max(run_ids, default=0) + 1}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Registered OmniPiano env id.")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-noise-scale", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Continue an existing --out-dir conversation; "
            "--tries is the total number of practices."
        ),
    )
    args = parser.parse_args(argv)
    if args.resume and not args.out_dir:
        parser.error("--resume requires --out-dir")
    # api_key = os.environ.get("DEEPSEEK_API_KEY")
    # if not api_key:
    #     parser.error("DEEPSEEK_API_KEY environment variable is required")

    # api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     parser.error("OPENAI_API_KEY environment variable is required")

    # api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     parser.error("GEMINI_API_KEY environment variable is required")

    # api_key = os.environ.get("GLM_API_KEY")
    # if not api_key:
    #     parser.error("GLM_API_KEY environment variable is required")

    # api_key = os.environ.get("GROK_API_KEY")
    # if not api_key:
    #     parser.error("GROK_API_KEY environment variable is required")

    api_key = os.environ.get("CLAUDE_API_KEY")
    
    llm_kwargs = {
        "model": "claude-opus-5",
        "base_url": "https://4router.net/v1",
        "api_key": api_key,
        # "thinking": True,
        "temperature": 0,
        "max_tokens": 65_536,
    }
    out_dir = Path(args.out_dir) if args.out_dir else _run_dir(
        Path(__file__).resolve().parent / "logs",
        f"{llm_kwargs['model'].replace('/', '_')}_keyframes_"
        f"{_short_env_token(args.env)}_seed{args.eval_seed}",
    )
    if args.resume and not (out_dir / f"keyframes_{args.env}_conversation.json").is_file():
        parser.error(f"no saved conversation in {out_dir} for {args.env}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_keyframes] log_dir={out_dir}")
    env = make(
        args.env,
        mode="eval",
        eval_noise_scale=args.eval_noise_scale,
        seed=args.eval_seed,
    )
    try:
        config = KeyframeOptimizerConfig(
            env=env,
            action_space=env.action_space,
            path=str(out_dir / f"keyframes_{args.env}.json"),
            practice_seed=args.eval_seed,
            num_optimization_steps=args.tries,
            resume=args.resume,
            **llm_kwargs,
        )
        KeyframeOptimizer(config).optimize()
    finally:
        env.close()


if __name__ == "__main__":
    main()
