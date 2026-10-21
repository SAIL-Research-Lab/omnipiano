"""Command-line validation and planning for unified OmniPiano experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from omnipiano.experiments.compiler import compile_plan
from omnipiano.experiments.schema import ExperimentValidationError, load_request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omnipiano.experiments",
        description="Validate or plan a backend-neutral OmniPiano experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate one JSON file")
    validate.add_argument("config", type=Path)

    plan = subparsers.add_parser(
        "plan", help="resolve defaults and expand one plan per seed"
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("--output", type=Path, default=None)
    plan.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = load_request(args.config)
        if args.command == "validate":
            print(
                f"VALID: {request.experiment.name} "
                f"({request.design.type}, {len(request.experiment.seeds)} seed(s))"
            )
            return 0

        bundle = compile_plan(request)
        indent = None if args.compact else 2
        rendered = json.dumps(
            bundle.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False
        )
        if args.output is None:
            print(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"WROTE: {args.output}")
        return 0
    except ExperimentValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
