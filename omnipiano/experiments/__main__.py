"""Command-line validation and planning for unified OmniPiano experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from omnipiano.experiments.catalog import (
    CapabilityError,
    TRACKS,
    catalog_dict,
    validate_request_capabilities,
)
from omnipiano.experiments.adapters import AdapterError
from omnipiano.experiments.compiler import bind_plan, compile_plan
from omnipiano.experiments.schema import ExperimentValidationError, load_request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omnipiano.experiments",
        description="Validate or plan a backend-neutral OmniPiano experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate one JSON file")
    validate.add_argument("config", type=Path)

    catalog = subparsers.add_parser(
        "catalog", help="print current track and algorithm capabilities"
    )
    catalog.add_argument("--track", choices=TRACKS, default=None)
    catalog.add_argument("--include-unavailable", action="store_true")
    catalog.add_argument("--compact", action="store_true")

    plan = subparsers.add_parser(
        "plan", help="resolve defaults and expand one plan per seed"
    )
    plan.add_argument("config", type=Path)
    plan.add_argument("--output", type=Path, default=None)
    plan.add_argument("--compact", action="store_true")

    bind = subparsers.add_parser(
        "bind",
        help="translate every seed plan to an existing backend command without running it",
    )
    bind.add_argument("config", type=Path)
    bind.add_argument("--output", type=Path, default=None)
    bind.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "catalog":
            indent = None if args.compact else 2
            print(
                json.dumps(
                    catalog_dict(
                        track=args.track,
                        include_unavailable=args.include_unavailable,
                    ),
                    indent=indent,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            return 0

        request = load_request(args.config)
        if args.command == "validate":
            validate_request_capabilities(request)
            print(
                f"VALID: {request.experiment.name} "
                f"({request.design.type}, {len(request.experiment.seeds)} seed(s))"
            )
            return 0

        bundle = bind_plan(request) if args.command == "bind" else compile_plan(request)
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
    except (ExperimentValidationError, CapabilityError, AdapterError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
