"""Compatibility entry point for algorithm-agnostic MARL evaluation."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from omnipiano.multiagent.evaluate import build_arg_parser as _build_arg_parser
from omnipiano.multiagent.evaluate import main as _evaluate_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _evaluate_main(argv)


if __name__ == "__main__":
    sys.exit(main())
