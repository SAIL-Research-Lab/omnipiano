"""Compatibility entry point for the canonical MAPPO baseline.

MAPPO and IPPO intentionally share the implementation in
:mod:`omnipiano.multiagent.train`; their registered ``AlgoSpec`` objects only
change the critic input and whether the environment emits global state.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from omnipiano.multiagent.train import run_algorithm_entrypoint


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the shared trainer with ``--algo mappo`` forced."""
    return run_algorithm_entrypoint("mappo", argv)


if __name__ == "__main__":
    sys.exit(main())
