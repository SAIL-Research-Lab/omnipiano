"""Compatibility entry point for the canonical IPPO baseline.

The implementation lives in :mod:`omnipiano.multiagent.train`. Keeping this
module as a thin adapter preserves existing cluster commands and checkpoint
metadata without allowing a second training loop to drift away from MAPPO.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from omnipiano.multiagent.train import (
    DEFAULT_ENV_ID,
    RLLIB_TARGET_VERSION,
    build_arg_parser as _build_arg_parser,
    make_env_for_rllib as _make_env_for_rllib,
    probe_agent_spaces as _probe_agent_spaces,
    run_algorithm_entrypoint,
)

ALGORITHM_NAME = "IPPO (independent PPO; decentralized per-agent critics)"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the shared trainer with ``--algo ippo`` forced."""
    return run_algorithm_entrypoint("ippo", argv)


if __name__ == "__main__":
    sys.exit(main())
