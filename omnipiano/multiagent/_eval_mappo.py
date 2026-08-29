"""Evaluate a MAPPO checkpoint.

Evaluation is algorithm-agnostic: it restores the checkpoint, reads the env id
and the ``include_global_state`` flag from ``run_config.json``, and rolls out
deterministic PettingZoo episodes.  The CTDE module's actor reads only
``obs[own_slice]``, so a centralized critic changes nothing at inference time.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from omnipiano.multiagent._eval_ippo import main as _shared_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _shared_main(argv)


if __name__ == "__main__":
    sys.exit(main())