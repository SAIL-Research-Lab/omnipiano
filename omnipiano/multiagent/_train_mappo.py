"""Deprecated compatibility entry point for the correctly named IPPO trainer."""

from __future__ import annotations

import sys
import warnings

from omnipiano.multiagent._train_ippo import main


if __name__ == "__main__":
    warnings.warn(
        "_train_mappo was misnamed: this baseline has no centralized critic. "
        "Use `python -m omnipiano.multiagent._train_ippo` instead.",
        FutureWarning,
        stacklevel=1,
    )
    sys.exit(main())
