"""MAPPO trainer: per-agent actor over o_i + centralized critic over s (CTDE).

This is a thin alias for the shared multi-agent PPO trainer with
``--algo mappo``.  Sharing one training loop with IPPO is deliberate: it makes
"the two baselines drifted apart in their training loop" structurally
impossible, so an IPPO/MAPPO comparison is a genuine single-factor ablation.

BREAKING CHANGE: previous revisions of this file were a deprecation shim that
forwarded to the IPPO trainer.  It now trains MAPPO.  For IPPO use
``python -m omnipiano.multiagent._train_ippo``.

Example::

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \\
    python -m omnipiano.multiagent._train_mappo \\
        --env-id OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0 \\
        --seed 0 --total-steps 10000000
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from omnipiano.multiagent._train_ippo import main as _shared_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--algo" in argv:
        raise SystemExit(
            "_train_mappo implies --algo mappo; use _train_ippo to select an "
            "algorithm explicitly."
        )
    return _shared_main(["--algo", "mappo", *argv])


if __name__ == "__main__":
    sys.exit(main())