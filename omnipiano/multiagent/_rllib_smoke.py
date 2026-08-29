"""Run the canonical IPPO trainer's 5k-step end-to-end smoke test.

This remains a standalone manual check rather than a pytest test because it
starts Ray and MuJoCo. It deliberately disables W&B; smoke success is recorded
by the trainer's local JSON artifacts and final checkpoint.
"""

from __future__ import annotations

import sys

from omnipiano.multiagent.train import run_algorithm_entrypoint


SMOKE_ENV_ID = "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0"


def main() -> int:
    return run_algorithm_entrypoint(
        "ippo",
        [
            "--env-id",
            SMOKE_ENV_ID,
            "--smoke-test",
            "--wandb-mode",
            "disabled",
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
