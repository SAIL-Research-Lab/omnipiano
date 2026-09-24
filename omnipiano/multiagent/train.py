"""Public command-line entry point for OmniPiano MARL training.

Implementation lives in :mod:`omnipiano.multiagent.training`; this module
keeps the stable CLI and the small compatibility surface used by tests and
external launchers.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.multiagent.algos import AlgoSpec
from omnipiano.multiagent.compile import DEFAULT_TRAIN_CONFIG_PATH
from omnipiano.multiagent.training import config as _config
from omnipiano.multiagent.training import runner as _runner
from omnipiano.multiagent.training.rllib import (
    RLLIB_TARGET_VERSION,
    make_env_for_rllib,
    probe_agent_spaces,
)


def _sync_entrypoint_dependencies() -> None:
    """Propagate intentionally patchable entry-point dependencies."""
    _config.BenchmarkProtocolConfig = BenchmarkProtocolConfig
    _runner.BenchmarkProtocolConfig = BenchmarkProtocolConfig
    _runner.probe_agent_spaces = probe_agent_spaces


def build_arg_parser(config_path=None, *, algo_override=None):
    _sync_entrypoint_dependencies()
    return _config.build_arg_parser(
        config_path=config_path, algo_override=algo_override
    )


def _parse_args(argv: Optional[Sequence[str]]):
    _sync_entrypoint_dependencies()
    return _config._parse_args(argv)


def _resolve_args(args) -> AlgoSpec:
    _sync_entrypoint_dependencies()
    return _config._resolve_args(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _sync_entrypoint_dependencies()
    return _runner.main(argv)


if __name__ == "__main__":
    sys.exit(main())
