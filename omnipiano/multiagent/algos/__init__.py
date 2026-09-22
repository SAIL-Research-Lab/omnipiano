"""Algorithm registry. Importing this package registers every baseline."""

from omnipiano.multiagent.algos.base import (  # noqa: F401
    AlgoSpec, algo_table, get_algo, list_algos, register_algo, resolve_dotted,
)

# Import for registration side effects. Add new algorithms here.
# These modules must stay import-light -- no torch, no ray -- so `--list-algos`
# is sub-second and a broken optional dependency cannot take down the registry.
# Heavy classes are referenced by DOTTED PATH in the AlgoSpec and imported
# lazily by train.py only for the algorithm actually being run.
from omnipiano.multiagent.algos import ippo as _ippo                    # noqa: F401,E402
from omnipiano.multiagent.algos import mappo as _mappo                  # noqa: F401,E402
from omnipiano.multiagent.algos import happo as _happo                  # noqa: F401,E402
from omnipiano.multiagent.algos import a2po as _a2po                    # noqa: F401,E402
from omnipiano.multiagent.algos import mat as _mat                      # noqa: F401,E402
from omnipiano.multiagent.algos import facmac as _facmac                # noqa: F401,E402
from omnipiano.multiagent.algos import masac as _masac                  # noqa: F401,E402
from omnipiano.multiagent.algos import ppo_monolithic as _ppo_mono      # noqa: F401,E402
# NOTE: no standalone `sac` import. Single-agent SAC is not a MARL baseline;
# MASAC is the multi-agent member of that family. Importing it here would put a
# non-MARL algorithm in the baseline table.

__all__ = ["AlgoSpec", "algo_table", "get_algo", "list_algos", "register_algo",
           "resolve_dotted"]
