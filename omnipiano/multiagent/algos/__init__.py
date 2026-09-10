"""Algorithm registry. Importing this package registers every baseline."""

from omnipiano.multiagent.algos.base import (  # noqa: F401
    AlgoSpec,
    algo_table,
    get_algo,
    list_algos,
    register_algo,
    resolve_dotted,
)

# Import for registration side effects. Add new algorithms here.
# NOTE: these modules must stay import-light -- no torch, no ray -- so that
# `--list-algos` is a sub-second operation and a broken optional dependency
# cannot take down the registry. Heavy classes are referenced by DOTTED PATH in
# the AlgoSpec and imported lazily by train.py only for the algo actually run.
from omnipiano.multiagent.algos import ippo as _ippo              # noqa: F401,E402
from omnipiano.multiagent.algos import mappo as _mappo            # noqa: F401,E402
from omnipiano.multiagent.algos import happo as _happo            # noqa: F401,E402
from omnipiano.multiagent.algos import mat as _mat                # noqa: F401,E402
from omnipiano.multiagent.algos import facmac as _facmac          # noqa: F401,E402
from omnipiano.multiagent.algos import masac as _masac            # noqa: F401,E402
from omnipiano.multiagent.algos import ppo_monolithic as _ppo_mono  # noqa: F401,E402

__all__ = ["AlgoSpec", "algo_table", "get_algo", "list_algos", "register_algo",
           "resolve_dotted"]