"""Algorithm registry. Importing this package registers every baseline."""

from omnipiano.multiagent.algos.base import (  # noqa: F401
    AlgoSpec,
    algo_table,
    get_algo,
    list_algos,
    register_algo,
)

# Import for registration side effects. Add new algorithms here.
from omnipiano.multiagent.algos import ippo as _ippo  # noqa: F401,E402
from omnipiano.multiagent.algos import mappo as _mappo  # noqa: F401,E402

__all__ = ["AlgoSpec", "algo_table", "get_algo", "list_algos", "register_algo"]