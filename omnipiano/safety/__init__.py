"""Safety experiments and compatibility for their pinned NumPy 1.x runtime."""

import sys as _sys

import numpy as _np

# The old main registry installs a NumPy-2 shim even under NumPy 1.x. That shim
# drops array-only keywords (e.g. subok), breaking dm_env's broadcast_to calls.
# NumPy 1.x already supports copy=False; restore its original implementation only
# when the active replacement is that exact upstream shim. Leave NumPy 2 alone.
_registration = _sys.modules.get("omnipiano.envs.registration")
if (
    _np.__version__.split(".")[0] == "1"
    and _registration is not None
    and _np.array is getattr(_registration, "_np_array_compat", None)
):
    _np.array = _registration._orig_np_array

del _registration
