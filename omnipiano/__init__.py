"""OmniPiano: A Safety and Robustness Benchmark for Robot Piano Playing."""

import os as _os
import sys as _sys

# Ensure our vendored robopianist (under envs/) is found before any other
# robopianist on sys.path (e.g., the repo-level robopianist/ directory).
_envs_dir = _os.path.join(_os.path.dirname(__file__), "envs")
if _envs_dir not in _sys.path:
    _sys.path.insert(0, _envs_dir)

from omnipiano.envs.registration import make, register  # noqa: E402

# Importing envs triggers all register() calls in envs/__init__.py
from omnipiano import envs  # noqa: E402, F401
