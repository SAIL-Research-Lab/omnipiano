"""OmniPiano: a benchmark for dexterous robot piano playing.

Environment registration is lazy so dependency-light tooling such as the
experiment composer can import ``omnipiano.experiments`` without initializing
MuJoCo or a graphics backend.  Existing public calls (``omnipiano.make`` and
``omnipiano.register``) retain the same interface and trigger registration on
first access.
"""

import os as _os
import sys as _sys

# Ensure our vendored robopianist (under envs/) is found before any other
# robopianist on sys.path (e.g., the repo-level robopianist/ directory).
_envs_dir = _os.path.join(_os.path.dirname(__file__), "envs")
if _envs_dir not in _sys.path:
    _sys.path.insert(0, _envs_dir)

# ---------------------------------------------------------------------------
# Environment entrypoints, exposed lazily (PEP 562).
#
# ``envs/registration.make()`` tells the user to call
# ``omnipiano.make_parallel(...)`` when it receives a ``-MA-`` id, and
# ``multi_agent_design.md`` § 4 documents it as a top-level entrypoint. An
# eager import would make PettingZoo a hard dependency of ``import omnipiano``
# even though it is an optional ``[marl]`` extra, so resolve on first access.
# ---------------------------------------------------------------------------
_MA_LAZY_ATTRS = {
    "make_parallel": "omnipiano.multiagent",
    "make_parallel_from_task": "omnipiano.multiagent",
    "register_parallel": "omnipiano.multiagent",
    "list_parallel_envs": "omnipiano.multiagent",
    "AGENT_ASSIGNMENTS": "omnipiano.multiagent",
}

_LAZY_ATTRS = {
    "make": "omnipiano.envs.registration",
    "register": "omnipiano.envs.registration",
    **_MA_LAZY_ATTRS,
}


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'omnipiano' has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if name not in _MA_LAZY_ATTRS:
            raise
        raise ImportError(
            f"omnipiano.{name} requires the multi-agent extra: "
            f"pip install -e '.[marl]'"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache; __getattr__ is not called again
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
