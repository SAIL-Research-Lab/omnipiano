"""Policies selected for the current evaluation run."""

_POLICIES = {}


def register_policy(name: str, factory, config=None) -> None:
    if name in _POLICIES:
        raise ValueError(f"policy already registered: {name}")
    _POLICIES[name] = (factory, config)


def registered_policies():
    return tuple(_POLICIES.items())
