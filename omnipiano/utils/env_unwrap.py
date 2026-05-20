"""Helpers for unwrapping Gym / dm_env wrapper chains.

Walks ``env.env`` (gym.Wrapper convention) down to the gym↔dm_env
adapter, then reads its ``_env`` attribute to enter the dm_env chain.
The adapter is ``omnipiano.envs.dm_env_adapter.DmEnvToGymnasium``;
the same access pattern also works for ``shimmy.DmControlCompatibilityV0``
since both expose ``_env``.
"""


def get_dm_env_from_gym(env):
    """Unwrap Gymnasium wrappers and return the underlying dm_env object."""
    env_ptr = env
    while hasattr(env_ptr, "env"):
        env_ptr = env_ptr.env

    if not hasattr(env_ptr, "_env"):
        raise AttributeError(
            "Expected a dm_env adapter exposing '_env' after unwrapping Gym wrappers."
        )
    return env_ptr._env


def unwrap_dm_env_chain(dm_env):
    """Unwrap dm_env wrappers via public `.environment` until the base env."""
    env_ptr = dm_env
    while hasattr(env_ptr, "environment"):
        next_env = env_ptr.environment
        if next_env is env_ptr:
            raise RuntimeError("Detected self-referential dm_env wrapper chain.")
        env_ptr = next_env
    return env_ptr


def get_composer_env_from_gym(env):
    """Return underlying composer-like env exposing `physics` and `task`."""
    dm_env = get_dm_env_from_gym(env)
    composer_env = unwrap_dm_env_chain(dm_env)
    if not hasattr(composer_env, "physics") or not hasattr(composer_env, "task"):
        raise AttributeError(
            "Failed to unwrap to a composer environment exposing physics/task."
        )
    return composer_env


def find_dm_env_wrapper(dm_env, wrapper_type):
    """Find the first wrapper instance of `wrapper_type` in dm_env chain."""
    env_ptr = dm_env
    while True:
        if isinstance(env_ptr, wrapper_type):
            return env_ptr
        if not hasattr(env_ptr, "environment"):
            return None
        next_env = env_ptr.environment
        if next_env is env_ptr:
            raise RuntimeError("Detected self-referential dm_env wrapper chain.")
        env_ptr = next_env
