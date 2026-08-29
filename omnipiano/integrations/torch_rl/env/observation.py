"""Observation slices shared by robust TorchRL algorithms."""

from __future__ import annotations

import numpy as np
import torch


def observation_indices(gym_env, mode: str):
    observation_dim = int(np.prod(gym_env.observation_space.shape))
    if mode == "full":
        return torch.arange(observation_dim)

    from dm_env_wrappers import ConcatObservationWrapper
    from omnipiano.utils.env_unwrap import find_dm_env_wrapper, get_dm_env_from_gym

    dm_env = get_dm_env_from_gym(gym_env)
    concat = find_dm_env_wrapper(dm_env, ConcatObservationWrapper)
    inner_spec = concat._environment.observation_spec()
    excluded = ("goal", "lookahead", "reward", "counter", "steps_left")
    indices = []
    offset = 0
    for name in sorted(concat._obs_names):
        width = int(np.prod(inner_spec[name].shape)) if inner_spec[name].shape else 1
        if not any(token in name.lower() for token in excluded):
            indices.extend(range(offset, offset + width))
        offset += width
    return torch.tensor(indices, dtype=torch.long)
