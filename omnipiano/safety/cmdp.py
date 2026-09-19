"""OmniSafe cost channel; no algorithm implementation is modified."""
import numpy as np
import torch
from omnisafe.envs.core import CMDP, env_register

import omnipiano
from omnipiano.safety.suite import register_all


@env_register
class SafetySemanticCMDP(CMDP):
    _support_envs = sorted(register_all())
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(self, env_id, **kwargs):
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("Only one environment per process is supported")
        self._num_envs = 1
        self._device = kwargs.get("device", torch.device("cpu"))
        self._env = omnipiano.make(env_id)
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

    def step(self, action):
        obs, reward, term, trunc, info = self._env.step(action.detach().cpu().numpy().astype(np.float64))
        cost = float(info["step_safety/cost_total"])
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("Invalid environment cost")
        values = [torch.as_tensor(x, dtype=torch.float32, device=self._device) for x in (obs, reward, cost)]
        flags = [torch.as_tensor(x, dtype=torch.bool, device=self._device) for x in (term, trunc)]
        return (*values, *flags, info)

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed):
        self._env.reset(seed=seed)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
