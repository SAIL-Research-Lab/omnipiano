"""SB3 per-environment adapter for benchmark reward curves."""

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env as _make_vec_env

from omnipiano.utils.info_keys import InfoKeys


class TrueRewardWrapper(gym.Wrapper):
    """Select the benchmark reward metric without changing training rewards."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_return = 0.0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        true_reward = float(info[InfoKeys.TASK_TRUE_REWARD])
        self._episode_return += true_reward

        if terminated or truncated:
            if "episode" in info:
                # Monitor has already created this dictionary. SB3 reads it
                # into ep_info_buffer for rollout/ep_rew_mean.
                info["episode"]["r"] = self._episode_return
            self._episode_return = 0.0

        return observation, reward, terminated, truncated, info


def make_vec_env(*args, **kwargs):
    """Call SB3's ``make_vec_env`` with OmniPiano's Monitor adapter."""
    if "wrapper_class" in kwargs or "wrapper_kwargs" in kwargs:
        raise TypeError(
            "OmniPiano make_vec_env manages wrapper_class internally."
        )
    return _make_vec_env(
        *args,
        wrapper_class=TrueRewardWrapper,
        **kwargs,
    )
