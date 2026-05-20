"""Action-noise injection at the gym layer.

Background — paper-chain refactor (see omnipiano/envs/registration.py)
---------------------------------------------------------------------

In the paper chain, observations are flattened into a single Box by
``ConcatObservationWrapper`` at the dm_env level, before reaching the
gym layer. A gym-level wrapper that wants per-key obs noise can no
longer iterate ``obs.keys()`` (the flat ndarray has no keys). So
per-key obs noise was moved to the dm_env layer
(``omnipiano/envs/dm_env_obs_noise.DmEnvObsNoiseWrapper``), where the
observation is still a Dict.

What this wrapper does
----------------------

* Injects action noise (operates on the flat action vector — no
  dependency on obs format).
* Reads the L2 norm of obs noise injected at the dm_env layer
  (via the cached ``DmEnvObsNoiseWrapper`` reference) and reports it
  through ``info["robust/noise_obs_l2"]``.
* Logs ``info["robust/noise_action_l2"]`` for action noise.

If the chain doesn't contain a ``DmEnvObsNoiseWrapper`` (e.g.,
``robust_config.obs_noise_std == 0``), the obs-noise lookup returns 0
and the info key is reported as 0.0.
"""
import gymnasium as gym
import numpy as np

from omnipiano.configs import RobustConfig
from omnipiano.utils.info_keys import InfoKeys


class RobustWrapper(gym.Wrapper):
    """Inject action noise (gym layer) + report obs noise (read from dm_env layer)."""

    def __init__(self, env, config: RobustConfig):
        super().__init__(env)
        self.config = config

    def step(self, action):
        # 1. Action noise — operates on flat action vector,
        #    completely independent of obs format.
        action_noise_l2 = 0.0
        if self.config.action_noise_std > 0:
            noise = self.np_random.normal(
                0, self.config.action_noise_std, size=action.shape
            )
            action_noise_l2 = float(np.linalg.norm(noise))
            action = np.clip(
                action + noise, self.action_space.low, self.action_space.high
            )

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. Obs noise — already injected at dm_env layer by
        #    DmEnvObsNoiseWrapper.step (if present). Just read its L2
        #    snapshot for logging.
        obs_noise_l2 = 0.0
        if self.config.obs_noise_std > 0:
            obs_noise_l2 = self._read_obs_noise_l2()

        info[InfoKeys.ROBUST_NOISE_ACTION_L2] = action_noise_l2
        info[InfoKeys.ROBUST_NOISE_OBS_L2] = obs_noise_l2

        return obs, reward, terminated, truncated, info

    def _read_obs_noise_l2(self) -> float:
        """Walk dm_env chain → DmEnvObsNoiseWrapper.last_step_noise_l2.

        Re-walked on every call (no caching) for two reasons:

        1. ``DmEnvToGymnasium.reset(seed=X)`` rebuilds the dm_env chain
           in-place (``self._env = self._builder(seed)``). A reference
           cached at ``__init__`` time would point at the *pre-rebuild*
           wrapper instance — which is no longer being stepped, so its
           ``last_step_noise_l2`` is stuck at the value from the last
           step before the rebuild. Re-walking each call always lands
           on the *current* wrapper instance.

        2. If the wrapper isn't found (chain mis-wired or someone
           forgot to insert it when ``obs_noise_std > 0``), raise
           a ``RuntimeError`` with a clear diagnostic, instead of
           silently falling back to 0.0 and letting training proceed
           against a non-noised env.

        Walk cost is ~10 attribute accesses per step (the gym-side
        chain has 3-4 wrappers, dm_env-side has ~7). Negligible vs
        the policy forward pass.
        """
        from omnipiano.envs.dm_env_obs_noise import DmEnvObsNoiseWrapper
        from omnipiano.utils.env_unwrap import (
            find_dm_env_wrapper,
            get_dm_env_from_gym,
        )

        dm_env = get_dm_env_from_gym(self.env)
        wrapper = find_dm_env_wrapper(dm_env, DmEnvObsNoiseWrapper)
        if wrapper is None:
            raise RuntimeError(
                "RobustConfig.obs_noise_std > 0 but DmEnvObsNoiseWrapper "
                "is not present in the dm_env chain. Verify "
                "omnipiano.envs.registration._build_dm_env_chain inserts "
                "DmEnvObsNoiseWrapper BEFORE ConcatObservationWrapper "
                "whenever robust_config.obs_noise_std > 0. If you build "
                "the env outside of omnipiano.make(), insert it yourself "
                "or set obs_noise_std=0."
            )
        return float(wrapper.last_step_noise_l2)
