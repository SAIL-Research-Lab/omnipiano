import gymnasium as gym
import numpy as np
from OmniPiano.configs import RobustConfig
from OmniPiano.utils.info_keys import InfoKeys

class RobustWrapper(gym.Wrapper):
    """Injects robustness perturbations into the environment."""
    
    def __init__(self, env, config: RobustConfig):
        super().__init__(env)
        self.config = config

    def step(self, action):
        # 1. Inject Action Noise
        action_noise_l2 = 0.0
        if self.config.action_noise_std > 0:
            noise = self.np_random.normal(0, self.config.action_noise_std, size=action.shape)
            # Calculate the L2 norm (magnitude) of the noise vector to quantify the perturbation severity for logging
            action_noise_l2 = np.linalg.norm(noise)
            action = np.clip(action + noise, self.action_space.low, self.action_space.high)
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Inject Observation Noise
        obs_noise_l2 = 0.0
        if self.config.obs_noise_std > 0:
            for key in obs.keys():
                # Only inject noise into continuous (floating-point) arrays.
                # This prevents errors when trying to add continuous noise to discrete/boolean 
                # observations (like goal states, step counters, or categorical flags).
                if isinstance(obs[key], np.ndarray) and obs[key].dtype in [np.float32, np.float64]:
                    noise = self.np_random.normal(0, self.config.obs_noise_std, size=obs[key].shape)
                    obs_noise_l2 += np.linalg.norm(noise)
                    obs[key] = obs[key] + noise
                    
        # Log robustness metrics
        info[InfoKeys.ROBUST_NOISE_ACTION_L2] = action_noise_l2
        info[InfoKeys.ROBUST_NOISE_OBS_L2] = obs_noise_l2
        
        return obs, reward, terminated, truncated, info
