import gymnasium as gym
import numpy as np
from safe_robopianist.configs import SafetyConfig
from safe_robopianist.metrics.info_keys import InfoKeys, EpisodeInfoKeys

class SafetyWrapper(gym.Wrapper):
    """Calculates safety costs and violations using modular constraints."""
    
    def __init__(self, env, config: SafetyConfig):
        super().__init__(env)
        self.config = config
        self.constraints = self.config.constraints
        
        # Episode accumulators
        self.ep_cost = 0.0
        self.ep_violations = 0
        
    def reset(self, **kwargs):
        self.ep_cost = 0.0
        self.ep_violations = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        step_cost = 0.0
        step_violation = False
        
        # Evaluate all constraints
        for constraint in self.constraints:
            cost = constraint.compute_cost(self.env, action, obs, info)
            if cost > 0:
                step_cost += cost
                step_violation = True
            info[constraint.get_info_key()] = cost
                
        # Accumulate
        self.ep_cost += step_cost
        if step_violation:
            self.ep_violations += 1
            
        # Modify reward
        reward -= step_cost
        
        # Step info
        info[InfoKeys.SAFETY_COST_TOTAL] = step_cost
        info[InfoKeys.SAFETY_VIOLATION_JOINT_LIMIT] = step_violation
        
        # Episode info
        if terminated or truncated:
            info[EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL] = self.ep_cost
            info[EpisodeInfoKeys.EPISODE_SAFETY_VIOLATIONS] = self.ep_violations
            
        return obs, reward, terminated, truncated, info
