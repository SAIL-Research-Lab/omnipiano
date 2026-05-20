import gymnasium as gym
import numpy as np
from omnipiano.configs import SafetyConfig
from omnipiano.utils.info_keys import InfoKeys, EpisodeInfoKeys

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
        
        # Evaluate all constraints — each constraint returns a cost already scaled by
        # its own penalty_coef. The total cost is the sum of all weighted constraint costs.
        # Per-constraint costs are also logged individually via constraint.get_info_key().
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
            
        # CRITICAL FIX: As a Benchmark, we MUST NOT modify the reward directly.
        # Safe RL algorithms (like CPO, PPO-Lagrangian) expect the environment to
        # return the raw reward and expose the cost via the `info` dictionary.
        # It is the algorithm's responsibility to decide how to balance reward and cost.
        # reward -= step_cost  <-- REMOVED
        
        # Step info
        info[InfoKeys.STEP_SAFETY_COST_TOTAL] = step_cost
        info[InfoKeys.STEP_SAFETY_VIOLATION_ANY] = step_violation
        
        # Episode info
        if terminated or truncated:
            info[EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL] = self.ep_cost
            info[EpisodeInfoKeys.EPISODE_SAFETY_VIOLATIONS] = self.ep_violations
            
        return obs, reward, terminated, truncated, info
