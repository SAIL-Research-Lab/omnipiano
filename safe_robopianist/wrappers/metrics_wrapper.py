import gymnasium as gym
from robopianist.wrappers.evaluation import MidiEvaluationWrapper
from safe_robopianist.utils.info_keys import InfoKeys, EpisodeInfoKeys


class MetricsWrapper(gym.Wrapper):
    """
    Extracts and standardizes task metrics (musical performance, reward terms).
    This wrapper is fundamentally required for ALL tasks in the benchmark to ensure
    consistent logging of base performance metrics, regardless of safety/robustness configs.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.ep_reward_terms = {}
        
    def reset(self, **kwargs):
        self.ep_reward_terms = {}
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Extract Reward Terms
        try:
            env_ptr = self.env
            while hasattr(env_ptr, 'env'):
                env_ptr = env_ptr.env
            dm_env = env_ptr._env
            if hasattr(dm_env._environment.task, 'reward_fn'):
                reward_terms = dm_env._environment.task.reward_fn.reward_terms
                for term_name, term_val in reward_terms.items():
                    # Map to standardized keys
                    key = getattr(InfoKeys, f"TASK_{term_name.upper()}", f"task/{term_name}")
                    info[key] = term_val
                    self.ep_reward_terms[term_name] = self.ep_reward_terms.get(term_name, 0.0) + term_val
        except Exception:
            pass # TODO: Add strict mode
            
        # 2. Episode Metrics
        if terminated or truncated:
            # Reward terms
            for term_name, term_val in self.ep_reward_terms.items():
                key = getattr(EpisodeInfoKeys, f"EPISODE_TASK_{term_name.upper()}", f"episode_task/{term_name}")
                info[key] = term_val
                
            # Musical metrics
            try:
                env_ptr = self.env
                while hasattr(env_ptr, 'env'):
                    env_ptr = env_ptr.env
                dm_env = env_ptr._env
                
                if isinstance(dm_env, MidiEvaluationWrapper):
                    metrics = dm_env.get_musical_metrics()
                    info[EpisodeInfoKeys.EPISODE_TASK_F1] = metrics['f1']
                    info[EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION] = metrics['precision']
                    info[EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL] = metrics['recall']
                    info[EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_F1] = metrics['sustain_f1']
                    info[EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_PRECISION] = metrics['sustain_precision']
                    info[EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_RECALL] = metrics['sustain_recall']
            except Exception:
                pass # TODO: Add strict mode
                
        return obs, reward, terminated, truncated, info