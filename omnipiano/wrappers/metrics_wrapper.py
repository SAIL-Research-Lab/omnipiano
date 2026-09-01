import gymnasium as gym
from robopianist.wrappers.evaluation import MidiEvaluationWrapper
from omnipiano.utils.info_keys import (
    InfoKeys,
    EpisodeInfoKeys,
    benchmark_metrics_to_episode_info,
)
from omnipiano.utils.env_unwrap import (
    get_composer_env_from_gym,
    get_dm_env_from_gym,
    find_dm_env_wrapper,
)


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
        # Pipeline reminder (with source files):
        # 1) Unwrap to composer env: `get_composer_env_from_gym(...)`
        #    - file: omnipiano/utils/env_unwrap.py
        # 2) dm_control step calls `task.get_reward(...)`
        #    - file: dm_control/composer/environment.py (in site-packages)
        # 3) Task forwards reward call to `self._reward_fn.compute(physics)`
        #    - file: omnipiano/envs/robopianist/suite/tasks/piano_with_shadow_hands.py
        # 4) Composite reward computes each term and updates `reward_terms`
        #    - file: omnipiano/envs/robopianist/suite/composite_reward.py
        #
        # This wrapper runs right after env.step(...) returns, so we read
        # `composer_env.task.reward_fn.reward_terms` to capture per-step sub-reward values
        # (energy, key_press, sustain, fingering, etc.) and then aggregate them per episode.
        composer_env = get_composer_env_from_gym(self.env)
        if not hasattr(composer_env.task, "reward_fn"):
            raise AttributeError(
                "MetricsWrapper expects task.reward_fn for reward term extraction."
            )
        reward_terms = composer_env.task.reward_fn.reward_terms
        for term_name, term_val in reward_terms.items():
            attr_name = f"TASK_{term_name.upper()}"
            if not hasattr(InfoKeys, attr_name):
                raise AttributeError(
                    f"InfoKeys has no attribute '{attr_name}'. "
                    f"Add it to info_keys.py or check if the reward term "
                    f"name '{term_name}' has changed in RoboPianist."
                )
            info[getattr(InfoKeys, attr_name)] = term_val
            self.ep_reward_terms[term_name] = self.ep_reward_terms.get(term_name, 0.0) + term_val
            
        # 2. Episode Metrics
        if terminated or truncated:
            # Reward terms
            for term_name, term_val in self.ep_reward_terms.items():
                attr_name = f"EPISODE_TASK_{term_name.upper()}"
                if not hasattr(EpisodeInfoKeys, attr_name):
                    raise AttributeError(
                        f"EpisodeInfoKeys has no attribute '{attr_name}'. "
                        f"Add it to info_keys.py or check if the reward term "
                        f"name '{term_name}' has changed in RoboPianist."
                    )
                info[getattr(EpisodeInfoKeys, attr_name)] = term_val
                
            # Musical metrics
            dm_env = get_dm_env_from_gym(self.env)
            midi_eval_wrapper = find_dm_env_wrapper(dm_env, MidiEvaluationWrapper)
            if midi_eval_wrapper is None:
                raise RuntimeError(
                    "MetricsWrapper expects MidiEvaluationWrapper in dm_env chain."
                )
            metrics = midi_eval_wrapper.get_musical_metrics()
            info.update(benchmark_metrics_to_episode_info(metrics))
                
        return obs, reward, terminated, truncated, info
