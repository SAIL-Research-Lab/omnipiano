import gymnasium as gym
import csv
import os
import time
import uuid
from OmniPiano.utils.info_keys import EpisodeInfoKeys


class SafeRecordEpisodeStatistics(gym.Wrapper):
    """
    A Gymnasium wrapper that automatically logs episode statistics (returns, lengths, 
    safety costs, and musical metrics) to a CSV file.
    
    This makes the logging completely algorithm-agnostic. It works with Stable Baselines 3, 
    CleanRL, or any custom training loop.
    """
    def __init__(self, env, log_dir, env_id=None, split="train"):
        super().__init__(env)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.split = split
        
        # Use env_id to avoid collision in vectorized environments (e.g., SubprocVecEnv)
        self.env_id = env_id if env_id is not None else str(uuid.uuid4())[:8]
        self.csv_path = os.path.join(
            self.log_dir, f"{self.split}_episode_metrics_{self.env_id}.csv"
        )
        
        # Cumulative number of finished episodes for this env instance.
        # This intentionally does NOT reset on every `reset()`.
        self.completed_episode_count = 0
        self.env_step_count = 0  # Per-env local step counter since wrapper init.
        self.t0 = time.perf_counter()
        self.episode_return = 0.0
        self.episode_length = 0
        
        # Initialize CSV header
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                # `episode` is a per-env cumulative completed-episode index.
                'env_step_count', 'episode', 'time_elapsed', 'ep_return', 'ep_length',
                'ep_cost', 'ep_violations', 'ep_f1', 'ep_precision', 'ep_recall',
                'ep_sustain_f1', 'ep_sustain_precision', 'ep_sustain_recall',
                'energy_reward', 'fingering_reward', 'forearm_reward', 'key_press_reward', 'sustain_reward'
            ])

    def reset(self, **kwargs):
        # Reset the wrapped env first to obtain the initial observation/info pair
        # required by Gymnasium's reset API, then clear logger-local episode
        # accumulators so a new episode starts from zeroed stats.
        obs, info = self.env.reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_return += reward
        self.episode_length += 1
        self.env_step_count += 1
        
        if terminated or truncated:
            self.completed_episode_count += 1
            t = time.perf_counter() - self.t0
            
            ep_cost = info.get(EpisodeInfoKeys.EPISODE_SAFETY_COST_TOTAL, 0.0)
            ep_violations = info.get(EpisodeInfoKeys.EPISODE_SAFETY_VIOLATIONS, 0)
            
            ep_f1 = info.get(EpisodeInfoKeys.EPISODE_TASK_F1, "")
            ep_precision = info.get(EpisodeInfoKeys.EPISODE_TASK_KEY_PRECISION, "")
            ep_recall = info.get(EpisodeInfoKeys.EPISODE_TASK_KEY_RECALL, "")
            
            ep_sus_f1 = info.get(EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_F1, "")
            ep_sus_prec = info.get(EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_PRECISION, "")
            ep_sus_rec = info.get(EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_RECALL, "")
            
            energy_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_ENERGY_REWARD, "")
            fingering_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_FINGERING_REWARD, "")
            forearm_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_FOREARM_REWARD, "")
            key_press_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_KEY_PRESS_REWARD, "")
            sustain_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_REWARD, "")
            
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.env_step_count,
                    self.completed_episode_count,
                    round(t, 2),
                    self.episode_return,
                    self.episode_length,
                    ep_cost,
                    ep_violations,
                    ep_f1, ep_precision, ep_recall,
                    ep_sus_f1, ep_sus_prec, ep_sus_rec,
                    energy_rew, fingering_rew, forearm_rew, key_press_rew, sustain_rew
                ])
                
        return obs, reward, terminated, truncated, info
