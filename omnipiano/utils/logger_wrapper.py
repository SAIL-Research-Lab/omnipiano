import gymnasium as gym
import csv
import os
import time
import uuid

from omnipiano.utils.episode_csv import (
    EPISODE_CSV_HEADER,
    build_episode_csv_row,
)
from omnipiano.utils.info_keys import InfoKeys


class SafeRecordEpisodeStatistics(gym.Wrapper):
    """Episode-level CSV logger for OmniPiano eval rollouts.

    Writes one CSV row per completed episode (env_step_count, episode index,
    ep_return, ep_length, ep_cost, ep_violations, F1 / precision / recall,
    sustain_*, and reward decomposition) to
    ``<log_dir>/<split>_episode_metrics_<env_id>.csv``. Both this wrapper and
    ``examples/checkpoint_replay_eval.py`` use the shared
    :mod:`omnipiano.utils.episode_csv` serializer so SB3 and OmniSafe eval
    data plot with the same downstream code.

    NOT a drop-in for ``gymnasium.wrappers.RecordEpisodeStatistics``
    --------------------------------------------------------------------
    Despite the similar name, this wrapper does NOT populate
    ``info["episode"] = {"r", "l", "t"}`` — it only writes the CSV file
    as a side effect. Consumers that read ``info["episode"]`` (SB3's
    ``EvalCallback`` for ``ep_rew_mean`` / ``ep_len_mean`` tensorboard
    scalars, CleanRL's logging hooks, etc.) still need a separate
    ``Monitor`` / ``VecMonitor`` wrapper.

    Today this is invisible to users because ``stable_baselines3.common
    .env_util.make_vec_env`` auto-wraps each sub-env in ``Monitor``
    before we attach this wrapper, so both signals coexist. But any
    handwritten training loop that does ``omnipiano.make(env_id,
    log_dir=..., mode="eval")`` and skips ``Monitor`` will get a
    CSV but no ``info["episode"]``, silently breaking downstream
    ``EvalCallback``-style code.

    Activation
    ----------
    Attached only when ``omnipiano.make()`` is called with
    ``mode == "eval"`` AND a ``log_dir``
    (see ``omnipiano/envs/registration.py``).
    Used by the SB3 baseline templates' eval_env construction; not
    attached on training envs or on OmniSafe runs (OmniSafe eval CSVs
    are produced post-hoc by ``examples/checkpoint_replay_eval.py``).

    Required upstream wrapper order (set in ``registration.py``)
    -----------------------------------------------------------
    ``... → MetricsWrapper → SafetyWrapper → RobustWrapper →
    SafeRecordEpisodeStatistics``. By the time ``step()`` runs here,
    ``info`` already contains ``episode_task/*`` (from MetricsWrapper at
    terminal step) and ``episode_safety/*`` (from SafetyWrapper) so this
    wrapper just reads them out.
    """
    def __init__(self, env, log_dir, env_id=None, split="train",
                 eval_noise_scale=0.0):
        super().__init__(env)
        # The effective robustness eval scale this env was built with (0=clean,
        # 1=matched/training-level, >1=stress). Logged per row so a sweep can
        # locate the point on the robustness curve (§0.6, decision 10/11).
        self.eval_noise_scale = eval_noise_scale
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
        # Per-episode robust-noise aggregates, summed over the episode from
        # info["robust/noise_*"]. ep_noise_{action,obs}_l2 are summed per-step
        # L2 norms (dev tripwire); ep_noise_reward is the summed signed reward
        # noise (== ep_return - ep_return_true for reward tasks; §0.6).
        self.ep_noise_action_l2 = 0.0
        self.ep_noise_obs_l2 = 0.0
        self.ep_noise_reward = 0.0

        # Initialize CSV header
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(EPISODE_CSV_HEADER)

    def reset(self, **kwargs):
        # Reset the wrapped env first to obtain the initial observation/info pair
        # required by Gymnasium's reset API, then clear logger-local episode
        # accumulators so a new episode starts from zeroed stats.
        obs, info = self.env.reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        self.ep_noise_action_l2 = 0.0
        self.ep_noise_obs_l2 = 0.0
        self.ep_noise_reward = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_return += reward
        self.episode_length += 1
        self.env_step_count += 1
        self.ep_noise_action_l2 += info.get(InfoKeys.ROBUST_NOISE_ACTION_L2, 0.0)
        self.ep_noise_obs_l2 += info.get(InfoKeys.ROBUST_NOISE_OBS_L2, 0.0)
        self.ep_noise_reward += info.get(InfoKeys.ROBUST_NOISE_REWARD, 0.0)

        if terminated or truncated:
            self.completed_episode_count += 1
            t = time.perf_counter() - self.t0
            
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(build_episode_csv_row(
                    env_step_count=self.env_step_count,
                    episode=self.completed_episode_count,
                    time_elapsed=round(t, 2),
                    ep_return=self.episode_return,
                    ep_length=self.episode_length,
                    info=info,
                    eval_noise_scale=self.eval_noise_scale,
                    ep_noise_action_l2=self.ep_noise_action_l2,
                    ep_noise_obs_l2=self.ep_noise_obs_l2,
                    ep_noise_reward=self.ep_noise_reward,
                ))
                
        return obs, reward, terminated, truncated, info
