import gymnasium as gym
import csv
import json
import os
import time
import uuid
from omnipiano.utils.info_keys import EpisodeInfoKeys, InfoKeys


class SafeRecordEpisodeStatistics(gym.Wrapper):
    """Episode-level CSV logger for OmniPiano eval rollouts.

    Writes one CSV row per completed episode (env_step_count, episode index,
    ep_return, ep_length, ep_cost, ep_violations, F1 / precision / recall,
    sustain_*, and reward decomposition) to
    ``<log_dir>/<split>_episode_metrics_<env_id>.csv``. Schema is locked
    to ``examples/checkpoint_replay_eval.py``'s ``_CSV_HEADER`` so SB3 +
    OmniSafe eval data plot with the same downstream code.

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
        self.env_noise_gravity = 0.0
        self.env_noise_contact_friction = 0.0
        self.env_gravity_z = 0.0
        self.env_contact_friction_sliding = 0.0
        self.env_hand_position_l2 = 0.0
        self.env_hand_position_max_l2 = 0.0
        self.env_hand_position_offsets = {}

        # Initialize CSV header
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                # `episode` is a per-env cumulative completed-episode index.
                'env_step_count', 'episode', 'time_elapsed', 'ep_return', 'ep_length',
                'ep_cost', 'ep_violations', 'ep_f1', 'ep_precision', 'ep_recall',
                'ep_sustain_f1', 'ep_sustain_precision', 'ep_sustain_recall',
                # Reward decomposition. ``fingering_reward`` and
                # ``ot_fingering_reward`` are mutually exclusive per env
                # (controlled by ``disable_fingering_reward``); each row
                # populates exactly one and leaves the other blank. Keep
                # them in separate columns so they're never confused at
                # paper-writing time — the two functions have different
                # mathematical definitions (annotation-pair distance vs.
                # Hungarian-matched K-to-K distance) and are not directly
                # comparable.
                'energy_reward', 'fingering_reward', 'ot_fingering_reward',
                'forearm_reward', 'key_press_reward', 'sustain_reward',
                # Robust-eval columns (§0.6, decision 11), appended so existing
                # column indices are unchanged. `ep_return` (col 4) stays the
                # received/accumulated return (= noised for reward-noise tasks);
                # `ep_return_true` is the clean/denoised return = sum of the
                # reward-decomposition terms above. F1 remains the noise-immune
                # headline. `eval_noise_scale` locates the robustness-curve
                # point; `ep_noise_*` are the per-episode summed injected noise.
                'eval_noise_scale', 'ep_return_true',
                'ep_noise_action_l2', 'ep_noise_obs_l2', 'ep_noise_reward',
                'ep_noise_gravity', 'ep_noise_contact_friction',
                'env_gravity_z', 'env_contact_friction_sliding',
                'env_hand_position_l2', 'env_hand_position_max_l2',
                'env_hand_position_offsets',
            ])

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
        self.env_noise_gravity = 0.0
        self.env_noise_contact_friction = 0.0
        self.env_gravity_z = 0.0
        self.env_contact_friction_sliding = 0.0
        self.env_hand_position_l2 = 0.0
        self.env_hand_position_max_l2 = 0.0
        self.env_hand_position_offsets = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_return += reward
        self.episode_length += 1
        self.env_step_count += 1
        self.ep_noise_action_l2 += info.get(InfoKeys.ROBUST_NOISE_ACTION_L2, 0.0)
        self.ep_noise_obs_l2 += info.get(InfoKeys.ROBUST_NOISE_OBS_L2, 0.0)
        self.ep_noise_reward += info.get(InfoKeys.ROBUST_NOISE_REWARD, 0.0)
        # Physical parameters are constant within an episode. Keep the latest
        # snapshot instead of summing it once per step.
        self.env_noise_gravity = info.get(
            InfoKeys.ROBUST_NOISE_GRAVITY, 0.0
        )
        self.env_noise_contact_friction = info.get(
            InfoKeys.ROBUST_NOISE_CONTACT_FRICTION, 0.0
        )
        self.env_gravity_z = info.get(InfoKeys.ROBUST_ENV_GRAVITY_Z, 0.0)
        self.env_contact_friction_sliding = info.get(
            InfoKeys.ROBUST_ENV_CONTACT_FRICTION_SLIDING, 0.0
        )
        self.env_hand_position_l2 = info.get(
            InfoKeys.ROBUST_ENV_HAND_POSITION_L2, 0.0
        )
        self.env_hand_position_max_l2 = info.get(
            InfoKeys.ROBUST_ENV_HAND_POSITION_MAX_L2, 0.0
        )
        self.env_hand_position_offsets = info.get(
            InfoKeys.ROBUST_ENV_HAND_POSITION_OFFSETS, {}
        )

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
            ot_fingering_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_OT_FINGERING_REWARD, "")
            forearm_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_FOREARM_REWARD, "")
            key_press_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_KEY_PRESS_REWARD, "")
            sustain_rew = info.get(EpisodeInfoKeys.EPISODE_TASK_SUSTAIN_REWARD, "")

            # Clean/denoised return = sum of the reward-decomposition terms
            # (MetricsWrapper reads pre-noise physics, so this is noise-immune).
            # Inactive terms are logged blank ("") and skipped. Equals the
            # received ep_return minus the accumulated reward noise.
            ep_return_true = sum(
                float(x) for x in (
                    energy_rew, fingering_rew, ot_fingering_rew,
                    forearm_rew, key_press_rew, sustain_rew,
                ) if x != ""
            )

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
                    energy_rew, fingering_rew, ot_fingering_rew,
                    forearm_rew, key_press_rew, sustain_rew,
                    self.eval_noise_scale, ep_return_true,
                    self.ep_noise_action_l2, self.ep_noise_obs_l2,
                    self.ep_noise_reward,
                    self.env_noise_gravity,
                    self.env_noise_contact_friction,
                    self.env_gravity_z,
                    self.env_contact_friction_sliding,
                    self.env_hand_position_l2,
                    self.env_hand_position_max_l2,
                    json.dumps(
                        self.env_hand_position_offsets,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ])
                
        return obs, reward, terminated, truncated, info
