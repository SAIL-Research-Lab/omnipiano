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
* Overrides the ``reward`` slot of the flat observation with the
  *noised observed* reward when the reward channel is active
  (see below).

If the obs channel is inactive, the obs-noise lookup is skipped and the
info key is reported as 0.0. If the channel is active but the chain has
no ``DmEnvObsNoiseWrapper``, the lookup raises ``RuntimeError``
(fail-fast; see ``_read_obs_noise_l2``).

OAR slot semantics (decision 2026-07-21, supersedes Method 6 Option B)
----------------------------------------------------------------------

``ObservationActionRewardWrapper`` sits *below* ``CanonicalSpecWrapper``
in the dm_env chain, so what it naturally records into the flat obs:

* ``obs["action"]`` = the **noised executed** action (``a_exec``) in
  physical actuator units. This IS the intended semantics under the
  hardware-wear threat model (2026-07-21 meeting decision): actuator
  aging/wear means the truly executed action is the noised one, and the
  agent's proprioceptive action memory legitimately reflects it. No
  override is performed — the earlier Method 6 (Option B) clean-``a_cmd``
  override was REMOVED (see ``robust_task_design.md`` §4.7 supersession
  note for the history and the old rationale).
* ``obs["reward"]`` = the **clean** task reward — because reward noise
  is injected here at the gym layer, *above* OAR. That is NOT the
  intended semantics ("worn reward sensor": observed = learned = noised),
  so when the reward channel is active this wrapper overwrites the
  reward slot with the noised received reward ``r_obs``. This is the
  only obs-slot override that remains.

Non-robust envs (and action/obs-only robust envs) take a code path that
never touches (or copies) the observation, so existing baselines are
*structurally* unaffected — not merely numerically.

Layout caching is rebuild-safe: ``DmEnvToGymnasium.reset(seed=X)``
rebuilds the dm_env chain, but the obs-key structure is derived from the
static env config (same MJCF model, same wrapper kwargs), so the reward
slice cached at ``__init__`` remains valid across rebuilds. (Contrast
with ``last_step_noise_l2`` below, which lives on a *wrapper instance*
and must be re-walked every call.)

Limitations (fail-fast guarded):
* ``frame_stack > 1`` with an active reward channel is not supported —
  the flat layout becomes per-frame interleaved and only the newest
  frame's reward slot could be fixed. All OmniPiano / RoboPianist
  protocols use ``frame_stack=1``. (Action noise needs no slot fix, so
  action-channel tasks are stacking-compatible.)
* If the obs Dict has no "reward" key (``action_reward_observation=
  False``), there is nothing to fix and the override is skipped.
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
        # Reward-slot layout — computed once; rebuild-safe (see module
        # docstring).
        self._reward_slice = self._compute_reward_slice()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info[InfoKeys.TASK_TRUE_REWARD] = 0.0
        return obs, info

    # ------------------------------------------------------------------
    # Reward-slot override — construction-time layout computation
    # ------------------------------------------------------------------
    def _compute_reward_slice(self):
        """Locate the "reward" slot in the flat obs vector.

        Returns a slice, or None when the key is absent from the obs Dict
        (``action_reward_observation=False``) or the chain has no flat
        concat layout to address.
        """
        from dm_env_wrappers import (
            ConcatObservationWrapper,
            FrameStackingWrapper,
        )

        from omnipiano.utils.env_unwrap import (
            find_dm_env_wrapper,
            get_dm_env_from_gym,
        )

        dm_env = get_dm_env_from_gym(self.env)

        # Guard: frame stacking interleaves frames in the flat obs; the
        # slot position below would only address one frame. v1 protocols
        # are all frame_stack=1 (paper-same), so fail fast instead of
        # silently overriding the wrong slice. Only the reward channel
        # needs a slot fix (action noise needs none — decision 2026-07-21).
        if find_dm_env_wrapper(dm_env, FrameStackingWrapper) is not None:
            if self.config.is_channel_active("reward"):
                raise NotImplementedError(
                    "RobustWrapper's obs['reward'] override does not support "
                    "frame_stack > 1: the flat obs interleaves frames, so "
                    "only the newest frame's slot could be fixed. All "
                    "OmniPiano / RoboPianist protocols use frame_stack=1; "
                    "see omnipiano/docs/robust_task_design.md §4.7 before "
                    "enabling stacking on a reward robust task."
                )
            return None

        concat = find_dm_env_wrapper(dm_env, ConcatObservationWrapper)
        if concat is None:
            # Non-paper chain (Dict obs all the way up) — no flat slot
            # to override. Reward noise still applies to the scalar.
            return None

        # ConcatObservationWrapper flattens {k: obs[k] for k in _obs_names}
        # via tree.flatten, which iterates dict keys in SORTED (alphabetical)
        # order — see its class docstring ("fields ... in sorted order by
        # their names"). Scalars are promoted to 1-d by np.atleast_1d, so a
        # shape-() spec contributes one element.
        inner_spec = concat._environment.observation_spec()
        offset = 0
        reward_slice = None
        for key in sorted(concat._obs_names):
            dim = int(np.prod(inner_spec[key].shape)) if inner_spec[key].shape else 1
            if key == "reward":
                reward_slice = slice(offset, offset + dim)
            offset += dim

        expected_dim = int(np.prod(self.env.observation_space.shape))
        if offset != expected_dim:
            raise RuntimeError(
                f"RobustWrapper: flat-obs layout mismatch — alphabetical "
                f"concat of the dm_env Dict spec gives {offset} dims but "
                f"observation_space has {expected_dim}. The ConcatObs "
                f"ordering assumption may have changed upstream; do NOT "
                f"trust the computed slot. Keys: {sorted(concat._obs_names)}"
            )

        return reward_slice

    # ------------------------------------------------------------------
    # Noise dispatch — delegates to RobustConfig (single source of truth,
    # shared with the dm_env-layer DmEnvObsNoiseWrapper so the two layers
    # never diverge). See RobustConfig.is_channel_active / sample_noise.
    # ------------------------------------------------------------------
    def _channel_active(self, channel: str) -> bool:
        return self.config.is_channel_active(channel)

    def _sample_noise(self, rng, channel: str, shape):
        return self.config.sample_noise(rng, channel, shape)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        # 1. Action noise — operates on flat action vector, completely
        #    independent of obs format. Distribution per config.noise_dist.
        #    The noised action is what physics executes AND what OAR
        #    naturally records into obs["action"] (physical units) — the
        #    intended hardware-wear semantics (decision 2026-07-21), so no
        #    a_cmd bookkeeping or slot override is needed.
        action_noise_l2 = 0.0
        if self._channel_active("action"):
            noise = self._sample_noise(
                self.np_random, channel="action", shape=action.shape
            )
            action_noise_l2 = float(np.linalg.norm(noise))
            action = np.clip(
                action + noise, self.action_space.low, self.action_space.high
            )

        obs, reward, terminated, truncated, info = self.env.step(action)
        info[InfoKeys.TASK_TRUE_REWARD] = float(reward)

        # 2. Obs noise — already injected at dm_env layer by
        #    DmEnvObsNoiseWrapper.step (if present). Just read its L2
        #    snapshot for logging.
        obs_noise_l2 = 0.0
        if self._channel_active("obs"):
            obs_noise_l2 = self._read_obs_noise_l2()

        # 3. Reward noise (decision 11: matched at eval, NOT force-zeroed).
        #    Add noise to the scalar reward the agent receives and trains on.
        #    Shares the gym np_random stream with action noise. v1 tasks are
        #    single-channel (only one of action/reward active), but a
        #    combined-channel task (both active) is ALSO correct: action noise
        #    is sampled above BEFORE reward noise here, so the two consume the
        #    shared stream in a FIXED order → deterministic + reproducible
        #    (§12 decision 8, verified by test_robust_multichannel). Obs noise
        #    uses a separate dm_env-layer stream, so it never interferes.
        reward_noise = 0.0
        reward_active = self._channel_active("reward")
        if reward_active:
            reward_noise = float(
                self._sample_noise(self.np_random, channel="reward", shape=())
            )
            reward = float(reward) + reward_noise

        # 4. Reward-slot override — only when reward noise is active, so
        #    non-robust (and action/obs-only robust) envs return the inner
        #    obs object untouched (no copy), leaving existing baselines
        #    structurally unaffected.
        #    obs["reward"] -> the noised observed reward r_obs the agent
        #    received, so its reward memory matches the training signal and
        #    stays in-distribution at matched eval (Wang 2020; decision 11).
        #    (obs["action"] needs no override: OAR naturally records the
        #    noised executed action, which is the intended semantics —
        #    decision 2026-07-21.)
        if reward_active and self._reward_slice is not None:
            obs = obs.copy()
            obs[self._reward_slice] = np.asarray(reward, dtype=obs.dtype)

        info[InfoKeys.ROBUST_NOISE_ACTION_L2] = action_noise_l2
        info[InfoKeys.ROBUST_NOISE_OBS_L2] = obs_noise_l2
        info[InfoKeys.ROBUST_NOISE_REWARD] = reward_noise
        environment_state = self._read_environment_noise_state()
        info[InfoKeys.ROBUST_NOISE_GRAVITY] = environment_state["gravity_noise"]
        info[InfoKeys.ROBUST_NOISE_CONTACT_FRICTION] = environment_state[
            "contact_friction_noise"
        ]
        info[InfoKeys.ROBUST_ENV_GRAVITY_Z] = environment_state["gravity_z"]
        info[InfoKeys.ROBUST_ENV_CONTACT_FRICTION_SLIDING] = environment_state[
            "contact_friction_sliding"
        ]
        info[InfoKeys.ROBUST_ENV_HAND_POSITION_L2] = environment_state[
            "hand_position_l2"
        ]
        info[InfoKeys.ROBUST_ENV_HAND_POSITION_MAX_L2] = environment_state[
            "hand_position_max_l2"
        ]
        info[InfoKeys.ROBUST_ENV_HAND_POSITION_OFFSETS] = environment_state[
            "hand_position_offsets"
        ]

        return obs, reward, terminated, truncated, info

    def _read_environment_noise_state(self):
        """Read the current task instance after any seed-triggered rebuild."""
        environment_noise_active = any(
            self.config.is_channel_active(parameter)
            for parameter in self.config.environment_noise.PARAMETERS
        )

        from omnipiano.utils.env_unwrap import get_composer_env_from_gym

        composer_env = get_composer_env_from_gym(self.env)
        task = composer_env.task
        if not hasattr(task, "environment_noise_state"):
            if not environment_noise_active:
                return {
                    "gravity_noise": 0.0,
                    "gravity_z": 0.0,
                    "contact_friction_noise": 0.0,
                    "contact_friction_sliding": 0.0,
                    "hand_position_offsets": {},
                    "hand_position_l2": 0.0,
                    "hand_position_max_l2": 0.0,
                }
            raise RuntimeError(
                "Environment noise is active but the composer task does not "
                "expose environment_noise_state. Build robust environment "
                "tasks with OmniPianoTask through omnipiano.make()."
            )
        return task.environment_noise_state

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
                "RobustConfig obs channel is active but DmEnvObsNoiseWrapper "
                "is not present in the dm_env chain. Verify "
                "omnipiano.envs.registration._build_dm_env_chain inserts "
                "DmEnvObsNoiseWrapper BEFORE ConcatObservationWrapper "
                "whenever robust_config.is_channel_active('obs'). If you "
                "build the env outside of omnipiano.make(), insert it "
                "yourself or deactivate the obs channel."
            )
        return float(wrapper.last_step_noise_l2)
