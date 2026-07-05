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
* **Method 6 (Option B)** — when action noise is active, overrides the
  ``action`` slot of the flat observation with the *clean commanded*
  action (in physical actuator units), fixing the noise-information
  leak described in ``omnipiano/docs/robust_task_design.md`` §4.7.

If the chain doesn't contain a ``DmEnvObsNoiseWrapper`` (e.g.,
``robust_config.obs_noise_std == 0``), the obs-noise lookup returns 0
and the info key is reported as 0.0.

Method 6 (Option B) — obs["action"] principled-semantics fix
------------------------------------------------------------

Problem: ``ObservationActionRewardWrapper`` sits *below*
``CanonicalSpecWrapper`` in the dm_env chain, so the value it records
into the flat obs "action" slot is the **noised executed** action
(``a_exec``) after canonical→physical rescaling. Under the standard
robust-RL threat model (Disrupted-MDP, Gu et al. 2025), the policy's
own action memory must be the **clean commanded** action (``a_cmd``);
exposing ``a_exec`` leaks the injected noise to the policy, which can
then trivially learn to counteract it.

Fix: after the inner ``env.step`` returns, overwrite the "action"
slot of the flat obs with ``physical(a_cmd)``, computed by replaying
the *exact* transform pipeline the executed action goes through:

    DmEnvToGymnasium.step:       np.asarray(action, dtype=np.float32)
    SinglePrecisionWrapper.step: passes the action through unchanged
    CanonicalSpecWrapper.step:   _scale_nested_action(action, spec, clip)

We call dm_env_wrappers' own ``_scale_nested_action`` with the *same*
spec object and clip flag cached from the chain's
``CanonicalSpecWrapper``, so the computed value is bit-identical to
what the executed path would produce for the same input. Importing
from ``dm_env_wrappers._src`` is deliberate: reimplementing the
formula risks 1-ulp dtype-promotion differences that the A6 gate test
(``omnipiano/tests/test_robust_v1_method6.py``) would flag.

The override only runs when action noise is actually injected
(``action_noise_std > 0``); non-robust envs take a code path that
never touches (or copies) the observation, so all existing baselines
are *structurally* unaffected — not merely numerically.

Layout caching is rebuild-safe: ``DmEnvToGymnasium.reset(seed=X)``
rebuilds the dm_env chain, but the obs-key structure, physical action
spec values, and clip flag are all derived from the static env config
(same MJCF model, same wrapper kwargs), so values cached at
``__init__`` remain valid across rebuilds. (Contrast with
``last_step_noise_l2`` below, which lives on a *wrapper instance* and
must be re-walked every call.)

Limitations (fail-fast guarded):
* ``frame_stack > 1`` is not supported — the flat layout becomes
  per-frame interleaved and only the newest frame's slot could be
  fixed. All OmniPiano / RoboPianist protocols use ``frame_stack=1``.
* If the obs Dict has no "action" key (``action_reward_observation=
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
        # Method 6 layout — computed once; rebuild-safe (see module docstring).
        (
            self._action_slice,
            self._reward_slice,
            self._physical_action_spec,
            self._canonical_clip,
        ) = self._compute_override_layout()

    # ------------------------------------------------------------------
    # Method 6 — construction-time layout computation
    # ------------------------------------------------------------------
    def _compute_override_layout(self):
        """Locate the "action" / "reward" slots in the flat obs vector.

        Returns ``(action_slice, reward_slice, physical_action_spec, clip)``.
        Either slice may be None (key absent from the obs Dict); the
        spec/clip pair is (None, False) if the chain has no
        ``CanonicalSpecWrapper`` (then commanded == executed coordinates
        and the override degenerates to a float32 cast).
        """
        from dm_env_wrappers import (
            CanonicalSpecWrapper,
            ConcatObservationWrapper,
            FrameStackingWrapper,
        )

        from omnipiano.utils.env_unwrap import (
            find_dm_env_wrapper,
            get_dm_env_from_gym,
        )

        dm_env = get_dm_env_from_gym(self.env)

        # Guard: frame stacking interleaves frames in the flat obs; the
        # slot positions below would only address one frame. v1 protocols
        # are all frame_stack=1 (paper-same), so fail fast instead of
        # silently overriding the wrong slice.
        if find_dm_env_wrapper(dm_env, FrameStackingWrapper) is not None:
            if self.config.action_noise_std > 0:
                raise NotImplementedError(
                    "RobustWrapper's obs['action'] override (Method 6) does "
                    "not support frame_stack > 1. All OmniPiano / "
                    "RoboPianist protocols use frame_stack=1; see "
                    "omnipiano/docs/robust_task_design.md §4.7 before "
                    "enabling stacking on a robust task."
                )
            return None, None, None, False

        concat = find_dm_env_wrapper(dm_env, ConcatObservationWrapper)
        if concat is None:
            # Non-paper chain (Dict obs all the way up) — no flat slots
            # to override. Action noise still applies to physics.
            return None, None, None, False

        # ConcatObservationWrapper flattens {k: obs[k] for k in _obs_names}
        # via tree.flatten, which iterates dict keys in SORTED (alphabetical)
        # order — see its class docstring ("fields ... in sorted order by
        # their names"). Scalars are promoted to 1-d by np.atleast_1d, so a
        # shape-() spec contributes one element.
        inner_spec = concat._environment.observation_spec()
        offset = 0
        action_slice = None
        reward_slice = None
        for key in sorted(concat._obs_names):
            dim = int(np.prod(inner_spec[key].shape)) if inner_spec[key].shape else 1
            if key == "action":
                action_slice = slice(offset, offset + dim)
            elif key == "reward":
                reward_slice = slice(offset, offset + dim)
            offset += dim

        expected_dim = int(np.prod(self.env.observation_space.shape))
        if offset != expected_dim:
            raise RuntimeError(
                f"RobustWrapper: flat-obs layout mismatch — alphabetical "
                f"concat of the dm_env Dict spec gives {offset} dims but "
                f"observation_space has {expected_dim}. The ConcatObs "
                f"ordering assumption may have changed upstream; do NOT "
                f"trust the computed slots. Keys: {sorted(concat._obs_names)}"
            )

        canonical = find_dm_env_wrapper(dm_env, CanonicalSpecWrapper)
        if canonical is not None:
            # The wrapper's stored spec is the *physical* action spec of the
            # chain below it — exactly what _scale_nested_action rescales to.
            physical_spec = canonical._action_spec
            clip = canonical._clip
        else:
            physical_spec = None
            clip = False

        return action_slice, reward_slice, physical_spec, clip

    def _clean_physical(self, a_cmd: np.ndarray) -> np.ndarray:
        """physical(a_cmd): what the executed path would have stored for a
        noise-free command. Replays the exact transform pipeline (see
        module docstring) using dm_env_wrappers' own scaling function and
        the cached spec/clip, guaranteeing bit-identical arithmetic."""
        a = np.asarray(a_cmd, dtype=np.float32)  # DmEnvToGymnasium's cast
        if self._physical_action_spec is None:
            return a
        from dm_env_wrappers._src.canonical_spec import _scale_nested_action

        return _scale_nested_action(
            a, self._physical_action_spec, self._canonical_clip
        )

    # ------------------------------------------------------------------
    # Noise distribution dispatch (Option C.3 — per-distribution fields)
    # ------------------------------------------------------------------
    def _channel_active(self, channel: str) -> bool:
        """True if ``channel`` has nonzero noise magnitude for the currently
        active ``noise_dist`` (gaussian→std, uniform→low/high, shift→shift)."""
        dist = self.config.noise_dist
        if dist == "gaussian":
            return getattr(self.config, f"{channel}_noise_std") != 0.0
        if dist == "uniform":
            lo = getattr(self.config, f"{channel}_noise_uniform_low")
            hi = getattr(self.config, f"{channel}_noise_uniform_high")
            return (lo != 0.0) or (hi != 0.0)
        if dist == "shift":
            return getattr(self.config, f"{channel}_noise_shift") != 0.0
        raise ValueError(f"Unknown noise_dist: {self.config.noise_dist!r}")

    def _sample_noise(self, rng, channel: str, shape):
        """Sample noise for ``channel`` from the active distribution.

        - gaussian: read ``{channel}_noise_std`` as σ → N(0, σ²) per-dim
          (step-level). Kept **bit-identical** to the pre-Phase-0 direct
          ``rng.normal(0, std, size=shape)`` call (equivalence gate §0.7).
        - uniform:  read ``{channel}_noise_uniform_{low,high}`` → U[low, high]
          per-dim (step-level); bounds may be asymmetric.
        - shift:    read ``{channel}_noise_shift`` → constant offset broadcast
          to all dims; **NO RNG draw** (deterministic, program-run-level), so
          the shift branch never perturbs the gaussian/uniform stream ordering.
        """
        dist = self.config.noise_dist
        if dist == "gaussian":
            std = getattr(self.config, f"{channel}_noise_std")
            return rng.normal(0.0, std, size=shape)
        if dist == "uniform":
            lo = getattr(self.config, f"{channel}_noise_uniform_low")
            hi = getattr(self.config, f"{channel}_noise_uniform_high")
            return rng.uniform(lo, hi, size=shape)
        if dist == "shift":
            shift = getattr(self.config, f"{channel}_noise_shift")
            return np.full(shape, shift, dtype=float)
        raise ValueError(f"Unknown noise_dist: {self.config.noise_dist!r}")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        # 1. Action noise — operates on flat action vector, completely
        #    independent of obs format. Distribution per config.noise_dist.
        action_noise_l2 = 0.0
        a_cmd = None
        if self._channel_active("action"):
            # Save the clean commanded action BEFORE noise (Method 6).
            # .copy() guards against callers reusing the same buffer.
            a_cmd = np.asarray(action).copy()
            noise = self._sample_noise(
                self.np_random, channel="action", shape=action.shape
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
        if self._channel_active("obs"):
            obs_noise_l2 = self._read_obs_noise_l2()

        # 3. Method 6 (Option B): expose the clean commanded action to the
        #    policy instead of the noised executed one. Only runs when
        #    action noise was actually injected — non-robust envs return
        #    the inner obs object untouched (no copy), so existing
        #    baselines are structurally unaffected.
        if a_cmd is not None and self._action_slice is not None:
            obs = obs.copy()
            obs[self._action_slice] = np.asarray(
                self._clean_physical(a_cmd), dtype=obs.dtype
            )

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
