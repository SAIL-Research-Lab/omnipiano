"""dm_env-layer per-key observation noise injection (gaussian/uniform/shift).

This wrapper replaces the per-key obs noise that previously lived in
`omnipiano/wrappers/robust_wrapper.py`. The move was forced by the
paper-chain refactor: with `ConcatObservationWrapper` flattening the
Dict observation into a single ndarray before reaching the gym layer,
per-key selectivity is only possible BEFORE that flatten — i.e., at the
dm_env layer. The distribution and magnitude come from ``RobustConfig``
(``sample_noise("obs", ...)``), so all three distributions are handled
by the same code path.

Usage (within `omnipiano.envs.registration.make()`):

    if effective_robust_config.is_channel_active("obs"):
        dm_env = DmEnvObsNoiseWrapper(
            dm_env,
            robust_config=effective_robust_config,
            seed=...,
        )
    dm_env = ConcatObservationWrapper(dm_env)

Per-step noise L2 is exposed via the `last_step_noise_l2` attribute so
the gym-layer `RobustWrapper` can populate
`info["robust/noise_obs_l2"]` without dm_env's missing info-dict.
"""
from collections import OrderedDict
from typing import Optional, Sequence

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper


# Offset added to the master seed to derive the obs-noise RNG seed, so the
# obs-noise stream is independent of the task's internal random_state (episode
# init). SINGLE SOURCE OF TRUTH — imported by both the single-agent
# (envs/registration.py) and multi-agent (multiagent/compile/environment.py)
# factories
# so they can never drift to different offsets (the §0.5 dedup: this was
# previously hardcoded as +31415 in two places). Value chosen 20000.
OBS_NOISE_SEED_OFFSET = 20000


class DmEnvObsNoiseWrapper(EnvironmentWrapper):
    """Inject independent per-key noise (gaussian / uniform / shift) into
    selected dm_env obs keys.

    Only modifies keys whose name CONTAINS any pattern in
    `include_key_patterns`. Default allowlist covers continuous
    proprioceptive observations (joint positions, joint velocities,
    piano key state). Categorical / counter-style keys (e.g.
    ``goal``, ``step_counter``, ``action``, ``reward``) are skipped.

    Attributes:
        last_step_noise_l2: Sum of L2 norms of injected noise vectors
            on the most recent step. Read by gym-layer `RobustWrapper`
            to fill info["robust/noise_obs_l2"]. Reset to 0.0 on
            ``reset()``.

    Args:
        environment: Inner dm_env to wrap. Its `observation_spec()`
            must be a Dict (i.e., place this BEFORE
            `ConcatObservationWrapper`).
        robust_config: RobustConfig whose obs-channel fields + noise_dist
            define the injected noise (gaussian / uniform / shift). If the
            obs channel is inactive (``is_channel_active("obs")`` False) the
            wrapper is a no-op (still passes through).
        include_key_patterns: Substrings (case-sensitive) that select
            which obs keys to noise. A key is noised iff
            ``any(p in key for p in patterns)``. Defaults to
            (``"joints_pos"``, ``"joints_vel"``, ``"piano/state"``,
            ``"piano/sustain_state"``).
        seed: Seed for the wrapper's own RNG. Use a value derived from
            (but distinct from) the dm_env's master seed so that
            obs-noise sequences don't consume the same RNG stream as
            task initialization (e.g., ``master_seed + OBS_NOISE_SEED_OFFSET``).
    """

    DEFAULT_INCLUDE_PATTERNS: Sequence[str] = (
        "joints_pos",
        "joints_vel",
        "piano/state",
        "piano/sustain_state",
    )

    def __init__(
        self,
        environment: dm_env.Environment,
        robust_config,
        include_key_patterns: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(environment)
        # RobustConfig owns the noise semantics (gaussian / uniform / shift);
        # this wrapper injects the obs channel via config.sample_noise, so the
        # dm_env obs layer and the gym-layer RobustWrapper never diverge.
        self._config = robust_config
        self._active = robust_config.is_channel_active("obs")
        self._patterns: tuple = tuple(
            include_key_patterns
            if include_key_patterns is not None
            else self.DEFAULT_INCLUDE_PATTERNS
        )
        self._rng = np.random.default_rng(seed)
        self.last_step_noise_l2: float = 0.0

    def reset(self) -> dm_env.TimeStep:
        self.last_step_noise_l2 = 0.0
        return self._environment.reset()

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        if not self._active:
            self.last_step_noise_l2 = 0.0
            return timestep

        # Must be Dict obs at this layer. If somebody mis-wires the chain
        # (puts ConcatObs before us), fail fast rather than silently
        # corrupt a flat ndarray.
        if not isinstance(timestep.observation, (dict, OrderedDict)):
            raise ValueError(
                "DmEnvObsNoiseWrapper requires Dict observation. Place it "
                "BEFORE ConcatObservationWrapper in the dm_env chain."
            )

        new_obs = OrderedDict(timestep.observation)
        total_l2 = 0.0
        for key, value in list(new_obs.items()):
            if not isinstance(value, np.ndarray):
                continue
            if value.dtype.kind != "f":
                continue
            if not any(p in key for p in self._patterns):
                continue
            noise = self._config.sample_noise(self._rng, "obs", value.shape)
            total_l2 += float(np.linalg.norm(noise))
            new_obs[key] = value + noise

        self.last_step_noise_l2 = total_l2
        return timestep._replace(observation=new_obs)
