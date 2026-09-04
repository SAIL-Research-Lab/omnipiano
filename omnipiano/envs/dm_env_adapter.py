"""dm_env.Environment → gymnasium.Env adapter for OmniPiano's paper chain.

Replaces ``shimmy.DmControlCompatibilityV0`` with a focused ~30-line
adapter targeted at the paper-chain layout used in
``omnipiano.envs.registration.make()``.

Why a custom adapter rather than shimmy
---------------------------------------

* shimmy's ``DmControlCompatibilityV0._find_env_type`` only recognizes
  ``composer.Environment`` / ``control.Environment`` and raises
  ``ValueError`` on other dm_env subclasses. OmniPiano's paper chain
  ends in ``SinglePrecisionWrapper`` (a ``dm_env_wrappers.EnvironmentWrapper``
  subclass), so shimmy requires a ``_find_env_type`` monkey-patch that
  is fragile across shimmy upgrades.
* The paper chain has already done all the heavy lifting (Concat,
  Canonical [-1, 1], float32). The adapter only needs to translate
  step return / reset to gymnasium's 5-tuple and (5-tuple) interface.
  ~30 lines of explicit code beats 230 lines of generic shimmy logic
  for our narrow use case.

Seeding semantics
-----------------

dm_env's ``random_state`` is set ONCE at composer.Environment
construction time (via ``suite.load_with_task(seed=...)``); it is not
mutable through the dm_env API. SB3's ``make_vec_env`` seeds per-sub-env
through ``env.reset(seed=X+i)``, so for that to actually distribute
seeds we need ``reset(seed=X)`` to rebuild the underlying dm_env. The
adapter takes a builder callable ``env_builder(seed)`` and rebuilds the
chain when ``reset(seed=...)`` is called with a new seed.

Cost: rebuilding the dm_env is ~1 second (composer compile).
make_vec_env triggers this once per sub-env at init, after which the
seed is stable across resets — so rebuilds happen O(n_envs) times total,
not per-episode.

Compatibility with OmniPiano helpers
------------------------------------

The adapter exposes ``self._env`` as the live dm_env chain top, which
is the contract that ``omnipiano.utils.env_unwrap.get_dm_env_from_gym``
walks via ``hasattr(env, "env")`` / ``hasattr(env, "_env")``. This means
``MetricsWrapper``, ``SafetyWrapper``, and any future wrapper that uses
``get_composer_env_from_gym`` continue to work without modification.
"""
from typing import Callable, Optional

import dm_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DmEnvToGymnasium(gym.Env):
    """Minimal dm_env → gymnasium adapter with seed-aware reset.

    Args:
        env_builder: Callable ``(seed: Optional[int]) -> dm_env.Environment``
            that constructs and returns the full dm_env wrapper chain.
            Called once at adapter init and again on every ``reset(seed=X)``
            with a non-None seed (even an unchanged one — rebuilding is
            the only way to rewind the chain-internal RNGs; see reset()).
        seed: Initial seed for the dm_env; passed to ``env_builder(seed)``.

    Assumptions about the wrapped chain:
    * ``observation_spec()`` is a single ``Array`` / ``BoundedArray``
      (i.e., ``ConcatObservationWrapper`` is in the chain).
    * ``action_spec()`` is a single ``BoundedArray`` with bounds
      ``[-1, 1]`` (i.e., ``CanonicalSpecWrapper`` is in the chain).
    * ``observation`` dtype is float32 (i.e., ``SinglePrecisionWrapper``
      is in the chain).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_builder: Callable[[Optional[int]], dm_env.Environment],
        seed: Optional[int] = None,
    ) -> None:
        self._builder = env_builder
        self._env = env_builder(seed)

        act_spec = self._env.action_spec()
        obs_spec = self._env.observation_spec()
        self.action_space = spaces.Box(
            low=np.asarray(act_spec.minimum, dtype=np.float32),
            high=np.asarray(act_spec.maximum, dtype=np.float32),
            dtype=np.float32,
        )
        # OmniPiano's paper chain produces obs_spec.minimum / obs_spec.maximum
        # = scalar -inf / +inf (verified empirically): some underlying
        # observables are unbounded — joint velocities, MIDI goal lookahead,
        # prev_reward — so ConcatObservationWrapper's union degrades the
        # whole concat to unbounded. We could read obs_spec.minimum /
        # .maximum here, but they're always (-inf, +inf), so we hardcode
        # those for clarity. Matches Path C's adapter and shimmy's effective
        # behavior (shimmy.dm_spec2gym_space also produces -inf/+inf for
        # this same dm_env). PPO doesn't use obs bounds for anything in our
        # training loop (no obs sampling, no VecNormalize), so unbounded is
        # safe.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_spec.shape,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        # 1. Seed the gymnasium-level RNG (self._np_random). This is what
        #    gym.Wrapper.np_random returns to outer wrappers — notably
        #    RobustWrapper uses it to sample action noise. Without this
        #    line, self._np_random gets auto-seeded with OS entropy on
        #    first access → non-reproducible action noise on
        #    ActionRobust-v0 tasks despite the user passing a fixed seed.
        super().reset(seed=seed)

        # 2. Re-seed the dm_env-level RNGs. Gymnasium's spec says
        #    reset(seed=X) must re-seed the PRNG even if it was already
        #    seeded with X. But the dm_env layer has two RNGs that are
        #    NOT touched by _env.reset():
        #       - composer.Environment._random_state (fixed at
        #         suite.load_with_task construction; advances per step).
        #       - DmEnvObsNoiseWrapper._rng (set in __init__; advances
        #         per step, never rewound).
        #    The only way to rewind both is to rebuild the chain. So:
        #    any non-None seed triggers a rebuild, even one identical to
        #    the previous seed. Without this, reset(seed=42) called
        #    twice would give different obs_noise realizations on
        #    ObservationRobust-v0 tasks, violating gymnasium spec.
        #
        #    Cost is ~1s per rebuild, but SB3 make_vec_env only passes
        #    seed once per sub-env at init (subsequent resets during
        #    training use seed=None), so this never fires a second time
        #    in the training loop.
        if seed is not None:
            old_env, self._env = self._env, None
            old_env.close()
            self._env = self._builder(seed)

        ts = self._env.reset()
        return np.asarray(ts.observation, dtype=np.float32), {}

    def step(self, action):
        ts = self._env.step(np.asarray(action, dtype=np.float32))
        obs = np.asarray(ts.observation, dtype=np.float32)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        # dm_env semantics:
        #   ts.last() & discount == 0  →  task self-terminated (failure or success)
        #   ts.last() & discount  > 0  →  time-limit truncation (episode ended
        #                                  but task hadn't "completed")
        # PianoTask's MIDI-end is implemented as task self-terminate with
        # discount=1.0 (see piano_with_shadow_hands.get_discount), so MIDI
        # end maps to truncated=True here. SB3 uses the truncated flag
        # to bootstrap V(terminal_obs) via TimeLimit.truncated info key.
        terminated = bool(ts.last() and ts.discount == 0.0)
        truncated = bool(ts.last() and not terminated)
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
