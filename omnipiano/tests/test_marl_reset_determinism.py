"""Regression test for MA review item #4: reset(seed=S) must rewind RNGs.

Runs against an obs-noise MA env, because on a clean env the two episodes
coincidentally match even without rebuilding the chain (the composer RNG is
never consumed).  That coincidence is exactly why the bug survived review.
"""

from __future__ import annotations

import numpy as np
import pytest

from omnipiano.multiagent import make_parallel
from omnipiano.multiagent.compile.env_runtime import registry as ma_reg


_MA_ID = "OmniPianoTest-MA-ResetDeterminism-v0"


def _find_obs_noise_sa_env() -> str:
    from omnipiano.envs import registration as sa_reg

    for env_id, spec in sa_reg._registry.items():
        rc = getattr(spec, "robust_config", None)
        if rc is None or not rc.is_channel_active("obs"):
            continue
        if getattr(spec, "safety_config", None) is not None and spec.safety_config.constraints:
            continue
        if spec.hand_specs is None or any(s.key_range is None for s in spec.hand_specs):
            continue
        if len(spec.hand_specs) == 3:
            return env_id
    pytest.skip("no obs-noise 3-hand StaticPartition SA env is registered")


def _rollout(env, n_steps: int, rng_seed: int) -> np.ndarray:
    """Deterministic action sequence -> flattened obs/reward trace."""
    rng = np.random.default_rng(rng_seed)
    trace = []
    for _ in range(n_steps):
        if not env.agents:
            break
        actions = {
            a: rng.uniform(-1.0, 1.0, size=env.action_space(a).shape).astype(np.float32)
            for a in env.agents
        }
        obs, rewards, _, _, _ = env.step(actions)
        for agent in sorted(obs):
            for key in sorted(obs[agent]):
                value = obs[agent][key]
                if isinstance(value, dict):
                    for sub in sorted(value):
                        trace.append(np.asarray(value[sub]).ravel())
                else:
                    trace.append(np.asarray(value).ravel())
            trace.append(np.asarray([rewards[agent]], dtype=np.float64))
    return np.concatenate(trace)


def test_reset_same_seed_is_bit_identical() -> None:
    sa_env_id = _find_obs_noise_sa_env()
    if _MA_ID not in ma_reg._ma_registry:
        ma_reg.register_parallel(id=_MA_ID, sa_env_id=sa_env_id, morphology="ThreeHand")
    env = make_parallel(_MA_ID, seed=42)
    try:
        env.reset(seed=42)
        first = _rollout(env, 40, rng_seed=0)
        env.reset(seed=42)
        second = _rollout(env, 40, rng_seed=0)
        np.testing.assert_array_equal(
            first,
            second,
            err_msg=(
                "reset(seed=42) twice produced different episodes; the dm_env "
                "chain RNGs were not rewound (MA review item #4)"
            ),
        )
    finally:
        env.close()
        ma_reg._ma_registry.pop(_MA_ID, None)
