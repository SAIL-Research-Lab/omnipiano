"""Phase 0 · S5 gate — OBS_NOISE_SEED_OFFSET dedup + obs-noise seeding.

The obs-noise RNG seed offset was previously hardcoded as +31415 in TWO places
(single-agent + multi-agent registration), risking silent drift if one is
refactored and the other forgotten. It is now a single named constant
OBS_NOISE_SEED_OFFSET (=20000) imported by both.

Verifies:
  - the constant value (20000);
  - NEITHER registration source hardcodes an offset — both reference the
    constant (guards the exact regression the dedup fixes);
  - obs-noise draws are deterministic across builds at the same master seed
    (reproducibility) and differ across seeds.
"""
import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration
from omnipiano.envs.dm_env_obs_noise import OBS_NOISE_SEED_OFFSET
from omnipiano.utils.info_keys import InfoKeys

_BASE = "RoboPianist-repertoire-150-ForElise-v0"


def _reg(env_id: str, **kw) -> None:
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, **kw)


_reg("OmniPianoTest-S5-ObsGauss-v0",
     robust_config=RobustConfig(noise_dist="gaussian", obs_noise_std=0.05))


def test_offset_value():
    assert OBS_NOISE_SEED_OFFSET == 20000


def test_no_hardcoded_offset_in_registration_sources():
    import omnipiano.envs.registration as sa
    import omnipiano.multiagent.registration as ma
    for mod in (sa, ma):
        with open(mod.__file__) as f:
            src = f.read()
        assert "31415" not in src, f"{mod.__file__} still hardcodes 31415"
        assert "OBS_NOISE_SEED_OFFSET" in src, (
            f"{mod.__file__} does not reference the shared constant"
        )


def _first_obs_noise_l2(env_id: str, seed: int) -> float:
    env = registration.make(env_id, seed=seed)
    try:
        env.reset(seed=seed)
        act = np.zeros(env.action_space.shape, dtype=np.float32)  # fixed action
        _, _, _, _, info = env.step(act)
        return float(info[InfoKeys.ROBUST_NOISE_OBS_L2])
    finally:
        env.close()


def test_obs_noise_deterministic_across_builds():
    a = _first_obs_noise_l2("OmniPianoTest-S5-ObsGauss-v0", seed=7)
    b = _first_obs_noise_l2("OmniPianoTest-S5-ObsGauss-v0", seed=7)
    assert a > 0.0
    assert a == b  # same master seed → same obs-noise seed → identical draw


def test_obs_noise_differs_by_seed():
    a = _first_obs_noise_l2("OmniPianoTest-S5-ObsGauss-v0", seed=7)
    c = _first_obs_noise_l2("OmniPianoTest-S5-ObsGauss-v0", seed=99)
    assert a != c  # different master seed → different obs-noise draw
