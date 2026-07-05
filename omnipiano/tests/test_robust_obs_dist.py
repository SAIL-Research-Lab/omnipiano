"""Phase 0 · S3b gate — obs-channel noise across all 3 distributions at the
dm_env layer (DmEnvObsNoiseWrapper now dispatches via RobustConfig.sample_noise).

Verifies:
  - RobustConfig.is_channel_active picks the right fields per noise_dist;
  - end-to-end, a gaussian / uniform / shift OBS task injects obs noise
    (info["robust/noise_obs_l2"] > 0), and a clean env injects none;
  - the wrapper is only inserted when the obs channel is active.

Gaussian-obs behavior is unchanged (existing ObservationRobust envs); this
adds uniform/shift support that Phase 1 obs tasks need.
"""
import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration
from omnipiano.envs.dm_env_obs_noise import DmEnvObsNoiseWrapper
from omnipiano.utils.env_unwrap import find_dm_env_wrapper, get_dm_env_from_gym
from omnipiano.utils.info_keys import InfoKeys

_BASE = "RoboPianist-repertoire-150-ForElise-v0"


def _reg(env_id: str, **kw) -> None:
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, **kw)


_reg("OmniPianoTest-S3b-ObsGauss-v0",
     robust_config=RobustConfig(noise_dist="gaussian", obs_noise_std=0.05))
_reg("OmniPianoTest-S3b-ObsUniform-v0",
     robust_config=RobustConfig(noise_dist="uniform",
                                obs_noise_uniform_low=-0.05,
                                obs_noise_uniform_high=0.05))
_reg("OmniPianoTest-S3b-ObsShift-v0",
     robust_config=RobustConfig(noise_dist="shift", obs_noise_shift=0.03))
_reg("OmniPianoTest-S3b-ObsClean-v0")  # no noise → obs channel inactive


# --------------------------------------------------------------------------
# RobustConfig.is_channel_active — obs channel per distribution
# --------------------------------------------------------------------------
def test_is_channel_active_obs_gaussian():
    assert RobustConfig(noise_dist="gaussian", obs_noise_std=0.05).is_channel_active("obs")
    assert not RobustConfig(noise_dist="gaussian").is_channel_active("obs")


def test_is_channel_active_obs_uniform():
    assert RobustConfig(noise_dist="uniform", obs_noise_uniform_low=-0.05,
                        obs_noise_uniform_high=0.05).is_channel_active("obs")
    assert not RobustConfig(noise_dist="uniform").is_channel_active("obs")


def test_is_channel_active_obs_shift():
    assert RobustConfig(noise_dist="shift", obs_noise_shift=0.03).is_channel_active("obs")
    assert not RobustConfig(noise_dist="shift").is_channel_active("obs")


# --------------------------------------------------------------------------
# End-to-end injection at the dm_env layer
# --------------------------------------------------------------------------
def _obs_noise_l2(env_id: str, seed: int = 0) -> float:
    env = registration.make(env_id, seed=seed)  # train mode → scale 1.0
    try:
        env.reset(seed=seed)
        _, _, _, _, info = env.step(env.action_space.sample())
        return float(info[InfoKeys.ROBUST_NOISE_OBS_L2])
    finally:
        env.close()


def test_obs_gaussian_injects():
    assert _obs_noise_l2("OmniPianoTest-S3b-ObsGauss-v0") > 0.0


def test_obs_uniform_injects():
    assert _obs_noise_l2("OmniPianoTest-S3b-ObsUniform-v0") > 0.0


def test_obs_shift_injects():
    assert _obs_noise_l2("OmniPianoTest-S3b-ObsShift-v0") > 0.0


def test_obs_clean_no_injection():
    # obs channel inactive → DmEnvObsNoiseWrapper not inserted → L2 == 0
    assert _obs_noise_l2("OmniPianoTest-S3b-ObsClean-v0") == 0.0


def test_wrapper_present_iff_obs_active():
    for env_id, present in [
        ("OmniPianoTest-S3b-ObsUniform-v0", True),
        ("OmniPianoTest-S3b-ObsShift-v0", True),
        ("OmniPianoTest-S3b-ObsClean-v0", False),
    ]:
        env = registration.make(env_id)
        try:
            dm_env = get_dm_env_from_gym(env)
            found = find_dm_env_wrapper(dm_env, DmEnvObsNoiseWrapper) is not None
            assert found is present, f"{env_id}: wrapper present={found}, want {present}"
        finally:
            env.close()
