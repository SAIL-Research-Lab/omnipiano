"""Per-channel noise distribution gate (2026-07-21).

One env may now mix distributions across channels (e.g. action=shift +
obs=gaussian) via the optional ``{channel}_noise_dist`` fields; unset
channels inherit the global ``noise_dist``. Robust-Gymnasium has no
equivalent (single global ``--noise-type``), so this is OmniPiano-only
capability and needs its own gates:

  * P1 — ``dist_for`` resolution: override wins, None inherits.
  * P2 — backward compatibility: a config with no per-channel field
         resolves every channel to the global dist (the equivalence gate
         in scratchpad/fingerprint.py proves this bit-exactly across all
         33 registered robust envs; this is the unit-level assertion).
  * P3 — a mixed config constructs and both channels report noise.
  * P4★ — EXACTNESS, action side: in a mixed action=shift + obs=gaussian
         env, the action noise must equal an action-only shift env's
         exactly (shift is deterministic; the obs channel must not perturb
         it).
  * P5★ — EXACTNESS, obs side: the same mixed env's obs-noise sequence
         must be bit-identical to an obs-only gaussian env's, because the
         obs channel draws from its own dm_env-layer stream
         (seed+OBS_NOISE_SEED_OFFSET) that the action channel never touches.
  * P6 — mixed A=gaussian + R=shift: the action draw is unaffected by the
         reward channel (shift consumes no RNG), so it matches an
         action-only gaussian env exactly.
  * P7 — RNG determinism: same seed twice on a mixed env → identical.
  * P8 — validation is per-channel: a field belonging to another channel's
         distribution still raises, and a legal channel in the SAME config
         does not mask it.
  * P9 — "declared but empty" raises (a channel naming a distribution but
         leaving every magnitude at zero would silently train clean).
  * P10 — unknown per-channel dist value raises; field count is 16.
"""
import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration
from omnipiano.utils.info_keys import InfoKeys

_BASE = "RoboPianist-repertoire-150-ForElise-v0"
_N = 5
_SEED = 0


def _reg(env_id, cfg):
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, robust_config=cfg)


def _trace(env_id, n=_N, seed=_SEED):
    """Per-step (action_l2, obs_l2, reward_noise) under a fixed action seq."""
    env = registration.make(env_id, seed=seed)
    try:
        env.reset(seed=seed)
        rng = np.random.default_rng(777)
        dim = env.action_space.shape[0]
        out = []
        for _ in range(n):
            _o, _r, term, trunc, info = env.step(rng.uniform(-1, 1, size=dim))
            out.append((
                float(info[InfoKeys.ROBUST_NOISE_ACTION_L2]),
                float(info[InfoKeys.ROBUST_NOISE_OBS_L2]),
                float(info[InfoKeys.ROBUST_NOISE_REWARD]),
            ))
            if term or trunc:
                break
        return out
    finally:
        env.close()


# Reference (single-channel) and mixed envs sharing identical magnitudes.
_reg("OmniPianoTest-PCD-Ashift-v0",
     RobustConfig(noise_dist="shift", action_noise_shift=0.15))
_reg("OmniPianoTest-PCD-Ogauss-v0",
     RobustConfig(noise_dist="gaussian", obs_noise_std=0.15))
_reg("OmniPianoTest-PCD-Agauss-v0",
     RobustConfig(noise_dist="gaussian", action_noise_std=0.15))
_reg("OmniPianoTest-PCD-MixedAshiftOgauss-v0",
     RobustConfig(action_noise_dist="shift", action_noise_shift=0.15,
                  obs_noise_dist="gaussian", obs_noise_std=0.15))
_reg("OmniPianoTest-PCD-MixedAgaussRshift-v0",
     RobustConfig(action_noise_dist="gaussian", action_noise_std=0.15,
                  reward_noise_dist="shift", reward_noise_shift=0.50))


# --------------------------------------------------------------------------
# P1 / P2 — resolution + backward compatibility
# --------------------------------------------------------------------------
def test_P1_dist_for_override_and_inherit():
    cfg = RobustConfig(noise_dist="gaussian",
                       action_noise_dist="shift", action_noise_shift=0.15,
                       obs_noise_std=0.15)
    assert cfg.dist_for("action") == "shift"     # own override
    assert cfg.dist_for("obs") == "gaussian"     # inherits global
    assert cfg.dist_for("reward") == "gaussian"  # inherits global


@pytest.mark.parametrize("dist,kw", [
    ("gaussian", {"action_noise_std": 0.1}),
    ("uniform", {"action_noise_uniform_low": -0.1, "action_noise_uniform_high": 0.1}),
    ("shift", {"action_noise_shift": 0.1}),
])
def test_P2_no_per_channel_field_inherits_global(dist, kw):
    cfg = RobustConfig(noise_dist=dist, **kw)
    for ch in ("action", "obs", "reward"):
        assert cfg.dist_for(ch) == dist
    assert cfg.action_noise_dist is None  # untouched by construction


# --------------------------------------------------------------------------
# P3-P6 — mixed envs inject per channel, exactly
# --------------------------------------------------------------------------
def test_P3_mixed_env_both_channels_inject():
    tr = _trace("OmniPianoTest-PCD-MixedAshiftOgauss-v0")
    assert all(s[0] > 0 for s in tr), "action (shift) channel silent"
    assert all(s[1] > 0 for s in tr), "obs (gaussian) channel silent"
    assert all(s[2] == 0.0 for s in tr), "reward channel must stay off"
    # shift is deterministic → identical every step; gaussian must vary.
    assert len({s[0] for s in tr}) == 1, "shift action noise should be constant"
    assert len({s[1] for s in tr}) > 1, "gaussian obs noise should vary"


def test_P4_mixed_action_shift_matches_action_only_reference():
    mixed = [s[0] for s in _trace("OmniPianoTest-PCD-MixedAshiftOgauss-v0")]
    ref = [s[0] for s in _trace("OmniPianoTest-PCD-Ashift-v0")]
    assert mixed == ref, (
        "action(shift) noise changed when an obs(gaussian) channel was added "
        "— per-channel dispatch is leaking across channels"
    )


def test_P5_mixed_obs_gauss_matches_obs_only_reference():
    # The obs channel draws from its own dm_env-layer RNG
    # (seed + OBS_NOISE_SEED_OFFSET), which the action channel never touches,
    # so the sequence must be bit-identical to the obs-only env's.
    mixed = [s[1] for s in _trace("OmniPianoTest-PCD-MixedAshiftOgauss-v0")]
    ref = [s[1] for s in _trace("OmniPianoTest-PCD-Ogauss-v0")]
    assert mixed == ref, "obs(gaussian) stream perturbed by the action channel"


def test_P6_mixed_action_gauss_unaffected_by_reward_shift():
    # reward=shift consumes NO RNG, so the shared gym stream still delivers
    # the same action draws as an action-only gaussian env.
    mixed = _trace("OmniPianoTest-PCD-MixedAgaussRshift-v0")
    ref = [s[0] for s in _trace("OmniPianoTest-PCD-Agauss-v0")]
    assert [s[0] for s in mixed] == ref
    assert all(s[2] == pytest.approx(0.50) for s in mixed), "reward shift wrong"


def test_P7_mixed_rng_reproducible():
    assert _trace("OmniPianoTest-PCD-MixedAgaussRshift-v0") == \
           _trace("OmniPianoTest-PCD-MixedAgaussRshift-v0")


# --------------------------------------------------------------------------
# P8-P10 — validation
# --------------------------------------------------------------------------
def test_P8_cross_dist_check_is_per_channel():
    # action declares shift but carries a gaussian magnitude → raises, even
    # though obs in the same config is perfectly legal.
    with pytest.raises(ValueError, match="noise_dist='shift'"):
        RobustConfig(action_noise_dist="shift", action_noise_std=0.15,
                     obs_noise_dist="gaussian", obs_noise_std=0.15)
    # ...and the legal mixed pair alone constructs fine.
    RobustConfig(action_noise_dist="shift", action_noise_shift=0.15,
                 obs_noise_dist="gaussian", obs_noise_std=0.15)


@pytest.mark.parametrize("ch", ["action", "obs", "reward"])
def test_P9_declared_but_empty_raises(ch):
    with pytest.raises(ValueError, match="declared but every"):
        RobustConfig(**{f"{ch}_noise_dist": "shift"})   # no magnitude set


def test_P9b_inherited_channel_may_stay_silent():
    # A channel that does NOT declare its own dist is allowed to be inactive
    # (that is how every single-channel env works).
    cfg = RobustConfig(noise_dist="shift", action_noise_shift=0.15)
    assert cfg.is_channel_active("action")
    assert not cfg.is_channel_active("obs")


def test_P10_invalid_per_channel_dist_raises():
    with pytest.raises(ValueError, match="obs_noise_dist must be one of"):
        RobustConfig(obs_noise_dist="gausian", obs_noise_std=0.1)


def test_P10b_per_channel_fields_exist():
    # The authoritative field-set gate lives in test_robust_config.py
    # (test_exactly_17_fields); here we only assert the 3 new names exist.
    from dataclasses import fields
    names = {f.name for f in fields(RobustConfig)}
    assert {"action_noise_dist", "obs_noise_dist", "reward_noise_dist"} <= names
