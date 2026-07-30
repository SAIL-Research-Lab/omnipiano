"""Multi-channel / combined-channel compatibility gate (Phase 2 readiness audit,
2026-07-05).

v1 registers only single-channel tasks (§5.1.0), but the core wrapper / config /
logger logic is already multi-channel-safe. This test locks that in so a future
combined-channel registration (e.g. A+O) cannot silently regress:

  - A+O: both action AND obs noise inject in the same step; reward untouched.
  - A+R: both action AND reward inject; the reward obs slot is overridden to
    the noised received reward (the action slot keeps upstream OAR's noised
    executed action — decision 2026-07-21, no override).
  - §14 cross-check holds per-step under simultaneous action+reward noise:
    received reward == clean(decomposition sum) + reward_noise.
  - RNG determinism: action + reward share the gym stream but are sampled in a
    FIXED order (action→reward, §12 decision 8), so a combined task is bit-exact
    reproducible under the same seed.
  - Mixing distributions by MAGNITUDE FIELDS ALONE still raises: a channel
    inherits the global noise_dist unless it declares its own
    ``{channel}_noise_dist`` (per-channel dist landed 2026-07-21; its own
    gates live in test_robust_per_channel_dist.py). This keeps accidental
    mixing an error while the deliberate form is explicit.
  - frame_stack>1 with any override channel still fails fast under multi-channel.
"""
import numpy as np
import pytest

from omnipiano.configs import BenchmarkEnvConfig, RobustConfig
from omnipiano.envs import registration
from omnipiano.utils.info_keys import InfoKeys
from omnipiano.wrappers.robust_wrapper import RobustWrapper

_BASE = "RoboPianist-repertoire-150-ClairDeLune-v0"


def _reg(env_id, cfg, **kw):
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, robust_config=cfg, **kw)


def _robust(env) -> RobustWrapper:
    p = env
    while not isinstance(p, RobustWrapper):
        p = p.env
    return p


def _clean_decomp(info) -> float:
    return float(sum(
        v for k, v in info.items()
        if isinstance(k, str) and k.startswith("task/") and k.endswith("_reward")
    ))


def _first_step(env_id, seed=0):
    env = registration.make(env_id, seed=seed)
    env.reset(seed=seed)
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    return env, obs, float(reward), info


# --------------------------------------------------------------------------
# A+O: two channels inject simultaneously, reward untouched
# --------------------------------------------------------------------------
def test_action_plus_obs_both_inject():
    _reg("OmniPianoTest-MC-AO-v0",
         RobustConfig(noise_dist="gaussian", action_noise_std=0.10, obs_noise_std=0.10))
    env, obs, reward, info = _first_step("OmniPianoTest-MC-AO-v0")
    try:
        assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0
        assert info[InfoKeys.ROBUST_NOISE_OBS_L2] > 0.0
        assert info[InfoKeys.ROBUST_NOISE_REWARD] == 0.0   # reward channel inactive
    finally:
        env.close()


# --------------------------------------------------------------------------
# A+R: two channels inject; the reward obs slot overrides to noised reward
# --------------------------------------------------------------------------
def test_action_plus_reward_both_inject_and_override():
    _reg("OmniPianoTest-MC-AR-v0",
         RobustConfig(noise_dist="gaussian", action_noise_std=0.10, reward_noise_std=0.50))
    env, obs, reward, info = _first_step("OmniPianoTest-MC-AR-v0")
    try:
        rn = float(info[InfoKeys.ROBUST_NOISE_REWARD])
        assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0
        assert rn != 0.0
        # §14 cross-check per step: received == clean decomposition + reward noise
        assert reward == pytest.approx(_clean_decomp(info) + rn, abs=1e-4)
        # obs["reward"] slot overridden to the noised received reward
        # (the action slot needs no override — decision 2026-07-21).
        rw = _robust(env)
        assert rw._reward_slice is not None
        slot = float(np.asarray(obs[rw._reward_slice]).ravel()[0])
        assert slot == pytest.approx(reward, abs=1e-4)
    finally:
        env.close()


# --------------------------------------------------------------------------
# RNG determinism under the shared gym stream (fixed action→reward order)
# --------------------------------------------------------------------------
def test_multichannel_rng_reproducible():
    _reg("OmniPianoTest-MC-AR-repro-v0",
         RobustConfig(noise_dist="gaussian", action_noise_std=0.10, reward_noise_std=0.50))

    def run():
        env, obs, reward, info = _first_step("OmniPianoTest-MC-AR-repro-v0", seed=0)
        out = (float(info[InfoKeys.ROBUST_NOISE_ACTION_L2]),
               float(info[InfoKeys.ROBUST_NOISE_REWARD]))
        env.close()
        return out

    assert run() == run()   # bit-identical: same seed + fixed consumption order


# --------------------------------------------------------------------------
# Implicit mixing (magnitude fields only, no declared dist) still raises
# --------------------------------------------------------------------------
def test_implicit_mixed_dist_raises():
    # obs did not declare its own dist → it inherits noise_dist='gaussian',
    # so setting its UNIFORM bounds is a mismatch and must raise. Deliberate
    # mixing requires obs_noise_dist="uniform" (see
    # test_robust_per_channel_dist.py::test_P8_cross_dist_check_is_per_channel).
    with pytest.raises(ValueError, match="noise_dist='gaussian'"):
        RobustConfig(noise_dist="gaussian",
                     action_noise_std=0.10,
                     obs_noise_uniform_low=-0.10, obs_noise_uniform_high=0.10)


def test_same_dist_multichannel_config_valid():
    # A+O both gaussian, A+R both shift — same dist across channels is fine.
    RobustConfig(noise_dist="gaussian", action_noise_std=0.10, obs_noise_std=0.10)
    RobustConfig(noise_dist="shift", action_noise_shift=0.10, reward_noise_shift=0.30)


# --------------------------------------------------------------------------
# Per-channel DIFFERENT level (same dist): each channel injects at ITS OWN
# magnitude, no cross-channel coupling. Only noise_dist is shared; the 12
# magnitude fields are per-channel, so different levels use the same code path.
# --------------------------------------------------------------------------
def test_per_channel_different_levels_shift_exact():
    # obs-only reference at -0.15 → its deterministic obs L2
    _reg("OmniPianoTest-MC-Oshift-ref-v0",
         RobustConfig(noise_dist="shift", obs_noise_shift=-0.15))
    env_o, _, _, info_o = _first_step("OmniPianoTest-MC-Oshift-ref-v0")
    obs_l2_ref = float(info_o[InfoKeys.ROBUST_NOISE_OBS_L2])
    env_o.close()
    assert obs_l2_ref > 0.0

    # Combined O(-0.15) + R(-0.50): obs must inject at its OWN -0.15 (== ref,
    # unaffected by reward's -0.50), reward at its OWN -0.50 (not obs's -0.15).
    _reg("OmniPianoTest-MC-ORshift-diff-v0",
         RobustConfig(noise_dist="shift", obs_noise_shift=-0.15, reward_noise_shift=-0.50))
    env, obs, reward, info = _first_step("OmniPianoTest-MC-ORshift-diff-v0")
    try:
        assert float(info[InfoKeys.ROBUST_NOISE_REWARD]) == pytest.approx(-0.50)
        assert float(info[InfoKeys.ROBUST_NOISE_OBS_L2]) == pytest.approx(obs_l2_ref)
        assert float(info[InfoKeys.ROBUST_NOISE_ACTION_L2]) == 0.0  # action silent
    finally:
        env.close()


def test_per_channel_levels_scaled_proportionally_at_eval():
    # eval_noise_scale multiplies EVERY magnitude by the same factor → the
    # per-channel level ratio is preserved (matched-eval semantics).
    _reg("OmniPianoTest-MC-ORshift-scale-v0",
         RobustConfig(noise_dist="shift", obs_noise_shift=-0.15, reward_noise_shift=-0.50))
    env = registration.make("OmniPianoTest-MC-ORshift-scale-v0",
                            mode="eval", eval_noise_scale=0.5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    try:
        # -0.50 * 0.5 = -0.25 (reward), obs -0.15 * 0.5 = -0.075 (both halved).
        assert float(info[InfoKeys.ROBUST_NOISE_REWARD]) == pytest.approx(-0.25)
        assert float(info[InfoKeys.ROBUST_NOISE_OBS_L2]) > 0.0
    finally:
        env.close()


# --------------------------------------------------------------------------
# frame_stack>1 with a multi-channel override still fails fast
# --------------------------------------------------------------------------
def test_multichannel_frame_stack_gt1_raises():
    _reg("OmniPianoTest-MC-AR-fs-v0",
         RobustConfig(noise_dist="gaussian", action_noise_std=0.10, reward_noise_std=0.50),
         env_config=BenchmarkEnvConfig(frame_stack=4))
    with pytest.raises(NotImplementedError, match="frame_stack"):
        registration.make("OmniPianoTest-MC-AR-fs-v0")
