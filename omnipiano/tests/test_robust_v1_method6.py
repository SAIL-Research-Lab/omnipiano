"""OAR slot semantics gate tests.

Decision 2026-07-21 (supersedes Method 6 Option B): under the
hardware-wear threat model, ``obs["action"]`` keeps upstream OAR's
natural value — the **noised executed** action in physical units
(``physical(a_exec)``). The former clean-``a_cmd`` override was removed.
``obs["reward"]`` is still overridden to the noised observed reward when
the reward channel is active (reward noise is injected at the gym layer
*above* OAR, so the slot would otherwise stay clean).

Gates:

* N1 — tree.flatten iterates dict keys alphabetically (the ConcatObs
       layout assumption behind the reward-slice computation).
* N2 — reward-slice layout matches the live chain.
* N3 — non-robust env: the returned obs IS the inner obs object
       (identity ⇒ never copied ⇒ baselines structurally unaffected).
* N4 — action-noise env: SAME identity guarantee (no override runs at
       all for the action channel — the strongest structural proof the
       clean-a_cmd override is gone).
* N5 — action slot == physical(a_exec) bit-exact: spy-capture the
       noised action RobustWrapper passes down, replay the executed
       path's arithmetic (float32 cast → upstream _scale_nested_action
       with the live chain's spec/clip), assert equality; and assert the
       slot ≠ physical(a_cmd) (the leak is now BY DESIGN).
* N6 — same fixed action sequence into clean vs noised env → action
       slot DIVERGES nearly every step (inverse of the old A6
       invariant), other slots diverge too.
* N7 — reward-slot override survives reset(seed) chain rebuilds
       (cached slice is rebuild-safe).
* N8 — action-noise env: reward slot naturally equals the returned
       reward (no reward inconsistency, no override needed).
* Fail-fast / fallback: frame_stack>1 now constructs fine for
  action-channel tasks (no slot fix needed) but still raises for
  reward-channel tasks; action_reward_observation=False skips the
  reward override cleanly; SB3 DummyVecEnv smoke.

Run:
    python -m pytest \
        omnipiano/tests/test_robust_v1_method6.py -v
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
import tree

from omnipiano.configs import BenchmarkEnvConfig, RobustConfig
from omnipiano.envs import registration
from omnipiano.utils.info_keys import InfoKeys
from omnipiano.wrappers.robust_wrapper import RobustWrapper

# ForElise: shortest benchmark piece (399 steps) — fastest env build/step.
_BASE = "RoboPianist-repertoire-150-ForElise-v0"


def _register_once(env_id: str, **kwargs) -> None:
    if env_id not in registration._registry:
        registration.register(id=env_id, base_env_name=_BASE, **kwargs)


_register_once("OmniPianoTest-M6-Clean-v0")
_register_once(
    "OmniPianoTest-M6-ActionNoise-v0",
    robust_config=RobustConfig(action_noise_std=0.05),
)
_register_once(
    "OmniPianoTest-M6-RewardNoise-v0",
    robust_config=RobustConfig(reward_noise_std=0.5),
)
_register_once(
    "OmniPianoTest-M6-NoAR-v0",
    env_config=BenchmarkEnvConfig(action_reward_observation=False),
    robust_config=RobustConfig(action_noise_std=0.05),
)
_register_once(
    "OmniPianoTest-M6-FrameStack-v0",
    env_config=BenchmarkEnvConfig(frame_stack=4),
    robust_config=RobustConfig(action_noise_std=0.05),
)
_register_once(
    "OmniPianoTest-M6-RewardFrameStack-v0",
    env_config=BenchmarkEnvConfig(frame_stack=4),
    robust_config=RobustConfig(reward_noise_std=0.5),
)


def _find_robust_wrapper(env) -> RobustWrapper:
    ptr = env
    while not isinstance(ptr, RobustWrapper):
        ptr = ptr.env
    return ptr


def _chain_parts(env):
    """(concat, canonical) wrappers of the live dm_env chain."""
    from dm_env_wrappers import CanonicalSpecWrapper, ConcatObservationWrapper

    from omnipiano.utils.env_unwrap import (
        find_dm_env_wrapper,
        get_dm_env_from_gym,
    )

    dm_env = get_dm_env_from_gym(env)
    return (
        find_dm_env_wrapper(dm_env, ConcatObservationWrapper),
        find_dm_env_wrapper(dm_env, CanonicalSpecWrapper),
    )


def _action_slice_of(env):
    """Alphabetical-concat action slice, computed independently in-test
    (the wrapper no longer tracks it — only the test needs it)."""
    concat, _ = _chain_parts(env)
    spec = concat._environment.observation_spec()
    offset = 0
    for key in sorted(concat._obs_names):
        dim = int(np.prod(spec[key].shape)) if spec[key].shape else 1
        if key == "action":
            return slice(offset, offset + dim)
        offset += dim
    return None


def _physical(env, action):
    """Replay the executed path's arithmetic for a given (already noised)
    action: DmEnvToGymnasium's float32 cast → upstream _scale_nested_action
    with the live chain's spec object and clip flag."""
    from dm_env_wrappers._src.canonical_spec import _scale_nested_action

    _, canon = _chain_parts(env)
    a = np.asarray(action, dtype=np.float32)
    return _scale_nested_action(a, canon._action_spec, canon._clip)


def _fixed_actions(n: int, dim: int, seed: int = 2024):
    """Deterministic float64 caller actions (mimics OmniSafe's
    ``action.detach().cpu().numpy().astype(np.float64)``)."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1.0, 1.0, size=dim) for _ in range(n)]


@pytest.fixture(scope="module")
def clean_env():
    env = registration.make("OmniPianoTest-M6-Clean-v0")
    yield env
    env.close()


@pytest.fixture(scope="module")
def noise_env():
    env = registration.make("OmniPianoTest-M6-ActionNoise-v0")
    yield env
    env.close()


# ==========================================================================
# Layout gates
# ==========================================================================


def test_N1_tree_flatten_alphabetical_order():
    d = {
        "reward": 1.0,
        "action": 2.0,
        "goal": 3.0,
        "joints_pos": 4.0,
        "piano/state": 5.0,
    }
    # alphabetical: action, goal, joints_pos, piano/state, reward
    assert tree.flatten(d) == [2.0, 3.0, 4.0, 5.0, 1.0]


def test_N2_reward_slice_layout(clean_env):
    rw = _find_robust_wrapper(clean_env)
    assert rw._reward_slice is not None
    assert rw._reward_slice.stop - rw._reward_slice.start == 1
    obs_dim = clean_env.observation_space.shape[0]
    assert rw._reward_slice.stop <= obs_dim


# ==========================================================================
# Structural no-op gates — obs object identity
# ==========================================================================


def _assert_obs_identity(env, seed):
    """Step once with a spy on the inner step; assert the wrapper returned
    the inner obs OBJECT itself (never copied/modified)."""
    rw = _find_robust_wrapper(env)
    env.reset(seed=seed)

    captured = {}
    orig_step = rw.env.step

    def spy(action):
        out = orig_step(action)
        captured["obs"] = out[0]
        return out

    rw.env.step = spy
    try:
        a = _fixed_actions(1, env.action_space.shape[0], seed=seed)[0]
        obs, *_ = env.step(a)
    finally:
        del rw.env.step  # remove instance attr, restore bound method

    assert obs is captured["obs"], (
        "RobustWrapper copied/modified obs — breaks the structural no-op "
        "guarantee (only reward-channel tasks may copy)"
    )


def test_N3_non_robust_obs_object_untouched(clean_env):
    _assert_obs_identity(clean_env, seed=11)


def test_N4_action_noise_obs_object_untouched(noise_env):
    """[GATE] Action-channel env: NO override runs at all — the returned
    obs is the inner object. Structural proof the clean-a_cmd override is
    gone (decision 2026-07-21)."""
    _assert_obs_identity(noise_env, seed=13)


# ==========================================================================
# Action-slot semantics — noised executed action, by design
# ==========================================================================


def test_N5_action_slot_is_noised_executed_physical(noise_env):
    """obs[action_slice] == physical(a_exec) bit-exact, and != physical(
    a_cmd): the slot reflects what was EXECUTED (hardware-wear model)."""
    rw = _find_robust_wrapper(noise_env)
    sl = _action_slice_of(noise_env)
    noise_env.reset(seed=17)

    captured = {}
    orig_step = rw.env.step

    def spy(action):
        captured["a_exec"] = np.asarray(action).copy()
        return orig_step(action)

    rw.env.step = spy
    try:
        a_cmd = _fixed_actions(1, noise_env.action_space.shape[0], seed=17)[0]
        obs, _r, _t, _tr, info = noise_env.step(a_cmd)
    finally:
        del rw.env.step

    assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0, "noise not injected?"
    expected = np.asarray(_physical(noise_env, captured["a_exec"]),
                          dtype=np.float32)
    np.testing.assert_array_equal(
        obs[sl], expected,
        err_msg="obs[action_slice] != physical(executed action)",
    )
    # The leak is by design now: slot must NOT equal physical(a_cmd).
    assert not np.array_equal(
        obs[sl], np.asarray(_physical(noise_env, a_cmd), dtype=np.float32)
    ), "slot equals physical(a_cmd) — the removed override seems active"


def test_N6_action_slot_diverges_from_clean_env(clean_env, noise_env):
    """[Inverse of the old A6] Same fixed action sequence into clean vs
    noised env: the action slot must DIVERGE nearly every step (noise is
    visible in the slot by design), and other slots diverge too (physics
    runs on the noised action)."""
    sl = _action_slice_of(clean_env)
    assert sl == _action_slice_of(noise_env), "layout mismatch"

    clean_env.reset(seed=42)
    noise_env.reset(seed=42)

    dim = clean_env.action_space.shape[0]
    actions = _fixed_actions(30, dim, seed=2024)

    slot_diverged = 0
    for a in actions:
        obs_c, _rc, tc, trc, _ = clean_env.step(a)
        obs_n, _rn, tn, trn, _ = noise_env.step(a)
        if not np.array_equal(obs_c[sl], obs_n[sl]):
            slot_diverged += 1
        if tc or trc or tn or trn:
            break

    assert slot_diverged >= 25, (
        f"action slot diverged in only {slot_diverged}/30 steps — gaussian "
        f"noise should make it differ essentially every step; is the "
        f"removed clean-a_cmd override somehow back?"
    )


# ==========================================================================
# Reward-slot override — retained behavior
# ==========================================================================


def test_N7_reward_override_survives_seed_rebuild():
    """reset(seed=X) rebuilds the dm_env chain; the reward slice cached at
    __init__ must remain valid (static-config determinism)."""
    env = registration.make("OmniPianoTest-M6-RewardNoise-v0")
    try:
        rw = _find_robust_wrapper(env)
        assert rw._reward_slice is not None
        dim = env.action_space.shape[0]
        for rebuild_seed in (101, 202):
            env.reset(seed=rebuild_seed)
            for a in _fixed_actions(3, dim, seed=rebuild_seed):
                obs, reward, *_ = env.step(a)
                assert obs[rw._reward_slice][0] == np.float32(reward), (
                    f"reward slot != returned reward after "
                    f"reset(seed={rebuild_seed}) rebuild — cached slice stale?"
                )
    finally:
        env.close()


def test_N8_reward_slot_consistent_under_action_noise(noise_env):
    """Action-channel tasks create no reward inconsistency: the obs
    'reward' slot (recorded by OAR from raw physics reward) equals the
    reward returned to the policy — with NO override involved."""
    rw = _find_robust_wrapper(noise_env)
    noise_env.reset(seed=51)
    a = _fixed_actions(1, noise_env.action_space.shape[0], seed=51)[0]
    obs, reward, *_ = noise_env.step(a)
    assert obs[rw._reward_slice][0] == np.float32(reward)


# ==========================================================================
# Fail-fast / fallback coverage
# ==========================================================================


def test_action_frame_stack_gt1_constructs():
    """Action noise needs no obs-slot fix, so frame_stack>1 is now allowed
    for action-channel tasks (decision 2026-07-21)."""
    env = registration.make("OmniPianoTest-M6-FrameStack-v0")
    try:
        env.reset(seed=3)
        a = _fixed_actions(1, env.action_space.shape[0], seed=3)[0]
        _obs, _r, _t, _tr, info = env.step(a)
        assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0
    finally:
        env.close()


def test_reward_frame_stack_gt1_raises():
    with pytest.raises(NotImplementedError, match="frame_stack"):
        registration.make("OmniPianoTest-M6-RewardFrameStack-v0")


def test_no_action_reward_obs_skips_override():
    env = registration.make("OmniPianoTest-M6-NoAR-v0")
    try:
        rw = _find_robust_wrapper(env)
        assert rw._reward_slice is None

        env.reset(seed=3)
        a = _fixed_actions(1, env.action_space.shape[0], seed=3)[0]
        _obs, _r, _t, _tr, info = env.step(a)
        # Noise still applies to physics; only the (absent) obs slot fix
        # is skipped.
        assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0
    finally:
        env.close()


def test_sb3_dummy_vecenv_smoke():
    """DummyVecEnv copies/stacks obs — ensure the chain survives SB3's
    vec plumbing (info keys intact)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _mk():
        return registration.make("OmniPianoTest-M6-ActionNoise-v0")

    vec = DummyVecEnv([_mk, _mk])
    try:
        vec.seed(7)
        vec.reset()
        dim = vec.action_space.shape[0]
        rng = np.random.default_rng(7)
        for _ in range(5):
            acts = rng.uniform(-1, 1, size=(2, dim)).astype(np.float32)
            obs, rewards, dones, infos = vec.step(acts)
            assert obs.shape[0] == 2
            for info in infos:
                assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0
    finally:
        vec.close()
