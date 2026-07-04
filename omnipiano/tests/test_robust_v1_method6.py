"""Method 6 (Option B) prototype gate tests — robust_task_design.md §4.7.

Validates the obs["action"] principled-semantics fix in
``omnipiano/wrappers/robust_wrapper.py``:

* A1  — the obs "action" slot stores PHYSICAL values (post-CanonicalSpec),
        and our ``_clean_physical`` replicates that transform bit-exactly
        on the clean (no-noise) path.
* A2  — tree.flatten iterates dict keys alphabetically (the ConcatObs
        layout assumption behind slice computation).
* A3  — slice layout + cached physical spec match the live chain.
* A4  — non-robust envs: the returned obs IS the inner obs object
        (identity ⇒ never copied/modified ⇒ existing baselines are
        structurally unaffected).
* A5  — with action noise, obs[action_slice] equals the clean commanded
        action's physical version (bit-exact), and noise was injected.
* A6★ — THE cross-condition invariant: feeding the SAME fixed action
        sequence to a clean env and a noised env, obs[action_slice] must
        be bit-identical every step (clean env → CanonicalSpec's own
        arithmetic; noised env → our override arithmetic). Any formula /
        dtype / slice divergence fails here. Other obs slots MUST
        diverge (physics runs on the noised action) — checked as a
        sanity guard that noise is actually active.

Plus fail-fast coverage: frame_stack>1 raises; action_reward_observation
=False skips the override cleanly.

Run:
    /home/accelerator/miniforge3/envs/pianist/bin/python -m pytest \
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
    "OmniPianoTest-M6-NoAR-v0",
    env_config=BenchmarkEnvConfig(action_reward_observation=False),
    robust_config=RobustConfig(action_noise_std=0.05),
)
_register_once(
    "OmniPianoTest-M6-FrameStack-v0",
    env_config=BenchmarkEnvConfig(frame_stack=4),
    robust_config=RobustConfig(action_noise_std=0.05),
)


def _find_robust_wrapper(env) -> RobustWrapper:
    ptr = env
    while not isinstance(ptr, RobustWrapper):
        ptr = ptr.env
    return ptr


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
# Gate tests A1-A6
# ==========================================================================


def test_A2_tree_flatten_alphabetical_order():
    d = {
        "reward": 1.0,
        "action": 2.0,
        "goal": 3.0,
        "joints_pos": 4.0,
        "piano/state": 5.0,
    }
    # alphabetical: action, goal, joints_pos, piano/state, reward
    assert tree.flatten(d) == [2.0, 3.0, 4.0, 5.0, 1.0]


def test_A3_layout_and_physical_spec(clean_env):
    rw = _find_robust_wrapper(clean_env)
    act_dim = clean_env.action_space.shape[0]

    assert rw._action_slice is not None
    assert rw._reward_slice is not None
    assert rw._action_slice.stop - rw._action_slice.start == act_dim
    assert rw._reward_slice.stop - rw._reward_slice.start == 1
    obs_dim = clean_env.observation_space.shape[0]
    assert rw._action_slice.stop <= obs_dim
    assert rw._reward_slice.stop <= obs_dim

    # Cached physical spec must match the live CanonicalSpecWrapper's.
    from dm_env_wrappers import CanonicalSpecWrapper

    from omnipiano.utils.env_unwrap import (
        find_dm_env_wrapper,
        get_dm_env_from_gym,
    )

    canon = find_dm_env_wrapper(
        get_dm_env_from_gym(rw.env), CanonicalSpecWrapper
    )
    assert canon is not None
    np.testing.assert_array_equal(
        np.asarray(rw._physical_action_spec.minimum),
        np.asarray(canon._action_spec.minimum),
    )
    np.testing.assert_array_equal(
        np.asarray(rw._physical_action_spec.maximum),
        np.asarray(canon._action_spec.maximum),
    )
    assert rw._canonical_clip == canon._clip
    assert np.all(
        np.asarray(rw._physical_action_spec.minimum)
        < np.asarray(rw._physical_action_spec.maximum)
    )


def test_A1_action_slot_is_physical_and_formula_matches(clean_env):
    """Clean path: obs[action_slice] is produced by CanonicalSpec's own
    arithmetic. Our _clean_physical must reproduce it bit-exactly."""
    rw = _find_robust_wrapper(clean_env)
    clean_env.reset(seed=7)
    a = _fixed_actions(1, clean_env.action_space.shape[0], seed=7)[0]
    obs, *_ = clean_env.step(a)

    expected = np.asarray(rw._clean_physical(a), dtype=np.float32)
    np.testing.assert_array_equal(obs[rw._action_slice], expected)

    # Sanity: a physical rescale actually happened (the slot is NOT the raw
    # canonical command) — guards against a degenerate [-1,1] spec.
    assert not np.array_equal(
        obs[rw._action_slice], np.asarray(a, dtype=np.float32)
    )


def test_A4_non_robust_obs_object_untouched(clean_env):
    """[GATE] Zero-noise env: RobustWrapper must return the inner obs
    OBJECT itself (identity) — proving the override branch never runs and
    existing baselines are structurally unaffected (bit-exact trivially)."""
    rw = _find_robust_wrapper(clean_env)
    clean_env.reset(seed=11)

    captured = {}
    orig_step = rw.env.step

    def spy(action):
        out = orig_step(action)
        captured["obs"] = out[0]
        return out

    rw.env.step = spy
    try:
        a = _fixed_actions(1, clean_env.action_space.shape[0], seed=11)[0]
        obs, *_ = clean_env.step(a)
    finally:
        del rw.env.step  # remove instance attr, restore bound method

    assert obs is captured["obs"], (
        "RobustWrapper copied/modified obs on a zero-noise env — "
        "breaks the structural no-op guarantee for existing baselines"
    )


def test_A5_action_slot_clean_physical_under_noise(noise_env):
    rw = _find_robust_wrapper(noise_env)
    noise_env.reset(seed=13)
    a = _fixed_actions(1, noise_env.action_space.shape[0], seed=13)[0]
    obs, _r, _t, _tr, info = noise_env.step(a)

    assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0, "noise not injected?"
    expected = np.asarray(rw._clean_physical(a), dtype=np.float32)
    np.testing.assert_array_equal(
        obs[rw._action_slice],
        expected,
        err_msg=(
            "obs[action_slice] != physical(a_cmd): override missing or "
            "formula mismatch"
        ),
    )


def test_A6_action_slot_invariant_across_noise_levels(clean_env, noise_env):
    """[GATE ★★★] Same fixed action sequence into a clean env and a noised
    env → obs[action_slice] must be bit-identical every step. The clean
    env's value comes from CanonicalSpec's own computation; the noised
    env's from our override. Equality proves the two arithmetic paths are
    bit-exact twins. Other obs slots MUST diverge (physics executes the
    noised action) — sanity check that noise is genuinely active."""
    rw_c = _find_robust_wrapper(clean_env)
    rw_n = _find_robust_wrapper(noise_env)
    assert rw_c._action_slice == rw_n._action_slice, "layout mismatch"
    sl = rw_c._action_slice

    clean_env.reset(seed=42)
    noise_env.reset(seed=42)

    dim = clean_env.action_space.shape[0]
    actions = _fixed_actions(100, dim, seed=2024)

    obs_dim = clean_env.observation_space.shape[0]
    other_mask = np.ones(obs_dim, dtype=bool)
    other_mask[sl] = False

    mismatched_steps = []
    other_diverged = 0

    for i, a in enumerate(actions):
        obs_c, _rc, tc, trc, _ = clean_env.step(a)
        obs_n, _rn, tn, trn, _ = noise_env.step(a)

        if not np.array_equal(obs_c[sl], obs_n[sl]):
            mismatched_steps.append(
                (i, float(np.max(np.abs(obs_c[sl] - obs_n[sl]))))
            )
        if not np.array_equal(obs_c[other_mask], obs_n[other_mask]):
            other_diverged += 1
        if tc or trc or tn or trn:
            break

    assert not mismatched_steps, (
        f"[A6 FAILED] obs[action_slice] must be invariant to action noise "
        f"(both = physical(a_cmd)), but diverged at "
        f"{mismatched_steps[:5]} (step, max_abs_diff). Causes to check: "
        f"dtype promotion mismatch, clip semantics, wrong cached spec, "
        f"wrong slice indices."
    )
    assert other_diverged > 10, (
        f"[A6 sanity] other obs slots diverged in only {other_diverged} "
        f"steps — action noise may not be affecting physics at all."
    )


# ==========================================================================
# Fail-fast / fallback coverage
# ==========================================================================


def test_frame_stack_gt1_raises():
    with pytest.raises(NotImplementedError, match="frame_stack"):
        registration.make("OmniPianoTest-M6-FrameStack-v0")


def test_no_action_reward_obs_skips_override():
    env = registration.make("OmniPianoTest-M6-NoAR-v0")
    try:
        rw = _find_robust_wrapper(env)
        assert rw._action_slice is None
        assert rw._reward_slice is None

        env.reset(seed=3)
        a = _fixed_actions(1, env.action_space.shape[0], seed=3)[0]
        _obs, _r, _t, _tr, info = env.step(a)
        # Noise still applies to physics; only the (absent) obs slot fix
        # is skipped.
        assert info[InfoKeys.ROBUST_NOISE_ACTION_L2] > 0.0
    finally:
        env.close()
