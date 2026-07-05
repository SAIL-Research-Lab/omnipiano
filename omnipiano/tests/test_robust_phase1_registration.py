"""Phase 1 · registration gate — the 27 single-channel robust tasks + 1 Clean.

Verifies (§5.1 / §6):
  - exactly 28 `OmniPiano-ClairDeLune-*` robust ids are registered
    (3 channel × 3 dist × 3 level + Clean);
  - each id's RobustConfig is single-channel (only the named channel's field
    set, the other two channels 0) and carries the right distribution/level;
  - level is the natural parameter (Gaussian σ / Uniform ±half-range / Shift
    constant), no cross-dist normalization;
  - eval_noise_scale == 1.0 (matched-eval default, decisions 10/11) — NOT the
    stale 0.0 in the design-doc §5.1 snippet;
  - a representative env constructs + resets (smoke).
"""
import re

import numpy as np
import pytest

from omnipiano.configs import RobustConfig
from omnipiano.envs import registration

_PREFIX = "OmniPiano-ClairDeLune-"
_CHANNEL_FIELD = {"A": "action", "O": "obs", "R": "reward"}
_ACTION_OBS_LEVELS = {"P05": 0.05, "P10": 0.10, "P15": 0.15}
_REWARD_LEVELS = {"P10": 0.10, "P30": 0.30, "P50": 0.50}


# Canonical v1 matrix ids only: <C>-<Dist>-P<XX>-v0 or Clean-v0. Diagnostic /
# ablation envs (e.g. the negative-shift twin A-Shift-N15, §10 claim 5) are
# intentionally OUTSIDE the matrix and must not be counted here.
_CANONICAL_RE = re.compile(r"^OmniPiano-ClairDeLune-[AOR]-(Gauss|Uniform|Shift)-P\d{2}-v0$")


def _robust_ids():
    return sorted(
        i for i in registration._registry
        if _CANONICAL_RE.match(i) or i == f"{_PREFIX}Clean-v0"
    )


# --------------------------------------------------------------------------
# Count + id set
# --------------------------------------------------------------------------
def test_28_ids_registered():
    ids = _robust_ids()
    # 3 channel × 3 dist × 3 level = 27 robust + 1 clean
    assert len(ids) == 28, f"got {len(ids)}: {ids}"
    assert f"{_PREFIX}Clean-v0" in ids


def test_full_id_matrix_present():
    expected = set()
    for letter, levels in (("A", _ACTION_OBS_LEVELS),
                           ("O", _ACTION_OBS_LEVELS),
                           ("R", _REWARD_LEVELS)):
        for dist in ("Gauss", "Uniform", "Shift"):
            for plabel in levels:
                expected.add(f"{_PREFIX}{letter}-{dist}-{plabel}-v0")
    expected.add(f"{_PREFIX}Clean-v0")
    assert set(_robust_ids()) == expected


# --------------------------------------------------------------------------
# Per-id config correctness — single-channel + right dist/level
# --------------------------------------------------------------------------
def _levels_for(letter):
    return _ACTION_OBS_LEVELS if letter in ("A", "O") else _REWARD_LEVELS


def _other_channels(ch):
    return [c for c in ("action", "obs", "reward") if c != ch]


@pytest.mark.parametrize("letter", ["A", "O", "R"])
def test_gaussian_configs(letter):
    ch = _CHANNEL_FIELD[letter]
    for plabel, level in _levels_for(letter).items():
        cfg = registration._registry[f"{_PREFIX}{letter}-Gauss-{plabel}-v0"].robust_config
        assert cfg.noise_dist == "gaussian"
        assert getattr(cfg, f"{ch}_noise_std") == pytest.approx(level)
        # other channels silent
        for oc in _other_channels(ch):
            assert getattr(cfg, f"{oc}_noise_std") == 0.0


@pytest.mark.parametrize("letter", ["A", "O", "R"])
def test_uniform_configs_symmetric(letter):
    ch = _CHANNEL_FIELD[letter]
    for plabel, level in _levels_for(letter).items():
        cfg = registration._registry[f"{_PREFIX}{letter}-Uniform-{plabel}-v0"].robust_config
        assert cfg.noise_dist == "uniform"
        assert getattr(cfg, f"{ch}_noise_uniform_low") == pytest.approx(-level)
        assert getattr(cfg, f"{ch}_noise_uniform_high") == pytest.approx(+level)
        for oc in _other_channels(ch):
            assert getattr(cfg, f"{oc}_noise_uniform_low") == 0.0
            assert getattr(cfg, f"{oc}_noise_uniform_high") == 0.0


@pytest.mark.parametrize("letter", ["A", "O", "R"])
def test_shift_configs(letter):
    ch = _CHANNEL_FIELD[letter]
    for plabel, level in _levels_for(letter).items():
        cfg = registration._registry[f"{_PREFIX}{letter}-Shift-{plabel}-v0"].robust_config
        assert cfg.noise_dist == "shift"
        assert getattr(cfg, f"{ch}_noise_shift") == pytest.approx(level)
        for oc in _other_channels(ch):
            assert getattr(cfg, f"{oc}_noise_shift") == 0.0


# --------------------------------------------------------------------------
# Matched-eval default (decisions 10/11): eval_noise_scale == 1.0, NOT 0.0
# --------------------------------------------------------------------------
def test_all_robust_ids_matched_eval_default():
    for i in _robust_ids():
        cfg = registration._registry[i].robust_config
        assert cfg.eval_noise_scale == 1.0, (
            f"{i} has eval_noise_scale={cfg.eval_noise_scale} (expected 1.0 "
            f"matched default; the doc §5.1 snippet's 0.0 is stale)")


def test_clean_baseline_all_zero():
    cfg = registration._registry[f"{_PREFIX}Clean-v0"].robust_config
    for ch in ("action", "obs", "reward"):
        assert getattr(cfg, f"{ch}_noise_std") == 0.0
        assert getattr(cfg, f"{ch}_noise_uniform_low") == 0.0
        assert getattr(cfg, f"{ch}_noise_uniform_high") == 0.0
        assert getattr(cfg, f"{ch}_noise_shift") == 0.0
    assert not cfg.is_channel_active("action")
    assert not cfg.is_channel_active("obs")
    assert not cfg.is_channel_active("reward")


# --------------------------------------------------------------------------
# Smoke — a representative robust env constructs + resets
# --------------------------------------------------------------------------
def test_smoke_make_reset():
    env = registration.make(f"{_PREFIX}A-Gauss-P05-v0", seed=0)
    try:
        obs, info = env.reset(seed=0)
        assert np.all(np.isfinite(obs))
    finally:
        env.close()
