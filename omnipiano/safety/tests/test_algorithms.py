"""Configuration coverage is not a claim that all algorithms were trained."""
from pathlib import Path

import pytest

from omnipiano.safety.algorithms import (
    SUPPORTED_ALGORITHMS, ON_POLICY, OFF_POLICY, LAGRANGIAN, DIRECT_BUDGET,
    AUGMENTED, UNCONSTRAINED, evaluation_budget,
)
from omnipiano.safety.runtime import algorithm_config


def test_algorithm_partition():
    assert len(SUPPORTED_ALGORITHMS) == len(set(SUPPORTED_ALGORITHMS)) == 32
    assert len(ON_POLICY) == 23 and len(OFF_POLICY) == 9
    groups = UNCONSTRAINED + LAGRANGIAN + DIRECT_BUDGET + AUGMENTED
    assert len(groups) == len(set(groups)) == 32
    assert set(groups) == set(SUPPORTED_ALGORITHMS)


@pytest.mark.parametrize("algorithm", SUPPORTED_ALGORITHMS)
def test_native_config_keys_and_budget(algorithm):
    omnisafe = pytest.importorskip("omnisafe")
    import yaml
    family = "off-policy" if algorithm in OFF_POLICY else "on-policy"
    path = Path(omnisafe.__file__).parent / "configs" / family / f"{algorithm}.yaml"
    native = yaml.safe_load(path.read_text())["defaults"]
    for smoke in (False, True):
        cfg = algorithm_config(algorithm, 14.4, 2_000 if smoke else 5_000_000,
                               smoke=smoke, max_ep_len=720)
        for section, value in cfg.items():
            assert section in native
            if isinstance(value, dict):
                assert set(value) <= set(native[section]), (algorithm, section, value)
        assert cfg["algo_cfgs"]["gamma"] == .8
        assert not cfg["algo_cfgs"]["cost_normalize"]
        if algorithm in LAGRANGIAN:
            assert cfg["lagrange_cfgs"]["cost_limit"] == 14.4
        elif algorithm in DIRECT_BUDGET:
            assert cfg["algo_cfgs"]["cost_limit"] == 14.4
        elif algorithm in AUGMENTED:
            assert cfg["algo_cfgs"]["safety_budget"] == 14.4
        if algorithm in OFF_POLICY:
            assert "cost_gamma" not in cfg["algo_cfgs"]
            assert cfg["algo_cfgs"]["steps_per_epoch"] == 2000
            if smoke:
                assert cfg["algo_cfgs"]["start_learning_steps"] < 2000
        else:
            assert cfg["algo_cfgs"]["cost_gamma"] == .8


def test_fixed_target_augmentation():
    cfg = dict(saute_gamma=.999, max_ep_len=720, safety_budget=5., upper_budget=14.4)
    factor = (1 - .999 ** 720) / (1 - .999) / 720
    assert evaluation_budget("PPOSaute", cfg) == pytest.approx(5 * factor)
    assert evaluation_budget("PPOSimmerPID", cfg) == pytest.approx(14.4 * factor)
    assert evaluation_budget("SACLag", {}) is None
    with pytest.raises(ValueError):
        algorithm_config("PPOSaute", 0, max_ep_len=720)


def test_off_policy_checkpoint_schedule():
    cfg = algorithm_config("SACLag", 14.4)
    assert cfg["logger_cfgs"]["save_model_freq"] == 10
    assert cfg["algo_cfgs"]["size"] == 100_000
    with pytest.raises(ValueError):
        algorithm_config("SACLag", 14.4, steps=22_000)
    with pytest.raises(ValueError):
        algorithm_config("PPO", 14.4, buffer_size=100_000)


def test_zero_steps_not_replaced_by_default():
    from types import SimpleNamespace
    from omnipiano.safety.run import custom_cell
    from omnipiano.safety.suite import MAIN
    args = SimpleNamespace(steps=0, smoke_test=False, device="cpu", buffer_size=None)
    with pytest.raises(ValueError, match="positive multiple"):
        custom_cell("main", 0, MAIN[0], "PPO", 1, args)
