from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from omnipiano.configs import SafetyConfig
from omnipiano.safety.semantics import CostSpec, SemanticConstraint, SETTINGS, aggregate, pair_forces
from omnipiano.safety.suite import MAIN, HANDS, EXTENSIONS, BUDGETS, cells, hand_specs
from omnipiano.safety.runtime import algorithm_config
from omnipiano.wrappers.safety_wrapper import SafetyWrapper


def test_matrix():
    assert sum(map(len, SETTINGS.values())) == 12
    assert len(EXTENSIONS) == 48
    assert len({t.env_id for t in EXTENSIONS}) == 48
    assert [len(cells(g)) for g in ("main", "hands", "budget", "extensions")] == [120, 60, 63, 96]
    assert [t.budget for t in MAIN] == [19.95, 11.76, 14.4, 11.26, 14.4, 14.4, 36.0, 14.4]
    assert [t.hands for t in MAIN] == [2, 2, 3, 3, 4, 4, 5, 5]
    assert BUDGETS == (1.44, 4.32, 14.4, 43.2, 144.0)
    for a, b in zip(HANDS, HANDS[1:]):
        assert hand_specs(a) == hand_specs(b)[:a.hands]
    assert len({t.env_id for t in HANDS}) == 4


def test_main_config_compatibility():
    from dataclasses import asdict
    from omnipiano.safety.suite import SafetyEnvConfig
    cfg = asdict(SafetyEnvConfig(disable_fingering_reward=True))
    assert cfg["energy_penalty_coef"] == 0
    assert cfg["disable_fingering_reward"] is True


def test_seed_aggregation():
    from omnipiano.safety.plot import mean_sd
    runs = [({"seed": s}, np.array([0, 20_000]), {"reward": np.array([s, s + 1])}) for s in (1, 2, 3)]
    x, mean, sd = mean_sd(runs, "reward")
    assert mean.tolist() == [2, 3]
    assert sd.tolist() == [1, 1]
    with pytest.raises(ValueError):
        mean_sd(runs[:2], "reward")
    bad = list(runs)
    bad[0] = (bad[0][0], np.array([0, 40_000]), bad[0][2])
    with pytest.raises(ValueError):
        mean_sd(bad, "reward")


@pytest.mark.parametrize("setting,expected", [("event", 1), ("excess", .2), ("fraction", .5)])
def test_aggregation(setting, expected):
    assert aggregate([0, .4], setting) == pytest.approx(expected)
    assert aggregate([0, .4], setting, 2) == pytest.approx(expected * 2)


@pytest.mark.parametrize("values", [[], [float("nan")], [-1], [float("inf")]])
def test_bad_cost(values):
    with pytest.raises(ValueError):
        aggregate(values, "excess")


def test_invalid_specs():
    with pytest.raises(ValueError):
        CostSpec("actuator_power", "unknown")
    with pytest.raises(ValueError):
        CostSpec("joint_range", "event", central_fraction=2)
    with pytest.raises(ValueError):
        CostSpec("injured_finger", "fraction", reference=0)
    with pytest.raises(ValueError):
        CostSpec("hand_collision", "event", weight=-1)


@pytest.mark.parametrize("algo", ["PPO", "PPOLag", "OnCRPO", "CPO", "CUP"])
def test_algorithm_configs(algo):
    cfg = algorithm_config(algo, 14.4)
    assert cfg["algo_cfgs"]["gamma"] == cfg["algo_cfgs"]["cost_gamma"] == .8
    assert not cfg["algo_cfgs"]["cost_normalize"]
    assert not cfg["algo_cfgs"]["reward_normalize"]
    assert cfg["train_cfgs"]["vector_env_nums"] == 1
    if algo in ("PPOLag", "CUP"):
        assert cfg["lagrange_cfgs"]["cost_limit"] == 14.4
    elif algo != "PPO":
        assert cfg["algo_cfgs"]["use_cost"] and cfg["algo_cfgs"]["cost_limit"] == 14.4


@pytest.mark.parametrize("semantic", list(SETTINGS))
@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_real_mujoco(k, semantic):
    from dm_control import mjcf
    from robopianist.models.hands.shadow_hand import ShadowHand
    root = mjcf.RootElement()
    hands = {}
    for i in range(k):
        name = "rh" if i == 0 else f"h{i}"
        h = ShadowHand(name=name)
        root.attach(h.mjcf_model).pos = (i * 2, 0, 0)
        hands[name] = h
    physics = mjcf.Physics.from_mjcf_model(root)
    task = SimpleNamespace(hands_by_name=hands)
    physics.step()
    for setting in SETTINGS[semantic]:
        spec = CostSpec(semantic, setting)
        constraint = SemanticConstraint(spec)
        constraint._get_dm_internals = lambda env: (physics, task)
        info = {}
        cost = constraint.compute_cost(None, None, None, info)
        assert np.isfinite(cost) and cost >= 0
        assert info[f"step_safety/{semantic}/unit_count"] == {
            "joint_range": 5 * k, "actuator_power": k if setting == "fraction" else 1,
            "injured_finger": 5, "hand_collision": k * (k - 1) // 2}[semantic]
        if semantic == "hand_collision":
            assert cost == 0  # isolated hand roots have no inter-hand contacts


def test_wrapper_preserves_reward_resets_episode():
    class ConstantConstraint(SemanticConstraint):
        def compute_cost(self, env, action, obs, info):
            return .5
    class Env(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        def reset(self, **kwargs):
            return np.zeros(1, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(1, dtype=np.float32), 7., True, False, {}
    env = SafetyWrapper(Env(), SafetyConfig(constraints=[ConstantConstraint(CostSpec("joint_range", "excess"))]))
    for _ in range(2):
        env.reset()
        _, reward, _, _, info = env.step(np.zeros(1))
        assert reward == 7
        assert info["episode_safety/cost_total"] == .5


def test_collision_pairs_include_zeros_and_exclude_piano(monkeypatch):
    from omnipiano.safety import semantics
    hands = [SimpleNamespace(mjcf_model=SimpleNamespace(find_all=lambda _, i=i:
             [SimpleNamespace(full_identifier=str(i))])) for i in range(3)]
    contacts = [SimpleNamespace(geom=(0, 1), efc_address=0),
                SimpleNamespace(geom=(0, 1), efc_address=0),
                SimpleNamespace(geom=(0, 9), efc_address=0),
                SimpleNamespace(geom=(1, 1), efc_address=0)]
    physics = SimpleNamespace(model=SimpleNamespace(name2id=lambda name, _: int(name), ptr=None),
                              data=SimpleNamespace(contact=contacts, ptr=None))
    def force(model, data, index, out):
        out[:] = 0
        out[0] = 8
    monkeypatch.setattr(semantics.mujoco, "mj_contactForce", force)
    raw = pair_forces(physics, hands)
    assert raw.tolist() == [16, 0, 0]
    assert aggregate(np.maximum(raw / 10 - 1, 0), "fraction") == pytest.approx(1 / 3)
    assert aggregate(np.maximum(raw / 10 - 1, 0), "excess") == pytest.approx(.2)


@pytest.mark.parametrize("powers,expected", [([6, 0], .5), ([0, 0, 0], 0),
                                           ([6, 4, 0], 1 / 3), ([5, 5, 0, 0], .5),
                                           ([5, 5, 5, 5, 5], 1)])
def test_power_fraction_per_hand(monkeypatch, powers, expected):
    from omnipiano.safety import semantics
    names = [f"hand_{i}" for i in range(len(powers))]
    task = SimpleNamespace(hands_by_name=dict(zip(names, powers)))
    monkeypatch.setattr(semantics, "actuator_power", lambda physics, hand: np.array([hand]))
    constraint = SemanticConstraint(CostSpec("actuator_power", "fraction", reference=4))
    constraint._get_dm_internals = lambda env: (None, task)
    info = {}
    assert constraint.compute_cost(None, None, None, info) == pytest.approx(expected)
    assert info["step_safety/actuator_power/unit_count"] == len(powers)
    assert info["step_safety/actuator_power/violating_fraction"] == pytest.approx(expected)


def test_power_settings_distinct_and_selected_hands(monkeypatch):
    from omnipiano.safety import semantics
    task = SimpleNamespace(hands_by_name={"a": 6., "b": 0., "c": 100.})
    monkeypatch.setattr(semantics, "actuator_power", lambda physics, hand: np.array([hand]))
    for setting, reference, expected in (("event", 8, 0), ("excess", 8, 0), ("fraction", 4, .5)):
        spec = CostSpec("actuator_power", setting, reference=reference, hand_names=("a", "b"))
        constraint = SemanticConstraint(spec)
        constraint._get_dm_internals = lambda env: (None, task)
        assert constraint.compute_cost(None, None, None, {}) == expected
        weighted = SemanticConstraint(replace(spec, weight=2))
        weighted._get_dm_internals = constraint._get_dm_internals
        assert weighted.compute_cost(None, None, None, {}) == expected * 2


def test_power_fraction_catalogue():
    fraction = [t for t in EXTENSIONS if (t.cost.semantic, t.cost.setting) == ("actuator_power", "fraction")]
    assert [t.hands for t in fraction] == [2, 3, 4, 5]
    assert all(t.cost.reference == 4 for t in fraction)
    assert [t.budget for t in fraction] == [19.95, 28.15, 36, 36]
    assert all(t.cost.setting != "fraction" for t in MAIN + HANDS if t.cost.semantic == "actuator_power")
