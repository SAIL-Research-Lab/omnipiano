"""Behavioral readiness gates for the native A2PO learner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omnipiano.multiagent import train
from omnipiano.multiagent.algos import get_algo
from omnipiano.multiagent.algos._a2po_math import clipped_preceding_ratio
from omnipiano.multiagent.algos._native import make_model
from omnipiano.multiagent.training.native import native_options
from omnipiano.multiagent.training.native_io import add_gae
from omnipiano.multiagent.training.runner import _training_telemetry


def _args(**overrides):
    values = {
        "algo": "a2po",
        "hidden_sizes_parsed": [16, 16],
        "input_layer_norm": True,
        "activation": "tanh",
        "hidden_orthogonal_gain": 2 ** 0.5,
        "policy_output_gain": 0.01,
        "value_output_gain": 1.0,
        "initial_log_std": 0.0,
        "log_std_min": -5.0,
        "log_std_max": 2.0,
        "value_norm": True,
        "value_norm_beta": 0.99999,
        "value_norm_epsilon": 1e-5,
        "value_norm_variance_floor": 0.01,
        "lr": 3e-3,
        "critic_lr": 3e-3,
        "adam_epsilon": 1e-5,
        "num_epochs": 2,
        "minibatch_size": 8,
        "clip_param": 0.2,
        "entropy_coeff": 0.0,
        "grad_clip": 10.0,
        "vf_loss_coeff": 1.0,
        "vf_clip_param": 0.2,
        "gamma": 0.8,
        "gae_lambda": 0.95,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _options():
    return {
        "variant": "a2po_preopc_v1",
        "reference_commit": "28c11e6063bcf80caffc53a791c99dd7be1003b5",
        "parameter_sharing": False,
        "order": "semi_greedy",
        "order_score": "official_normalized_advantage",
        "order_score_epsilon": 1e-8,
        "preceding_ratio_clip": 0.1,
        "trace_clip_param": 1.0,
        "adaptive_clip_weight": 0.5,
        "two_stage": True,
        "agent_block_size": 1,
    }


def _meta():
    return {
        "agents": ["agent_1", "agent_2"],
        "action_dims": [2, 3],
        "own_slices": [[0, 3], [0, 4]],
        "state_dim": 5,
    }


def _model(seed=0, **overrides):
    torch.manual_seed(seed)
    return make_model(
        _args(**overrides), _meta(), _options(), torch.device("cpu"))


def _batch(model, size=16, streams=2, seed=0):
    if size % streams:
        raise ValueError("test fixture requires complete vector steps")
    generator = torch.Generator().manual_seed(seed)
    observations = {
        "o": torch.randn(size, sum(model.od), generator=generator),
        "s": torch.randn(size, model.sd, generator=generator),
    }
    following = {
        "o": torch.randn(size, sum(model.od), generator=generator),
        "s": torch.randn(size, model.sd, generator=generator),
    }
    with torch.no_grad():
        rollout = model.act(observations)
        _, next_values = model.values(following)

    keys = np.asarray([
        [stream, 0, timestep]
        for timestep in range(size // streams)
        for stream in range(streams)
    ], dtype=np.int64)
    numpy_batch = {
        "keys": keys,
        "r": torch.randn(size, generator=generator).numpy(),
        "term": np.zeros(size, np.float32),
        "trunc": np.zeros(size, np.float32),
        "v": rollout["v"].numpy(),
        "nv": next_values.numpy(),
    }
    add_gae(
        numpy_batch, gamma=model.args.gamma,
        gae_lambda=model.args.gae_lambda, streams=streams)
    return {
        **observations,
        "a": rollout["a"].detach().clone(),
        "lp": rollout["lp"].detach().clone(),
        "vn": rollout["vn"].detach().clone(),
        "v": rollout["v"].detach().clone(),
        "nv": next_values.detach().clone(),
        "r": torch.from_numpy(numpy_batch["r"]),
        "term": torch.from_numpy(numpy_batch["term"]),
        "trunc": torch.from_numpy(numpy_batch["trunc"]),
        "keys": torch.from_numpy(keys),
        "adv": torch.from_numpy(numpy_batch["adv"]),
        "ret": torch.from_numpy(numpy_batch["ret"]),
    }


def _parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module):
    return any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, module.parameters())
    )


def test_a2po_is_directly_launchable_with_canonical_defaults():
    spec = get_algo("a2po")
    spec.assert_launchable()
    assert spec.backend == "native"
    assert spec.execution == "decentralized"

    args = train._parse_args(["--algo", "a2po"])
    resolved = train._resolve_args(args)
    options = native_options(args)
    assert resolved is spec
    assert args.ray_num_cpus is None
    assert args.gamma == 0.8
    assert args.num_epochs == 5
    assert args.minibatch_size == args.train_batch_size == 4000
    assert options == _options()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"parameter_sharing": True}, "independent policies"),
        ({"trace_clip_param": 0.9}, "truncates each trace ratio at 1.0"),
    ],
)
def test_canonical_topology_and_preopc_trace_cannot_be_silently_changed(
    override, message,
):
    args = train._parse_args(["--algo", "a2po"])
    train._resolve_args(args)
    args._native_options = override
    with pytest.raises(ValueError, match=message):
        native_options(args)


def test_a2po_has_independent_heterogeneous_actors_and_global_critics():
    model = _model()
    assert len(model.actors) == len(model.critics) == 2
    assert model.actors[0] is not model.actors[1]
    assert model.critics[0] is not model.critics[1]
    assert model.actors[0].dim == 2
    assert model.actors[1].dim == 3
    assert set(model.optimizers) == {
        "actor_0", "actor_1", "critic_0", "critic_1"}

    batch = _batch(model)
    normalized, values = model.values(batch)
    assert normalized.shape == values.shape == (16, 2)


def test_two_stage_and_agent_by_agent_optimizer_order_is_strict():
    model = _model(num_epochs=2, minibatch_size=8)
    model.agent_update_order = lambda *_: [0, 1]
    batch = _batch(model)
    events = []
    for name, optimizer in model.optimizers.items():
        original = optimizer.step

        def step(*args, _name=name, _original=original, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        optimizer.step = step

    model.learn(batch)

    # Every PPO epoch contains a critic-only stage followed immediately by an
    # actor+critic stage. No optimizer for agent 1 runs before agent 0 finishes.
    one_epoch_0 = (
        ["critic_0"] * 2
        + [item for _ in range(2) for item in ("actor_0", "critic_0")]
    )
    one_epoch_1 = (
        ["critic_1"] * 2
        + [item for _ in range(2) for item in ("actor_1", "critic_1")]
    )
    one_agent = one_epoch_0 * 2
    other_agent = one_epoch_1 * 2
    assert events == one_agent + other_agent


def test_second_agent_uses_first_agents_exact_post_update_ratio():
    model = _model(seed=3)
    model.agent_update_order = lambda *_: [0, 1]
    batch = _batch(model, seed=7)
    old_logp = batch["lp"][:, 0].clone()

    stats = model.learn(batch)

    with torch.no_grad():
        new_logp, _ = model.evaluate_actor(batch, 0)
        expected = clipped_preceding_ratio(
            new_logp - old_logp, _options()["preceding_ratio_clip"])
    np.testing.assert_allclose(
        stats["actor_1_preceding_ratio_mean"], float(expected.mean()), rtol=1e-6)
    np.testing.assert_allclose(
        stats["actor_1_preceding_ratio_min"], float(expected.min()), rtol=1e-6)
    np.testing.assert_allclose(
        stats["actor_1_preceding_ratio_max"], float(expected.max()), rtol=1e-6)
    assert not np.isclose(stats["actor_1_preceding_ratio_max"], 1.0)


def test_one_update_changes_every_actor_critic_and_normalizer():
    model = _model(seed=5)
    model.agent_update_order = lambda *_: [0, 1]
    batch = _batch(model, seed=11)
    actors_before = [_parameters(actor) for actor in model.actors]
    critics_before = [_parameters(critic) for critic in model.critics]

    stats = model.learn(batch)

    assert all(_changed(before, actor)
               for before, actor in zip(actors_before, model.actors))
    assert all(_changed(before, critic)
               for before, critic in zip(critics_before, model.critics))
    assert all(normalizer.weight > 0 for normalizer in model.normalizers)
    assert np.isfinite(list(stats.values())).all()


def test_four_agent_heterogeneous_scalability_update():
    meta = {
        "agents": [f"agent_{i + 1}" for i in range(4)],
        "action_dims": [1, 2, 1, 3],
        "own_slices": [[0, 2], [0, 3], [0, 1], [0, 4]],
        "state_dim": 6,
    }
    model = make_model(
        _args(num_epochs=1, minibatch_size=8), meta, _options(),
        torch.device("cpu"),
    )
    model.agent_update_order = lambda *_: [0, 1, 2, 3]
    actors_before = [_parameters(actor) for actor in model.actors]
    critics_before = [_parameters(critic) for critic in model.critics]

    stats = model.learn(_batch(model, size=8, streams=2, seed=29))

    assert all(_changed(before, actor)
               for before, actor in zip(actors_before, model.actors))
    assert all(_changed(before, critic)
               for before, critic in zip(critics_before, model.critics))
    assert [stats[f"actor_{i}_update_position"] for i in range(4)] \
        == [0.0, 1.0, 2.0, 3.0]


def test_stale_rollout_log_probabilities_are_rejected():
    model = _model()
    batch = _batch(model)
    batch["lp"][:, 0].add_(0.1)
    with np.testing.assert_raises_regex(RuntimeError, "Stale rollout"):
        model.learn(batch)


def test_model_optimizer_normalizer_and_order_rng_round_trip():
    model = _model(seed=19)
    model.agent_update_order = lambda *_: [0, 1]
    model.learn(_batch(model, seed=23))
    model_state = model.state_dict()
    optimizer_state = model.optimizer_state()
    assert "normalizers.0.mean_acc" in model_state
    assert set(optimizer_state) == {
        "actor_0", "actor_1", "critic_0", "critic_1"}

    restored = _model(seed=101)
    restored.load_state_dict(model_state)
    restored.restore_optimizers(optimizer_state)
    probe = {"o": torch.randn(7, 7), "s": torch.randn(7, 5)}
    with torch.no_grad():
        first = model.act(probe, deterministic=True)
        second = restored.act(probe, deterministic=True)
    for key in ("a", "lp", "vn", "v"):
        torch.testing.assert_close(first[key], second[key])

    score_adv = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    score_value = torch.ones_like(score_adv)
    four_meta = {
        "agents": [f"agent_{i}" for i in range(4)],
        "action_dims": [1, 1, 1, 1],
        "own_slices": [[0, 1]] * 4,
        "state_dim": 2,
    }
    four = make_model(_args(), four_meta, _options(), torch.device("cpu"))
    np.random.seed(31)
    state = np.random.get_state()
    expected = four.agent_update_order(score_adv, score_value)
    np.random.set_state(state)
    assert four.agent_update_order(score_adv, score_value) == expected


def test_a2po_order_preopc_and_ratio_diagnostics_reach_tracking():
    telemetry = _training_telemetry({
        "learners": {"__all_modules__": {
            "actor_1_update_position": 1.0,
            "actor_1_preopc_adv_std": 0.75,
            "actor_1_preceding_ratio_max": 1.1,
            "critic_1_pretrain_loss": 0.25,
            "preceding_log_ratio_abs_max": 0.4,
        }},
        "env_runners": {},
    })
    assert telemetry["learner/__all__/actor_1_update_position"] == 1.0
    assert telemetry["learner/__all__/actor_1_preopc_adv_std"] == 0.75
    assert telemetry["learner/__all__/actor_1_preceding_ratio_max"] == 1.1
    assert telemetry["learner/__all__/critic_1_pretrain_loss"] == 0.25
    assert telemetry["learner/__all__/preceding_log_ratio_abs_max"] == 0.4
