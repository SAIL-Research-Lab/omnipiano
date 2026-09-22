"""FACMAC readiness gates: architecture, joint gradient, targets and replay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from omnipiano.multiagent import train
from omnipiano.multiagent.algos import get_algo
from omnipiano.multiagent.algos._facmac_mixer import FactoredMixer
from omnipiano.multiagent.algos._native import _facmac_compile_mode, make_model
from omnipiano.multiagent.training.native import materialize_scalar_stats
from omnipiano.multiagent.training.native_io import Replay


REFERENCE_COMMIT = "d7e62b8c51a5a77330de85f83c10553d0bd18fe5"


def _options(**overrides):
    values = {
        "variant": "facmac_continuous_qmix_v1",
        "reference_commit": REFERENCE_COMMIT,
        "parameter_sharing": False,
        "actor_hidden_sizes": [400, 400],
        "utility_hidden_sizes": [400, 400],
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "adam_epsilon": 1e-2,
        "grad_clip": 0.5,
        "tau": 1e-3,
        "action_l2": 1e-3,
        "mixer_embed": 64,
        "hypernet_embed": 64,
        "monotonic": True,
    }
    values.update(overrides)
    return values


def _model(seed=0, **option_overrides):
    torch.manual_seed(seed)
    args = SimpleNamespace(algo="facmac", gamma=0.8)
    meta = {
        "agents": ["agent_1", "agent_2"],
        "action_dims": [2, 3],
        "own_slices": [[0, 3], [0, 4]],
        "state_dim": 5,
    }
    return make_model(args, meta, _options(**option_overrides), torch.device("cpu"))


def _batch(size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "o": torch.randn(size, 7, generator=g),
        "s": torch.randn(size, 5, generator=g),
        "a": torch.rand(size, 5, generator=g) * 2 - 1,
        "no": torch.randn(size, 7, generator=g),
        "ns": torch.randn(size, 5, generator=g),
        "r": torch.randn(size, generator=g),
        "term": torch.zeros(size),
        "trunc": torch.zeros(size),
    }


def _parameters(module):
    return [value.detach().clone() for value in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(old, new.detach())
               for old, new in zip(before, module.parameters()))


def test_facmac_is_supported_by_the_native_backend():
    spec = get_algo("facmac")
    assert spec.status == "supported"
    assert spec.backend == "native"
    spec.assert_launchable()


def test_facmac_compile_mode_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv("OMNIPIANO_TORCH_COMPILE", raising=False)
    monkeypatch.delenv("OMNIPIANO_TORCH_COMPILE_MODE", raising=False)
    assert _facmac_compile_mode() is None

    monkeypatch.setenv("OMNIPIANO_TORCH_COMPILE", "1")
    assert _facmac_compile_mode() == "reduce-overhead"
    monkeypatch.setenv("OMNIPIANO_TORCH_COMPILE_MODE", "default")
    assert _facmac_compile_mode() == "default"
    monkeypatch.setenv("OMNIPIANO_TORCH_COMPILE_MODE", "max-autotune")
    with np.testing.assert_raises_regex(ValueError, "must be 'default'"):
        _facmac_compile_mode()


def test_facmac_defaults_match_pinned_mamujoco_reference():
    args = train._parse_args(["--algo", "facmac"])
    spec = train._resolve_args(args)
    options = args._native_options

    assert spec.backend == "native"
    assert options["reference_commit"] == REFERENCE_COMMIT
    assert options["actor_hidden_sizes"] == [400, 400]
    assert options["utility_hidden_sizes"] == [400, 400]
    assert options["actor_lr"] == options["critic_lr"] == 1e-3
    assert options["adam_epsilon"] == 1e-2
    assert options["grad_clip"] == 0.5
    assert options["batch_size"] == 100
    assert options["replay_capacity"] == 1_000_000
    assert options["buffer_warmup"] == 1_000
    assert options["random_action_steps"] == 10_000
    assert options["updates_per_env_step"] == 1.0
    assert options["tau"] == 1e-3
    assert options["noise_std"] == 0.1
    assert options["action_l2"] == 1e-3
    assert options["checkpoint_replay_transitions"] == 50_000
    # Deliberate OmniPiano protocol adaptation.
    assert args.gamma == 0.8


def test_mixer_matches_the_qmix_equation_and_is_monotonic():
    torch.manual_seed(3)
    mixer = FactoredMixer(3, 7, embed_dim=11, hypernet_embed=13)
    utilities = torch.randn(17, 3, requires_grad=True)
    states = torch.randn(17, 7)

    actual = mixer(utilities, states)
    w1 = mixer.hyper_w1(states).abs().view(17, 3, 11)
    b1 = mixer.hyper_b1(states).view(17, 1, 11)
    hidden = torch.nn.functional.elu(torch.bmm(utilities[:, None], w1) + b1)
    w2 = mixer.hyper_w2(states).abs().view(17, 11, 1)
    expected = (
        torch.bmm(hidden, w2).view(17)
        + mixer.state_value(states).view(17)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.sum().backward()
    assert (utilities.grad >= -1e-7).all()


def test_actor_and_utility_architecture_matches_reference_shape():
    model = _model()
    for network in list(model.actors) + list(model.q.utilities):
        linears = [layer for layer in network.modules()
                   if isinstance(layer, torch.nn.Linear)]
        assert [layer.out_features for layer in linears] == [400, 400, linears[-1].out_features]
        assert sum(isinstance(layer, torch.nn.ReLU)
                   for layer in network.modules()) == 2
        assert not any(isinstance(layer, torch.nn.LayerNorm)
                       for layer in network.modules())


def test_one_update_changes_every_actor_critic_and_soft_target():
    model = _model(seed=4, tau=0.2)
    actor_before = [_parameters(actor) for actor in model.actors]
    critic_before = _parameters(model.q)
    target_actor_before = [_parameters(actor) for actor in model.target_actors]
    target_q_before = _parameters(model.target_q)

    stats = model.learn(_batch(seed=9))

    assert np.isfinite(list(stats.values())).all()
    assert all(_changed(before, actor)
               for before, actor in zip(actor_before, model.actors))
    assert _changed(critic_before, model.q)
    for old_target, target_actor, actor in zip(
        target_actor_before, model.target_actors, model.actors
    ):
        for old, target, source in zip(old_target, target_actor.parameters(), actor.parameters()):
            torch.testing.assert_close(
                target, old.lerp(source.detach(), 0.2), rtol=1e-6, atol=1e-7
            )
    for old, target, source in zip(
        target_q_before, model.target_q.parameters(), model.q.parameters()
    ):
        torch.testing.assert_close(
            target, old.lerp(source.detach(), 0.2), rtol=1e-6, atol=1e-7
        )


def test_deferred_facmac_statistics_materialize_as_finite_scalars():
    model = _model(seed=8)
    pending = model.learn(_batch(seed=11), sync_stats=False)
    assert pending and all(torch.is_tensor(value) for value in pending.values())
    stats = materialize_scalar_stats(pending)
    assert set(stats) == set(pending)
    assert np.isfinite(list(stats.values())).all()


def test_true_termination_blocks_bootstrap_but_truncation_does_not():
    model = _model(seed=5)
    batch = _batch(size=3, seed=2)
    batch["r"] = torch.tensor([1.0, 1.0, 1.0])
    batch["term"] = torch.tensor([1.0, 0.0, 0.0])
    batch["trunc"] = torch.tensor([0.0, 1.0, 0.0])
    target = model.td_target(batch)

    assert target[0] == 1.0
    # Truncation and an ordinary transition use the same bootstrap rule.
    assert target[1] != 1.0
    assert target[2] != 1.0


def test_replay_sampling_is_unique_and_round_trips():
    codec = SimpleNamespace(own_dim=3, state_dim=4, action_dim=2)
    replay = Replay(codec, capacity=16, maximum_gib=0.01)
    values = np.arange(10, dtype=np.float32)
    batch = {
        "o": np.repeat(values[:, None], 3, axis=1),
        "s": np.repeat(values[:, None], 4, axis=1),
        "a": np.repeat(values[:, None], 2, axis=1),
        "no": np.repeat((values + 1)[:, None], 3, axis=1),
        "ns": np.repeat((values + 1)[:, None], 4, axis=1),
        "r": values,
        "term": np.zeros(10, np.float32),
        "trunc": np.zeros(10, np.float32),
    }
    replay.add(batch)
    sampled = replay.sample(10)
    assert len(np.unique(sampled["r"])) == 10

    np.random.seed(13)
    expected = [replay.sample(6) for _ in range(4)]
    np.random.seed(13)
    grouped = replay.sample_many(4, 6)
    for name in replay.arrays:
        np.testing.assert_array_equal(
            grouped[name], np.stack([batch[name] for batch in expected])
        )

    restored = Replay(codec, capacity=16, maximum_gib=0.01)
    restored.restore(replay.state())
    assert restored.size == replay.size and restored.position == replay.position
    for name in replay.arrays:
        np.testing.assert_array_equal(
            restored.arrays[name][:restored.size], replay.arrays[name][:replay.size]
        )


def test_replay_recovery_tail_keeps_newest_ring_entries_only():
    codec = SimpleNamespace(own_dim=1, state_dim=1, action_dim=1)
    replay = Replay(codec, capacity=5, maximum_gib=0.01)

    def add(values):
        values = np.asarray(values, dtype=np.float32)
        batch = {
            name: values[:, None]
            for name in ("o", "s", "a", "no", "ns")
        }
        batch.update(
            r=values,
            term=np.zeros(len(values), np.float32),
            trunc=np.zeros(len(values), np.float32),
        )
        replay.add(batch)

    add([0, 1, 2, 3, 4])
    add([5, 6, 7])
    assert replay.size == 5 and replay.position == 3

    full = replay.state()
    assert full["position"] == 3 and not full["truncated"]
    np.testing.assert_array_equal(full["arrays"]["r"], replay.arrays["r"])

    tail = replay.state(max_transitions=3)
    assert tail["size"] == 3 and tail["original_size"] == 5
    assert tail["position"] == 3 and tail["truncated"]
    np.testing.assert_array_equal(tail["arrays"]["r"], [5, 6, 7])

    restored = Replay(codec, capacity=5, maximum_gib=0.01)
    restored.restore(tail)
    assert restored.size == restored.position == 3
    np.testing.assert_array_equal(restored.arrays["r"][:3], [5, 6, 7])
