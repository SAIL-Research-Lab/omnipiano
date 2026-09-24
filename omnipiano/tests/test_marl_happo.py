"""Behavioral readiness gates for the strict native HAPPO learner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from omnipiano.multiagent import train
from omnipiano.multiagent.algos import get_algo
from omnipiano.multiagent.algos._native import make_model
from omnipiano.multiagent.training.runner import _training_telemetry


def _args(**overrides):
    values = {
        "algo": "happo",
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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _meta():
    return {
        "agents": ["agent_1", "agent_2"],
        "action_dims": [2, 3],
        "own_slices": [[0, 3], [0, 4]],
        "state_dim": 5,
    }


def _model(seed=0, **overrides):
    torch.manual_seed(seed)
    return make_model(_args(**overrides), _meta(), {}, torch.device("cpu"))


def _batch(model, size=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    observations = {
        "o": torch.randn(size, 7, generator=generator),
        "s": torch.randn(size, 5, generator=generator),
    }
    with torch.no_grad():
        rollout = model.act(observations)
    advantage = torch.randn(size, generator=generator)
    returns = torch.randn(size, generator=generator)
    return {
        **observations,
        "a": rollout["a"].detach().clone(),
        "lp": rollout["lp"].detach().clone(),
        "vn": rollout["vn"].detach().clone(),
        "v": rollout["v"].detach().clone(),
        "nv": rollout["v"].detach().clone(),
        "adv": advantage[:, None].expand(-1, 2).clone(),
        "ret": returns[:, None].expand(-1, 2).clone(),
    }


def _parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _changed(before, module):
    return any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, module.parameters())
    )


def test_happo_is_directly_launchable_with_native_defaults():
    spec = get_algo("happo")
    spec.assert_launchable()
    assert spec.backend == "native"

    args = train._parse_args(["--algo", "happo"])
    resolved = train._resolve_args(args)
    assert resolved is spec
    assert args.ray_num_cpus is None
    assert args.gamma == 0.8
    assert args.num_epochs == 5
    assert args.minibatch_size == args.train_batch_size == 4000


def test_happo_has_independent_heterogeneous_actors_and_one_shared_critic():
    model = _model()
    assert len(model.actors) == 2
    assert model.actors[0] is not model.actors[1]
    assert model.actors[0].dim == 2
    assert model.actors[1].dim == 3
    assert set(model.optimizers) == {"actor_0", "actor_1", "critic"}

    batch = _batch(model)
    normalized, values = model.values(batch)
    assert normalized.shape == values.shape == (16, 2)
    torch.testing.assert_close(normalized[:, 0], normalized[:, 1])
    torch.testing.assert_close(values[:, 0], values[:, 1])


def test_each_actor_finishes_all_epochs_before_the_next_actor_starts():
    model = _model(num_epochs=2, minibatch_size=8)
    model.agent_update_order = lambda: [0, 1]
    batch = _batch(model)
    events = []

    for name in ("actor_0", "actor_1", "critic"):
        optimizer = model.optimizers[name]
        original = optimizer.step

        def step(*args, _name=name, _original=original, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        optimizer.step = step

    model.learn(batch)

    # 16 rows / minibatch 8 * 2 epochs = 4 consecutive steps per component.
    assert events == ["actor_0"] * 4 + ["actor_1"] * 4 + ["critic"] * 4


def test_second_actor_receives_the_first_actors_exact_post_update_factor():
    model = _model(seed=3, num_epochs=2, minibatch_size=8)
    model.agent_update_order = lambda: [0, 1]
    batch = _batch(model, seed=7)
    old_first_logp = batch["lp"][:, 0].clone()

    stats = model.learn(batch)

    with torch.no_grad():
        new_first_logp, _ = model.evaluate_actor(batch, 0)
        expected = (new_first_logp - old_first_logp).exp()
    np.testing.assert_allclose(
        stats["actor_1_factor_mean"], float(expected.mean()), rtol=1e-6
    )
    np.testing.assert_allclose(
        stats["actor_1_factor_min"], float(expected.min()), rtol=1e-6
    )
    np.testing.assert_allclose(
        stats["actor_1_factor_max"], float(expected.max()), rtol=1e-6
    )
    assert not np.isclose(stats["actor_1_factor_max"], 1.0)


def test_one_happo_update_changes_every_actor_and_the_shared_critic():
    model = _model(seed=5)
    batch = _batch(model, seed=11)
    actors_before = [_parameters(actor) for actor in model.actors]
    critic_before = _parameters(model.critic)

    stats = model.learn(batch)

    assert all(
        _changed(before, actor)
        for before, actor in zip(actors_before, model.actors)
    )
    assert _changed(critic_before, model.critic)
    assert np.isfinite(list(stats.values())).all()


def test_ep_happo_rejects_agent_specific_advantages():
    model = _model()
    batch = _batch(model)
    batch["adv"][:, 1].add_(1.0)
    with np.testing.assert_raises_regex(RuntimeError, "shared adv"):
        model.learn(batch)


def test_stale_rollout_log_probabilities_are_rejected():
    model = _model()
    batch = _batch(model)
    batch["lp"][:, 0].add_(0.1)
    model.agent_update_order = lambda: [0, 1]
    with np.testing.assert_raises_regex(RuntimeError, "Stale rollout"):
        model.learn(batch)


def test_happo_model_optimizer_valuenorm_and_rng_round_trip():
    model = _model(seed=19)
    model.agent_update_order = lambda: [0, 1]
    model.learn(_batch(model, seed=23))

    model_state = model.state_dict()
    optimizer_state = model.optimizer_state()
    assert "normalizer.mean_acc" in model_state
    assert set(optimizer_state) == {"actor_0", "actor_1", "critic"}

    restored = _model(seed=101)
    restored.load_state_dict(model_state)
    restored.restore_optimizers(optimizer_state)
    probe = {
        "o": torch.randn(7, 7),
        "s": torch.randn(7, 5),
    }
    with torch.no_grad():
        first = model.act(probe, deterministic=True)
        second = restored.act(probe, deterministic=True)
    for key in ("a", "lp", "vn", "v"):
        torch.testing.assert_close(first[key], second[key])

    torch.manual_seed(31)
    state = torch.get_rng_state()
    expected = restored.agent_update_order()
    torch.set_rng_state(state)
    assert restored.agent_update_order() == expected


def test_happo_compound_factor_is_persisted_in_training_telemetry():
    result = {
        "learners": {
            "__all_modules__": {
                "compound_factor_abs_log_mean": 0.125,
                "compound_factor_abs_log_max": 0.75,
            }
        },
        "env_runners": {},
    }

    telemetry = _training_telemetry(result)

    assert telemetry["learner/__all__/compound_factor_abs_log_mean"] == 0.125
    assert telemetry["learner/__all__/compound_factor_abs_log_max"] == 0.75
