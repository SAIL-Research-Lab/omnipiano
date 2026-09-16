"""CTDE correctness tests. These run without Ray training or MuJoCo."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ray.rllib")

from omnipiano.multiagent.algos.ppo_module import (
    CtdePPOTorchRLModule,
    RunningValueNorm,
)


OWN_DIM, GLOBAL_DIM, ACT_DIM = 64, 128, 45
OBS_DIM = GLOBAL_DIM + OWN_DIM          # flatten order: [global_state | own]
OWN_SLICE = slice(GLOBAL_DIM, OBS_DIM)
GS_SLICE = slice(0, GLOBAL_DIM)


def _module(critic_input: str) -> CtdePPOTorchRLModule:
    return CtdePPOTorchRLModule(
        observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32),
        model_config={
            "own_obs_start": GLOBAL_DIM,
            "own_obs_dim": OWN_DIM,
            "global_state_start": 0,
            "global_state_dim": GLOBAL_DIM,
            "critic_input": critic_input,
            "hidden_sizes": (32, 32),
            "activation": "tanh",
        },
    )


def _obs(batch: int = 8) -> "torch.Tensor":
    return torch.randn(batch, OBS_DIM, requires_grad=True)


@pytest.mark.parametrize("critic_input", ["own", "global"])
def test_forward_shapes(critic_input: str) -> None:
    from ray.rllib.core.columns import Columns

    module = _module(critic_input)
    out = module.forward_inference({Columns.OBS: _obs()})
    # TorchDiagGaussian expects [loc | log_std].
    assert out[Columns.ACTION_DIST_INPUTS].shape == (8, 2 * ACT_DIM)
    assert module.compute_values({Columns.OBS: _obs()}).shape == (8,)


@pytest.mark.parametrize("critic_input", ["own", "global"])
def test_actor_never_reads_global_state(critic_input: str) -> None:
    """The defining CTDE property: decentralized EXECUTION.

    If this fails, the policy has access to information unavailable at
    deployment and every reported number is invalid.
    """
    from ray.rllib.core.columns import Columns

    module = _module(critic_input)
    obs = _obs()
    module.forward_train({Columns.OBS: obs})[
        Columns.ACTION_DIST_INPUTS
    ].sum().backward()
    grad = obs.grad
    assert grad is not None
    np.testing.assert_array_equal(
        grad[:, GS_SLICE].detach().numpy(),
        np.zeros((obs.shape[0], GLOBAL_DIM), dtype=np.float32),
        err_msg="actor gradient leaked into the global-state slice",
    )
    assert torch.any(grad[:, OWN_SLICE] != 0), "actor ignored its own observation"


def test_critic_reads_only_its_configured_slice() -> None:
    from ray.rllib.core.columns import Columns

    # MAPPO: V(s) must depend on the global state and NOT on the own slice.
    module = _module("global")
    obs = _obs()
    module.compute_values({Columns.OBS: obs}).sum().backward()
    assert torch.any(obs.grad[:, GS_SLICE] != 0), "centralized critic ignored s"
    np.testing.assert_array_equal(
        obs.grad[:, OWN_SLICE].detach().numpy(),
        np.zeros((obs.shape[0], OWN_DIM), dtype=np.float32),
    )

    # IPPO: V(o_i) must be the mirror image.
    module = _module("own")
    obs = _obs()
    module.compute_values({Columns.OBS: obs}).sum().backward()
    assert torch.any(obs.grad[:, OWN_SLICE] != 0)
    np.testing.assert_array_equal(
        obs.grad[:, GS_SLICE].detach().numpy(),
        np.zeros((obs.shape[0], GLOBAL_DIM), dtype=np.float32),
    )


def test_actor_parameter_count_is_critic_independent() -> None:
    """IPPO and MAPPO must have identical actors, or the ablation is confounded."""
    own = _module("own").ctde_spec
    glob = _module("global").ctde_spec
    assert own["actor_params"] == glob["actor_params"]
    assert own["own_slice"] == glob["own_slice"]
    assert glob["critic_in_dim"] == GLOBAL_DIM
    assert own["critic_in_dim"] == OWN_DIM


def test_actor_and_critic_optimizer_parameters_are_disjoint_and_complete() -> None:
    module = _module("global")
    actor_ids = {id(p) for p in module.actor_parameters()}
    critic_ids = {id(p) for p in module.critic_parameters()}
    trainable_ids = {id(p) for p in module.parameters() if p.requires_grad}
    assert actor_ids.isdisjoint(critic_ids)
    assert actor_ids | critic_ids == trainable_ids


def test_value_norm_round_trip_and_checkpoint_buffers() -> None:
    norm = RunningValueNorm(beta=0.5, epsilon=1e-5)
    values = torch.tensor([1.0, 3.0, 5.0])
    norm.update(values)
    assert norm.mean.item() == pytest.approx(3.0)
    assert norm.variance.item() == pytest.approx(8.0 / 3.0)
    torch.testing.assert_close(norm.denormalize(norm.normalize(values)), values)
    assert set(norm.state_dict()) == {
        "running_mean", "running_mean_sq", "debiasing_term"
    }


def test_compute_values_exposes_denormalized_critic_output() -> None:
    from ray.rllib.core.columns import Columns

    module = _module("own")
    batch = {Columns.OBS: _obs()}
    module.update_value_normalizer(torch.tensor([10.0, 20.0, 30.0]))
    normalized = module.compute_normalized_values(batch)
    torch.testing.assert_close(
        module.compute_values(batch), module.denormalize_values(normalized)
    )


def test_global_critic_requires_global_state() -> None:
    with pytest.raises(ValueError, match="global_state_dim"):
        CtdePPOTorchRLModule(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (OWN_DIM,), np.float32),
            action_space=gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32),
            model_config={
                "own_obs_start": 0, "own_obs_dim": OWN_DIM,
                "global_state_start": 0, "global_state_dim": 0,
                "critic_input": "global",
            },
        )
