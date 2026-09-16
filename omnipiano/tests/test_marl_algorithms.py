"""Registry invariants and algorithm mathematics.

Everything here runs on CPU, in under a second, with no MuJoCo, no Ray and no
GPU -- which is precisely why it is the only place that can catch a wrong loss
before it costs eight hours.
"""

from __future__ import annotations

import pytest
import torch

from omnipiano.multiagent.algos import algo_table, get_algo, list_algos
from omnipiano.multiagent.algos.base import AlgoSpec
from omnipiano.multiagent.algos._happo_math import (
    compound_log_factor_update, happo_surrogate, ppo_surrogate)
from omnipiano.multiagent.algos._facmac_mixer import FactoredMixer
from omnipiano.multiagent.algos._mat_module import MATDecoder, MATEncoder


# --- registry ---------------------------------------------------------------
def test_expected_algorithms_are_registered():
    assert set(list_algos()) == {
        "ippo", "ippo-rllib-module", "mappo", "mappo-own-critic",
        "mappo-noshuffle", "happo", "happo-m1-control",
        "mat", "facmac", "masac", "ppo-monolithic"}


def test_only_validated_algorithms_are_supported():
    """A 'supported' algo must be launchable with no extra flags."""
    assert set(list_algos(status="supported")) == {
        "ippo", "ippo-rllib-module", "mappo", "mappo-own-critic"}
    for name in list_algos(status="supported"):
        get_algo(name).assert_launchable()


def test_unimplemented_algorithms_refuse_to_launch():
    # ppo-monolithic joins this set: num_agents is not an env_config knob here.
    for name in ("mat", "facmac", "masac", "ppo-monolithic"):
        with pytest.raises(ValueError, match="NOT IMPLEMENTED"):
            get_algo(name).assert_launchable(allow_experimental=True)


def test_experimental_algorithms_need_an_explicit_flag():
    for name in ("happo", "happo-m1-control", "mappo-noshuffle"):
        with pytest.raises(ValueError, match="EXPERIMENTAL"):
            get_algo(name).assert_launchable()
        get_algo(name).assert_launchable(allow_experimental=True)


def test_ippo_and_mappo_differ_only_in_the_critic():
    ippo, mappo = get_algo("ippo"), get_algo("mappo")
    for f in ("family", "rl_module", "execution", "critic_head",
              "learner_class", "num_agents_override"):
        assert getattr(ippo, f) == getattr(mappo, f), f
    assert (ippo.critic_input, mappo.critic_input) == ("own", "global")
    assert dict(ippo.training_overrides) == dict(mappo.training_overrides) == {}


def test_sequential_family_cannot_silently_degrade_to_mappo():
    """Guards the single most dangerous failure mode in this registry."""
    with pytest.raises(ValueError, match="requires learner_class"):
        AlgoSpec(name="fake-happo", display_name="x", reference="x",
                 family="on_policy_ppo_sequential",
                 critic_input="global", needs_global_state=True)


def test_q_critic_cannot_claim_the_ppo_loop():
    with pytest.raises(ValueError, match="off-policy family"):
        AlgoSpec(name="fake", display_name="x", reference="x",
                 family="on_policy_ppo", critic_input="global_joint_action",
                 critic_head="q", needs_global_state=True)


def test_nondecentralized_execution_must_be_documented():
    for name in ("mat", "ppo-monolithic"):
        spec = get_algo(name)
        assert spec.execution != "decentralized"
        assert len(spec.notes) > 80, f"{name}: caveat too short to be a warning"


def test_happo_control_is_only_a_control():
    """It must never be mistaken for a reportable baseline."""
    spec = get_algo("happo-m1-control")
    assert spec.status == "experimental"
    assert "must never appear in a results table" in spec.blocking
    assert dict(spec.training_overrides) == dict(
        get_algo("mappo-noshuffle").training_overrides), (
        "the control and its comparison partner must share the shuffle setting")


def test_happo_learner_inherits_the_ctde_learner():
    """Guards the failure mode where HAPPO silently loses MAPPO's learner.

    Skipped without ray/torch so the registry tests stay import-light, but it
    runs in CI and before any launch, which is when it matters.
    """
    ray = pytest.importorskip("ray")            # noqa: F841
    from omnipiano.multiagent.algos.ppo_learner import OmniPianoPPOTorchLearner
    for name in ("happo", "happo-m1-control"):
        cls = get_algo(name).resolve_learner_class()
        assert cls is not None, f"{name}: learner_class did not resolve"
        assert issubclass(cls, OmniPianoPPOTorchLearner), (
            f"{name}: {cls.__name__} must subclass OmniPianoPPOTorchLearner, or "
            f"it silently drops the separate actor/critic optimizers, critic_lr, "
            f"adam_epsilon, value normalisation and prediction-delta vf clipping "
            f"-- five extra variables in a one-variable ablation")    


def test_algo_table_renders():
    assert "launchable now" in algo_table()


# --- HAPPO mathematics ------------------------------------------------------
def test_happo_reduces_to_ppo_when_factor_is_one():
    """The identity that makes the whole implementation auditable."""
    g = torch.Generator().manual_seed(0)
    logp_new = torch.randn(64, generator=g)
    logp_old = torch.randn(64, generator=g)
    adv = torch.randn(64, generator=g)
    a = happo_surrogate(logp_new, logp_old, adv, torch.zeros(64), 0.2)
    b = ppo_surrogate(logp_new, logp_old, adv, 0.2)
    torch.testing.assert_close(a, b)


def test_compound_factor_scales_the_advantage_multiplicatively():
    logp_new, logp_old = torch.zeros(4), torch.zeros(4)   # ratio == 1
    adv = torch.ones(4)
    for factor in (0.5, 1.0, 2.0):
        out = happo_surrogate(logp_new, logp_old, adv,
                              torch.full((4,), float(torch.log(torch.tensor(factor)))), 0.2)
        torch.testing.assert_close(out, torch.full((4,), factor), rtol=1e-5, atol=1e-6)


def test_compound_factor_is_detached_from_the_graph():
    """M must never backpropagate into a previously updated agent."""
    logp_prev = torch.zeros(4, requires_grad=True)
    m = compound_log_factor_update(torch.zeros(4), logp_prev, torch.zeros(4))
    assert not m.requires_grad


def test_compound_factor_is_clamped_both_ways():
    huge = torch.full((4,), 50.0)
    assert compound_log_factor_update(torch.zeros(4), huge, torch.zeros(4)).max() < 2.31
    assert compound_log_factor_update(torch.zeros(4), -huge, torch.zeros(4)).min() > -2.31


def test_clipping_still_bounds_the_update_under_a_large_factor():
    """A big M must not let a huge ratio through the clip."""
    logp_new = torch.full((8,), 5.0)          # ratio = e^5
    logp_old = torch.zeros(8)
    adv = torch.ones(8)
    out = happo_surrogate(logp_new, logp_old, adv, torch.zeros(8), 0.2)
    torch.testing.assert_close(out, torch.full((8,), 1.2), rtol=1e-5, atol=1e-6)


# --- FACMAC mixer -----------------------------------------------------------
def test_monotonic_mixer_is_monotonic_in_every_agent_utility():
    torch.manual_seed(0)
    mix = FactoredMixer(n_agents=3, state_dim=7, monotonic=True)
    q = torch.zeros(16, 3, requires_grad=True)
    mix(q, torch.randn(16, 7)).sum().backward()
    assert (q.grad >= -1e-6).all(), "monotonic mixer violated dQ_tot/dQ_i >= 0"


def test_nonmonotonic_mixer_is_allowed_to_break_monotonicity():
    torch.manual_seed(0)
    mix = FactoredMixer(n_agents=3, state_dim=7, monotonic=False)
    q = torch.zeros(256, 3, requires_grad=True)
    mix(q, torch.randn(256, 7)).sum().backward()
    assert (q.grad < 0).any(), "non-monotonic mode should not be constrained"


# --- MAT network ------------------------------------------------------------
def test_mat_shapes():
    enc, dec = MATEncoder(obs_dim=11, dim=32, n_heads=4), MATDecoder(act_dim=5, dim=32, n_heads=4)
    latent, values = enc(torch.randn(8, 4, 11))
    assert latent.shape == (8, 4, 32) and values.shape == (8, 4)
    mean, log_std = dec(latent, torch.zeros(8, 4, 5))
    assert mean.shape == log_std.shape == (8, 4, 5)


def test_mat_decoder_is_causal_across_agents():
    """Agent m's output must not depend on agent m+1's action, or the
    autoregressive factorisation of the joint policy is invalid."""
    torch.manual_seed(0)
    dec = MATDecoder(act_dim=5, dim=32, n_heads=4)
    latent = torch.randn(2, 4, 32)
    a = torch.zeros(2, 4, 5)
    b = a.clone()
    b[:, 3] = 99.0                       # perturb the LAST agent's action token
    with torch.no_grad():
        ma, _ = dec(latent, a)
        mb, _ = dec(latent, b)
    torch.testing.assert_close(ma[:, :3], mb[:, :3])       # earlier agents unchanged
    assert not torch.allclose(ma[:, 3], mb[:, 3])          # last agent does change
