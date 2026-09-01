"""One CTDE actor-critic RLModule shared by the IPPO and MAPPO baselines.

The two baselines differ in exactly one respect -- which slice of the flat
observation vector the critic reads:

    critic_input="own"     ->  V(o_i)  : IPPO   (decentralized critic)
    critic_input="global"  ->  V(s)    : MAPPO  (centralized critic, CTDE)

The actor *always* reads only ``obs[own_slice]``, so decentralized execution
holds for both baselines and the policy input is bit-identical between them.
Everything else -- layer sizes, activation, initialization order, optimizer,
every PPO hyperparameter -- is shared, which makes an IPPO/MAPPO pair a clean
single-factor ablation rather than a confounded architecture comparison.

References
----------
Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022).
    The surprising effectiveness of PPO in cooperative multi-agent games.
    Advances in Neural Information Processing Systems 35 (Datasets and
    Benchmarks Track).
de Witt, C. S., Gupta, T., Makoviichuk, D., Makoviychuk, V., Torr, P. H. S.,
    Sun, M., & Whiteson, S. (2020). Is independent learning all you need in the
    StarCraft multi-agent challenge? arXiv:2011.09533.
Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., &
    Madry, A. (2020). Implementation matters in deep RL: A case study on PPO and
    TRPO. ICLR.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

try:  # Ray >= 2.40
    from ray.rllib.core.rl_module.apis import ValueFunctionAPI
except ImportError:  # pragma: no cover - older layouts
    from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI

torch, nn = try_import_torch()

CRITIC_INPUTS = ("own", "global")

def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str,
    hidden_gain: float,
    out_gain: float,
    input_layer_norm: bool,
) -> "nn.Sequential":
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers: list = []
    prev = int(in_dim)
    if input_layer_norm:
        # Official MAPPO calls this "feature normalization": actor and critic
        # each normalize only the feature vector they are allowed to observe.
        # In particular, the actor never normalizes (and therefore never reads)
        # the global-state prefix.
        layers.append(nn.LayerNorm(prev))
    for hidden in hidden_dims:
        linear = nn.Linear(prev, int(hidden))
        nn.init.orthogonal_(linear.weight, gain=hidden_gain)
        nn.init.zeros_(linear.bias)
        layers += [linear, act_cls()]
        prev = int(hidden)
    out = nn.Linear(prev, int(out_dim))
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class RunningValueNorm(nn.Module):
    """Bias-corrected running return normalization used by official MAPPO.

    The statistics are registered buffers rather than optimizer parameters, so
    they are checkpointed and synchronized with the RLModule but never receive
    gradients.  A scalar normalizer is sufficient because each critic predicts
    one shared-return value per observation.
    """

    def __init__(
        self,
        *,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        variance_floor: float = 1e-2,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"ValueNorm beta must be in [0, 1), got {beta}")
        if epsilon <= 0.0:
            raise ValueError(f"ValueNorm epsilon must be positive, got {epsilon}")
        if variance_floor <= 0.0:
            raise ValueError(
                f"ValueNorm variance_floor must be positive, got {variance_floor}"
            )
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.variance_floor = float(variance_floor)
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_mean_sq", torch.zeros(()))
        self.register_buffer("debiasing_term", torch.zeros(()))

    @property
    def mean(self) -> "torch.Tensor":
        debias = torch.clamp(self.debiasing_term, min=self.epsilon)
        return self.running_mean / debias

    @property
    def variance(self) -> "torch.Tensor":
        debias = torch.clamp(self.debiasing_term, min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        # The reference MAPPO implementation floors variance at 1e-2.  This
        # avoids a huge critic target during the first nearly-constant batch.
        return torch.clamp(mean_sq - mean.square(), min=self.variance_floor)

    @property
    def std(self) -> "torch.Tensor":
        return torch.sqrt(self.variance)

    @torch.no_grad()
    def update(self, values: "torch.Tensor") -> None:
        values = values.detach()
        if values.numel() == 0:
            return
        values = values.to(device=self.running_mean.device,
                           dtype=self.running_mean.dtype)
        batch_mean = values.mean()
        batch_mean_sq = values.square().mean()
        one_minus_beta = 1.0 - self.beta
        self.running_mean.mul_(self.beta).add_(batch_mean, alpha=one_minus_beta)
        self.running_mean_sq.mul_(self.beta).add_(
            batch_mean_sq, alpha=one_minus_beta
        )
        self.debiasing_term.mul_(self.beta).add_(one_minus_beta)

    def normalize(self, values: "torch.Tensor") -> "torch.Tensor":
        return (values - self.mean) / self.std

    def denormalize(self, values: "torch.Tensor") -> "torch.Tensor":
        return values * self.std + self.mean


class CtdePPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """Gaussian actor over ``obs[own_slice]`` + critic over a selectable slice.

    Required ``model_config`` keys
    ------------------------------
    own_obs_start, own_obs_dim : int
        Slice of the flat observation the ACTOR reads.  Also the critic input
        when ``critic_input == "own"``.
    global_state_start, global_state_dim : int
        Slice the critic reads when ``critic_input == "global"``.  Must be 0/0
        when the environment does not expose a global state.
    critic_input : {"own", "global"}
    hidden_sizes : sequence of int, default (256, 256)
    activation : {"tanh", "relu"}, default "tanh"
    input_layer_norm : bool, default True
        Apply MAPPO-style feature LayerNorm independently to actor and critic
        inputs.
    value_norm : bool, default True
        Train the critic against Running ValueNorm targets while exposing
        denormalized values to GAE and evaluation.
    """

    @override(RLModule)
    def setup(self) -> None:
        cfg: Mapping[str, Any] = self.model_config or {}

        self._critic_input = str(cfg.get("critic_input", "own"))
        if self._critic_input not in CRITIC_INPUTS:
            raise ValueError(
                f"critic_input must be one of {CRITIC_INPUTS}, "
                f"got {self._critic_input!r}"
            )

        own_start = int(cfg["own_obs_start"])
        own_dim = int(cfg["own_obs_dim"])
        gs_start = int(cfg.get("global_state_start", 0))
        gs_dim = int(cfg.get("global_state_dim", 0))
        if own_dim <= 0:
            raise ValueError("own_obs_dim must be positive")
        if self._critic_input == "global" and gs_dim <= 0:
            raise ValueError(
                "critic_input='global' requires global_state_dim > 0; build the "
                "environment with include_global_state=True"
            )

        obs_dim = int(np.prod(self.observation_space.shape))
        for label, start, dim in (
            ("own", own_start, own_dim),
            ("global_state", gs_start, gs_dim),
        ):
            if dim and not (0 <= start and start + dim <= obs_dim):
                raise ValueError(
                    f"{label} slice [{start}, {start + dim}) is outside the "
                    f"{obs_dim}-dim observation space"
                )

        self._own_slice = slice(own_start, own_start + own_dim)
        self._gs_slice = slice(gs_start, gs_start + gs_dim)

        hidden = tuple(int(h) for h in cfg.get("hidden_sizes", (256, 256)))
        activation = str(cfg.get("activation", "tanh"))
        hidden_gain = float(cfg.get("hidden_orthogonal_gain", np.sqrt(2.0)))
        policy_output_gain = float(cfg.get("policy_output_gain", 0.01))
        value_output_gain = float(cfg.get("value_output_gain", 1.0))
        initial_log_std = float(cfg.get("initial_log_std", 0.0))
        self._log_std_min = float(cfg.get("log_std_min", -5.0))
        self._log_std_max = float(cfg.get("log_std_max", 2.0))
        input_layer_norm = bool(cfg.get("input_layer_norm", True))
        self._value_norm_enabled = bool(cfg.get("value_norm", True))

        action_dim = int(np.prod(self.action_space.shape))
        self._action_dim = action_dim

        # Actor: mean network + a STATE-INDEPENDENT log-std parameter, matching
        # the MAPPO reference implementation for continuous control.  RLlib's
        # TorchDiagGaussian.from_logits() expects [loc | log_std], so the two
        # are concatenated on the way out.
        self._pi = _mlp(
            own_dim, hidden, action_dim, activation, hidden_gain,
            policy_output_gain,
            input_layer_norm,
        )
        self._log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

        critic_in = own_dim if self._critic_input == "own" else gs_dim
        self._vf = _mlp(
            critic_in, hidden, 1, activation, hidden_gain, value_output_gain,
            input_layer_norm,
        )
        self._critic_in_dim = critic_in
        self._input_layer_norm = input_layer_norm
        self._value_normalizer = RunningValueNorm(
            beta=float(cfg.get("value_norm_beta", 0.99999)),
            epsilon=float(cfg.get("value_norm_epsilon", 1e-5)),
            variance_floor=float(cfg.get("value_norm_variance_floor", 1e-2)),
        )
        self._hidden_orthogonal_gain = hidden_gain
        self._policy_output_gain = policy_output_gain
        self._value_output_gain = value_output_gain
        self._initial_log_std = initial_log_std

    # ---------------- slicing helpers ----------------

    def _actor_features(self, obs: "torch.Tensor") -> "torch.Tensor":
        # The single line that guarantees decentralized execution: the actor
        # can only ever see its own observation slice.
        return obs[..., self._own_slice]

    def _critic_features(self, obs: "torch.Tensor") -> "torch.Tensor":
        if self._critic_input == "own":
            return obs[..., self._own_slice]
        return obs[..., self._gs_slice]

    def _action_dist_inputs(self, obs: "torch.Tensor") -> "torch.Tensor":
        mean = self._pi(self._actor_features(obs))
        log_std = torch.clamp(
            self._log_std, self._log_std_min, self._log_std_max
        )
        return torch.cat([mean, log_std.expand_as(mean)], dim=-1)

    # ---------------- RLModule forward passes ----------------

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {Columns.ACTION_DIST_INPUTS: self._action_dist_inputs(batch[Columns.OBS])}

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return {Columns.ACTION_DIST_INPUTS: self._action_dist_inputs(batch[Columns.OBS])}

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # PPO's Learner calls compute_values() separately (see ValueFunctionAPI),
        # mirroring DefaultPPOTorchRLModule, so no VF_PREDS here.
        return {Columns.ACTION_DIST_INPUTS: self._action_dist_inputs(batch[Columns.OBS])}

    # ---------------- ValueFunctionAPI ----------------

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> "torch.Tensor":
        # ``embeddings`` is ignored on purpose: actor and critic share no trunk,
        # so there is nothing to reuse. Keeping them fully separate is also what
        # makes the critic-input ablation clean -- no gradient path from the
        # critic loss can reach the actor.
        del embeddings
        values = self.compute_normalized_values(batch)
        return self.denormalize_values(values)

    def compute_normalized_values(self, batch: Dict[str, Any]) -> "torch.Tensor":
        """Return the critic network's native (normalized) output."""
        return self._vf(self._critic_features(batch[Columns.OBS])).squeeze(-1)

    @torch.no_grad()
    def update_value_normalizer(self, targets: "torch.Tensor") -> None:
        if self._value_norm_enabled:
            self._value_normalizer.update(targets)

    def normalize_value_targets(self, targets: "torch.Tensor") -> "torch.Tensor":
        if not self._value_norm_enabled:
            return targets
        return self._value_normalizer.normalize(targets)

    def denormalize_values(self, values: "torch.Tensor") -> "torch.Tensor":
        if not self._value_norm_enabled:
            return values
        return self._value_normalizer.denormalize(values)

    def actor_parameters(self) -> list:
        """Parameters owned by the actor optimizer (including log-std)."""
        return [*self._pi.parameters(), self._log_std]

    def critic_parameters(self) -> list:
        """Parameters owned by the critic optimizer."""
        return list(self._vf.parameters())

    @property
    def value_norm_mean(self) -> "torch.Tensor":
        return self._value_normalizer.mean

    @property
    def value_norm_std(self) -> "torch.Tensor":
        return self._value_normalizer.std

    # ---------------- action distributions ----------------
    # RLlib defaults to TorchDiagGaussian for Box action spaces, but we pin it
    # so a future default change cannot silently alter the policy class.

    def get_inference_action_dist_cls(self):
        from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

        return TorchDiagGaussian

    def get_exploration_action_dist_cls(self):
        return self.get_inference_action_dist_cls()

    def get_train_action_dist_cls(self):
        return self.get_inference_action_dist_cls()

    # ---------------- introspection (used by tests / run_config) ----------------

    @property
    def ctde_spec(self) -> Dict[str, Any]:
        return {
            "critic_input": self._critic_input,
            "own_slice": [self._own_slice.start, self._own_slice.stop],
            "global_state_slice": [self._gs_slice.start, self._gs_slice.stop],
            "critic_in_dim": int(self._critic_in_dim),
            "action_dim": int(self._action_dim),
            "hidden_orthogonal_gain": float(self._hidden_orthogonal_gain),
            "policy_output_gain": float(self._policy_output_gain),
            "value_output_gain": float(self._value_output_gain),
            "initial_log_std": float(self._initial_log_std),
            "log_std_range": [float(self._log_std_min), float(self._log_std_max)],
            "input_layer_norm": bool(self._input_layer_norm),
            "value_norm": bool(self._value_norm_enabled),
            "actor_params": int(
                sum(p.numel() for p in self.actor_parameters())
            ),
            "critic_params": int(sum(p.numel() for p in self.critic_parameters())),
        }


def build_ctde_module_spec(
    *,
    observation_space,
    action_space,
    own_slice: Tuple[int, int],
    global_state_slice: Tuple[int, int],
    critic_input: str,
    hidden_sizes: Sequence[int],
    activation: str,
    hidden_orthogonal_gain: float = float(np.sqrt(2.0)),
    policy_output_gain: float = 0.01,
    value_output_gain: float = 1.0,
    initial_log_std: float = 0.0,
    log_std_min: float = -5.0,
    log_std_max: float = 2.0,
    input_layer_norm: bool = True,
    value_norm: bool = True,
    value_norm_beta: float = 0.99999,
    value_norm_epsilon: float = 1e-5,
    value_norm_variance_floor: float = 1e-2,
):
    """Version-tolerant ``RLModuleSpec`` factory for the CTDE module."""
    try:
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    except ImportError:  # pragma: no cover - pre-2.40 naming
        from ray.rllib.core.rl_module.rl_module import (
            SingleAgentRLModuleSpec as RLModuleSpec,
        )

    model_config = {
        "own_obs_start": int(own_slice[0]),
        "own_obs_dim": int(own_slice[1] - own_slice[0]),
        "global_state_start": int(global_state_slice[0]),
        "global_state_dim": int(global_state_slice[1] - global_state_slice[0]),
        "critic_input": str(critic_input),
        "hidden_sizes": [int(h) for h in hidden_sizes],
        "activation": str(activation),
        "hidden_orthogonal_gain": float(hidden_orthogonal_gain),
        "policy_output_gain": float(policy_output_gain),
        "value_output_gain": float(value_output_gain),
        "initial_log_std": float(initial_log_std),
        "log_std_min": float(log_std_min),
        "log_std_max": float(log_std_max),
        "input_layer_norm": bool(input_layer_norm),
        "value_norm": bool(value_norm),
        "value_norm_beta": float(value_norm_beta),
        "value_norm_epsilon": float(value_norm_epsilon),
        "value_norm_variance_floor": float(value_norm_variance_floor),
    }
    kwargs = dict(
        module_class=CtdePPOTorchRLModule,
        observation_space=observation_space,
        action_space=action_space,
    )
    try:
        return RLModuleSpec(**kwargs, model_config=model_config)
    except TypeError:  # pragma: no cover - older kwarg name
        return RLModuleSpec(**kwargs, model_config_dict=model_config)


def build_multi_module_spec(specs: Mapping[str, Any]):
    """Version-tolerant ``MultiRLModuleSpec`` factory."""
    try:
        from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    except ImportError:  # pragma: no cover
        from ray.rllib.core.rl_module.marl_module import (
            MultiAgentRLModuleSpec as MultiRLModuleSpec,
        )
    for kwarg in ("rl_module_specs", "module_specs"):
        try:
            return MultiRLModuleSpec(**{kwarg: dict(specs)})
        except TypeError:
            continue
    raise RuntimeError(
        "could not construct MultiRLModuleSpec; the installed Ray version "
        "expects an unrecognized keyword"
    )
