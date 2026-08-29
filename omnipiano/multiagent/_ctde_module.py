# ===== omnipiano/multiagent/_ctde_module.py（新建）=====
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

# Standard PPO initialization (Engstrom et al., 2020): sqrt(2) orthogonal on
# hidden layers, a *small* gain on the policy output so the initial action
# distribution is near-isotropic and exploration is not collapsed at step 0,
# and gain 1.0 on the value head.
_HIDDEN_GAIN = float(np.sqrt(2.0))
_PI_OUT_GAIN = 0.01
_VF_OUT_GAIN = 1.0

# Keeps the Gaussian from collapsing to a delta or exploding.
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0


def _mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str,
    out_gain: float,
) -> "nn.Sequential":
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers: list = []
    prev = int(in_dim)
    for hidden in hidden_dims:
        linear = nn.Linear(prev, int(hidden))
        nn.init.orthogonal_(linear.weight, gain=_HIDDEN_GAIN)
        nn.init.zeros_(linear.bias)
        layers += [linear, act_cls()]
        prev = int(hidden)
    out = nn.Linear(prev, int(out_dim))
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


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

        action_dim = int(np.prod(self.action_space.shape))
        self._action_dim = action_dim

        # Actor: mean network + a STATE-INDEPENDENT log-std parameter, matching
        # the MAPPO reference implementation for continuous control.  RLlib's
        # TorchDiagGaussian.from_logits() expects [loc | log_std], so the two
        # are concatenated on the way out.
        self._pi = _mlp(own_dim, hidden, action_dim, activation, _PI_OUT_GAIN)
        self._log_std = nn.Parameter(torch.zeros(action_dim))

        critic_in = own_dim if self._critic_input == "own" else gs_dim
        self._vf = _mlp(critic_in, hidden, 1, activation, _VF_OUT_GAIN)
        self._critic_in_dim = critic_in

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
        log_std = torch.clamp(self._log_std, _LOG_STD_MIN, _LOG_STD_MAX)
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
        return self._vf(self._critic_features(batch[Columns.OBS])).squeeze(-1)

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
            "actor_params": int(
                sum(p.numel() for p in self._pi.parameters()) + self._log_std.numel()
            ),
            "critic_params": int(sum(p.numel() for p in self._vf.parameters())),
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