"""Declarative algorithm registry for the OmniPiano multi-agent baselines.

Adding a baseline must never require editing the trainer.  An algorithm is
therefore a *data* object that declares:

  * which slice of the flat observation its critic reads,
  * whether the environment must emit the centralized state,
  * which RLModule implementation to use,
  * any PPO-hyperparameter overrides specific to it.

``train.py`` consumes an ``AlgoSpec`` and knows nothing about IPPO or MAPPO, so
every registered algorithm shares bit-identical bookkeeping, checkpointing,
evaluation, protocol recording and W&B logging.  A cross-algorithm comparison
can then only differ where the AlgoSpec says it differs.

To add an algorithm: create ``algos/<name>.py``, build an ``AlgoSpec``, call
``register_algo(...)``, and import it from ``algos/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

CRITIC_INPUTS = ("own", "global")
RL_MODULES = ("ctde", "rllib_default")


@dataclass(frozen=True)
class AlgoSpec:
    """Everything that distinguishes one on-policy MARL baseline from another."""

    name: str                     # CLI token, run-dir prefix, W&B group prefix
    display_name: str             # human-readable, written into artifacts
    critic_input: str             # "own" -> V(o_i) ; "global" -> V(s)
    reference: str                # citation, written into run_config.json
    needs_global_state: bool = False
    rl_module: str = "ctde"
    # PPO kwargs this algorithm forces regardless of CLI. Empty for IPPO/MAPPO:
    # the whole point is that they share identical hyperparameters.
    training_overrides: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.critic_input not in CRITIC_INPUTS:
            raise ValueError(f"{self.name}: critic_input must be in {CRITIC_INPUTS}")
        if self.rl_module not in RL_MODULES:
            raise ValueError(f"{self.name}: rl_module must be in {RL_MODULES}")
        if self.critic_input == "global" and not self.needs_global_state:
            raise ValueError(
                f"{self.name}: a centralized critic requires needs_global_state=True"
            )
        if self.rl_module == "rllib_default" and self.critic_input != "own":
            raise ValueError(
                f"{self.name}: RLlib's built-in PPO module has no centralized critic"
            )
        if self.rl_module == "rllib_default" and self.needs_global_state:
            raise ValueError(
                f"{self.name}: RLlib's built-in module would feed the global state "
                f"to the ACTOR too, breaking decentralized execution"
            )

    @property
    def is_centralized_critic(self) -> bool:
        return self.critic_input == "global"


_REGISTRY: Dict[str, AlgoSpec] = {}


def register_algo(spec: AlgoSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"algorithm {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def get_algo(name: str) -> AlgoSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown --algo {name!r}. Registered: {list_algos()}"
        ) from None


def list_algos() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def algo_table() -> str:
    """Human-readable table for ``--list-algos``."""
    rows = [f"{'algo':22s} {'critic':8s} {'module':14s} description"]
    rows.append("-" * 92)
    for name in list_algos():
        s = _REGISTRY[name]
        rows.append(f"{s.name:22s} {s.critic_input:8s} {s.rl_module:14s} {s.display_name}")
    return "\n".join(rows)