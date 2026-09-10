"""Declarative algorithm registry for the OmniPiano multi-agent baselines.

Adding a baseline must never require editing the trainer.  An algorithm is
therefore a *data* object that declares:

  * which slice of the flat observation its critic reads, and what the critic
    emits (a state value, a joint Q, a factored joint Q),
  * whether the environment must emit the centralized state,
  * which optimisation loop it needs (the PPO family shares one; the off-policy
    family does not),
  * which RLModule / Learner implementation to use (dotted paths, resolved
    lazily so ``--list-algos`` stays a sub-second, import-free operation),
  * what an agent needs AT EXECUTION TIME -- the field every MARL paper gets
    wrong,
  * and how finished it is, so an unimplemented algorithm can be documented and
    cited long before it can be launched.

``train.py`` consumes an ``AlgoSpec`` and knows nothing about IPPO or MAPPO, so
every registered algorithm shares bit-identical bookkeeping, checkpointing,
evaluation, protocol recording and W&B logging.  A cross-algorithm comparison
can then only differ where the AlgoSpec says it differs.

To add an algorithm: create ``algos/<name>.py``, build an ``AlgoSpec``, call
``register_algo(...)``, and import it from ``algos/__init__.py``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

# --- Vocabulary -------------------------------------------------------------
# What the critic READS.
CRITIC_INPUTS = ("own", "global", "global_joint_action")
# What the critic EMITS.  A Q head cannot be trained by our PPO loop.
CRITIC_HEADS = ("v", "q", "q_mixed")
# Which network topology train.py must build.
RL_MODULES = ("ctde", "rllib_default", "joint_autoregressive", "joint_actor_critic")
# Which optimisation loop the algorithm needs.  Everything in the
# ``on_policy_ppo*`` families is served by the existing PPOConfig trainer.
FAMILIES = (
    "on_policy_ppo",             # simultaneous update of every agent (IPPO/MAPPO)
    "on_policy_ppo_sequential",  # agents updated one at a time (HAPPO/MAT)
    "off_policy_ddpg",           # replay + target nets + deterministic policy
    "off_policy_sac",            # replay + target nets + squashed Gaussian
)
# What an agent needs AT EXECUTION TIME.  Anything other than "decentralized"
# is a DIFFERENT DEPLOYMENT CLASS: it must not be presented as a drop-in
# comparison against IPPO/MAPPO without saying so out loud.
EXECUTIONS = ("decentralized", "sequential_autoregressive", "centralized")
# Honest gate.  "planned" specs are printed, cited and documented, but refused
# by train.py -- so nobody burns three GPU-days on a stub.
STATUSES = ("supported", "experimental", "planned")

# Families whose optimisation loop the current PPO-based trainer can run.
_PPO_TRAINABLE_FAMILIES = ("on_policy_ppo", "on_policy_ppo_sequential")


def resolve_dotted(path: str) -> Any:
    """Import ``pkg.mod:Attr`` or ``pkg.mod.Attr`` lazily.

    Deliberately NOT called at registration time: importing a Learner drags in
    torch and ray, which would make ``--list-algos`` cost seconds and would make
    a broken optional dependency break the whole registry.
    """
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        mod_name, _, attr = path.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"not a dotted path: {path!r}")
    return getattr(importlib.import_module(mod_name), attr)


@dataclass(frozen=True)
class AlgoSpec:
    """Everything that distinguishes one cooperative MARL baseline from another."""

    name: str                     # CLI token, run-dir prefix, W&B group prefix
    display_name: str             # human-readable, written into artifacts
    critic_input: str             # "own" -> V(o_i) ; "global" -> V(s)
    reference: str                # citation, written into run_config.json
    needs_global_state: bool = False
    rl_module: str = "ctde"

    # --- fields added for the HAPPO / MAT / FACMAC / MASAC expansion ---------
    # Defaults are chosen so that ippo.py and mappo.py need NO edits.
    family: str = "on_policy_ppo"
    critic_head: str = "v"
    execution: str = "decentralized"
    status: str = "supported"
    # Dotted paths, resolved by train.py only when this algo is actually run.
    learner_class: Optional[str] = None
    rl_module_class: Optional[str] = None
    # Forces the env's agent count (ppo-monolithic := one agent owning all hands).
    num_agents_override: Optional[int] = None
    # PPO kwargs this algorithm forces regardless of CLI. Empty for IPPO/MAPPO:
    # the whole point is that they share identical hyperparameters.
    training_overrides: Mapping[str, Any] = field(default_factory=dict)
    # For status != "supported": what exactly is missing. Printed on refusal.
    blocking: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for value, allowed, label in (
            (self.critic_input, CRITIC_INPUTS, "critic_input"),
            (self.critic_head, CRITIC_HEADS, "critic_head"),
            (self.rl_module, RL_MODULES, "rl_module"),
            (self.family, FAMILIES, "family"),
            (self.execution, EXECUTIONS, "execution"),
            (self.status, STATUSES, "status"),
        ):
            if value not in allowed:
                raise ValueError(f"{self.name}: {label} must be in {allowed}, got {value!r}")

        if self.critic_input in ("global", "global_joint_action") and not self.needs_global_state:
            raise ValueError(
                f"{self.name}: a centralized critic requires needs_global_state=True")
        if self.rl_module == "rllib_default" and self.critic_input != "own":
            raise ValueError(
                f"{self.name}: RLlib's built-in PPO module has no centralized critic")
        if self.rl_module == "rllib_default" and self.needs_global_state:
            raise ValueError(
                f"{self.name}: RLlib's built-in module would feed the global state "
                f"to the ACTOR too, breaking decentralized execution")

        # A Q head implies bootstrapping off stored transitions, which the PPO
        # loop cannot do. Catch the mismatch at import time, not at hour six.
        if self.critic_head in ("q", "q_mixed") and self.family in _PPO_TRAINABLE_FAMILIES:
            raise ValueError(
                f"{self.name}: critic_head={self.critic_head!r} needs an off-policy "
                f"family; the PPO trainer has no replay buffer or target network")
        if self.family.startswith("off_policy") and self.critic_head == "v":
            raise ValueError(
                f"{self.name}: an off-policy actor-critic needs a Q critic "
                f"(critic_head='q' or 'q_mixed'), not a state-value head")

        # A custom update rule or a custom topology must SAY which class implements
        # it, otherwise train.py would silently fall back to plain PPO and produce
        # numbers labelled HAPPO that are really MAPPO. This is the single most
        # dangerous failure mode in a benchmark paper.
        if self.family == "on_policy_ppo_sequential" and not self.learner_class:
            raise ValueError(
                f"{self.name}: family='on_policy_ppo_sequential' requires "
                f"learner_class, else the sequential update is silently dropped")
        if self.rl_module in ("joint_autoregressive", "joint_actor_critic") \
                and not self.rl_module_class:
            raise ValueError(f"{self.name}: rl_module={self.rl_module!r} requires rl_module_class")

        # Non-decentralized execution is a publication-level caveat. Force it to
        # be written down where it will be copied into run_config.json.
        if self.execution != "decentralized" and not self.notes:
            raise ValueError(
                f"{self.name}: execution={self.execution!r} is not a drop-in "
                f"replacement for IPPO/MAPPO; document it in notes=")
        if self.status != "supported" and not self.blocking:
            raise ValueError(f"{self.name}: status={self.status!r} requires blocking=")
        if self.num_agents_override is not None and self.num_agents_override < 1:
            raise ValueError(f"{self.name}: num_agents_override must be >= 1")

    # --- derived ------------------------------------------------------------
    @property
    def is_centralized_critic(self) -> bool:
        return self.critic_input in ("global", "global_joint_action")

    @property
    def is_on_policy(self) -> bool:
        return self.family in _PPO_TRAINABLE_FAMILIES

    @property
    def is_decentralized_execution(self) -> bool:
        return self.execution == "decentralized"

    def resolve_learner_class(self) -> Optional[Any]:
        return resolve_dotted(self.learner_class) if self.learner_class else None

    def resolve_rl_module_class(self) -> Optional[Any]:
        return resolve_dotted(self.rl_module_class) if self.rl_module_class else None

    def assert_launchable(self, allow_experimental: bool = False) -> None:
        """Raise a message a human can act on, instead of training garbage."""
        if self.status == "planned":
            raise ValueError(
                f"--algo {self.name!r} is DECLARED but NOT IMPLEMENTED.\n"
                f"  Blocking work: {self.blocking}\n"
                f"  Runnable now:  {list_algos(status='supported')}")
        if self.status == "experimental" and not allow_experimental:
            raise ValueError(
                f"--algo {self.name!r} is EXPERIMENTAL and unvalidated.\n"
                f"  Caveat: {self.blocking}\n"
                f"  Pass --allow-experimental to run it anyway (and do NOT put "
                f"the numbers in a paper before the validation run below passes).")
        if not self.is_on_policy:
            raise ValueError(
                f"--algo {self.name!r} is family={self.family!r}; the current "
                f"trainer builds a PPOConfig and cannot run it.\n"
                f"  Blocking work: {self.blocking}")

    def metadata(self) -> Dict[str, Any]:
        """Copied verbatim into run_config.json so every artifact is self-describing."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "family": self.family,
            "critic_input": self.critic_input,
            "critic_head": self.critic_head,
            "execution": self.execution,
            "rl_module": self.rl_module,
            "needs_global_state": self.needs_global_state,
            "status": self.status,
            "learner_class": self.learner_class,
            "rl_module_class": self.rl_module_class,
            "num_agents_override": self.num_agents_override,
            "training_overrides": dict(self.training_overrides),
            "reference": self.reference,
            "blocking": self.blocking,
            "notes": self.notes,
        }


_REGISTRY: Dict[str, AlgoSpec] = {}


def register_algo(spec: AlgoSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"algorithm {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def get_algo(name: str) -> AlgoSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown --algo {name!r}. Registered: {list_algos()}") from None


def list_algos(status: Optional[str] = None) -> Tuple[str, ...]:
    if status is None:
        return tuple(sorted(_REGISTRY))
    if status not in STATUSES:
        raise ValueError(f"status must be in {STATUSES}")
    return tuple(sorted(n for n, s in _REGISTRY.items() if s.status == status))


def algo_table() -> str:
    """Human-readable table for ``--list-algos``."""
    hdr = (f"{'algo':20s} {'status':12s} {'family':26s} {'critic':22s} "
           f"{'execution':26s} description")
    rows = [hdr, "-" * len(hdr)]
    for name in list_algos():
        s = _REGISTRY[name]
        critic = f"{s.critic_input}->{s.critic_head}"
        rows.append(f"{s.name:20s} {s.status:12s} {s.family:26s} {critic:22s} "
                    f"{s.execution:26s} {s.display_name}")
    rows.append("")
    rows.append(f"launchable now: {list(list_algos(status='supported'))}")
    rows.append(f"needs --allow-experimental: {list(list_algos(status='experimental'))}")
    rows.append(f"declared but NOT implemented: {list(list_algos(status='planned'))}")
    return "\n".join(rows)