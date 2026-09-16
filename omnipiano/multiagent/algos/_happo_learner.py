"""RLlib Learner adding HAPPO's compound-ratio advantage reweighting.

Why this subclasses OmniPianoPPOTorchLearner and not PPOTorchLearner
--------------------------------------------------------------------
Our CTDE learner owns separate actor/critic Adam optimizers, critic_lr,
adam_epsilon, value normalisation and the official prediction-delta value
clipping. HAPPO must inherit ALL of it, because the entire point of the
experiment is that HAPPO and MAPPO differ in ONE thing -- the actor's surrogate.
Subclassing RLlib's plain PPO learner would make them differ in six.

Why this does NOT override _update
----------------------------------
Strict HAPPO steps agent i_m's optimiser, re-forwards it, and only then folds
its realised change into the compound factor M used by agent i_{m+1}, all inside
one minibatch. In this Ray version ``TorchLearner._update`` is a dispatcher into
``_possibly_compiled_update``, which owns gradient post-processing, grad_clip_by
semantics, LR scheduling, metrics bookkeeping and the torch.compile path.
Replacing it would silently drop all of that.

So we reweight the advantage in ``compute_loss_for_module`` and touch no
optimiser mechanics. M for agent i_m is the product of the ratios of the agents
BEFORE it in this permutation, evaluated with their parameters as of the START of
the current minibatch -- i.e. reflecting every previous epoch's update, but not
this minibatch's step. Consequences, recorded in run_config.json:

  * epoch 1 has M == 1 EXACTLY, so it is bit-identical to MAPPO;
  * from epoch 2 on, M carries the accumulated drift of the earlier agents in
    the permutation, which is the mechanism HAPPO's surrogate relies on;
  * the one-epoch staleness means the monotonic-improvement theorem of Kuba et
    al. does not transfer verbatim. This is an approximation of HAPPO.

Reference: Kuba et al. (2022), ICLR (arXiv:2109.11251), Sec. 4 / Eq. (10).
"""

from __future__ import annotations

import inspect
import random
from typing import Any, Dict, Mapping

import torch
from ray.rllib.core.columns import Columns

from omnipiano.multiagent.algos.ppo_learner import OmniPianoPPOTorchLearner
from omnipiano.multiagent.algos._happo_math import (
    DEFAULT_MAX_ABS_LOG_FACTOR,
    clamp_log_factor,
    happo_surrogate,
    ppo_surrogate,
)

try:  # RLlib keeps moving this; the VALUE has been "advantages" throughout.
    from ray.rllib.evaluation.postprocessing import Postprocessing
    ADVANTAGES = Postprocessing.ADVANTAGES
except Exception:  # pragma: no cover
    ADVANTAGES = "advantages"


def _check_hooks() -> None:
    """Fail at import with an actionable message, never degrade into MAPPO."""
    for name in ("compute_losses", "compute_loss_for_module"):
        if not hasattr(OmniPianoPPOTorchLearner, name):
            raise RuntimeError(
                f"HAPPOTorchLearner requires the {name!r} hook, absent here. "
                f"Inspect the current flow with:\n"
                f"  python -c \"import inspect, "
                f"ray.rllib.core.learner.torch.torch_learner as m; "
                f"print(inspect.getsource(m.TorchLearner.compute_losses))\"")
    params = inspect.signature(OmniPianoPPOTorchLearner.compute_losses).parameters
    missing = [p for p in ("fwd_out", "batch") if p not in params]
    if missing:
        raise RuntimeError(
            f"compute_losses is missing keyword parameters {missing}; HAPPO reads "
            f"the whole multi-agent batch through it. Signature seen: "
            f"{inspect.signature(OmniPianoPPOTorchLearner.compute_losses)}")


_check_hooks()


class HAPPOTorchLearner(OmniPianoPPOTorchLearner):
    """MAPPO's learner plus HAPPO's compound-ratio advantage reweighting."""

    # Flipped off by HAPPOCompoundDisabledLearner, which MUST reproduce mappo.
    COMPOUND_FACTOR_ENABLED = True
    # M is a PRODUCT over agents, so it explodes geometrically in n. Clamp in log
    # space and log the hit rate: a high rate means the trust region is being
    # violated and HAPPO's improvement argument no longer applies.
    MAX_ABS_LOG_FACTOR = DEFAULT_MAX_ABS_LOG_FACTOR

    def build(self) -> None:
        super().build()
        # The permutation is part of the ALGORITHM, so it must be part of the
        # seed: a rerun has to reproduce the same sequence of orderings.
        self._happo_rng = random.Random(
            (int(getattr(self.config, "seed", 0) or 0)) * 7919 + 104729)
        self._happo_log_factor: Dict[str, torch.Tensor] = {}
        self._happo_clamped = 0
        self._happo_seen = 0

    # -- step 1: build M for every module, before any loss is computed -------
    def compute_losses(self, *, fwd_out, batch, **kwargs):  # noqa: ANN001,ANN201
        self._happo_log_factor = self._build_compound_factors(fwd_out, batch)
        return super().compute_losses(fwd_out=fwd_out, batch=batch, **kwargs)

    def _logp_of_stored_actions(
        self, module_id: str, module_batch: Mapping[str, Any],
        module_fwd: Mapping[str, Any],
    ) -> torch.Tensor:
        """log pi_theta(a_stored | o) from an ALREADY COMPUTED forward pass."""
        module = self.module[module_id]
        try:
            dist = module.get_train_action_dist_cls().from_logits(
                module_fwd[Columns.ACTION_DIST_INPUTS])
            return dist.logp(module_batch[Columns.ACTIONS])
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                f"HAPPO cannot recover log pi(a|o) for module {module_id!r}: the "
                f"RLModule must expose get_train_action_dist_cls() and emit "
                f"Columns.ACTION_DIST_INPUTS from forward_train(). Keys seen: "
                f"{sorted(module_fwd)}") from exc

    def _build_compound_factors(self, fwd_out, batch) -> Dict[str, torch.Tensor]:
        ids = [m for m in fwd_out if m in self.module and m in batch]
        # One agent => M == 1 identically => HAPPO IS PPO. Single-agent debugging
        # and any future monolithic control stay bit-identical to PPO.
        if len(ids) < 2 or not self.COMPOUND_FACTOR_ENABLED:
            return {}

        # M pairs agent A's row t with agent B's row t, so the per-agent
        # minibatches must be row-aligned in time. True for OmniPiano's duet
        # (fixed-length episodes, every hand commanded every control step) and
        # guaranteed by shuffle_batch_per_epoch=False. Check it, loudly.
        sizes = {m: int(batch[m][Columns.ACTIONS].shape[0]) for m in ids}
        if len(set(sizes.values())) != 1:
            raise RuntimeError(
                f"HAPPO requires row-aligned per-agent minibatches, got {sizes}. "
                f"Confirm shuffle_batch_per_epoch=False and that every agent acts "
                f"at every control step.")

        log_ratio: Dict[str, torch.Tensor] = {}
        for m in ids:
            with torch.no_grad():
                logp = self._logp_of_stored_actions(m, batch[m], fwd_out[m])
            # DETACHED: M measures how much the world already moved; it is not a
            # quantity we optimise. A gradient through it would make agent i_m
            # responsible for agent i_1's parameters -- exactly the cross-agent
            # coupling the sequential scheme exists to remove.
            log_ratio[m] = (logp - batch[m][Columns.ACTION_LOGP]).detach()

        order = list(ids)
        self._happo_rng.shuffle(order)

        ref = log_ratio[order[0]]
        cum = torch.zeros_like(ref)
        out: Dict[str, torch.Tensor] = {}
        for m in order:
            out[m] = cum                      # M for i_m excludes i_m itself
            cum = clamp_log_factor(cum + log_ratio[m], self.MAX_ABS_LOG_FACTOR)
            self._happo_clamped += int(
                (cum.abs() >= self.MAX_ABS_LOG_FACTOR - 1e-9).sum().item())
            self._happo_seen += cum.numel()

        if self._happo_seen:
            self.metrics.log_value(
                key=("happo", "compound_factor_clamp_rate"),
                value=self._happo_clamped / self._happo_seen, window=1)
        self.metrics.log_value(
            key=("happo", "compound_factor_abs_log_mean"),
            value=float(torch.stack(list(out.values())).abs().mean().item()),
            window=1)
        # Proof-of-life in progress.jsonl: if this key is absent, the custom
        # learner never ran and you are looking at MAPPO numbers.
        self.metrics.log_value(key=("happo", "num_agents_in_permutation"),
                               value=float(len(order)), window=1)
        return out

    # -- step 2: correct ONLY the surrogate ---------------------------------
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):  # noqa: ANN001,ANN201
        # Everything non-HAPPO (value loss, entropy, KL, clipping semantics,
        # value normalisation) comes from our MAPPO learner VERBATIM, so those
        # code paths are bit-identical to the runs we compare against.
        base = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out)
        log_factor = self._happo_log_factor.get(module_id)
        if log_factor is None:
            return base

        logp_new = self._logp_of_stored_actions(module_id, batch, fwd_out)
        logp_old = batch[Columns.ACTION_LOGP].detach()
        adv = batch[ADVANTAGES].detach()
        clip = float(config.clip_param)
        # loss = -surrogate + ... , so ADD the negated surrogate difference.
        # For the first agent in the permutation log_factor == 0 exactly, so
        # delta == 0 and that agent is bit-identical to MAPPO -- which is what
        # HAPPO prescribes.
        delta = (happo_surrogate(logp_new, logp_old, adv, log_factor, clip).mean()
                 - ppo_surrogate(logp_new, logp_old, adv, clip).mean())
        return base - delta


class HAPPOCompoundDisabledLearner(HAPPOTorchLearner):
    """HAPPO with M forced to 1: MUST reproduce mappo (with shuffle disabled).

    This is the only decisive test that the surrogate-delta above did not corrupt
    the loss. It runs every line of the HAPPO code path -- the permutation, the
    per-module bookkeeping, the extra forward -- while the mathematics collapses
    to MAPPO's. If its learning curve diverges from mappo-noshuffle, the delta is
    wrong and no HAPPO number may be quoted.
    """

    COMPOUND_FACTOR_ENABLED = False
