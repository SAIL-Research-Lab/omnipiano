"""RLlib Learner implementing HAPPO's sequential update scheme.

Why this needs a custom Learner and not just a custom loss
----------------------------------------------------------
HAPPO's compound factor M for agent i_m is built from the ratios of agents
i_1..i_{m-1} *after those agents' parameters have already been stepped*. A loss
function alone cannot express that: RLlib's default flow computes every module's
loss from one forward pass and then applies all gradients at once. So the
per-minibatch update itself has to be re-ordered, which is ``_update``.

Assumption this implementation makes (asserted at runtime)
---------------------------------------------------------
All agents act at every control step, so the per-module batches are row-aligned
in time: row t of agent A and row t of agent B are the same joint transition.
This is TRUE for OmniPiano's territorial duet (fixed-length episodes, all hands
commanded every step) and is checked below. A turn-based environment would need
a joint buffer instead.

Because IPPO/MAPPO/HAPPO give every agent its own module with NO shared
parameters, stepping agent i_m's optimiser cannot perturb agent i_k's forward
pass. That is what makes the cheap "re-forward only the agent we just updated"
trick exact rather than an approximation.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from omnipiano.multiagent.algos._happo_math import (
    DEFAULT_MAX_ABS_LOG_FACTOR,
    compound_log_factor_update,
    happo_surrogate,
    ppo_surrogate,
)

# Hooks this Learner relies on. If a future Ray renames one, fail at import with
# a message that says what to do -- never silently degrade into MAPPO.
_REQUIRED_HOOKS = ("_update", "compute_loss_for_module", "get_optimizers_for_module")


def _check_rllib_compat() -> None:
    missing = [h for h in _REQUIRED_HOOKS if not hasattr(PPOTorchLearner, h)]
    if missing:
        raise RuntimeError(
            f"HAPPOTorchLearner needs these RLlib hooks, absent in this Ray "
            f"version: {missing}. Inspect the current flow with:\n"
            f"  python -c \"import inspect, "
            f"ray.rllib.core.learner.torch.torch_learner as m; "
            f"print(inspect.getsource(m.TorchLearner._update))\"")


_check_rllib_compat()


class HAPPOTorchLearner(PPOTorchLearner):
    """Sequential, compound-ratio PPO (Kuba et al., ICLR 2022)."""

    # Set from AlgoSpec.training_overrides via the config; see happo.py.
    MAX_ABS_LOG_FACTOR = DEFAULT_MAX_ABS_LOG_FACTOR
    # Tolerance for the M==1 self-check on the very first minibatch.
    SELFCHECK_TOL = 1e-4

    def build(self) -> None:
        super().build()
        # Seeded so a rerun reproduces the permutation sequence exactly. The
        # permutation is part of the algorithm, so it must be part of the seed.
        self._happo_rng = random.Random(
            (getattr(self.config, "seed", None) or 0) * 7919 + 104729)
        self._happo_selfchecked = False
        self._happo_clamp_hits = 0
        self._happo_updates = 0

    # -- helpers -----------------------------------------------------------
    def _logp_of_stored_actions(self, module_id: str, mb: Dict[str, Any]) -> torch.Tensor:
        """log pi_theta(a_stored | o) under the module's CURRENT parameters."""
        module = self.module[module_id]
        fwd = module.forward_train(mb)
        dist = module.get_train_action_dist_cls().from_logits(
            fwd[Columns.ACTION_DIST_INPUTS])
        return dist.logp(mb[Columns.ACTIONS]), fwd

    def _params_of(self, module_id: str) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for _name, optim in self.get_optimizers_for_module(module_id=module_id):
            for group in optim.param_groups:
                params.extend(group["params"])
        return params

    # -- the sequential update --------------------------------------------
    def _update(self, batch: Dict[str, Any], **kwargs):  # noqa: ANN201
        module_ids = [mid for mid in batch.keys() if mid in self.module]
        if len(module_ids) <= 1:
            # One agent: M == 1 identically, HAPPO IS PPO. Take the fast path so
            # ppo-monolithic and single-agent debugging are bit-identical to PPO.
            return super()._update(batch, **kwargs)

        # Row alignment is what makes M meaningful. Check it, loudly.
        sizes = {mid: len(batch[mid][Columns.ACTIONS]) for mid in module_ids}
        if len(set(sizes.values())) != 1:
            raise RuntimeError(
                f"HAPPO requires row-aligned per-agent minibatches, got {sizes}. "
                f"Set shuffle_batch_per_epoch=False and make sure every agent "
                f"acts at every control step.")

        order = list(module_ids)
        self._happo_rng.shuffle(order)

        n = next(iter(sizes.values()))
        dev = batch[order[0]][Columns.ACTIONS].device
        log_factor = torch.zeros(n, device=dev)

        loss_per_module: Dict[str, torch.Tensor] = {}
        fwd_out_all: Dict[str, Any] = {}

        for module_id in order:
            mb = batch[module_id]
            logp_new, fwd = self._logp_of_stored_actions(module_id, mb)
            fwd_out_all[module_id] = fwd
            logp_old = mb[Columns.ACTION_LOGP].detach()
            adv = mb[Postprocessing.ADVANTAGES].detach()

            # Reuse RLlib's PPO loss verbatim for the value-function and entropy
            # terms, then correct ONLY the surrogate. This guarantees that every
            # non-HAPPO-specific detail (vf clipping, entropy coeff, KL term,
            # masking) is bit-identical to the MAPPO runs we compare against.
            base_loss = super().compute_loss_for_module(
                module_id=module_id, config=self.config.get_config_for_module(module_id),
                batch=mb, fwd_out=fwd)

            surr_ppo = ppo_surrogate(logp_new, logp_old, adv,
                                     self.config.clip_param).mean()
            surr_happo = happo_surrogate(logp_new, logp_old, adv, log_factor,
                                         self.config.clip_param).mean()
            loss = base_loss - (surr_happo - surr_ppo)  # loss = -surrogate + ...

            # One-off proof that our surrogate reproduces RLlib's when M == 1.
            # If this fires, the delta trick above is invalid for this module and
            # the HAPPO numbers would be quietly wrong.
            if not self._happo_selfchecked:
                with torch.no_grad():
                    zero = torch.zeros_like(log_factor)
                    delta = (happo_surrogate(logp_new, logp_old, adv, zero,
                                             self.config.clip_param).mean()
                             - surr_ppo).abs().item()
                if delta > self.SELFCHECK_TOL:
                    raise RuntimeError(
                        f"HAPPO self-check failed on {module_id}: |M=1 delta| "
                        f"= {delta:.3e} > {self.SELFCHECK_TOL}. Our surrogate "
                        f"disagrees with PPO's at M=1; do not trust these runs.")
                self._happo_selfchecked = True

            # Step ONLY this agent's optimiser.
            optims = list(self.get_optimizers_for_module(module_id=module_id))
            for _name, optim in optims:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = getattr(self.config, "grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self._params_of(module_id), grad_clip)
            for _name, optim in optims:
                optim.step()

            # Re-forward THIS agent only, with its new parameters, to fold its
            # actual (not predicted) change into M for the next agent.
            with torch.no_grad():
                logp_after, _ = self._logp_of_stored_actions(module_id, mb)
            new_log_factor = compound_log_factor_update(
                log_factor, logp_after, logp_old,
                max_abs_log_factor=self.MAX_ABS_LOG_FACTOR)
            self._happo_clamp_hits += int(
                (new_log_factor.abs() >= self.MAX_ABS_LOG_FACTOR - 1e-9).sum().item())
            self._happo_updates += n
            log_factor = new_log_factor

            loss_per_module[module_id] = loss.detach()

        # Monitorable in W&B: a high clamp rate means the trust region is being
        # violated and HAPPO's monotonic-improvement argument no longer applies.
        if self._happo_updates:
            self.metrics.log_value(
                key=("happo", "compound_factor_clamp_rate"),
                value=self._happo_clamp_hits / self._happo_updates,
                window=1)
        self.metrics.log_value(key=("happo", "update_order"),
                               value=float(hash(tuple(order)) % 1000), window=1)

        # Ray's TorchLearner._update returns (fwd_out, loss_per_module, metrics).
        # Verify for your Ray with the command in _check_rllib_compat().
        return fwd_out_all, loss_per_module, {}