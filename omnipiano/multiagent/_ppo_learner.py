"""MAPPO-style PPO learner details for OmniPiano's CTDE modules.

RLlib 2.55.1's stock PPO learner clips the *squared value error* to a ceiling.
That makes samples above the ceiling contribute zero critic gradient.  MAPPO
instead clips the change in the value prediction relative to the rollout-time
prediction, then takes the larger of the clipped and unclipped losses.  This
module supplies that semantics together with running ValueNorm and independent
actor/critic Adam optimizers.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
    PPOConfig,
)
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.typing import EpisodeType, ModuleID, TensorType

from omnipiano.multiagent._ctde_module import CtdePPOTorchRLModule

torch, _ = try_import_torch()

# The stock GAE connector intentionally drops rollout-time value predictions.
# Correct PPO value clipping needs them throughout every optimization epoch.
OLD_NORMALIZED_VALUE_PREDS = "_omnipiano_old_normalized_value_preds"

VALUE_NORM_MEAN_KEY = "value_norm_mean"
VALUE_NORM_STD_KEY = "value_norm_std"
VALUE_CLIP_FRACTION_KEY = "value_clip_fraction"


class GAEWithOldValuePredictions(GeneralAdvantageEstimation):
    """Run stock GAE and retain the old critic output for value clipping."""

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module,
        episodes: List[EpisodeType],
        batch: Dict[str, Any],
        **kwargs,
    ):
        batch = super().__call__(
            rl_module=rl_module, episodes=episodes, batch=batch, **kwargs
        )

        # GAE above uses compute_values(), which is denormalized.  The MAPPO
        # value clip, however, compares the critic's old and new *normalized*
        # outputs.  The Learner has not optimized yet, so recomputing here is
        # the exact rollout-time network prediction needed by every PPO epoch.
        def _old_prediction(module_id, module):
            if module_id not in batch or not isinstance(module, ValueFunctionAPI):
                return None
            raw_module = (
                module.unwrapped() if hasattr(module, "unwrapped") else module
            )
            if not hasattr(raw_module, "compute_normalized_values"):
                return None
            with torch.no_grad():
                return raw_module.compute_normalized_values(batch[module_id]).detach()

        old_predictions = rl_module.foreach_module(
            func=_old_prediction, return_dict=True
        )
        for module_id, predictions in old_predictions.items():
            if predictions is not None:
                batch[module_id][OLD_NORMALIZED_VALUE_PREDS] = predictions
        return batch


class OmniPianoPPOTorchLearner(PPOTorchLearner):
    """PPO learner matching the implementation details used by MAPPO."""

    @override(PPOLearner)
    def build(self) -> None:
        super().build()
        if (
            self._learner_connector is not None
            and self.config.add_default_connectors_to_learner_pipeline
        ):
            self._learner_connector.remove(GeneralAdvantageEstimation)
            self._learner_connector.append(
                GAEWithOldValuePredictions(
                    gamma=self.config.gamma, lambda_=self.config.lambda_
                )
            )

    @override(TorchLearner)
    def configure_optimizers_for_module(
        self, module_id: ModuleID, config: PPOConfig = None
    ) -> None:
        module = self.module[module_id].unwrapped()
        if not isinstance(module, CtdePPOTorchRLModule):
            raise TypeError(
                "OmniPianoPPOTorchLearner requires CtdePPOTorchRLModule, got "
                f"{type(module).__name__} for module {module_id!r}"
            )

        options = dict(config.learner_config_dict)
        actor_lr = float(config.lr)
        critic_lr = float(options.get("critic_lr", actor_lr))
        adam_epsilon = float(options.get("adam_epsilon", 1e-5))
        actor_params = module.actor_parameters()
        critic_params = module.critic_parameters()

        # Registering two optimizers makes RLlib apply grad_clip independently
        # to these disjoint parameter sets and exposes separate norm metrics.
        actor_optimizer = torch.optim.Adam(
            actor_params, lr=actor_lr, eps=adam_epsilon
        )
        critic_optimizer = torch.optim.Adam(
            critic_params, lr=critic_lr, eps=adam_epsilon
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="actor",
            optimizer=actor_optimizer,
            params=actor_params,
            lr_or_lr_schedule=config.lr,
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="critic",
            optimizer=critic_optimizer,
            params=critic_params,
            lr_or_lr_schedule=critic_lr,
        )

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        module = self.module[module_id].unwrapped()
        if not isinstance(module, CtdePPOTorchRLModule):
            raise TypeError(
                "custom PPO loss received an unsupported module: "
                f"{type(module).__name__}"
            )

        mask = None
        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK].bool()
            num_valid = torch.sum(mask)

            def possibly_masked_mean(data):
                return torch.sum(data[mask]) / num_valid

        else:
            possibly_masked_mean = torch.mean

        action_dist_class_train = module.get_train_action_dist_cls()
        action_dist_class_exploration = module.get_exploration_action_dist_cls()
        curr_action_dist = action_dist_class_train.from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        prev_action_dist = action_dist_class_exploration.from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )
        logp_ratio = torch.exp(
            curr_action_dist.logp(batch[Columns.ACTIONS])
            - batch[Columns.ACTION_LOGP]
        )

        if config.use_kl_loss:
            mean_kl_loss = possibly_masked_mean(
                prev_action_dist.kl(curr_action_dist)
            )
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = possibly_masked_mean(curr_entropy)
        advantages = batch[Postprocessing.ADVANTAGES]
        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages
            * torch.clamp(
                logp_ratio, 1.0 - config.clip_param, 1.0 + config.clip_param
            ),
        )

        if config.use_critic:
            if OLD_NORMALIZED_VALUE_PREDS not in batch:
                raise KeyError(
                    f"learner batch is missing {OLD_NORMALIZED_VALUE_PREDS!r}; "
                    "GAEWithOldValuePredictions was not installed"
                )
            value_targets = batch[Postprocessing.VALUE_TARGETS]
            normalizer_targets = (
                value_targets[mask] if mask is not None else value_targets
            )
            module.update_value_normalizer(normalizer_targets)
            normalized_targets = module.normalize_value_targets(value_targets)

            values = module.compute_normalized_values(batch)
            old_values = batch[OLD_NORMALIZED_VALUE_PREDS]
            clipped_values = old_values + torch.clamp(
                values - old_values,
                -config.vf_clip_param,
                config.vf_clip_param,
            )
            # Official MAPPO uses 0.5 * squared error and the pessimistic
            # maximum of the original and value-delta-clipped objectives.
            vf_loss_unclipped = 0.5 * (normalized_targets - values).square()
            vf_loss_clipped = 0.5 * (normalized_targets - clipped_values).square()
            vf_loss = torch.maximum(vf_loss_unclipped, vf_loss_clipped)
            mean_vf_loss = possibly_masked_mean(vf_loss)
            mean_vf_unclipped_loss = possibly_masked_mean(vf_loss_unclipped)
            value_clip_fraction = possibly_masked_mean(
                ((values - old_values).abs() > config.vf_clip_param).float()
            )
            denormalized_values = module.denormalize_values(values.detach())
        else:
            zero = torch.tensor(0.0, device=surrogate_loss.device)
            mean_vf_loss = mean_vf_unclipped_loss = value_clip_fraction = zero
            vf_loss = denormalized_values = zero

        total_loss = possibly_masked_mean(
            -surrogate_loss
            + config.vf_loss_coeff * vf_loss
            - (
                self.entropy_coeff_schedulers_per_module[
                    module_id
                ].get_current_value()
                * curr_entropy
            )
        )
        if config.use_kl_loss:
            total_loss += self.curr_kl_coeffs_per_module[module_id] * mean_kl_loss

        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -possibly_masked_mean(surrogate_loss),
                VF_LOSS_KEY: mean_vf_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: mean_vf_unclipped_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], denormalized_values
                ),
                ENTROPY_KEY: mean_entropy,
                LEARNER_RESULTS_KL_KEY: mean_kl_loss,
                VALUE_NORM_MEAN_KEY: module.value_norm_mean.detach(),
                VALUE_NORM_STD_KEY: module.value_norm_std.detach(),
                VALUE_CLIP_FRACTION_KEY: value_clip_fraction,
            },
            key=module_id,
            window=1,
        )
        return total_loss
