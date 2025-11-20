"""Actor-Critic policy with action masking support for invalid actions.

This policy applies action masks during both action selection and policy evaluation,
ensuring masked actions have zero probability of being selected.
"""

import logging

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

logger = logging.getLogger(__name__)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Actor-Critic policy with action masking support for invalid actions.

    This policy applies action masks during both action selection and policy evaluation,
    ensuring masked actions have zero probability of being selected.

    Action masking is particularly useful for:
    - Preventing selection of actions that have no effect (e.g., useless jumps)
    - Reducing exploration waste on invalid actions
    - Improving sample efficiency

    The implementation overrides the distribution creation to apply masks, which is
    the cleanest way to integrate with stable-baselines3's PPO algorithm.
    """

    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict action with action mask support.

        Overrides base ActorCriticPolicy._predict() to properly handle action_mask
        in observation dictionaries during inference (model.predict() calls this).

        Args:
            observation: Observation tensor or dict (may contain action_mask)
            deterministic: Whether to use deterministic action selection

        Returns:
            Predicted action tensor
        """
        # Extract action mask before feature extraction (if present)
        action_mask = observation.get("action_mask", None)
        if action_mask is None:
            raise ValueError("Action mask not found in observation")

        # Create a copy without action_mask for feature extraction
        obs_for_features = {k: v for k, v in observation.items() if k != "action_mask"}

        # Get device from model parameters
        device = next(self.parameters()).device

        # Convert numpy arrays to tensors if needed
        if isinstance(obs_for_features, dict):
            obs_for_features = {
                k: torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v
                for k, v in obs_for_features.items()
            }
        elif isinstance(obs_for_features, np.ndarray):
            obs_for_features = torch.as_tensor(obs_for_features, device=device)

        # Preprocess the observation if needed
        # Extract features (handles both shared and separate extractors)
        features = self.extract_features(obs_for_features, self.features_extractor)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, _ = features
        else:
            features_pi = features

        # Get policy latent using appropriate method
        if hasattr(self.mlp_extractor, "forward_policy"):
            latent_pi = self.mlp_extractor.forward_policy(features_pi)
        else:
            latent_pi, _ = self.mlp_extractor(features_pi)

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Apply action mask
        # Convert mask to tensor if needed
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.as_tensor(action_mask, device=device)
        elif not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, device=device)

        action_logits = self._apply_action_mask(action_logits, action_mask)

        # Create distribution and sample action
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)

        return actions

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None
    ) -> CategoricalDistribution:
        """
        Retrieve action distribution given the latent codes.

        This method is called by predict() path only. Since we don't have access
        to observations here, we can't apply masking. The masking is instead
        applied in forward(), evaluate_actions(), and _predict() which handle
        action masks properly.

        Args:
            latent_pi: Latent code for the actor
            latent_sde: Latent code for state-dependent exploration (not used for discrete)

        Returns:
            Action distribution
        """
        # Get action logits from the actor network
        action_logits = self.action_net(latent_pi)

        # Create and return distribution
        # Note: Masking handled in forward(), evaluate_actions(), and _predict()
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic).

        Args:
            obs: Observation
            deterministic: Whether to sample or use deterministic actions

        Returns:
            action, value and log probability of the action
        """
        # Extract action mask before feature extraction (if present)
        # This prevents feature extractor from trying to process it
        action_mask = None
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # VALIDATION: action_mask must be present in dictionary observations
            if action_mask is None:
                raise ValueError(
                    "action_mask not found in observation dictionary! "
                    "This is required for masked action selection. "
                    f"Available keys: {list(obs.keys())}"
                )
            # Debug logging for action_mask extraction
            logger.debug(
                f"[forward] Extracted action_mask from observation dict. "
                f"Mask type: {type(action_mask)}, "
                f"Mask shape: {getattr(action_mask, 'shape', 'N/A')}, "
                f"Mask dtype: {getattr(action_mask, 'dtype', 'N/A')}"
            )
            # Create a copy without action_mask for feature extraction
            # to avoid issues if feature extractor expects certain keys
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Preprocess the observation if needed
        # Extract features (handles both shared and separate extractors)
        features = self.extract_features(obs_for_features)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, features_vf = features
        else:
            features_pi = features_vf = features

        # Extract latents using separate forward methods if available (DeepResNet), otherwise use standard approach
        if hasattr(self.mlp_extractor, "forward_policy") and hasattr(
            self.mlp_extractor, "forward_value"
        ):
            latent_pi = self.mlp_extractor.forward_policy(features_pi)
            latent_vf = self.mlp_extractor.forward_value(features_vf)
        else:
            # Standard MLP extractor: extract both latents separately
            latent_pi, _ = self.mlp_extractor(features_pi)
            _, latent_vf = self.mlp_extractor(features_vf)

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Apply action mask if present
        if action_mask is not None:
            # Validate action_mask freshness and correctness before applying
            self._validate_action_mask_freshness(obs, action_mask)
            logger.debug(
                f"[forward] Applying action_mask to logits. "
                f"Logits shape: {action_logits.shape}, "
                f"Mask shape: {getattr(action_mask, 'shape', 'N/A')}"
            )
            action_logits = self._apply_action_mask(action_logits, action_mask)
            logger.debug(
                f"[forward] Action mask applied. "
                f"Masked logits shape: {action_logits.shape}, "
                f"Valid actions per sample: {action_mask.sum(dim=-1) if hasattr(action_mask, 'sum') and action_mask.dim() > 1 else 'N/A'}"
            )

        # Create distribution and sample
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        logger.debug(
            f"[forward] Sampled actions: {actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions}, "
            f"Deterministic: {deterministic}"
        )
        log_prob = distribution.log_prob(actions)

        # Get values
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        Args:
            obs: Observation
            actions: Actions

        Returns:
            estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Extract action mask before feature extraction (if present)
        # This prevents feature extractor from trying to process it
        action_mask = None
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # VALIDATION: action_mask must be present in dictionary observations
            if action_mask is None:
                raise ValueError(
                    "action_mask not found in observation dictionary during evaluate_actions! "
                    "This is required for masked action selection. "
                    f"Available keys: {list(obs.keys())}"
                )
            # Debug logging for action_mask extraction
            logger.debug(
                f"[evaluate_actions] Extracted action_mask from observation dict. "
                f"Mask type: {type(action_mask)}, "
                f"Mask shape: {getattr(action_mask, 'shape', 'N/A')}, "
                f"Actions to evaluate: {actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions}"
            )
            # Create a copy without action_mask for feature extraction
            # to avoid issues if feature extractor expects certain keys
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Preprocess the observation if needed
        # Extract features (handles both shared and separate extractors)
        features = self.extract_features(obs_for_features)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, features_vf = features
        else:
            features_pi = features_vf = features

        # Extract latents using separate forward methods if available (DeepResNet), otherwise use standard approach
        if hasattr(self.mlp_extractor, "forward_policy") and hasattr(
            self.mlp_extractor, "forward_value"
        ):
            latent_pi = self.mlp_extractor.forward_policy(features_pi)
            latent_vf = self.mlp_extractor.forward_value(features_vf)
        else:
            # Standard MLP extractor: extract both latents separately
            latent_pi, _ = self.mlp_extractor(features_pi)
            _, latent_vf = self.mlp_extractor(features_vf)

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Apply action mask if present
        if action_mask is not None:
            # Validate action_mask freshness and correctness before applying
            self._validate_action_mask_freshness(obs, action_mask)
            logger.debug(
                f"[evaluate_actions] Applying action_mask to logits. "
                f"Logits shape: {action_logits.shape}, "
                f"Mask shape: {getattr(action_mask, 'shape', 'N/A')}"
            )
            action_logits = self._apply_action_mask(action_logits, action_mask)
            logger.debug(
                f"[evaluate_actions] Action mask applied. "
                f"Evaluating actions: {actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions}"
            )

        # Create distribution and evaluate
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Get values
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def _apply_action_mask(
        self, action_logits: torch.Tensor, action_mask: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Apply action mask to logits by setting masked actions to -inf.

        Args:
            action_logits: Raw action logits from actor network
            action_mask: Boolean mask where True = valid, False = invalid
                        (can be int8/uint8, will be converted to bool)

        Returns:
            Masked action logits
        """
        # VALIDATION: Check that action_mask is provided
        if action_mask is None:
            raise ValueError(
                "action_mask is None! Action masking must always be applied during training. "
                "This indicates a bug in observation preparation."
            )

        # Convert mask to tensor if needed
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask)

        # Ensure mask is boolean type (it might be int8 from numpy)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()

        # Ensure mask is on the same device as logits
        action_mask = action_mask.to(action_logits.device)

        # VALIDATION: Check action mask has correct number of actions
        if action_mask.shape[-1] != 6:
            raise ValueError(
                f"action_mask has wrong shape! Expected 6 actions, got {action_mask.shape[-1]}. "
                f"Full mask shape: {action_mask.shape}, logits shape: {action_logits.shape}"
            )

        # VALIDATION: Check that each sample has at least one valid action
        if action_mask.dim() == 1:
            if not action_mask.any():
                raise ValueError(
                    "action_mask has no valid actions! This should never happen. "
                    f"Mask: {action_mask}"
                )
        else:
            has_valid = action_mask.any(dim=-1)
            if not has_valid.all():
                invalid_indices = (~has_valid).nonzero(as_tuple=True)[0]
                raise ValueError(
                    f"action_mask has samples with no valid actions at indices {invalid_indices}! "
                    f"This should never happen. Mask shape: {action_mask.shape}"
                )

        # CRITICAL: Validate batch sizes match BEFORE attempting any broadcasting
        if action_mask.dim() > 1 and action_logits.dim() > 1:
            if action_mask.shape[0] != action_logits.shape[0]:
                raise ValueError(
                    f"BATCH SIZE MISMATCH: action_mask batch {action_mask.shape[0]} != "
                    f"logits batch {action_logits.shape[0]}. Shape mismatch indicates "
                    f"mask wasn't properly preserved from environment observations."
                )

        # Handle batch dimension - mask might be (batch, actions) or just (actions,)
        if action_mask.dim() == 1:
            # Single mask for all batch elements, expand it
            # VALIDATION: This should only happen during inference, not training
            action_mask = action_mask.unsqueeze(0).expand_as(action_logits)
        else:
            # VALIDATION: Batch dimensions must match
            if action_mask.shape[0] != action_logits.shape[0]:
                raise ValueError(
                    f"Batch size mismatch! action_mask batch: {action_mask.shape[0]}, "
                    f"logits batch: {action_logits.shape[0]}. This indicates a bug in "
                    f"vectorized environment handling or observation batching."
                )

        # Set masked actions to -inf (will have zero probability after softmax)
        masked_logits = torch.where(
            action_mask,
            action_logits,
            torch.tensor(
                float("-inf"), dtype=action_logits.dtype, device=action_logits.device
            ),
        )

        return masked_logits

    def _validate_action_mask_freshness(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action_mask: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """
        Validate that action_mask matches current observation state.

        This method performs comprehensive validation to detect when action_mask
        is stale, incorrect, or mismatched with the observation batch.

        Args:
            obs: Observation (dict or tensor)
            action_mask: Action mask to validate

        Raises:
            ValueError: If action_mask is invalid, stale, or mismatched
        """
        if action_mask is None:
            raise ValueError(
                "action_mask is None during validation! "
                "This indicates action_mask was not provided in the observation."
            )

        # Convert to tensor for validation if needed
        if not isinstance(action_mask, torch.Tensor):
            if isinstance(action_mask, np.ndarray):
                action_mask_tensor = torch.from_numpy(action_mask)
            else:
                action_mask_tensor = torch.tensor(action_mask)
        else:
            action_mask_tensor = action_mask

        # Ensure boolean dtype
        if action_mask_tensor.dtype != torch.bool:
            action_mask_tensor = action_mask_tensor.bool()

        # Validate shape
        if action_mask_tensor.dim() == 1:
            # Single mask - should have 6 actions
            if action_mask_tensor.shape[0] != 6:
                raise ValueError(
                    f"action_mask has wrong shape for single mask! "
                    f"Expected (6,), got {action_mask_tensor.shape}"
                )
        elif action_mask_tensor.dim() == 2:
            # Batched masks - should be (batch_size, 6)
            if action_mask_tensor.shape[1] != 6:
                raise ValueError(
                    f"action_mask has wrong shape for batched masks! "
                    f"Expected (batch_size, 6), got {action_mask_tensor.shape}"
                )

            # Check batch size matches observation batch size
            if isinstance(obs, dict):
                # Try to infer batch size from observation dict
                batch_size = None
                for key, value in obs.items():
                    if key != "action_mask" and isinstance(value, torch.Tensor):
                        if value.dim() > 0:
                            batch_size = value.shape[0]
                            break

                if batch_size is not None and action_mask_tensor.shape[0] != batch_size:
                    raise ValueError(
                        f"action_mask batch size mismatch! "
                        f"Mask batch: {action_mask_tensor.shape[0]}, "
                        f"Observation batch: {batch_size}. "
                        f"This indicates a synchronization bug between environments."
                    )
        else:
            raise ValueError(
                f"action_mask has invalid number of dimensions! "
                f"Expected 1 or 2, got {action_mask_tensor.dim()}"
            )

        # Validate that each sample has at least one valid action
        if action_mask_tensor.dim() == 1:
            if not action_mask_tensor.any():
                raise ValueError(
                    "action_mask has no valid actions! "
                    f"Mask: {action_mask_tensor}. "
                    "This should never happen - at least one action must be valid."
                )
        else:
            has_valid = action_mask_tensor.any(dim=1)
            if not has_valid.all():
                invalid_indices = (~has_valid).nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    f"action_mask has samples with no valid actions at indices {invalid_indices}! "
                    f"Mask shape: {action_mask_tensor.shape}. "
                    "This indicates corrupted or stale action masks."
                )

        # Log validation success at debug level
        logger.debug(
            f"action_mask validation passed. "
            f"Shape: {action_mask_tensor.shape}, "
            f"Valid actions per sample: {action_mask_tensor.sum(dim=-1) if action_mask_tensor.dim() > 1 else action_mask_tensor.sum()}"
        )
