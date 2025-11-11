"""Actor-Critic policy with action masking support for invalid actions.

This policy applies action masks during both action selection and policy evaluation,
ensuring masked actions have zero probability of being selected.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


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
        action_mask = None
        if isinstance(observation, dict):
            action_mask = observation.get("action_mask", None)
            # Create a copy without action_mask for feature extraction
            obs_for_features = {
                k: v for k, v in observation.items() if k != "action_mask"
            }
        else:
            obs_for_features = observation

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

        # Apply action mask if present
        if action_mask is not None:
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

        # Validate features after extraction
        if torch.isnan(features_pi).any():
            nan_mask = torch.isnan(features_pi)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_FWD] NaN in policy features from extract_features in batch indices: {batch_indices.tolist()}"
            )
        if torch.isnan(features_vf).any():
            nan_mask = torch.isnan(features_vf)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_FWD] NaN in value features from extract_features in batch indices: {batch_indices.tolist()}"
            )

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

        # Validate latent after MLP extractor
        if torch.isnan(latent_pi).any():
            nan_mask = torch.isnan(latent_pi)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_FWD] NaN in latent_pi from mlp_extractor in batch indices: {batch_indices.tolist()}"
            )

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Validate action logits before masking
        if torch.isnan(action_logits).any():
            nan_mask = torch.isnan(action_logits)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_FWD] NaN in action_logits BEFORE masking in batch indices: {batch_indices.tolist()}. "
                f"Logits shape: {action_logits.shape}, range: [{action_logits[~nan_mask].min():.4f}, {action_logits[~nan_mask].max():.4f}]"
            )

        # Apply action mask if present
        if action_mask is not None:
            action_logits = self._apply_action_mask(action_logits, action_mask)

            # Validate action logits after masking
            if torch.isnan(action_logits).any():
                nan_mask = torch.isnan(action_logits)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[POLICY_FWD] NaN in action_logits AFTER masking in batch indices: {batch_indices.tolist()}. "
                    f"Logits shape: {action_logits.shape}. "
                    f"Action mask for failed batches: {action_mask[batch_indices].cpu().numpy()}"
                )

        # Create distribution and sample
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
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

        # Validate features after extraction
        if torch.isnan(features_pi).any():
            nan_mask = torch.isnan(features_pi)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_EVAL] NaN in policy features from extract_features in batch indices: {batch_indices.tolist()}"
            )
        if torch.isnan(features_vf).any():
            nan_mask = torch.isnan(features_vf)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_EVAL] NaN in value features from extract_features in batch indices: {batch_indices.tolist()}"
            )

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

        # Validate latent after MLP extractor
        if torch.isnan(latent_pi).any():
            nan_mask = torch.isnan(latent_pi)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_EVAL] NaN in latent_pi from mlp_extractor in batch indices: {batch_indices.tolist()}"
            )

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Validate action logits before masking
        if torch.isnan(action_logits).any():
            nan_mask = torch.isnan(action_logits)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_EVAL] NaN in action_logits BEFORE masking in batch indices: {batch_indices.tolist()}"
            )

        # Apply action mask if present
        if action_mask is not None:
            action_logits = self._apply_action_mask(action_logits, action_mask)

            # Validate action logits after masking
            if torch.isnan(action_logits).any():
                nan_mask = torch.isnan(action_logits)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[POLICY_EVAL] NaN in action_logits AFTER masking in batch indices: {batch_indices.tolist()}. "
                    f"Action mask for failed batches: {action_mask[batch_indices].cpu().numpy()}"
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
        # Convert mask to tensor if needed
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask)

        # Ensure mask is boolean type (it might be int8 from numpy)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()

        # Ensure mask is on the same device as logits
        action_mask = action_mask.to(action_logits.device)

        # Handle batch dimension - mask might be (batch, actions) or just (actions,)
        if action_mask.dim() == 1:
            # Single mask for all batch elements, expand it
            action_mask = action_mask.unsqueeze(0).expand_as(action_logits)

        # Validate action mask - check for NaN and ensure at least one valid action per batch
        if torch.isnan(action_mask.float()).any():
            nan_mask = torch.isnan(action_mask.float())
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[POLICY_MASK] NaN detected in action_mask for batch indices: {batch_indices.tolist()}"
            )

        # Check if any batch has all actions masked (would lead to all -inf logits)
        valid_actions_per_batch = action_mask.sum(dim=1)
        all_masked_batches = torch.where(valid_actions_per_batch == 0)[0]
        if len(all_masked_batches) > 0:
            import sys

            print(
                f"\n{'=' * 60}\n"
                f"[POLICY_MASK_CRITICAL] All actions masked for batch indices: {all_masked_batches.tolist()}!\n"
                f"This should never happen due to ninja.py fallback.\n"
                f"Action masks: {action_mask[all_masked_batches].cpu().numpy()}\n"
                f"{'=' * 60}\n",
                file=sys.stderr,
            )
            raise ValueError(
                f"[POLICY_MASK] All actions masked for batch indices: {all_masked_batches.tolist()}. "
                f"This would create all -inf logits leading to NaN in distribution. "
                f"Action masks: {action_mask[all_masked_batches].cpu().numpy()}. "
                f"This indicates a bug in ninja.get_valid_action_mask() - the fallback should prevent this!"
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
